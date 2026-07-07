"""Adjoint Phase 0 spike: differentiable Ikarus forward solve in JAX.

Mirrors ikarus/core/solver.py exactly (engineering-convention bridge, li
factorization, forward-branch selection, Redheffer cascade) for a 1-D
high-contrast TM grating, plus a custom differentiable non-Hermitian eig
(Boeddeker 2019 eq. 4.77 with scale-aware Lorentzian broadening, mirroring
FMMax's eig.py conventions).

GATES:
  A. forward R_total matches the NumPy core to <= 1e-12;
  B. jax.grad of R w.r.t. per-pixel density rho and w.r.t. layer height
     matches central finite differences to rel. err <= 1e-5.
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from ikarus.core.fourier import HarmonicGrid
from ikarus.core import solver as npsolver

# ---------------------------------------------------------------------------
# Differentiable non-Hermitian eig (custom VJP)
# ---------------------------------------------------------------------------
_EPS_REL, _EPS_MIN = 1e-12, 1e-24


@jax.custom_vjp
def eig(matrix):
    return jax.lax.linalg.eig(matrix, compute_left_eigenvectors=False)


def _eig_fwd(matrix):
    w, v = jax.lax.linalg.eig(matrix, compute_left_eigenvectors=False)
    return (w, v), (w, v)


def _eig_bwd(res, grads):
    w, v = res
    gw, gv = grads
    delta = w[None, :] - w[:, None]
    rng = jnp.max(jnp.abs(delta) ** 2)
    eps = jnp.maximum(_EPS_REL * rng, _EPS_MIN)
    F = delta.conj() / (jnp.abs(delta) ** 2 + eps)
    n = w.shape[0]
    di = jnp.arange(n)
    F = F.at[di, di].set(0.0)
    gw_c, gv_c = jnp.conj(gw), jnp.conj(gv)
    vH = jnp.conj(v.T)
    eye = jnp.eye(n, dtype=bool)
    rhs = (jnp.diag(gw_c)
           + jnp.conj(F) * (vH @ gv_c)
           - jnp.conj(F) * (vH @ v) @ jnp.where(eye, jnp.real(vH @ gv_c), 0.0)
           ) @ vH
    gm = jnp.linalg.solve(vH, rhs)
    return (jnp.conj(gm),)


eig.defvjp(_eig_fwd, _eig_bwd)

# ---------------------------------------------------------------------------
# JAX mirror of the solver pipeline (static bookkeeping stays NumPy)
# ---------------------------------------------------------------------------
_ANOMALY_LOSS = 1e-9j


def _rdiv(M, K):
    return jnp.linalg.solve(K.T, M.T).T


def convolution_matrix_j(cell, grid: HarmonicGrid):
    nx, ny = cell.shape
    coeffs = jnp.fft.fftshift(jnp.fft.fft2(cell)) / (nx * ny)
    cx, cy = nx // 2, ny // 2
    p, q = grid.index_arrays()                       # static numpy
    dp = p[:, None] - p[None, :]
    dq = q[:, None] - q[None, :]
    return coeffs[cx + dp, cy + dq]


def mixed_convolution_j(ge, grid: HarmonicGrid, axis: str):
    nx, ny = ge.shape
    binv = 1.0 / ge
    p, q = grid.index_arrays()
    if axis == "x":
        ox, Mx, cx = grid.orders_x, grid.n_orders_x, nx // 2
        idx = cx + (ox[:, None] - ox[None, :])
        cof = jnp.fft.fftshift(jnp.fft.fft(binv, axis=0), axes=0) / nx
        toe = jnp.moveaxis(cof[idx], 2, 0)
        seq = jnp.linalg.inv(toe)
        seq_hat = jnp.fft.fft(seq, axis=0) / ny
        dy = (q[:, None] - q[None, :]) % ny
        return seq_hat[dy, p[:, None] + Mx, p[None, :] + Mx]
    oy, My, cy = grid.orders_y, grid.n_orders_y, ny // 2
    idy = cy + (oy[:, None] - oy[None, :])
    cof = jnp.fft.fftshift(jnp.fft.fft(binv, axis=1), axes=1) / ny
    seq = jnp.linalg.inv(cof[:, idy])
    seq_hat = jnp.fft.fft(seq, axis=0) / nx
    dx = (p[:, None] - p[None, :]) % nx
    return seq_hat[dx, q[:, None] + My, q[None, :] + My]


def forward_branch_j(eigvals):
    lam = jnp.sqrt(eigvals.astype(complex))
    marginal = jnp.abs(lam.real) <= 1e-8 * jnp.abs(lam)
    flip = (lam.real < 0) & ~marginal
    flip = flip | (marginal & (lam.imag < 0))
    lam = jnp.where(flip, -lam, lam)
    tiny = jnp.abs(lam) < 1e-7
    return jnp.where(tiny, 1e-7 * (1.0 + 1j), lam)


def wavevector_matrices_j(grid, kx0, ky0, period_x, period_y, wavelength):
    p, q = grid.index_arrays()
    kx = kx0 - p * (wavelength / period_x)
    ky = ky0 - q * (wavelength / period_y)
    return kx.astype(complex), ky.astype(complex)      # 1-D diagonals


def uniform_modes_j(eps, kx, ky):
    P = kx.shape[0]
    lam_order = forward_branch_j(kx ** 2 + ky ** 2 - eps)
    lam = jnp.concatenate([lam_order, lam_order])
    Q = jnp.block([[jnp.diag(kx * ky), jnp.diag(eps - kx * kx)],
                   [jnp.diag(ky * ky - eps), jnp.diag(-ky * kx)]])
    V = Q / lam[None, :]
    kz = -1j * lam_order
    return V, kz


def layer_modes_j(ERC, kx, ky, Exx, Eyy):
    P = kx.shape[0]
    I = jnp.eye(P, dtype=complex)
    Einv = jnp.linalg.inv(ERC)
    kxc, kyc = kx[:, None], ky[None, :]
    Pm = jnp.block([[kxc * Einv * kyc, I - kxc * Einv * kxc.T],
                    [(kyc.T * Einv * kyc) - I, -kyc.T * Einv * kxc.T]])
    Qm = jnp.block([[jnp.diag(kx * ky), Eyy - jnp.diag(kx * kx)],
                    [jnp.diag(ky * ky) - Exx, -jnp.diag(ky * kx)]])
    eigvals, W = eig(Pm @ Qm)
    lam = forward_branch_j(eigvals)
    V = (Qm @ W) / lam[None, :]
    return W, V, lam


def layer_smatrix_j(ERC, kx, ky, k0L, V0, Exx, Eyy):
    W, V, lam = layer_modes_j(ERC, kx, ky, Exx, Eyy)
    Winv = jnp.linalg.inv(W)
    VinvV0 = jnp.linalg.solve(V, V0)
    A = Winv + VinvV0
    B = Winv - VinvV0
    x = jnp.exp(-lam * k0L)[:, None]
    Ainv = jnp.linalg.inv(A)
    XB = x * B
    fac = jnp.linalg.inv(A - XB @ Ainv @ XB)
    S11 = fac @ (XB @ Ainv @ (x * A) - B)
    S12 = fac @ (x * (A - B @ Ainv @ B))
    return S11, S12, S12, S11


def region_smatrix_j(eps, kx, ky, V0inv, kind):
    V, kz = uniform_modes_j(eps, kx, ky)
    M = V0inv @ V
    I = jnp.eye(M.shape[0], dtype=complex)
    A, B = I + M, I - M
    Ainv = jnp.linalg.inv(A)
    if kind == "ref":
        return (-Ainv @ B, 2 * Ainv, 0.5 * (A - B @ Ainv @ B), B @ Ainv), kz
    return (B @ Ainv, 0.5 * (A - B @ Ainv @ B), 2 * Ainv, -Ainv @ B), kz


def redheffer_j(a, b):
    a11, a12, a21, a22 = a
    b11, b12, b21, b22 = b
    I = jnp.eye(a11.shape[0], dtype=complex)
    D = _rdiv(a12, I - b11 @ a22)
    F = _rdiv(b21, I - a22 @ b11)
    return (a11 + D @ b11 @ a21, D @ b12, F @ a21, b22 + F @ a22 @ b12)


def solve_R_total(eps_grid, height, eps_ref, eps_trn, grid, period_x, period_y,
                  wavelength, pol_xy):
    """R_total of cover / one patterned layer / substrate (li factorization)."""
    kx, ky = wavevector_matrices_j(grid, 0.0, 0.0, period_x, period_y, wavelength)
    V0, _ = uniform_modes_j(1.0 - _ANOMALY_LOSS, kx, ky)
    V0inv = jnp.linalg.inv(V0)
    k0 = 2.0 * jnp.pi / wavelength

    eps_ref_e = jnp.conj(eps_ref) - _ANOMALY_LOSS
    eps_trn_e = jnp.conj(eps_trn) - _ANOMALY_LOSS
    ge = jnp.conj(eps_grid) - _ANOMALY_LOSS
    ERC = convolution_matrix_j(ge, grid)
    Exx = mixed_convolution_j(ge, grid, "x")
    Eyy = mixed_convolution_j(ge, grid, "y")

    S_ref, kz_ref = region_smatrix_j(eps_ref_e, kx, ky, V0inv, "ref")
    S_lay = layer_smatrix_j(ERC, kx, ky, k0 * height, V0, Exx, Eyy)
    S_trn, kz_trn = region_smatrix_j(eps_trn_e, kx, ky, V0inv, "trn")
    S = redheffer_j(redheffer_j(S_ref, S_lay), S_trn)

    P = grid.size
    delta = jnp.zeros(P, dtype=complex).at[grid.zero_order_index()].set(1.0)
    px, py = jnp.conj(pol_xy[0]), jnp.conj(pol_xy[1])
    c_src = jnp.concatenate([px * delta, py * delta])
    c_ref = S[0] @ c_src
    rx, ry = c_ref[:P], c_ref[P:]
    rz = -(kx * rx + ky * ry) / kz_ref

    kz_inc = jnp.sqrt(eps_ref_e + 0j)
    kz_inc = jnp.where(kz_inc.real < 0, -kz_inc, kz_inc)
    R2 = jnp.abs(rx) ** 2 + jnp.abs(ry) ** 2 + jnp.abs(rz) ** 2
    return jnp.sum(R2 * jnp.real(kz_ref) / jnp.real(kz_inc))


# ---------------------------------------------------------------------------
# The spike case: validated 1-D high-contrast TM grating
# ---------------------------------------------------------------------------
NX, NY = 512, 2
PERIOD, WL, H = 400e-9, 700e-9, 300e-9
N_HI = 3.5
M = 12
grid = HarmonicGrid(M, 0)
POL = (1.0 + 0j, 0.0 + 0j)          # E || x (TM for an x-varying grating)


def eps_from_rho(rho):
    """Density rho (NX,) -> physics-convention eps grid (NX, NY)."""
    line = 1.0 + rho * (N_HI ** 2 - 1.0)
    return jnp.tile(line[:, None], (1, NY)).astype(complex)


def R_of_rho(rho, height=H):
    return solve_R_total(eps_from_rho(rho), height, 1.0 + 0j, 1.0 + 0j,
                         grid, PERIOD, PERIOD, WL, POL)


def numpy_reference(rho, height=H):
    eps_grid = np.array(eps_from_rho(jnp.asarray(rho)))
    sol = npsolver.solve_stack(
        eps_grids=[eps_grid], heights=[height], eps_ref=1.0, eps_trn=1.0,
        grid=grid, kx0=0.0, ky0=0.0, period_x=PERIOD, period_y=PERIOD,
        wavelength=WL, polarization_xy=(1.0, 0.0), factorization="li")
    return sol.R_total


if __name__ == "__main__":
    rng = np.random.default_rng(7)
    rho0 = np.zeros(NX)
    rho0[NX // 2:] = 1.0                                   # 50% duty binary
    rho_smooth = np.clip(rho0 + 0.15 * rng.standard_normal(NX), 0.05, 0.95)

    # ---- GATE A: forward agreement -------------------------------------
    print("GATE A -- forward vs NumPy core")
    for name, rho in (("binary 50% duty", rho0), ("randomized gray", rho_smooth)):
        Rj = float(R_of_rho(jnp.asarray(rho)))
        Rn = numpy_reference(rho)
        print(f"  {name:18s} jax={Rj:.12f}  numpy={Rn:.12f}  |d|={abs(Rj-Rn):.2e}")

    # ---- GATE B1: gradient w.r.t. pixel densities ----------------------
    print("GATE B1 -- d(R)/d(rho_i) vs central finite differences")
    g = np.array(jax.grad(R_of_rho)(jnp.asarray(rho_smooth)))
    step = 1e-5
    idxs = [37, 128, 250, 384, 470]
    worst = 0.0
    for i in idxs:
        rp = rho_smooth.copy(); rp[i] += step
        rm = rho_smooth.copy(); rm[i] -= step
        fd = (numpy_reference(rp) - numpy_reference(rm)) / (2 * step)
        rel = abs(g[i] - fd) / max(abs(fd), 1e-12)
        worst = max(worst, rel)
        print(f"  pixel {i:3d}: adjoint={g[i]:+.8e}  FD={fd:+.8e}  rel={rel:.2e}")
    print(f"  worst pixel rel err: {worst:.2e}")

    # ---- GATE B2: gradient w.r.t. layer height -------------------------
    print("GATE B2 -- d(R)/d(height) vs central finite differences")
    gh = float(jax.grad(R_of_rho, argnums=1)(jnp.asarray(rho_smooth), H))
    hstep = 1e-12                                          # meters
    fdh = (numpy_reference(rho_smooth, H + hstep)
           - numpy_reference(rho_smooth, H - hstep)) / (2 * hstep)
    relh = abs(gh - fdh) / abs(fdh)
    print(f"  adjoint={gh:+.6e}  FD={fdh:+.6e}  rel={relh:.2e}")

    # ---- adjoint cost sanity: one grad = O(1) forward solves ------------
    import time
    f = jax.jit(R_of_rho)
    gf = jax.jit(jax.grad(R_of_rho))
    _ = float(f(jnp.asarray(rho_smooth))); _ = np.array(gf(jnp.asarray(rho_smooth)))
    t0 = time.perf_counter(); [float(f(jnp.asarray(rho_smooth))) for _ in range(5)]
    tf = (time.perf_counter() - t0) / 5
    t0 = time.perf_counter(); [np.array(gf(jnp.asarray(rho_smooth))) for _ in range(5)]
    tg = (time.perf_counter() - t0) / 5
    print(f"TIMING: forward {tf*1e3:.1f} ms | forward+grad(512 DOFs) {tg*1e3:.1f} ms "
          f"| ratio {tg/tf:.2f}x")
