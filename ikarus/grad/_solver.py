"""Differentiable (JAX) mirror of the Ikarus forward solve.

This is a line-for-line reimplementation of :mod:`ikarus.core.solver` in JAX --
the engineering-conjugation bridge, the Fourier factorizations (Laurent, Li
two-step, normal-vector), the forward-branch selection and the Redheffer
S-matrix cascade -- so that ``jax.grad`` of any figure of merit built from the
result *is* the adjoint method: the gradient with respect to every input pixel
costs about one extra forward solve, independent of the DOF count.

Pinned against the NumPy core to ~1e-13 by ``ikarus/tests/test_grad.py``; the
NumPy core remains the canonical implementation (this module must follow it,
never the other way around).

Notes specific to the differentiable path
-----------------------------------------
* **Every layer goes through the factorized-operator path** (no uniform-layer
  short-circuit): under JAX tracing we cannot branch on traced values, and the
  factorizations all reduce exactly to Laurent for a uniform grid, so the
  result is identical -- verified by the pinning tests.
* **The normal-vector tangent field is a constant, not a variable.**  Under
  topology optimization the field would depend on the density, but its
  contribution to the gradient is a second-order effect and a first-order
  source of noise; we ``stop_gradient`` it.  Under ``jit`` the field cannot be
  computed from a traced grid at all (it uses SciPy), so callers precompute it
  with :func:`tangent_fields_for` outside the traced region -- typically once
  per optimizer iteration from the current density.
* **Anisotropic layers are not supported yet** (raise ``NotImplementedError``);
  the gradient path is isotropic scalar grids for now.
"""

from __future__ import annotations

from typing import NamedTuple, Optional, Sequence

import numpy as np
import jax
import jax.numpy as jnp

from ..core.fourier import HarmonicGrid
from ._eig import eig

_ANOMALY_LOSS = 1e-9j


# ---------------------------------------------------------------------------
# Small linear-algebra helpers (mirror solver.py)
# ---------------------------------------------------------------------------
def _rdiv(M, K):
    """``M @ inv(K)`` via a solve."""
    return jnp.linalg.solve(K.T, M.T).T


def _forward_branch(eigvals):
    """Forward/outgoing modal eigenvalues ``lam = sqrt(eigvals)`` (see core)."""
    lam = jnp.sqrt(eigvals.astype(complex))
    marginal = jnp.abs(lam.real) <= 1e-8 * jnp.abs(lam)
    flip = (lam.real < 0) & ~marginal
    flip = flip | (marginal & (lam.imag < 0))
    lam = jnp.where(flip, -lam, lam)
    # Rayleigh/Wood grazing regularization.
    return jnp.where(jnp.abs(lam) < 1e-7, 1e-7 * (1.0 + 1j), lam)


# ---------------------------------------------------------------------------
# Fourier machinery (mirror fourier.py / solver.py; index bookkeeping stays
# static NumPy so only the field data is traced)
# ---------------------------------------------------------------------------
def convolution_matrix(cell, grid: HarmonicGrid):
    nx, ny = cell.shape
    coeffs = jnp.fft.fftshift(jnp.fft.fft2(cell)) / (nx * ny)
    cx, cy = nx // 2, ny // 2
    p, q = grid.index_arrays()
    dp = p[:, None] - p[None, :]
    dq = q[:, None] - q[None, :]
    if np.abs(dp).max() > cx or np.abs(dq).max() > cy:
        raise ValueError("cell resolution too coarse for the harmonic orders")
    return coeffs[cx + dp, cy + dq]


def _mixed_convolution(ge, grid: HarmonicGrid, axis: str):
    """Li's two-step rule: inverse rule along ``axis``, direct along the other."""
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


def _inplane_tensor_normal(ge, tx, ty, grid: HarmonicGrid):
    """Normal-vector (FFF) in-plane operators (mirror _normalvector.py).

    ``tx``/``ty`` is the (constant) tangent field; returns Ikarus-convention
    ``(Exx, Eyy, Exy, Eyx)``.
    """
    tx = jax.lax.stop_gradient(jnp.asarray(tx))
    ty = jax.lax.stop_gradient(jnp.asarray(ty))
    cm = lambda f: convolution_matrix(jnp.asarray(f, dtype=complex), grid)
    eps_hat = cm(ge)
    eta_inv = jnp.linalg.inv(cm(1.0 / ge))
    delta = eps_hat - eta_inv

    denom = jnp.abs(tx) ** 2 + jnp.abs(ty) ** 2
    denom = jnp.where(jnp.isclose(denom, 0.0), 1.0, denom)
    Pyy = jnp.abs(tx) ** 2 / denom
    Pyx = jnp.conj(tx) * ty / denom
    Pxy = tx * jnp.conj(ty) / denom
    Pxx = jnp.abs(ty) ** 2 / denom

    E00 = eps_hat - delta @ cm(Pyy)          # eps_yy
    E01 = -delta @ cm(Pyx)                   # eps_yx
    E10 = -delta @ cm(Pxy)                   # eps_xy
    E11 = eps_hat - delta @ cm(Pxx)          # eps_xx
    return E11, E00, E10, E01                # (Exx, Eyy, Exy, Eyx)


# ---------------------------------------------------------------------------
# Modes and scattering matrices (mirror solver.py)
# ---------------------------------------------------------------------------
def _uniform_modes(eps, kx, ky):
    lam_order = _forward_branch(kx ** 2 + ky ** 2 - eps)
    lam = jnp.concatenate([lam_order, lam_order])
    Q = jnp.block([[jnp.diag(kx * ky), jnp.diag(eps - kx * kx)],
                   [jnp.diag(ky * ky - eps), jnp.diag(-ky * kx)]])
    V = Q / lam[None, :]
    kz = -1j * lam_order
    return V, kz


def _layer_modes(ERC, kx, ky, Exx, Eyy, Exy, Eyx):
    P = kx.shape[0]
    I = jnp.eye(P, dtype=complex)
    Einv = jnp.linalg.inv(ERC)
    kxc, kyc = kx[:, None], ky[None, :]
    Pm = jnp.block([[kxc * Einv * kyc, I - kxc * Einv * kxc.T],
                    [(kyc.T * Einv * kyc) - I, -kyc.T * Einv * kxc.T]])
    Q00 = jnp.diag(kx * ky)
    Q11 = -jnp.diag(ky * kx)
    if Eyx is not None:
        Q00 = Q00 - Eyx
    if Exy is not None:
        Q11 = Q11 + Exy
    Qm = jnp.block([[Q00, Eyy - jnp.diag(kx * kx)],
                    [jnp.diag(ky * ky) - Exx, Q11]])
    eigvals, W = eig(Pm @ Qm)
    lam = _forward_branch(eigvals)
    V = (Qm @ W) / lam[None, :]
    return W, V, lam


def _layer_smatrix(ERC, kx, ky, k0L, V0, Exx, Eyy, Exy, Eyx):
    W, V, lam = _layer_modes(ERC, kx, ky, Exx, Eyy, Exy, Eyx)
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
    return (S11, S12, S12, S11)


def _region_smatrix(eps, kx, ky, V0inv, kind):
    V, kz = _uniform_modes(eps, kx, ky)
    M = V0inv @ V
    I = jnp.eye(M.shape[0], dtype=complex)
    A, B = I + M, I - M
    Ainv = jnp.linalg.inv(A)
    if kind == "ref":
        return (-Ainv @ B, 2 * Ainv, 0.5 * (A - B @ Ainv @ B), B @ Ainv), kz
    return (B @ Ainv, 0.5 * (A - B @ Ainv @ B), 2 * Ainv, -Ainv @ B), kz


def _redheffer(a, b):
    a11, a12, a21, a22 = a
    b11, b12, b21, b22 = b
    I = jnp.eye(a11.shape[0], dtype=complex)
    D = _rdiv(a12, I - b11 @ a22)
    F = _rdiv(b21, I - a22 @ b11)
    return (a11 + D @ b11 @ a21, D @ b12, F @ a21, b22 + F @ a22 @ b12)


# ---------------------------------------------------------------------------
# Public solve
# ---------------------------------------------------------------------------
class GradSolution(NamedTuple):
    """Differentiable solve result (a JAX pytree; complex amplitudes are in the
    physics ``exp(-i w t)`` convention, matching the NumPy core)."""

    R_orders: jnp.ndarray
    T_orders: jnp.ndarray
    rx: jnp.ndarray
    ry: jnp.ndarray
    rz: jnp.ndarray
    tx: jnp.ndarray
    ty: jnp.ndarray
    tz: jnp.ndarray

    @property
    def R_total(self):
        return jnp.sum(self.R_orders)

    @property
    def T_total(self):
        return jnp.sum(self.T_orders)


def tangent_fields_for(eps_grids: Sequence, physics_convention: bool = True):
    """Precompute the normal-vector tangent fields for each layer (NumPy).

    Call this *outside* any jitted/traced region -- typically once per
    optimizer iteration from the current density -- and pass the result to
    :func:`solve` via ``tangent_fields``.  The fields are treated as constants
    (``stop_gradient``) inside the solve.
    """
    from ..core._normalvector import tangent_field
    fields = []
    for g in eps_grids:
        g = np.asarray(g)
        ge = (np.conj(g) if physics_convention else g) - _ANOMALY_LOSS
        fields.append(tangent_field(ge))
    return tuple(fields)


def solve(
    eps_grids: Sequence,
    heights: Sequence,
    eps_ref,
    eps_trn,
    grid: HarmonicGrid,
    kx0,
    ky0,
    period_x: float,
    period_y: float,
    wavelength: float,
    polarization_xy,
    factorization: str = "auto",
    tangent_fields: Optional[Sequence] = None,
) -> GradSolution:
    """Differentiable mirror of :func:`ikarus.core.solver.solve_stack`.

    Same conventions as the core: ``eps_grids`` are physics-convention
    ``(Nx, Ny)`` permittivity grids for the interior layers (JAX arrays or
    anything convertible; gradients flow through them), ``eps_ref``/``eps_trn``
    the scalar cover/substrate permittivities, ``kx0``/``ky0`` the normalized
    in-plane incident wavevector.

    ``factorization`` is ``"auto"`` (normal-vector; needs ``tangent_fields``
    under jit -- see :func:`tangent_fields_for`), ``"li"`` or ``"laurent"``.
    Anisotropic (tuple) layer entries are not supported yet.

    Returns a :class:`GradSolution`; build any real scalar figure of merit from
    it and take ``jax.grad`` -- that is the adjoint method.
    """
    for g in eps_grids:
        if isinstance(g, tuple):
            raise NotImplementedError(
                "anisotropic layers are not yet supported in ikarus.grad "
                "(use the NumPy core / GA path for tensor materials)")
    if factorization not in ("auto", "normal", "li", "laurent"):
        raise ValueError(f"unknown factorization {factorization!r}")
    use_normal = factorization in ("auto", "normal")

    if use_normal and tangent_fields is None:
        if any(isinstance(g, jax.core.Tracer) for g in eps_grids):
            raise ValueError(
                "factorization='auto'/'normal' under jit/grad tracing needs "
                "precomputed tangent fields: call ikarus.grad."
                "tangent_fields_for(...) outside the traced region and pass "
                "the result as tangent_fields=...")
        tangent_fields = tangent_fields_for(eps_grids)

    p, q = grid.index_arrays()
    kx = (kx0 - p * (wavelength / period_x)).astype(complex)
    ky = (ky0 - q * (wavelength / period_y)).astype(complex)
    V0, _ = _uniform_modes(1.0 - _ANOMALY_LOSS, kx, ky)
    V0inv = jnp.linalg.inv(V0)
    k0 = 2.0 * jnp.pi / wavelength

    # Physics -> engineering bridge (componentwise conjugation).
    eps_ref_e = jnp.conj(jnp.asarray(eps_ref, dtype=complex)) - _ANOMALY_LOSS
    eps_trn_e = jnp.conj(jnp.asarray(eps_trn, dtype=complex)) - _ANOMALY_LOSS

    S_ref, kz_ref = _region_smatrix(eps_ref_e, kx, ky, V0inv, "ref")
    S = S_ref
    for i, (g, h) in enumerate(zip(eps_grids, heights)):
        ge = jnp.conj(jnp.asarray(g, dtype=complex)) - _ANOMALY_LOSS
        ERC = convolution_matrix(ge, grid)
        if factorization == "laurent":
            Exx, Eyy, Exy, Eyx = ERC, ERC, None, None
        elif factorization == "li":
            Exx = _mixed_convolution(ge, grid, "x")
            Eyy = _mixed_convolution(ge, grid, "y")
            Exy, Eyx = None, None
        else:                                    # auto / normal
            tf = tangent_fields[i]
            Exx, Eyy, Exy, Eyx = _inplane_tensor_normal(ge, tf[0], tf[1], grid)
        S = _redheffer(S, _layer_smatrix(ERC, kx, ky, k0 * h, V0,
                                         Exx, Eyy, Exy, Eyx))
    S_trn, kz_trn = _region_smatrix(eps_trn_e, kx, ky, V0inv, "trn")
    S = _redheffer(S, S_trn)

    P = grid.size
    delta = jnp.zeros(P, dtype=complex).at[grid.zero_order_index()].set(1.0)
    px = jnp.conj(jnp.asarray(polarization_xy[0], dtype=complex))
    py = jnp.conj(jnp.asarray(polarization_xy[1], dtype=complex))
    c_src = jnp.concatenate([px * delta, py * delta])

    c_ref = S[0] @ c_src
    c_trn = S[2] @ c_src
    rx, ry = c_ref[:P], c_ref[P:]
    tx, ty = c_trn[:P], c_trn[P:]
    rz = -(kx * rx + ky * ry) / kz_ref
    tz = -(kx * tx + ky * ty) / kz_trn

    kz_inc = jnp.sqrt(eps_ref_e - kx0 ** 2 - ky0 ** 2 + 0j)
    kz_inc = jnp.where(kz_inc.real < 0, -kz_inc, kz_inc)
    R2 = jnp.abs(rx) ** 2 + jnp.abs(ry) ** 2 + jnp.abs(rz) ** 2
    T2 = jnp.abs(tx) ** 2 + jnp.abs(ty) ** 2 + jnp.abs(tz) ** 2
    R_orders = R2 * jnp.real(kz_ref) / jnp.real(kz_inc)
    T_orders = T2 * jnp.real(kz_trn) / jnp.real(kz_inc)

    # Back to the physics convention for the complex amplitudes.
    return GradSolution(
        R_orders=R_orders, T_orders=T_orders,
        rx=jnp.conj(rx), ry=jnp.conj(ry), rz=jnp.conj(rz),
        tx=jnp.conj(tx), ty=jnp.conj(ty), tz=jnp.conj(tz),
    )
