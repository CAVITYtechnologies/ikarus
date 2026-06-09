"""Real-space field reconstruction from the modal RCWA solution.

After a stack is solved, the tangential field amplitudes are known at every layer
boundary through the cascaded scattering matrices.  This module recovers the
*internal* mode amplitudes of each layer, evaluates the full vector field
``(Ex, Ey, Ez, Hx, Hy, Hz)`` in Fourier space at an arbitrary depth ``z`` and
inverse-transforms it to a real-space grid.

z convention: ``z = 0`` is the cover / first-interior-layer interface and ``z``
increases *into* the stack (towards the substrate).  Negative ``z`` is inside the
cover; ``z`` beyond the total thickness is inside the substrate.

The internal solve runs in the engineering convention used by
:mod:`ikarus.core.solver`; the returned complex fields are conjugated back to the
package's physics ``exp(-i w t)`` convention.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .solver import FieldSolution, redheffer_star, SMatrix, uniform_modes

_I = lambda n: np.eye(n, dtype=complex)
_inv = np.linalg.inv


@dataclass
class FieldMap:
    """Real-space vector fields on a sampling grid at one ``z`` (or a plane)."""

    E: np.ndarray  # complex, shape (..., 3) -> (Ex, Ey, Ez)
    H: np.ndarray  # complex, shape (..., 3)
    coords: dict   # axis-name -> 1-D coordinate array (meters)
    z: float | None = None

    @property
    def intensity(self) -> np.ndarray:
        """``|E|^2`` summed over components."""
        return np.sum(np.abs(self.E) ** 2, axis=-1)


def _gap_amplitudes(sol: FieldSolution):
    """Forward/backward gap-mode amplitudes ``(a_i, b_i)`` at every gap port.

    Gap ``i`` sits below layer ``i`` (gap 0 is between the cover and layer 1,
    gap ``N`` is between layer ``N`` and the substrate).
    """
    layer_S = sol.layer_smatrices
    n = len(layer_S)
    dim = sol.W0.shape[0]
    I = _I(dim)

    # Cumulative scattering matrices to the left (cover side) of each gap.
    SL = [sol.s_ref]
    for k in range(n):
        SL.append(redheffer_star(SL[-1], layer_S[k]))
    # SL[i] couples (cover_in, gap_i_back) -> (cover_out, gap_i_fwd), i = 0..n.

    # Cumulative scattering matrices to the right (substrate side).
    SR = [None] * (n + 1)
    SR[n] = sol.s_trn
    for k in range(n - 1, -1, -1):
        SR[k] = redheffer_star(layer_S[k], SR[k + 1])
    # SR[i] couples (gap_i_fwd, sub_in) -> (gap_i_back, sub_out).

    c_src = sol.c_src
    amps = []
    for i in range(n + 1):
        a = _inv(I - SL[i].S22 @ SR[i].S11) @ (SL[i].S21 @ c_src)
        b = SR[i].S11 @ a
        amps.append((a, b))
    return amps


def _layer_amplitudes(sol: FieldSolution, k: int, amps):
    """Forward/backward modal amplitudes ``(c_fwd, c_bwd)`` inside layer ``k``.

    ``c_fwd`` is referenced to the layer top, ``c_bwd`` to the layer bottom, so
    both modal exponentials stay bounded.
    """
    lm = sol.layer_modes[k]
    W, V = lm.W, lm.V
    Winv, Vinv = _inv(W), _inv(V)
    A = Winv @ sol.W0 + Vinv @ sol.V0
    B = Winv @ sol.W0 - Vinv @ sol.V0
    a_top, b_top = amps[k]       # gap above layer k
    a_bot, b_bot = amps[k + 1]   # gap below layer k
    c_fwd = 0.5 * (A @ a_top + B @ b_top)
    c_bwd = 0.5 * (B @ a_bot + A @ b_bot)
    return c_fwd, c_bwd


def _modal_fields_to_st(W, V, lam, c_fwd, c_bwd, zeta, k0L):
    """Tangential Fourier amplitudes ``(s, u)`` at normalized depth ``zeta``."""
    fwd = c_fwd * np.exp(-lam * zeta)
    bwd = c_bwd * np.exp(-lam * (k0L - zeta))
    s = W @ (fwd + bwd)
    u = V @ (fwd - bwd)
    return s, u


def _longitudinal(s, u, Kx, Ky, ERC):
    """Return ``(Ez, Hz)`` Fourier amplitudes from the tangential ones."""
    P = Kx.shape[0]
    sx, sy = s[:P], s[P:]
    ux, uy = u[:P], u[P:]
    Ez = -_inv(ERC) @ (Kx @ uy - Ky @ ux)
    Hz = -(Kx @ sy - Ky @ sx)
    return Ez, Hz


def _fourier_field_at_z(sol: FieldSolution, z: float, amps):
    """Full Fourier-space ``(E, H)`` (each a length-3P stacked vector) at ``z``."""
    Kx, Ky, k0 = sol.Kx, sol.Ky, sol.k0
    heights = sol.heights
    total = sum(heights)

    if z < 0:  # cover
        lam = 1j * np.concatenate([np.diag(sol.Kz_ref)] * 2)
        W, V, _ = uniform_modes(np.conj(sol.eps_ref), Kx, Ky)
        # cover amplitudes: incident c_src (forward), reflected c_ref.
        c_ref = sol.global_smatrix.S11 @ sol.c_src
        zeta = k0 * z  # <= 0
        s = W @ (sol.c_src * np.exp(-lam * zeta) + c_ref * np.exp(lam * zeta))
        u = V @ (sol.c_src * np.exp(-lam * zeta) - c_ref * np.exp(lam * zeta))
        ERC = np.conj(sol.eps_ref) * _I(Kx.shape[0])
    elif z > total:  # substrate
        lam = 1j * np.concatenate([np.diag(sol.Kz_trn)] * 2)
        W, V, _ = uniform_modes(np.conj(sol.eps_trn), Kx, Ky)
        c_trn = sol.global_smatrix.S21 @ sol.c_src
        zeta = k0 * (z - total)  # >= 0
        s = W @ (c_trn * np.exp(-lam * zeta))
        u = V @ (c_trn * np.exp(-lam * zeta))
        ERC = np.conj(sol.eps_trn) * _I(Kx.shape[0])
    else:  # interior layer
        edges = np.concatenate([[0.0], np.cumsum(heights)])
        k = int(np.searchsorted(edges, z, side="right") - 1)
        k = min(max(k, 0), len(heights) - 1)
        lm = sol.layer_modes[k]
        c_fwd, c_bwd = _layer_amplitudes(sol, k, amps)
        zeta = k0 * (z - edges[k])
        s, u = _modal_fields_to_st(lm.W, lm.V, lm.lam, c_fwd, c_bwd, zeta, lm.k0L)
        ERC = lm.ERC

    Ez, Hz = _longitudinal(s, u, Kx, Ky, ERC)
    P = Kx.shape[0]
    E = np.concatenate([s[:P], s[P:], Ez])      # Ex, Ey, Ez (each length P)
    H = np.concatenate([u[:P], u[P:], Hz])
    return E, H


def _inverse_fourier(coeffs_xyz, sol: FieldSolution, x, y):
    """Inverse-transform stacked ``(Fx, Fy, Fz)`` Fourier amplitudes to (x, y).

    ``coeffs_xyz`` is length ``3P``; returns array of shape ``(len(x), len(y), 3)``.
    """
    P = sol.grid.size
    kx = np.diag(sol.Kx).real * sol.k0  # physical kx per harmonic
    ky = np.diag(sol.Ky).real * sol.k0
    x = np.atleast_1d(x)
    y = np.atleast_1d(y)
    # phase[p, ix, iy] = exp(i (kx_p x + ky_p y))
    px = np.exp(1j * np.outer(kx, x))         # (P, Nx)
    py = np.exp(1j * np.outer(ky, y))         # (P, Ny)
    out = np.empty((x.size, y.size, 3), dtype=complex)
    for c in range(3):
        amp = coeffs_xyz[c * P:(c + 1) * P]    # (P,)
        # sum_p amp_p * px[p,ix] * py[p,iy]
        out[..., c] = np.einsum("p,px,py->xy", amp, px, py)
    return out


def reconstruct(
    sol: FieldSolution,
    z_positions,
    nx: int = 64,
    ny: int = 64,
    plane: str = "xy",
    x_position: float = 0.0,
    y_position: float = 0.0,
) -> dict:
    """Reconstruct real-space fields.

    Parameters
    ----------
    z_positions:
        Iterable of depths (meters) at which to build an ``xy`` field map.  Used
        only when ``plane == 'xy'``.
    plane:
        ``'xy'`` returns one :class:`FieldMap` per requested ``z``; ``'xz'`` /
        ``'yz'`` return a single cross-section sweeping ``z`` across the full
        stack (substrate-side margin included) at fixed ``y`` / ``x``.

    Returns
    -------
    dict mapping a label to :class:`FieldMap`.
    """
    amps = _gap_amplitudes(sol)
    Lx, Ly = sol.period_x, sol.period_y

    if plane == "xy":
        x = np.linspace(0, Lx, nx, endpoint=False)
        y = np.linspace(0, Ly, ny, endpoint=False)
        result = {}
        for z in np.atleast_1d(z_positions):
            E_f, H_f = _fourier_field_at_z(sol, float(z), amps)
            E = np.conj(_inverse_fourier(E_f, sol, x, y))
            H = np.conj(_inverse_fourier(H_f, sol, x, y))
            result[f"z_{z:.6e}"] = FieldMap(E=E, H=H,
                                            coords={"x": x, "y": y}, z=float(z))
        return result

    # Cross-section planes: sweep z over [margin above, total + margin below].
    total = sum(sol.heights)
    margin = 0.3 * (total if total > 0 else sol.period_x)
    z = np.linspace(-margin, total + margin, max(nx, ny))
    if plane == "xz":
        axis = np.linspace(0, Lx, nx, endpoint=False)
        E = np.empty((axis.size, z.size, 3), dtype=complex)
        H = np.empty_like(E)
        for j, zz in enumerate(z):
            E_f, H_f = _fourier_field_at_z(sol, float(zz), amps)
            E[:, j, :] = np.conj(_inverse_fourier(E_f, sol, axis, [y_position])[:, 0, :])
            H[:, j, :] = np.conj(_inverse_fourier(H_f, sol, axis, [y_position])[:, 0, :])
        return {"xz": FieldMap(E=E, H=H, coords={"x": axis, "z": z})}
    else:  # yz
        axis = np.linspace(0, Ly, ny, endpoint=False)
        E = np.empty((axis.size, z.size, 3), dtype=complex)
        H = np.empty_like(E)
        for j, zz in enumerate(z):
            E_f, H_f = _fourier_field_at_z(sol, float(zz), amps)
            E[:, j, :] = np.conj(_inverse_fourier(E_f, sol, [x_position], axis)[0, :, :])
            H[:, j, :] = np.conj(_inverse_fourier(H_f, sol, [x_position], axis)[0, :, :])
        return {"yz": FieldMap(E=E, H=H, coords={"y": axis, "z": z})}
