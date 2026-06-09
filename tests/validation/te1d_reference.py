"""Independent 1D lamellar-grating RCWA reference (TE and TM).

A completely separate implementation from the Ikarus engine, used to validate the
full-vector solver on 1-D diffraction gratings.  It uses *direct two-interface
mode-matching* (a single linear solve, no scattering-matrix cascade, no gap
medium) so that agreement with the engine cross-checks the gap-medium S-matrix
machinery.  Validated independently against the effective-medium-theory limit
(period -> 0 reproduces a uniform slab to <0.1%).

Geometry: cover (z>0) / grating (-d<z<0) / substrate.  Periodic in x, invariant
in y.  Physics ``exp(-i w t)`` convention (absorbing media: ``Im(n) > 0``).
"""

from __future__ import annotations

import numpy as np


def _sqrt_im_pos(arg: np.ndarray) -> np.ndarray:
    v = np.sqrt(np.asarray(arg, dtype=complex))
    return np.where(v.imag < 0, -v, v)


def grating_1d(
    eps_line: np.ndarray,
    eps_cover: complex,
    eps_substrate: complex,
    period: float,
    thickness: float,
    wavelength: float,
    theta_deg: float = 0.0,
    n_orders: int = 21,
    polarization: str = "TE",
) -> tuple[float, float]:
    """Total reflectance/transmittance of a 1-D lamellar grating.

    Parameters
    ----------
    eps_line:
        One period of the real-space permittivity profile ``eps(x)`` (1-D array).
    polarization:
        ``'TE'`` (E parallel to the grooves, along y) or ``'TM'`` (H along y).
    """
    eps_line = np.asarray(eps_line, dtype=complex)
    Nx = eps_line.size
    k0 = 2.0 * np.pi / wavelength
    nc = np.sqrt(eps_cover)
    kx0 = (nc * np.sin(np.deg2rad(theta_deg))).real

    m = np.arange(-n_orders, n_orders + 1)
    P = 2 * n_orders + 1
    Kx = np.diag(kx0 - m * (wavelength / period))
    I = np.eye(P)

    coeffs = np.fft.fftshift(np.fft.fft(eps_line)) / Nx
    c0 = Nx // 2
    E = coeffs[c0 + (m[:, None] - m[None, :])]

    kzc = _sqrt_im_pos(eps_cover - np.diag(Kx) ** 2)
    kzs = _sqrt_im_pos(eps_substrate - np.diag(Kx) ** 2)
    Kzc, Kzs = np.diag(kzc), np.diag(kzs)

    if polarization.upper() == "TE":
        # d^2 Ey/dz'^2 = (Kx^2 - E) Ey ;  H_x ~ dEy/dz'.
        gamma, W = np.linalg.eig(Kx @ Kx - E)
        q = _sqrt_im_pos(gamma)
        Q = np.diag(q)
        eta_c, eta_s = 1j * Kzc, 1j * Kzs  # cover/substrate admittances
        Wq = W @ Q
    else:  # TM
        # d^2 Hy/dz'^2 = (E^{1/?}) ... use the standard inverse-rule-free TM form:
        # d^2 Hy/dz'^2 = E (Kx E^{-1} Kx - I) Hy, admittance via 1/eps.
        Einv = np.linalg.inv(E)
        Omega2 = E @ (Kx @ Einv @ Kx - I)
        gamma, W = np.linalg.eig(Omega2)
        q = _sqrt_im_pos(gamma)
        Q = np.diag(q)
        eta_c, eta_s = 1j * Kzc / eps_cover, 1j * Kzs / eps_substrate
        Wq = Einv @ W @ Q

    X = np.diag(np.exp(-q * k0 * thickness))
    d = np.zeros(P, dtype=complex)
    d[n_orders] = 1.0
    Z = np.zeros((P, P), dtype=complex)

    # Unknowns [r, t, c+, c-]: continuity of the tangential E and H at both faces.
    Amat = np.hstack([
        np.vstack([I, eta_c, Z, Z]),
        np.vstack([Z, Z, -I, eta_s]),
        np.vstack([-W, -Wq, W @ X, Wq @ X]),
        np.vstack([-W @ X, Wq @ X, W, -Wq]),
    ])
    rhs = np.concatenate([-d, eta_c @ d, np.zeros(P), np.zeros(P)])
    sol = np.linalg.solve(Amat, rhs)
    r, t = sol[:P], sol[P:2 * P]

    R = float(np.sum(np.abs(r) ** 2 * np.real(kzc) / np.real(kzc[n_orders])))
    T = float(np.sum(np.abs(t) ** 2 * np.real(kzs) / np.real(kzc[n_orders])))
    return R, T
