"""Independent analytic reference: stratified-medium transfer matrix.

This is a completely separate implementation from the RCWA engine -- the classic
characteristic-matrix (Abelès / Macleod) method for a stack of homogeneous,
isotropic layers at oblique incidence.  It is used only to validate Ikarus on
problems where both methods must agree exactly (uniform layers, any angle, s and
p polarization, with or without absorption).

Time convention matches the solver: ``exp(-i w t)`` so ``n = n' + i k'`` with
``k' > 0`` for loss.
"""

from __future__ import annotations

import numpy as np


def _kz_over_k0(n: complex, n0_sin_theta: float) -> complex:
    """Normalized longitudinal wavevector ``kz / k0 = sqrt(n^2 - (n0 sinθ)^2)``."""
    val = np.sqrt(complex(n) ** 2 - n0_sin_theta**2)
    # Decaying branch (Im >= 0) for evanescent layers.
    if val.imag < 0:
        val = -val
    return val


def fresnel_stack(
    n_list: list[complex],
    d_list: list[float],
    wavelength: float,
    theta_deg: float = 0.0,
    polarization: str = "s",
) -> tuple[float, float]:
    """Reflectance/transmittance of a layered stack.

    Parameters
    ----------
    n_list:
        Complex refractive indices ``[n_cover, n_1, ..., n_substrate]``.
    d_list:
        Thicknesses of the *finite* interior layers (length ``len(n_list) - 2``).
    theta_deg:
        Incidence angle in the cover, degrees.
    polarization:
        ``'s'`` (TE) or ``'p'`` (TM).

    Returns
    -------
    (R, T): power reflectance and transmittance (floats).
    """
    # Physics convention (exp(-i w t), absorbing -> Im(n) > 0).  The admittance
    # algebra below is written in the engineering convention, so conjugate the
    # indices at entry; real (lossless) indices are unaffected.
    n_list = [np.conj(complex(n)) for n in n_list]
    n0 = complex(n_list[0])
    ns = complex(n_list[-1])
    k0 = 2.0 * np.pi / wavelength
    theta0 = np.deg2rad(theta_deg)
    n0_sin = (n0 * np.sin(theta0)).real  # conserved tangential index (real cover)

    def admittance(n: complex) -> complex:
        kz = _kz_over_k0(n, n0_sin)
        if polarization == "s":
            return kz  # n cosθ
        return complex(n) ** 2 / kz  # n / cosθ

    eta0 = admittance(n0)
    etas = admittance(ns)

    # Characteristic matrix of the finite interior layers.
    M = np.eye(2, dtype=complex)
    for n, d in zip(n_list[1:-1], d_list):
        kz = _kz_over_k0(n, n0_sin)
        delta = k0 * d * kz
        eta = admittance(n)
        Mi = np.array(
            [[np.cos(delta), 1j * np.sin(delta) / eta],
             [1j * eta * np.sin(delta), np.cos(delta)]],
            dtype=complex,
        )
        M = M @ Mi

    B, C = M @ np.array([1.0, etas], dtype=complex)
    r = (eta0 * B - C) / (eta0 * B + C)
    R = float(np.abs(r) ** 2)
    T = float(4.0 * eta0.real * etas.real / np.abs(eta0 * B + C) ** 2)
    return R, T
