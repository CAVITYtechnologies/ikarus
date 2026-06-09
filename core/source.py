"""Source / incidence definition for RCWA.

A :class:`Source` stores the illuminating plane wave: its vacuum wavelength,
propagation direction (elevation ``theta`` and azimuth ``phi``) and polarization
state.  It exposes the (normalized) in-plane wavevector components seen by the
solver and the complex tangential polarization vector ``(p_x, p_y, p_z)`` used to
build the incident-mode coefficients.

Angle conventions
-----------------
``theta`` is measured from the +z axis (0 = normal incidence, wave travelling in
-z into the structure).  ``phi`` is the azimuth in the xy-plane measured from +x.
The (real, normalized-by-k0) incident wavevector in the cover medium of index
``n_inc`` is

    k_inc = n_inc * (sin th cos ph,  sin th sin ph,  cos th).

Polarization
------------
* ``'linear'`` with ``linear_pol_angle`` psi (degrees): the E-field lies in the
  plane transverse to ``k`` spanned by the TE unit vector ``a_TE`` and TM unit
  vector ``a_TM``; ``psi`` is measured from ``a_TE`` (``psi=0`` -> pure TE/s,
  ``psi=90`` -> pure TM/p).
* ``'RCP' / 'LCP'``: right/left circular, ``(a_TE ± i a_TM)/sqrt(2)``.

For normal incidence ``a_TE`` is taken along +y and ``a_TM`` along +x so that the
``linear_pol_angle`` degenerates to the physical E-field angle in the xy-plane.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

_VALID_POL = ("linear", "RCP", "LCP")


@dataclass
class Source:
    wavelength: float
    theta: float = 0.0  # degrees, from +z
    phi: float = 0.0  # degrees, azimuth from +x
    polarization: str = "linear"
    linear_pol_angle: float = 0.0  # degrees, measured from a_TE

    # Refractive index of the medium the wave is launched from (the cover /
    # reflection region).  Set by the solver from the top semi-infinite layer.
    n_incident: complex = field(default=1.0)

    def __post_init__(self) -> None:
        if self.polarization not in _VALID_POL:
            raise ValueError(
                f"polarization must be one of {_VALID_POL}, got {self.polarization!r}"
            )
        if self.wavelength <= 0:
            raise ValueError("wavelength must be positive (meters)")

    # -- derived quantities ------------------------------------------------
    @property
    def k0(self) -> float:
        """Vacuum wavenumber ``2*pi/lambda`` (rad/m)."""
        return 2.0 * np.pi / self.wavelength

    @property
    def theta_rad(self) -> float:
        return np.deg2rad(self.theta)

    @property
    def phi_rad(self) -> float:
        return np.deg2rad(self.phi)

    def incident_wavevector(self) -> np.ndarray:
        """Normalized (by k0) incident wavevector ``(kx, ky, kz)`` in the cover."""
        n = self.n_incident
        st, ct = np.sin(self.theta_rad), np.cos(self.theta_rad)
        cp, sp = np.cos(self.phi_rad), np.sin(self.phi_rad)
        return np.array([n * st * cp, n * st * sp, n * ct], dtype=complex)

    def kx0_ky0(self) -> tuple[complex, complex]:
        """Normalized in-plane wavevector ``(kx0, ky0)`` (continuous across layers)."""
        k = self.incident_wavevector()
        return k[0], k[1]

    def polarization_vector(self) -> np.ndarray:
        """Complex unit polarization vector ``(p_x, p_y, p_z)`` (transverse to k)."""
        theta = self.theta_rad
        phi = self.phi_rad

        if np.isclose(theta, 0.0):
            # Degenerate: define TE along +y, TM along +x for an intuitive
            # in-plane E-field angle at normal incidence.
            a_te = np.array([0.0, 1.0, 0.0])
            a_tm = np.array([1.0, 0.0, 0.0])
        else:
            khat = np.array(
                [np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)]
            )
            zhat = np.array([0.0, 0.0, 1.0])
            a_te = np.cross(zhat, khat)
            a_te /= np.linalg.norm(a_te)
            a_tm = np.cross(a_te, khat)
            a_tm /= np.linalg.norm(a_tm)

        if self.polarization == "linear":
            psi = np.deg2rad(self.linear_pol_angle)
            pol = np.cos(psi) * a_te + np.sin(psi) * a_tm
        elif self.polarization == "RCP":
            pol = (a_te + 1j * a_tm) / np.sqrt(2.0)
        else:  # LCP
            pol = (a_te - 1j * a_tm) / np.sqrt(2.0)

        return pol.astype(complex)

    def te_tm_vectors(self) -> tuple[np.ndarray, np.ndarray]:
        """Return the real TE and TM unit vectors for the current geometry."""
        theta = self.theta_rad
        phi = self.phi_rad
        if np.isclose(theta, 0.0):
            return np.array([0.0, 1.0, 0.0]), np.array([1.0, 0.0, 0.0])
        khat = np.array(
            [np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)]
        )
        a_te = np.cross([0.0, 0.0, 1.0], khat)
        a_te /= np.linalg.norm(a_te)
        a_tm = np.cross(a_te, khat)
        a_tm /= np.linalg.norm(a_tm)
        return a_te, a_tm

    def copy_with(self, **changes) -> "Source":
        """Return a copy with selected fields overridden (for sweeps)."""
        params = dict(
            wavelength=self.wavelength,
            theta=self.theta,
            phi=self.phi,
            polarization=self.polarization,
            linear_pol_angle=self.linear_pol_angle,
            n_incident=self.n_incident,
        )
        params.update(changes)
        return Source(**params)
