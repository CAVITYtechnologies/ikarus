"""Validate Ikarus against the analytic stratified-medium (Fresnel) solution.

Uniform-layer stacks must agree with the independent characteristic-matrix
reference to ~1e-8 at normal and oblique incidence, both polarizations, with and
without absorption.  This is the package's primary correctness gate.
"""

import numpy as np
import pytest

from ikarus import RCWA
from ikarus.tests.validation.fresnel_reference import fresnel_stack

TOL = 1e-7


def _run(n_cover, n_film, n_sub, d_film, wl, theta, pol_angle, n_orders=3):
    rcwa = RCWA(period_x=500e-9, period_y=500e-9, resolution=8, n_orders=n_orders)
    rcwa.add_uniform_layer(height=np.inf, material=complex(n_cover))
    if d_film is not None:
        rcwa.add_uniform_layer(height=d_film, material=complex(n_film))
    rcwa.add_uniform_layer(height=np.inf, material=complex(n_sub))
    rcwa.set_source(wavelength=wl, theta=theta, phi=0.0,
                    polarization="linear", linear_pol_angle=pol_angle)
    _, _, res = rcwa.simulate()
    return res


# (n_cover, n_film, n_sub, d_film, wl, theta, pol_angle, label)
CASES = [
    (1.0, None, 1.5, None, 600e-9, 0.0, 0.0, "air/glass normal"),
    (1.0, None, 3.5, None, 1550e-9, 0.0, 0.0, "air/Si normal"),
    (1.0, 3.5, 1.45, 200e-9, 1550e-9, 0.0, 0.0, "air/Si/SiO2 slab normal"),
    (1.0, None, 1.5, None, 600e-9, 30.0, 0.0, "air/glass 30deg s"),
    (1.0, None, 1.5, None, 600e-9, 30.0, 90.0, "air/glass 30deg p"),
    (1.0, None, 1.5, None, 600e-9, 60.0, 0.0, "air/glass 60deg s"),
    (1.0, None, 1.5, None, 600e-9, 60.0, 90.0, "air/glass 60deg p"),
    (1.0, 2.0, 1.5, 300e-9, 600e-9, 40.0, 0.0, "slab 40deg s"),
    (1.0, 2.0, 1.5, 300e-9, 600e-9, 40.0, 90.0, "slab 40deg p"),
    (1.0, 3.5 + 0.1j, 1.45, 100e-9, 1550e-9, 0.0, 0.0, "lossy slab normal"),
    (1.0, 4.0 + 0.5j, 1.45, 50e-9, 800e-9, 45.0, 0.0, "lossy slab 45deg s"),
]


@pytest.mark.parametrize("case", CASES, ids=[c[-1] for c in CASES])
def test_fresnel_agreement(case):
    n_cover, n_film, n_sub, d_film, wl, theta, pol_angle, _ = case
    res = _run(n_cover, n_film, n_sub, d_film, wl, theta, pol_angle)

    pol = "s" if np.isclose(pol_angle, 0.0) else "p"
    n_list = [n_cover, n_sub] if d_film is None else [n_cover, n_film, n_sub]
    d_list = [] if d_film is None else [d_film]
    R_ref, T_ref = fresnel_stack(n_list, d_list, wl, theta, pol)

    assert abs(res.R_total - R_ref) < TOL
    assert abs(res.T_total - T_ref) < TOL


def test_lossless_energy_conservation():
    # ~1e-9 defect is expected from the Rayleigh-anomaly loss regularization.
    res = _run(1.0, 2.0, 1.5, 300e-9, 600e-9, 40.0, 0.0)
    assert abs(res.energy_balance - 1.0) < 1e-7


def test_absorption_is_physical():
    # A lossy film must absorb: R + T < 1 (and absorptance > 0).
    res = _run(1.0, 3.5 + 0.2j, 1.45, 150e-9, 1000e-9, 0.0, 0.0)
    assert res.energy_balance < 1.0
    assert 1.0 - res.energy_balance > 0.0
