"""Validate the patterned-layer (diffraction) machinery.

Three independent checks:
* agreement with a direct mode-matching 1-D grating reference (TE and TM),
* the effective-medium limit (subwavelength grating == uniform slab),
* energy conservation for lossless gratings and positive absorptance for lossy.
"""

import numpy as np
import pytest

from ikarus import RCWA
from ikarus.tests.validation.te1d_reference import grating_1d
from ikarus.tests.validation.fresnel_reference import fresnel_stack


def _binary_topology(n_pixels=512, duty=0.5):
    topo = np.zeros((n_pixels, 2), dtype=int)
    topo[int(n_pixels * (1 - duty)):, :] = 1
    return topo


def _eps_line(n_pixels=2048, duty=0.5, eps_hi=6.25):
    line = np.ones(n_pixels)
    line[int(n_pixels * (1 - duty)):] = eps_hi
    return line


def _ikarus_grating(n_orders, pol_angle, period=800e-9, thickness=300e-9,
                    wl=633e-9, factorization="li"):
    topo = _binary_topology()
    rcwa = RCWA(period_x=period, period_y=period, resolution=(1024, 2),
                n_orders=(n_orders, 0), factorization=factorization)
    rcwa.add_uniform_layer(np.inf, 1.0)
    rcwa.add_layer(thickness, topo, [1.0, 2.5])
    rcwa.add_uniform_layer(np.inf, 1.0)
    rcwa.set_source(wavelength=wl, theta=0.0, polarization="linear",
                    linear_pol_angle=pol_angle)
    return rcwa.simulate()[2]


@pytest.mark.parametrize("pol_angle,pol", [(0.0, "TE"), (90.0, "TM")])
def test_matches_mode_matching_reference(pol_angle, pol):
    # The reference (te1d_reference.grating_1d) uses the Laurent/direct rule for
    # the TM admittance (Einv = inv(<<eps>>)), so this cross-check pins Ikarus's
    # *Laurent* path against it.  Under the default Li inverse rule the two
    # intentionally differ for TM -- Li converges faster to the true value while
    # the Laurent reference drifts (see test_factorization.py).
    M = 30
    res = _ikarus_grating(M, pol_angle, factorization="laurent")
    R_ref, T_ref = grating_1d(_eps_line(), 1.0, 1.0, 800e-9, 300e-9, 633e-9,
                              0.0, M, pol)
    assert abs(res.R_total - R_ref) < 1e-3
    assert abs(res.T_total - T_ref) < 1e-3


def test_effective_medium_limit():
    # A deeply subwavelength TE grating behaves as a uniform slab with
    # eps_eff = <eps> (arithmetic mean for E parallel to the grooves).
    eps_eff = (1.0 + 6.25) / 2
    R_emt, _ = fresnel_stack([1.0, np.sqrt(eps_eff), 1.0], [300e-9], 633e-9, 0.0, "s")

    topo = _binary_topology()
    rcwa = RCWA(period_x=12e-9, period_y=12e-9, resolution=(1024, 2), n_orders=(20, 0))
    rcwa.add_uniform_layer(np.inf, 1.0)
    rcwa.add_layer(300e-9, topo, [1.0, 2.5])
    rcwa.add_uniform_layer(np.inf, 1.0)
    rcwa.set_source(wavelength=633e-9, theta=0.0, polarization="linear",
                    linear_pol_angle=0.0)
    res = rcwa.simulate()[2]
    assert abs(res.R_total - R_emt) < 5e-3


@pytest.mark.parametrize("n_orders", [11, 31, 61])
def test_lossless_energy_conservation(n_orders):
    res = _ikarus_grating(n_orders, 0.0)
    assert abs(res.energy_balance - 1.0) < 1e-6


def test_high_order_stability():
    # Regression for the branch-cut bug that blew up R+T to ~1e8 at high order.
    res = _ikarus_grating(81, 0.0)
    assert abs(res.energy_balance - 1.0) < 1e-5
    assert 0.0 <= res.R_total <= 1.0


def test_symmetric_grating_diffraction_symmetry():
    # A grating symmetric about its center sends equal power into +/-1 orders.
    topo = np.zeros((128, 2), dtype=int)
    topo[32:96, :] = 1  # centered stripe
    rcwa = RCWA(period_x=1200e-9, period_y=1200e-9, resolution=(256, 2), n_orders=(12, 0))
    rcwa.add_uniform_layer(np.inf, 1.0)
    rcwa.add_layer(250e-9, topo, [1.0, 2.5])
    rcwa.add_uniform_layer(np.inf, 1.0)
    rcwa.set_source(wavelength=633e-9, theta=0.0, polarization="linear",
                    linear_pol_angle=0.0)
    res = rcwa.simulate()[2]
    ip = res.order_index(1, 0)
    im = res.order_index(-1, 0)
    assert abs(res.T_orders[ip] - res.T_orders[im]) < 1e-9
    assert abs(res.energy_balance - 1.0) < 1e-8


def test_lossy_grating_absorbs():
    topo = _binary_topology(n_pixels=128)
    rcwa = RCWA(period_x=600e-9, period_y=600e-9, resolution=(256, 2), n_orders=(12, 0))
    rcwa.add_uniform_layer(np.inf, 1.0)
    rcwa.add_layer(100e-9, topo, [1.0, 3.5 + 0.5j])
    rcwa.add_uniform_layer(np.inf, 1.45)
    rcwa.set_source(wavelength=633e-9, theta=20.0, polarization="linear",
                    linear_pol_angle=0.0)
    res = rcwa.simulate()[2]
    assert res.energy_balance < 1.0
    assert 1.0 - res.energy_balance > 0.01
