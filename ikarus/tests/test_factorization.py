"""Tests for Fourier factorization rules (Laurent vs. Li's inverse rule).

Li's inverse rule restores fast convergence for high-index-contrast TM / 1-D
gratings, where the Laurent (direct) rule converges only as O(1/M).  These tests
pin the improvement and guard the Laurent default against regressions.
"""

import numpy as np
import pytest

from ikarus import RCWA


def _grating(factorization, M, n_hi=3.5, period=400e-9, wl=700e-9, h=300e-9, Nx=1024):
    """Zeroth-order TM reflection of a lossless high-contrast 1-D lamellar grating."""
    topo = np.zeros((Nx, 2), dtype=int)
    topo[: Nx // 2, :] = 1                      # 50% duty cycle
    rc = RCWA(period_x=period, period_y=period, resolution=(Nx, 2),
              n_orders=(M, 0), factorization=factorization)
    rc.add_uniform_layer(np.inf, "Air")
    rc.add_layer(h, topo, ["Air", n_hi])
    rc.add_uniform_layer(np.inf, "Air")
    rc.set_source(wavelength=wl, theta=0, polarization="linear", linear_pol_angle=90)
    _, _, r = rc.simulate()
    return r.R_total, np.degrees(np.angle(r.R)), r.energy_balance


def test_li_converges_fast_for_high_contrast_tm():
    """Li converges by M~12; Laurent is still far off there.

    The converged target (R~0.100, phase~+80.5 deg) is the M->infinity limit
    obtained independently by 1/M extrapolation of the Laurent sequence.
    """
    R_target, phase_target = 0.100, 80.5

    R12, p12, eb12 = _grating("li", 12)
    R20, p20, _ = _grating("li", 20)

    # Converged to the independent target already at modest M...
    assert abs(R12 - R_target) < 5e-3
    assert abs(p12 - phase_target) < 1.5
    # ...and stable between M=12 and M=20 (the hallmark of true convergence).
    assert abs(R20 - R12) < 2e-3
    assert abs(p20 - p12) < 0.5
    assert abs(eb12 - 1.0) < 1e-3            # lossless: energy conserved

    # Laurent at the same low M has NOT converged -- proves Li is doing work.
    RL12, pL12, _ = _grating("laurent", 12)
    assert abs(RL12 - R_target) > 0.05       # Laurent still ~0.16, far from 0.10


def test_laurent_is_unchanged_default():
    """Default factorization is Laurent, and it matches an explicit request."""
    rc = RCWA(period_x=1e-6, period_y=1e-6, n_orders=(6, 0))
    assert rc.factorization == "laurent"
    a = _grating("laurent", 10)
    b = _grating(None or "laurent", 10)
    assert a == b


def test_li_equals_laurent_for_uniform_stack():
    """With no in-plane discontinuity, the inverse rule must reduce to Laurent."""
    def fresnel(factorization):
        rc = RCWA(period_x=1e-6, period_y=1e-6, n_orders=0, factorization=factorization)
        rc.add_uniform_layer(np.inf, 1.0)
        rc.add_uniform_layer(200e-9, 2.5)       # uniform high-index film
        rc.add_uniform_layer(np.inf, 1.5)
        rc.set_source(wavelength=600e-9, theta=0, polarization="linear")
        return rc.simulate()[2].R_total

    assert fresnel("li") == pytest.approx(fresnel("laurent"), abs=1e-12)


def test_invalid_factorization_rejected():
    with pytest.raises(ValueError, match="factorization"):
        RCWA(period_x=1e-6, period_y=1e-6, factorization="bogus")
