"""Bad-but-plausible input must fail loudly, not quietly return a wrong answer.

The dangerous failure in a solver is not a crash -- it is a run that completes
and hands back a confident, plausible, wrong number. These tests pin the checks
that make the common mistakes audible, especially the unit slip
(``wavelength=1550`` for ``1550e-9``) that every existing positivity check
happily accepts.
"""

import numpy as np
import pytest

from ikarus import RCWA, Layer, Source


# -- unit slips: warn, never raise (mm-wave work is legitimately millimetric) --

@pytest.mark.parametrize("make, name", [
    (lambda: RCWA(period_x=500, period_y=500), "period_x"),
    (lambda: Source(wavelength=1550), "wavelength"),
    (lambda: Layer(height=200, material=3.5), "layer height"),
])
def test_nanometres_passed_as_metres_warns(make, name):
    with pytest.warns(UserWarning, match="metres"):
        make()


@pytest.mark.parametrize("make", [
    lambda: RCWA(period_x=500e-9, period_y=500e-9),
    lambda: Source(wavelength=1550e-9),
    lambda: Layer(height=200e-9, material=3.5),
    lambda: Layer(height=np.inf, material=1.0),   # semi-infinite: normal, not a slip
])
def test_sane_lengths_are_silent(make, recwarn):
    make()
    assert [w for w in recwarn if "metres" in str(w.message)] == []


# -- counts -------------------------------------------------------------------

def test_one_dimensional_n_orders_is_allowed():
    # A 1-D grating legitimately has zero harmonics in y. Rejecting this would
    # break every 1-D simulation, so it is pinned deliberately.
    RCWA(period_x=5e-7, period_y=5e-7, n_orders=(21, 0))


@pytest.mark.parametrize("kwargs", [
    {"n_orders": -5},
    {"n_orders": (21, -1)},
    {"resolution": 0},
    {"resolution": (64, 0)},
])
def test_meaningless_counts_raise(kwargs):
    with pytest.raises(ValueError):
        RCWA(period_x=5e-7, period_y=5e-7, **kwargs)


# -- cost ---------------------------------------------------------------------

@pytest.mark.parametrize("n_orders", [(8, 8), (16, 16), (200, 0)])
def test_affordable_truncations_are_silent(n_orders, recwarn):
    # (200, 0) is a 1-D grating: linear in M, not quadratic, so it is cheap.
    RCWA(period_x=5e-7, period_y=5e-7, n_orders=n_orders)
    assert [w for w in recwarn if "harmonics" in str(w.message)] == []


@pytest.mark.parametrize("n_orders", [(25, 25), 200])
def test_expensive_truncation_warns(n_orders):
    # A bare scalar is the trap: n_orders=200 means 200 in *both* axes, i.e.
    # 160801 harmonics, which never finishes.
    with pytest.warns(UserWarning, match="harmonics"):
        RCWA(period_x=5e-7, period_y=5e-7, n_orders=n_orders)


# -- angles -------------------------------------------------------------------

@pytest.mark.parametrize("theta", [0, 45, -45, 89.9])
def test_valid_incidence_angles(theta):
    Source(wavelength=1.55e-6, theta=theta)


@pytest.mark.parametrize("theta", [90, -90, 120, 180])
def test_impossible_incidence_angle_raises(theta):
    # theta is measured from the normal, so |theta| >= 90 is not an incident wave.
    with pytest.raises(ValueError, match="theta"):
        Source(wavelength=1.55e-6, theta=theta)


# -- topology -----------------------------------------------------------------

def test_non_integer_topology_raises():
    # A greyscale/density map is for inverse design; truncating it silently would
    # change the geometry without saying so.
    with pytest.raises(ValueError, match="integer material indices"):
        Layer(height=1e-7, topology=np.full((8, 8), 0.5), materials=[1, 2])


def test_integral_float_topology_is_accepted_and_cast():
    lay = Layer(height=1e-7, topology=np.ones((8, 8)), materials=[1, 2])
    assert np.issubdtype(lay.topology.dtype, np.integer)


def test_negative_topology_index_raises():
    # -1 would wrap around the materials list and silently pick the last material.
    with pytest.raises(ValueError, match="negative"):
        Layer(height=1e-7, topology=-np.ones((8, 8), int), materials=[1, 2])
