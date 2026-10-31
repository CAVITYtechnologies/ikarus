"""The sign of the diffraction-order label.

Ikarus reports orders in the convention of the grating equation,

    kx_p = kx0 + p * 2*pi/period_x,

so order ``+1`` is deflected towards ``+x`` -- the same as every textbook and
every other solver.  Internally the solver labels its harmonics the other way
round (``kx = kx0 - p*lambda/period``, self-consistently with its convolution
matrix) and ``HarmonicGrid.diffraction_orders`` translates at the boundary.

This file exists because the mismatch went unnoticed for a long time: the
physics was right and only the *name* of each beam was mirrored, so totals,
energy balance and efficiencies all looked correct.  Worse, **a mirror-symmetric
test structure cannot see it at all** -- orders +1 and -1 are then degenerate.
The decisive test is therefore the kinematic one: at oblique incidence the
grating equation decides *which orders can propagate at all*, regardless of what
the structure is, so a mirrored label shows up as the wrong order carrying the
power.
"""

import numpy as np
import pytest

import ikarus

PERIOD = 2e-6
LAMBDA = 1.55e-6


def _grating(cover="Air", substrate="SiO2", asymmetric=True):
    topo = np.zeros((20, 1), dtype=int)
    topo[8:, 0] = 1 if asymmetric else 1
    if not asymmetric:
        topo[:] = 0
        topo[5:15, 0] = 1                      # centred -> mirror-symmetric
    rcwa = ikarus.RCWA(period_x=PERIOD, period_y=PERIOD, n_orders=(12, 0),
                       resolution=(256, 1))
    rcwa.add_uniform_layer(np.inf, cover)
    rcwa.add_layer(500e-9, topo, ["Air", "Si"])
    rcwa.add_uniform_layer(np.inf, substrate)
    return rcwa


def test_order_plus_one_is_deflected_towards_plus_x():
    rcwa = _grating()
    rcwa.set_source(wavelength=LAMBDA, theta=0, polarization="linear")
    _, _, res = rcwa.simulate()
    # phi_out is the exit azimuth: 0 deg means +x, 180 deg means -x.
    assert res.phi_out_trn[res.order_index(1, 0)] == pytest.approx(0.0, abs=1e-6)
    assert res.phi_out_trn[res.order_index(-1, 0)] == pytest.approx(180.0, abs=1e-6)


def test_only_the_grating_equations_order_propagates_at_oblique_incidence():
    # From glass (n=1.45) at +18 deg into air: sin_inc = 1.45*sin(18) = 0.4481.
    # Grating equation: sin_p = sin_inc + p*lambda/period = 0.4481 + 0.775 p
    #   p = -1 -> -0.327, propagates;  p = +1 -> +1.223, evanescent.
    # This is kinematics only -- it does not depend on the structure, which is
    # what makes it able to catch a mirrored label on a symmetric grating.
    rcwa = _grating(cover="SiO2", substrate="Air")
    rcwa.set_source(wavelength=LAMBDA, theta=18.0, polarization="linear")
    _, _, res = rcwa.simulate()

    i_minus, i_plus = res.order_index(-1, 0), res.order_index(1, 0)
    assert res.T_orders[i_minus] > 1e-4, "order -1 must carry the power"
    # An evanescent order sits at the numerical floor, not exactly zero.
    assert res.T_orders[i_plus] < res.T_orders[i_minus] * 1e-6
    assert np.isnan(res.theta_out_trn[i_plus]), "order +1 must be evanescent"

    sin_expected = 1.45 * np.sin(np.radians(18.0)) - LAMBDA / PERIOD
    assert np.sin(np.radians(res.theta_out_trn[i_minus])) == pytest.approx(
        abs(sin_expected), abs=2e-3)


def test_a_symmetric_grating_cannot_see_the_sign_so_use_the_oblique_test():
    # Guards the guard: if this ever starts failing, orders +-1 have stopped
    # being degenerate for a mirror-symmetric structure and the test above is
    # no longer the only thing standing between us and a silent sign flip.
    rcwa = _grating(asymmetric=False)
    rcwa.set_source(wavelength=LAMBDA, theta=0, polarization="linear")
    _, _, res = rcwa.simulate()
    assert res.T_orders[res.order_index(1, 0)] == pytest.approx(
        res.T_orders[res.order_index(-1, 0)], rel=1e-9)


def test_grid_labels_are_the_negated_internal_harmonic_index():
    grid = ikarus.core.HarmonicGrid(3, 2)
    ip, iq = grid.index_arrays()
    p, q = grid.diffraction_orders()
    assert np.array_equal(p, -ip) and np.array_equal(q, -iq)
    # The specular order is its own mirror, so this index is sign-independent.
    i0 = grid.zero_order_index()
    assert (p[i0], q[i0]) == (0, 0)


def test_adjoint_resolves_an_order_the_same_way_as_the_forward_result():
    # Target(order=(1, 0)) must mean the same physical beam through the GA and
    # the adjoint engine; they resolve the index by separate code paths.
    pytest.importorskip("jax")
    from ikarus.core.fourier import HarmonicGrid
    from ikarus.inverse.adjoint import _order_flat_index

    rcwa = _grating()
    rcwa.set_source(wavelength=LAMBDA, theta=0, polarization="linear")
    _, _, res = rcwa.simulate()
    grid = HarmonicGrid(12, 0)
    for order in [(0, 0), (1, 0), (-1, 0), (3, 0), (-5, 0)]:
        assert _order_flat_index(grid, order) == res.order_index(*order)
