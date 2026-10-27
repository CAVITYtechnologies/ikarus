"""Conical incidence (``phi != 0``) -- symmetries the solver must obey.

``phi`` is wired through `Source` and used by the solver, but before this file
the whole test suite contained exactly one mention of it, and that was
``phi=0.0``.  The conical path was unexercised.

These are symmetries derived from the geometry, not from any reference tool, so
they check the solver against physics rather than against another
implementation.  Each one is paired with the case where the symmetry must
*fail*, so a test cannot pass by the quantity being trivially zero or the
rotation being silently ignored.
"""

import numpy as np
import pytest

from ikarus import RCWA, shapes

LAM = 1550e-9


def _uniform_stack(phi, theta=35.0, pol_angle=0.0):
    rcwa = RCWA(period_x=400e-9, period_y=400e-9, resolution=(32, 32), n_orders=(3, 3))
    rcwa.add_uniform_layer(np.inf, "Air")
    rcwa.add_uniform_layer(220e-9, "Si")
    rcwa.add_uniform_layer(np.inf, "SiO2")
    rcwa.set_source(wavelength=LAM, theta=theta, phi=phi,
                    polarization="linear", linear_pol_angle=pol_angle)
    return rcwa.simulate()[2]


def _square_lattice(phi, pol_angle=0.0, theta=25.0):
    """Circular pillar on a square lattice: C4 symmetric in the xy-plane."""
    mask = shapes.circle(center=(0.5, 0.5), radius=0.3, grid_shape=(64, 64))
    rcwa = RCWA(period_x=700e-9, period_y=700e-9, resolution=(64, 64), n_orders=(5, 5))
    rcwa.add_uniform_layer(np.inf, "Air")
    rcwa.add_layer(300e-9, mask, ["Air", "Si"])
    rcwa.add_uniform_layer(np.inf, "SiO2")
    rcwa.set_source(wavelength=LAM, theta=theta, phi=phi,
                    polarization="linear", linear_pol_angle=pol_angle)
    return rcwa.simulate()[2]


def _grating_order(phi, order, pol_angle=0.0, theta=20.0):
    """50% duty binary grating -- periodic in x, invariant in y, and its unit
    cell is mirror-symmetric about the bar centre."""
    topo = np.zeros((256, 2), dtype=int)
    topo[:128, :] = 1
    rcwa = RCWA(period_x=2.0e-6, period_y=2.0e-6, resolution=(256, 2), n_orders=(8, 0))
    rcwa.add_uniform_layer(np.inf, "Air")
    rcwa.add_layer(600e-9, topo, ["Air", "Si"])
    rcwa.add_uniform_layer(np.inf, "SiO2")
    rcwa.set_source(wavelength=LAM, theta=theta, phi=phi,
                    polarization="linear", linear_pol_angle=pol_angle)
    res = rcwa.simulate()[2]
    return res.T_orders[res.order_index(order, 0)]


# -- an unpatterned stack has no azimuth to break --------------------------

@pytest.mark.parametrize("phi", [30.0, 45.0, 90.0, 137.0, 180.0, 270.0])
def test_uniform_stack_is_azimuthally_isotropic(phi):
    """With no in-plane structure the answer cannot depend on phi. This is the
    bluntest check that the polarization basis is rotated consistently with the
    wavevector -- get one and not the other and this breaks immediately."""
    base, rotated = _uniform_stack(0.0), _uniform_stack(phi)
    assert rotated.R_total == pytest.approx(base.R_total, abs=1e-12)
    assert rotated.T_total == pytest.approx(base.T_total, abs=1e-12)


# -- a C4 structure is invariant under a quarter turn ----------------------

@pytest.mark.parametrize("pol_angle", [0.0, 90.0])
@pytest.mark.parametrize("phi", [0.0, 17.0])
def test_square_lattice_is_invariant_under_90_degree_azimuth(phi, pol_angle):
    """A circular pillar on a square lattice maps onto itself under a quarter
    turn, so rotating the illumination by 90 deg must change nothing."""
    a, b = _square_lattice(phi, pol_angle), _square_lattice(phi + 90.0, pol_angle)
    assert b.R_total == pytest.approx(a.R_total, abs=1e-11)
    assert b.T_total == pytest.approx(a.T_total, abs=1e-11)


def test_square_lattice_does_change_at_45_degrees():
    """Guard against the above passing because phi is ignored: a square lattice
    is NOT invariant under 45 deg, so that rotation must move the answer."""
    a, b = _square_lattice(0.0), _square_lattice(45.0)
    assert abs(b.R_total - a.R_total) > 1e-6


# -- reversing the azimuth mirrors the diffraction orders ------------------

@pytest.mark.parametrize("pol_angle", [0.0, 90.0])
def test_phi_180_mirrors_the_orders(pol_angle):
    """phi=180 reverses the in-plane wavevector, so order +m must take over the
    efficiency of order -m. This is also the convention that lets an Ikarus run
    be compared against tools whose x-axis points the other way."""
    assert _grating_order(0.0, +1, pol_angle) == pytest.approx(
        _grating_order(180.0, -1, pol_angle), abs=1e-12)


# -- a true conical mount: in-plane k along the invariant axis -------------

@pytest.mark.parametrize("pol_angle", [0.0, 90.0])
@pytest.mark.parametrize("order", [1, 2])
def test_grating_orders_are_symmetric_at_phi_90(order, pol_angle):
    """At phi=90 the incident wavevector has no x-component, and the grating
    only diffracts in x, so +m and -m must come out equal. This is the case
    that genuinely exercises conical mounting on a patterned layer."""
    plus, minus = (_grating_order(90.0, +order, pol_angle),
                   _grating_order(90.0, -order, pol_angle))
    assert plus == pytest.approx(minus, abs=1e-11)


@pytest.mark.parametrize("pol_angle", [0.0, 90.0])
def test_grating_orders_are_asymmetric_at_phi_0(pol_angle):
    """The companion to the above: at phi=0 there *is* an x-component, so the
    same orders must differ. Without this, the phi=90 test would also pass on a
    solver that quietly symmetrised everything."""
    plus, minus = _grating_order(0.0, +1, pol_angle), _grating_order(0.0, -1, pol_angle)
    assert abs(plus - minus) > 1e-3


# -- the azimuth sign itself -----------------------------------------------
# Every structure above is mirror-symmetric about the x-z plane, so flipping the
# sign of phi leaves all of them unchanged -- verified by mutation, none of them
# catch it. The zeroth order's outgoing azimuth does, and it is the convention a
# cross-tool comparison depends on.

@pytest.mark.parametrize("phi", [-45.0, 0.0, 30.0, 45.0, 90.0])
def test_zero_order_azimuth_tracks_the_incident_azimuth(phi):
    """The specular order keeps the incident in-plane wavevector, so its
    outgoing azimuth must equal phi -- sign and all."""
    res = _uniform_stack(phi)
    i0 = res.order_index(0, 0)
    assert res.phi_out_trn[i0] == pytest.approx(phi, abs=1e-6)
    assert res.phi_out_ref[i0] == pytest.approx(phi, abs=1e-6)
