"""Semantic contracts -- what every public number *means*.

``test_public_api.py`` pins **which** symbols exist.  This pins **what they
mean**: units, reference frame, amplitude-vs-power, and which attribute a tool's
metric name actually maps to.

Why this file exists.  Three independent users have now reported the same class
of bug: the code runs, returns a plausible number, and the number means
something other than the caller assumed.  Documentation demonstrably does not
prevent it -- in two cases the *author of the documentation* got it wrong
afterwards, and in a third the guide contradicted itself for weeks.  The one
mechanism that did catch it was an independent check against an analytic
answer, which is what these tests are.

Each contract names the mistake it prevents.  If one fails, do not "fix" the
number to match: either the behaviour changed (and the guide plus every caller's
mental model is now wrong), or the contract was written down wrong.  Both need a
human.
"""

import numpy as np
import pytest

from ikarus import RCWA, default_library
from ikarus.tools import convergence_curve

LAM = 729e-9


# -- oracles ------------------------------------------------------------------

def _bare_interface():
    """Air/SiO2 half-space: the one case with a closed-form answer."""
    rcwa = RCWA(period_x=200e-9, period_y=200e-9, resolution=8, n_orders=0)
    rcwa.add_uniform_layer(np.inf, "Air")
    rcwa.add_uniform_layer(np.inf, "SiO2")
    rcwa.set_source(wavelength=LAM, theta=0, polarization="linear",
                    linear_pol_angle=0.0)
    return rcwa


def _binary_grating(period=4.0e-6, lam=1550e-9):
    """A grating with enough propagating orders to check exit angles."""
    rcwa = RCWA(period_x=period, period_y=period, resolution=(256, 4),
                n_orders=(10, 0))
    rcwa.add_uniform_layer(np.inf, "Air")
    topo = np.zeros((256, 4), dtype=int)
    topo[:128, :] = 1
    rcwa.add_layer(600e-9, topo, ["Air", "Si"])
    rcwa.add_uniform_layer(np.inf, "SiO2")
    rcwa.set_source(wavelength=lam, theta=0)
    return rcwa, period, lam


# -- the solver agrees with closed-form physics -------------------------------

def test_matches_analytic_fresnel():
    """The anchor. If this drifts, every contract below is measuring nothing."""
    _, _, res = _bare_interface().simulate()
    n2 = default_library.get("SiO2", LAM).real
    assert res.R_total == pytest.approx(((1 - n2) / (1 + n2)) ** 2, abs=1e-12)
    assert res.T_total == pytest.approx(4 * n2 / (1 + n2) ** 2, abs=1e-12)


# -- what simulate() actually hands back --------------------------------------

def test_tuple_elements_are_the_result_objects():
    """`T, R, result = simulate()` -- the first two are not separate numbers."""
    T, R, res = _bare_interface().simulate()
    assert T is res.T and R is res.R


def test_T_and_R_are_amplitude_not_power():
    """Prevents: `power = T`.

    The guide once claimed these were "plain floats in [0, 1]". They are complex
    zero-order amplitudes whose squared modulus is the power, so reading them as
    power overstates transmission (0.9827 vs 0.9657 here).
    """
    T, R, res = _bare_interface().simulate()
    assert not isinstance(T, float) and not isinstance(R, float)
    assert abs(T) ** 2 == pytest.approx(res.T_total)
    assert abs(R) ** 2 == pytest.approx(res.R_total)
    assert abs(T) != pytest.approx(res.T_total)      # amplitude != power


def test_T_is_power_normalised_not_the_raw_fresnel_coefficient():
    """Prevents: comparing `T` against a textbook `t = 2n1/(n1+n2)`.

    Ikarus normalises so that ``|T|**2`` is the power transmittance, i.e.
    ``T = sqrt(n2/n1) * t_field``. `R` needs no such factor, so it *is* the raw
    Fresnel r -- including its sign.
    """
    T, R, _ = _bare_interface().simulate()
    n2 = default_library.get("SiO2", LAM).real
    t_field = 2.0 / (1.0 + n2)
    assert T.real == pytest.approx(np.sqrt(n2) * t_field)
    assert T.real != pytest.approx(t_field)
    assert R.real == pytest.approx((1 - n2) / (1 + n2))    # signed, negative


def test_phase_attributes_are_radians():
    """Prevents: reading `result.T_phase` as degrees.

    Also pins the physics: reflection off a denser medium flips the sign, so
    `R_phase` is pi, not 180.
    """
    T, R, res = _bare_interface().simulate()
    assert res.T_phase == pytest.approx(np.angle(T))
    assert res.R_phase == pytest.approx(np.angle(R))
    assert res.R_phase == pytest.approx(np.pi, abs=1e-9)


def test_totals_are_sums_over_orders_and_energy_balance_is_their_sum():
    rcwa, _, _ = _binary_grating()
    _, _, res = rcwa.simulate()
    assert res.T_orders.sum() == pytest.approx(res.T_total)
    assert res.R_orders.sum() == pytest.approx(res.R_total)
    assert res.energy_balance == pytest.approx(res.R_total + res.T_total)


# -- reference frames ---------------------------------------------------------

def test_theta_out_trn_is_measured_inside_the_substrate():
    """Prevents: quoting the transmitted angle as the angle in air.

    Both numbers are plausible deflection angles and nothing in the value says
    which frame it is in. Checked against the grating equation in each medium.
    """
    rcwa, period, lam = _binary_grating()
    _, _, res = rcwa.simulate()
    n_sub = float(np.sqrt(res.solution.eps_trn).real.flat[0])
    for order in (1, 2):
        i = res.order_index(order, 0)
        in_substrate = np.degrees(np.arcsin(order * lam / (period * n_sub)))
        in_air = np.degrees(np.arcsin(order * lam / period))
        assert res.theta_out_trn[i] == pytest.approx(in_substrate, abs=1e-6)
        assert res.theta_out_trn[i] != pytest.approx(in_air, abs=1e-3)
        assert res.theta_out_trn_in()[i] == pytest.approx(in_air, abs=1e-6)


def test_theta_out_ref_needs_no_conversion():
    """The cover *is* the medium the reflected light is in."""
    rcwa, period, lam = _binary_grating()
    _, _, res = rcwa.simulate()
    i = res.order_index(1, 0)
    assert res.theta_out_ref[i] == pytest.approx(
        np.degrees(np.arcsin(lam / period)), abs=1e-6)   # cover is Air, n=1


# -- convergence_curve: every metric name maps where it claims ----------------

def test_convergence_curve_metric_names_map_to_the_right_attribute():
    """Prevents: a metric name quietly evaluating something else.

    `metric="T_total"` used to fall through to the energy defect, which is ~0 on
    a converged structure and so looked like a clean result.
    """
    rcwa, _, _ = _binary_grating()
    M = 6
    rcwa.n_orders = (M, 0)
    _, _, res = rcwa.simulate()
    i0 = res.order_index(0, 0)
    expected = {
        "T": res.T_total,
        "R": res.R_total,
        "T0": res.T_orders[i0],
        "energy": abs(res.R_total + res.T_total - 1.0),
        "T_phase": np.degrees(res.T_phase),      # degrees here, radians on the result
        "R_phase": np.degrees(res.R_phase),
    }
    for metric, want in expected.items():
        _, got = convergence_curve(rcwa, [M], metric=metric)
        assert got[0] == pytest.approx(want, abs=1e-9), metric


def test_convergence_curve_phase_is_degrees_while_the_result_is_radians():
    """The inconsistency is pinned, not silently fixed: changing the unit would
    leave every caller running while their tolerance shifted by 57.3x."""
    rcwa, _, _ = _binary_grating()
    rcwa.n_orders = (6, 0)
    _, _, res = rcwa.simulate()
    _, deg = convergence_curve(rcwa, [6], metric="T_phase")
    assert deg[0] == pytest.approx(np.degrees(res.T_phase))
    assert abs(deg[0] / res.T_phase) == pytest.approx(180 / np.pi, rel=1e-6)


# -- inverse design: achieved vs the internal loss -----------------------------

def test_achieved_is_metric_units_and_F_is_the_minimisation_loss():
    """Prevents: quoting `F` as the achieved efficiency. For a `maximize`
    target the guide states `F = 1 - achieved`."""
    from ikarus.inverse import MetaAtom, Target, free, optimize
    from ikarus.shapes import Rectangle
    import warnings

    atom = MetaAtom(period=900e-9, cover="Air", substrate="SiO2")
    atom.add_pattern(Rectangle(width=free(0.2, 0.8), height=1.0),
                     ["Air", "Si"], height=400e-9)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = optimize(atom, Target.maximize("R", at=1550e-9), n_orders=(4, 0),
                       pop=4, n_gen=2, algorithm="ga", verbose=False)
    loss = float(np.ravel(res.F)[0])
    assert res.achieved == pytest.approx(1.0 - loss)
    assert 0.0 <= res.achieved <= 1.0
