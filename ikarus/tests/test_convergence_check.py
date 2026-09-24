"""Phase/R-aware convergence checking (the reliability safety net).

These guard the rule that **energy balance is not a convergence test**: the
checks track the complex zeroth-order R/T coefficients (magnitude *and* phase).
"""

import warnings

import numpy as np
import pytest

from ikarus import RCWA
from ikarus.tools.convergence import auto_converge_orders, check_convergence


def _tm_grating(M, factorization="li", n_hi=3.5):
    """Lossless high-contrast 1-D grating in TM -- the slow-converging case."""
    topo = np.zeros((1024, 2), dtype=int)
    topo[512:, :] = 1
    rc = RCWA(period_x=400e-9, period_y=400e-9, resolution=(1024, 2),
              n_orders=(M, 0), factorization=factorization)
    rc.add_uniform_layer(np.inf, "Air")
    rc.add_layer(300e-9, topo, [1.0, n_hi])
    rc.add_uniform_layer(np.inf, "Air")
    rc.set_source(wavelength=700e-9, theta=0, polarization="linear", linear_pol_angle=90)
    return rc


def test_check_convergence_flags_underresolved():
    """Laurent at low M on high-contrast TM is unconverged -> must warn."""
    rc = _tm_grating(8, factorization="laurent")
    with pytest.warns(RuntimeWarning, match="may not be converged"):
        ok, delta = check_convergence(rc, tol=1e-3, step=6)
    assert not ok and delta > 1e-3


def test_check_convergence_passes_when_resolved():
    """Li at modest M on the same grating is converged -> no warning."""
    rc = _tm_grating(16, factorization="li")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        ok, delta = check_convergence(rc, tol=3e-3, step=4)
    assert ok and delta < 3e-3
    assert not any("may not be converged" in str(w.message) for w in caught)


def test_check_convergence_is_side_effect_free():
    rc = _tm_grating(10)
    before = rc.n_orders
    check_convergence(rc, tol=1.0)            # loose tol -> no warning
    assert rc.n_orders == before              # n_orders restored


def test_energy_balance_is_not_convergence():
    """The footgun, pinned: a lossless grating balances energy at *every* M, yet
    is not converged -- check_convergence must catch what energy_balance misses."""
    rc = _tm_grating(8, factorization="laurent")
    _, _, res = rc.simulate()
    assert abs(res.energy_balance - 1.0) < 1e-6   # energy looks perfect ...
    ok, _ = check_convergence(rc, tol=1e-3, step=6)
    assert not ok                                 # ... but it is NOT converged


def test_auto_converge_settles_the_coefficients():
    rc = _tm_grating(5, factorization="li")
    auto_converge_orders(rc, mode="once", tol=1e-3)
    ok, _ = check_convergence(rc, tol=3e-3, step=4)
    assert ok                                     # converged result stays put


def test_auto_converge_handles_absorbing_structure():
    """Regression: an absorbing patterned layer has R+T<1 (legitimately), which
    the old energy-defect criterion never satisfied. It must still converge."""
    topo = np.zeros((256, 2), dtype=int)
    topo[128:, :] = 1
    rc = RCWA(period_x=500e-9, period_y=500e-9, resolution=(512, 2), n_orders=(5, 0))
    rc.add_uniform_layer(np.inf, "Air")
    rc.add_layer(150e-9, topo, [1.0, 3.5 + 0.4j])     # lossy high-index
    rc.add_uniform_layer(np.inf, 1.45)
    rc.set_source(wavelength=633e-9, theta=0, polarization="linear", linear_pol_angle=0)
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)   # 'not reached' would fail
        Mx, _ = auto_converge_orders(rc, mode="once", tol=1e-3, max_orders=60)
    assert Mx < 60                                        # converged before the ceiling
    assert rc.simulate()[2].energy_balance < 1.0         # genuinely absorbing


# --- the Pareto path gets the same safety net ------------------------------
# A multi-objective run used to skip verification entirely, so the recommended
# metamirror idiom -- maximize(R) paired with match(r_phase) -- could hand back
# an artifact-mined design with no warning at all. These guard the fix.

def _front(atom, params_list, n_orders):
    """A Pareto result whose F is the REAL objective at ``n_orders``.

    Inventing F by hand would make the check compare a made-up number against a
    computed one, and warn for the wrong reason.
    """
    from ikarus.inverse import Target
    from ikarus.inverse.optimize import OptimizeResult, _objective_of

    targets = [Target.maximize("R", at=1550e-9),
               Target.match("r_phase", value=0.0, at=1550e-9)]
    result = OptimizeResult(atom, targets, n_orders, list(params_list),
                            np.zeros((len(params_list), 2)), None, algorithm="ga")
    result.F = np.array(
        [[float(np.ravel(np.asarray(o))[0]) for o in _objective_of(result, p, n_orders)]
         for p in params_list], dtype=float)
    return result


def _hard_atom():
    """High-contrast silicon: badly under-resolved at a small truncation."""
    from ikarus.inverse import MetaAtom, free
    from ikarus.shapes import Rectangle
    atom = MetaAtom(period=900e-9, cover="Air", substrate="SiO2",
                    polarization="linear", pol_angle=90.0)
    atom.add_pattern(Rectangle(width=free(0.15, 0.85), height=1.0),
                     ["Air", "Si"], height=free(300e-9, 900e-9))
    return atom, {"shape__width": 0.25, "height": 400e-9}


def test_pareto_front_is_convergence_checked():
    """An under-resolved front must warn, exactly as a single design does."""
    from ikarus.inverse.optimize import _verify_convergence

    atom, params = _hard_atom()
    result = _front(atom, [params], n_orders=3)
    assert result.multi
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _verify_convergence(result, None)
    msgs = [str(w.message) for w in caught if "converged" in str(w.message)]
    assert msgs, "an under-resolved Pareto front produced no warning"
    assert "Pareto" in msgs[0]


def test_pareto_checks_every_metrics_champion():
    """``.achieved`` reports the best of EACH metric, so each champion is
    checked -- not merely the first point on the front."""
    from ikarus.inverse.optimize import _verify_convergence

    atom, bad = _hard_atom()
    good = {"shape__width": 0.55, "height": 700e-9}
    # the under-resolved design is second, so a first-point-only check misses it
    result = _front(atom, [good, bad], n_orders=3)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _verify_convergence(result, None)
    assert [w for w in caught if "converged" in str(w.message)]


def test_pareto_check_is_quiet_when_resolved():
    """No false alarm: a converged front must not warn."""
    from ikarus.inverse import MetaAtom, free
    from ikarus.inverse.optimize import _verify_convergence
    from ikarus.shapes import Rectangle

    atom = MetaAtom(period=400e-9, cover="Air", substrate="Air",
                    polarization="linear", pol_angle=0.0)
    atom.add_pattern(Rectangle(width=free(0.2, 0.8), height=1.0),
                     ["Air", 1.5], height=free(100e-9, 300e-9))
    result = _front(atom, [{"shape__width": 0.5, "height": 200e-9}], n_orders=6)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _verify_convergence(result, None)
    assert not [w for w in caught if "converged" in str(w.message)]


def test_pareto_check_never_breaks_a_finished_run():
    """Verification is a courtesy on an optimisation that already finished; if
    re-evaluating a front point fails, the result must still come back."""
    from ikarus.inverse.optimize import _verify_convergence

    atom, params = _hard_atom()
    result = _front(atom, [params], n_orders=3)
    result.X = [{"shape__width": 0.25}]          # missing 'height' -> build fails
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        _verify_convergence(result, None)        # must not raise


# --- an unrecognized metric must raise, not quietly become the energy defect --

def test_convergence_curve_rejects_unknown_metric():
    """`metric="T_total"` is the plausible typo: T_total is the attribute name
    used everywhere else in the API. It used to fall through to the energy
    defect, which is ~0 for any converged structure -- so it looked like a
    clean convergence result for a metric that was never evaluated."""
    from ikarus import RCWA
    from ikarus.tools import convergence_curve

    rc = RCWA(period_x=500e-9, period_y=500e-9, resolution=(32, 32), n_orders=(3, 3))
    rc.add_uniform_layer(np.inf, "Air")
    rc.add_uniform_layer(220e-9, "Si")
    rc.add_uniform_layer(np.inf, "SiO2")
    rc.set_source(wavelength=729e-9, theta=0)
    with pytest.raises(ValueError, match="unknown metric"):
        convergence_curve(rc, [3, 5], metric="T_total")


def test_convergence_curve_accepts_every_documented_metric():
    from ikarus import RCWA
    from ikarus.tools import convergence_curve
    from ikarus.tools.convergence import _CURVE_METRICS

    rc = RCWA(period_x=500e-9, period_y=500e-9, resolution=(32, 32), n_orders=(3, 3))
    rc.add_uniform_layer(np.inf, "Air")
    rc.add_uniform_layer(220e-9, "Si")
    rc.add_uniform_layer(np.inf, "SiO2")
    rc.set_source(wavelength=729e-9, theta=0)
    for m in _CURVE_METRICS:
        _, vals = convergence_curve(rc, [3, 5], metric=m)
        assert len(vals) == 2 and np.isfinite(vals).all(), m


def test_convergence_curve_phase_is_degrees_not_radians():
    """Pinning the inconsistency rather than silently changing it: switching the
    unit would keep every caller running while shifting their tolerance by 57x,
    which is the worst kind of breaking change."""
    from ikarus import RCWA
    from ikarus.tools import convergence_curve

    rc = RCWA(period_x=500e-9, period_y=500e-9, resolution=(32, 32), n_orders=(3, 3))
    rc.add_uniform_layer(np.inf, "Air")
    rc.add_uniform_layer(220e-9, "Si")
    rc.add_uniform_layer(np.inf, "SiO2")
    rc.set_source(wavelength=729e-9, theta=0)
    _, ph = convergence_curve(rc, [3], metric="T_phase")
    _, _, res = rc.simulate()
    assert np.isclose(ph[0], np.degrees(res.T_phase), atol=1e-6)
