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
