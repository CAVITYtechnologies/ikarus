"""Tests for adjoint (gradient-based) inverse design -- the plug-and-play path.

The contract: ``optimize(atom, target)`` keeps the exact same user experience
as the GA (same MetaAtom/Target in, same OptimizeResult out), while ``auto``
silently dispatches to the adjoint engine when the problem is differentiable.
Skipped wholesale without the [grad] extra.
"""

import numpy as np
import pytest

jax = pytest.importorskip("jax")
pytest.importorskip("optax")

from ikarus.inverse import MetaAtom, Target, free, optimize, pixels  # noqa: E402
from ikarus.inverse.adjoint import AdjointIncompatible, check_compatible  # noqa: E402
from ikarus.inverse.optimize import OptimizeResult, _auto_algorithm  # noqa: E402
from ikarus.shapes import Circle  # noqa: E402


def _film_atom():
    """Uniform quarter-wave-AR candidate: only the height is free."""
    atom = MetaAtom(period=400e-9, cover="Air", substrate=1.5)
    atom.add_pattern(topology=np.zeros((4, 4), dtype=int), materials=[1.2247],
                     height=free(60e-9, 200e-9))
    return atom


def test_height_dof_reaches_analytic_optimum():
    """Quarter-wave AR: n = sqrt(n_sub) film, R -> 0 at h = lambda/(4n)."""
    res = optimize(_film_atom(), Target.minimize("R", at=600e-9),
                   n_orders=1, steps=80, verbose=False)
    assert isinstance(res, OptimizeResult)
    assert abs(res.params["height"] - 122.47e-9) < 1e-9      # analytic optimum
    assert float(np.ravel(res.F)[0]) < 1e-4                   # R ~ 0
    assert res.history[0] > res.history[-1]                   # it optimized


def test_pixel_topology_improves_and_binarizes():
    atom = MetaAtom(period=650e-9, cover="Air", substrate="SiO2")
    atom.add_pattern(topology=pixels(24, 24, symmetry="c4v"),
                     materials=["Air", 3.5], height=350e-9)
    res = optimize(atom, Target.maximize("R", at=1050e-9, order=None),
                   n_orders=4, steps=60, verbose=False)
    # the loss went down substantially (reference run: 0.70 -> 0.01)
    assert res.history[-1] < 0.5 * res.history[0]
    # the shipped design is truly binary and rebuildable
    topo = atom.pattern["topology"]
    bits = np.array([res.params[f"px{k}"] for k in range(topo.n_free)])
    assert set(np.unique(bits)) <= {0, 1}
    rcwa = res.rcwa
    rcwa.set_source(wavelength=1050e-9, theta=0, polarization="linear")
    _, _, result = rcwa.simulate()
    # the reported objective is exactly the numpy-core value of the design
    assert abs((1.0 - result.R_total) - float(np.ravel(res.F)[0])) < 1e-9
    assert "objective" in res.report()


def test_worst_case_multiwavelength_target_runs():
    """min-max over wavelengths -- the many-objectives mode of the brief."""
    atom = _film_atom()
    t = Target.minimize("R", at=[550e-9, 650e-9], worst_case=True)
    res = optimize(atom, t, n_orders=1, steps=25, verbose=False)
    assert np.isfinite(res.history).all()
    assert res.history[-1] <= res.history[0]


def test_phase_target_differentiates():
    """r_phase metrics flow gradients (metalens/metamirror objectives)."""
    atom = _film_atom()
    t = Target.match("r_phase", value=np.pi / 2, at=600e-9)
    res = optimize(atom, t, n_orders=1, steps=15, verbose=False)
    assert np.isfinite(res.history).all()


def test_auto_selection_rules():
    # differentiable single-target problem -> adjoint
    assert _auto_algorithm(_film_atom(), [Target.minimize("R", at=600e-9)]) \
        == "adjoint"
    # parametric shapes are not differentiable -> GA
    shaped = MetaAtom(period=500e-9, cover="Air", substrate="SiO2")
    shaped.add_pattern(Circle(radius=free(0.1, 0.4), grid_shape=(32, 32)),
                       ["Air", "Si"], height=300e-9)
    assert _auto_algorithm(shaped, [Target.minimize("R", at=600e-9)]) == "ga"
    # a full Pareto front stays with NSGA-III
    two = [Target.maximize("R", at=1064e-9), Target.maximize("R", at=1550e-9)]
    assert _auto_algorithm(_film_atom(), two) == "nsga3"


def test_incompatible_problems_raise_clearly():
    shaped = MetaAtom(period=500e-9, cover="Air", substrate="SiO2")
    shaped.add_pattern(Circle(radius=free(0.1, 0.4), grid_shape=(32, 32)),
                       ["Air", "Si"], height=300e-9)
    with pytest.raises(AdjointIncompatible, match="parametric-shape"):
        check_compatible(shaped, [Target.minimize("R", at=600e-9)])
    aniso = MetaAtom(period=500e-9, cover="Air", substrate="SiO2")
    aniso.add_pattern(np.zeros((8, 8), dtype=int), [(1.5, 1.6, 1.7)],
                      height=300e-9)
    with pytest.raises(AdjointIncompatible, match="anisotropic"):
        check_compatible(aniso, [Target.minimize("R", at=600e-9)])


def test_ga_path_rejects_adjoint_kwargs():
    with pytest.raises(TypeError, match="adjoint-only"):
        optimize(_film_atom(), Target.minimize("R", at=600e-9),
                 algorithm="ga", steps=10)
