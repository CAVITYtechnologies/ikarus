"""One-line gradient-free inverse design, backed by pymoo.

``optimize(atom, targets)`` turns a :class:`~ikarus.inverse.dof.MetaAtom` plus a
list of :class:`~ikarus.inverse.targets.Target` into a mixed-variable optimization
problem (binary pixels with bit-flip, continuous parameters with SBX/PM) and runs
a genetic algorithm (single objective) or NSGA-III (multi-objective).

pymoo is an optional dependency::

    pip install pymoo            # or: pip install ikarus-rcwa[inverse]
"""

from __future__ import annotations

import numpy as np

from .targets import Target


def _require_pymoo():
    try:
        import pymoo  # noqa: F401
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "inverse.optimize needs pymoo -- install with `pip install pymoo` "
            "(or `pip install ikarus-rcwa[inverse]`)."
        ) from exc


def _build_problem(atom, targets, n_orders):
    from pymoo.core.problem import ElementwiseProblem
    from pymoo.core.variable import Binary, Real

    wavelengths = sorted({wl for t in targets for wl in t.wavelengths})
    variables = {}
    for name, spec in atom.variables().items():
        if spec[0] == "real":
            variables[name] = Real(bounds=spec[1])
        else:
            variables[name] = Binary()

    class _MetaProblem(ElementwiseProblem):
        def __init__(self):
            super().__init__(vars=variables, n_obj=len(targets))

        def _evaluate(self, X, out, *args, **kwargs):
            rcwa = atom.build(X, n_orders)
            results = {}
            for wl in wavelengths:
                rcwa.set_source(wavelength=wl, theta=0.0,
                                polarization=atom.polarization,
                                linear_pol_angle=atom.pol_angle)
                results[wl] = rcwa.simulate()[2]
            out["F"] = [t.objective(results) for t in targets]

    return _MetaProblem()


def _make_algorithm(name, pop, n_obj):
    from pymoo.core.mixed import (MixedVariableGA, MixedVariableMating,
                                  MixedVariableSampling,
                                  MixedVariableDuplicateElimination)
    if n_obj == 1:
        return MixedVariableGA(pop_size=pop)

    dedup = MixedVariableDuplicateElimination()
    kw = dict(pop_size=pop, sampling=MixedVariableSampling(),
              mating=MixedVariableMating(eliminate_duplicates=dedup),
              eliminate_duplicates=dedup)
    if name in ("nsga2",):
        from pymoo.algorithms.moo.nsga2 import NSGA2
        return NSGA2(**kw)
    from pymoo.algorithms.moo.nsga3 import NSGA3
    from pymoo.util.ref_dirs import get_reference_directions
    ref_dirs = get_reference_directions("das-dennis", n_obj, n_partitions=12)
    return NSGA3(ref_dirs=ref_dirs, **kw)


class OptimizeResult:
    """Outcome of :func:`optimize`."""

    def __init__(self, atom, targets, n_orders, X, F, history):
        self.atom = atom
        self.targets = targets
        self.n_orders = n_orders
        self.X = X            # best params (dict) or Pareto set (list of dicts)
        self.F = F            # objective value(s)
        self.history = history
        self.multi = len(targets) > 1

    @property
    def params(self) -> dict:
        """Best parameter dict (for multi-objective, the first Pareto point)."""
        return self.X[0] if self.multi else self.X

    @property
    def metaatom(self):
        """The optimized structure as a ready-to-simulate :class:`~ikarus.RCWA`."""
        return self.atom.build(self.params, self.n_orders)

    def report(self) -> str:
        lines = ["Inverse-design result:"]
        if self.multi:
            lines.append(f"  {len(self.X)} Pareto-optimal designs")
            for t, col in zip(self.targets, np.asarray(self.F).T):
                lines.append(f"    {t.name:<18} best={np.min(col):.4f}")
        else:
            lines.append(f"  objective = {float(np.ravel(self.F)[0]):.5f}  "
                         f"({self.targets[0].name})")
            for k, v in self.params.items():
                if not k.startswith("px"):
                    lines.append(f"    {k} = {v:.4g}")
        return "\n".join(lines)


def optimize(atom, targets, n_orders: int = 8, algorithm: str = "auto",
             pop: int = 100, n_gen: int = 60, seed: int = 0,
             verbose: bool = True) -> OptimizeResult:
    """Optimize ``atom`` against one or more ``targets``.

    Parameters
    ----------
    atom: :class:`~ikarus.inverse.dof.MetaAtom`
    targets: a :class:`Target` or list of them (>=2 -> multi-objective).
    n_orders: harmonic truncation for every forward solve.
    algorithm: ``'auto'`` (GA if one objective, NSGA-III if several), or one of
        ``'ga'``, ``'nsga2'``, ``'nsga3'``.
    pop, n_gen, seed: optimizer settings.
    """
    _require_pymoo()
    from pymoo.optimize import minimize as pymoo_minimize

    if isinstance(targets, Target):
        targets = [targets]
    problem = _build_problem(atom, targets, n_orders)
    if algorithm == "auto":
        algorithm = "ga" if len(targets) == 1 else "nsga3"
    algo = _make_algorithm(algorithm, pop, len(targets))

    res = pymoo_minimize(problem, algo, ("n_gen", n_gen), seed=seed,
                         verbose=verbose, save_history=False)
    return OptimizeResult(atom, targets, n_orders, res.X, res.F, None)
