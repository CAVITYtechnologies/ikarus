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

    @property
    def rcwa(self):
        """Alias of :attr:`metaatom` -- the optimized design as a ready
        :class:`~ikarus.RCWA` (clearer when the design is a ``Structure``)."""
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
                if not (k.startswith("px") or "__px" in k):   # hide binary pixel bits
                    lines.append(f"    {k} = {v:.4g}")
        return "\n".join(lines)


def _auto_algorithm(atom, targets) -> str:
    """Pick the engine for ``algorithm='auto'`` -- the user never has to.

    Gradient-based (adjoint) when the problem suits it: a single scalarizable
    objective and differentiable DOFs (pixel maps, heights, periods), with the
    ``[grad]`` extra installed.  Otherwise the GA family: NSGA-III for a full
    Pareto front (>= 2 targets), GA for discrete/parametric-shape DOFs.
    """
    if len(targets) > 1:
        return "nsga3"          # a full Pareto front is a GA-family capability
    try:
        import jax  # noqa: F401
        import optax  # noqa: F401
    except ModuleNotFoundError:
        return "ga"
    from .adjoint import AdjointIncompatible, check_compatible
    try:
        check_compatible(atom, targets)
    except AdjointIncompatible:
        return "ga"
    return "adjoint"


def optimize(atom, targets, n_orders: int = 8, algorithm: str = "auto",
             pop: int = 100, n_gen: int = 60, seed: int = 0,
             verbose: bool = True, progress: bool = False,
             **adjoint_options) -> OptimizeResult:
    """Optimize ``atom`` against one or more ``targets``.

    Parameters
    ----------
    atom: :class:`~ikarus.inverse.dof.MetaAtom`
    targets: a :class:`Target` or list of them (>=2 -> multi-objective).
    n_orders: harmonic truncation for every forward solve.
    algorithm: ``'auto'`` (the default -- picks the best engine for the
        problem: gradient-based ``'adjoint'`` for differentiable DOFs such as
        pixel maps and heights, the GA family otherwise), or explicitly one of
        ``'adjoint'``, ``'ga'``, ``'nsga2'``, ``'nsga3'``.
    pop, n_gen: GA-family settings (ignored by ``'adjoint'``).
    seed: optimizer seed (both engines).
    verbose: print progress (the pymoo table, or the adjoint loss every few
        steps).
    progress: show a single progress bar instead (sets ``verbose=False``).
    adjoint_options: forwarded to the adjoint engine -- ``steps``,
        ``learning_rate``, ``min_feature`` (meters), ``beta``.  All defaulted;
        see :func:`ikarus.inverse.adjoint.adjoint_optimize`.

    Whatever the engine, the result is the same :class:`OptimizeResult`:
    ``result.rcwa`` is the optimized, ready-to-simulate structure and
    ``result.report()`` summarizes it.
    """
    if isinstance(targets, Target):
        targets = [targets]
    if algorithm == "auto":
        algorithm = _auto_algorithm(atom, targets)

    if algorithm == "adjoint":
        from .adjoint import adjoint_optimize
        return adjoint_optimize(atom, targets, n_orders=n_orders, seed=seed,
                                verbose=verbose, progress=progress,
                                **adjoint_options)
    if adjoint_options:
        raise TypeError(f"unexpected keyword(s) for the GA path: "
                        f"{sorted(adjoint_options)} (adjoint-only options)")

    _require_pymoo()
    from pymoo.optimize import minimize as pymoo_minimize

    problem = _build_problem(atom, targets, n_orders)
    algo = _make_algorithm(algorithm, pop, len(targets))

    # Only pass a callback when one exists -- pymoo's default is a no-op Callback,
    # and explicitly handing it ``callback=None`` makes it crash.
    extra, bar = {}, None
    if progress:
        from .._progress import counter
        from pymoo.core.callback import Callback

        bar = counter(n_gen, desc="optimize")

        class _ProgressCallback(Callback):
            def notify(self, algorithm):  # called once per generation
                bar.update(1)

        extra["callback"] = _ProgressCallback()
        verbose = False

    res = pymoo_minimize(problem, algo, ("n_gen", n_gen), seed=seed,
                         verbose=verbose, save_history=False, **extra)
    if bar is not None:
        bar.close()
    return OptimizeResult(atom, targets, n_orders, res.X, res.F, None)
