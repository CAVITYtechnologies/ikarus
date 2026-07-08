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

    def __init__(self, atom, targets, n_orders, X, F, history, algorithm=None):
        self.atom = atom
        self.targets = targets
        self.n_orders = n_orders
        self.X = X            # best params (dict) or Pareto set (list of dicts)
        self.F = F            # objective value(s)
        self.history = history
        self.multi = len(targets) > 1
        self.algorithm = algorithm   # which engine ran: 'adjoint'/'ga'/'nsga3'/...

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

    @property
    def achieved(self):
        """The optimized result in **metric units** -- what you asked for.

        A float for a single target (e.g. the achieved ``R``), a list for
        multi-objective runs (the best value of each metric across the Pareto
        front).  This is the number to quote; :attr:`F` is the internal
        minimization loss (for ``maximize`` targets, ``F = 1 - achieved``).
        """
        if self.multi:
            return [t.achieved(float(np.min(col)))
                    for t, col in zip(self.targets, np.asarray(self.F).T)]
        return self.targets[0].achieved(float(np.ravel(self.F)[0]))

    def report(self) -> str:
        lines = ["Inverse-design result:"]
        if self.multi:
            lines.append(f"  {len(self.X)} Pareto-optimal designs")
            for t, col in zip(self.targets, np.asarray(self.F).T):
                best = float(np.min(col))
                lines.append(f"    {t.name:<18} best {t.achieved_label} = "
                             f"{t.achieved(best):.4f}  (loss {best:.4f})")
        else:
            t = self.targets[0]
            loss = float(np.ravel(self.F)[0])
            eng = f"  [{self.algorithm}]" if self.algorithm else ""
            lines.append(f"  {t.name}:  {t.achieved_label} = "
                         f"{t.achieved(loss):.4f}   (loss = {loss:.5f}){eng}")
            for k, v in self.params.items():
                if not (k.startswith("px") or "__px" in k):   # hide binary pixel bits
                    lines.append(f"    {k} = {v:.4g}")
        return "\n".join(lines)

    def plot(self, ax=None, savefig: str | None = None):
        """Convergence curve in **metric units** (single-objective runs).

        Plots the optimizer's own objective history translated to the metric
        (positive = better for ``maximize`` targets), plus a marker for the
        final **verified** design (the packaged result, re-simulated with the
        standard solver).  For the adjoint engine the curve is the relaxed
        (gray-density) objective during optimization; for the GA it is the
        best-so-far fitness per generation.
        """
        if self.multi:
            raise ValueError("plot() supports single-objective runs; for a "
                             "Pareto front, scatter np.asarray(result.F)")
        if not self.history:
            raise ValueError("no optimization history was recorded for this run")
        import matplotlib.pyplot as plt
        t = self.targets[0]
        y = [t.achieved(h) for h in self.history]
        if ax is None:
            _, ax = plt.subplots(figsize=(6.5, 4))
        ax.plot(range(1, len(y) + 1), y, lw=2)
        ax.scatter([len(y)], [self.achieved], marker="*", s=140, zorder=5,
                   label=f"final verified: {t.achieved_label} = {self.achieved:.4f}")
        ax.set_xlabel("optimizer iteration")
        ax.set_ylabel(t.achieved_label)
        ax.set_title(t.name)
        ax.legend(frameon=False)
        ax.grid(alpha=0.3)
        if savefig:
            ax.figure.savefig(savefig, dpi=150, bbox_inches="tight")
        return ax


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


def optimize(atom, targets, n_orders=8, algorithm: str = "auto",
             pop: int = 100, n_gen: int = 60, seed: int = 0,
             verbose: bool = True, progress: bool = False,
             verify_n_orders=None, restarts: int = 1,
             **adjoint_options) -> OptimizeResult:
    """Optimize ``atom`` against one or more ``targets``.

    Parameters
    ----------
    atom: :class:`~ikarus.inverse.dof.MetaAtom`
    targets: a :class:`Target` or list of them (>=2 -> multi-objective).
    n_orders: harmonic truncation for every forward solve -- an ``int`` (square
        truncation) or an ``(Mx, My)`` tuple.  Use ``(M, 0)`` for a genuinely
        1-D structure (uniform in y) to avoid paying the full 2-D ``O(M^6)``.
    algorithm: ``'auto'`` (the default -- picks the best engine for the
        problem: gradient-based ``'adjoint'`` for differentiable DOFs such as
        pixel maps and heights, the GA family otherwise), or explicitly one of
        ``'adjoint'``, ``'ga'``, ``'nsga2'``, ``'nsga3'``.
    pop, n_gen: GA-family settings (ignored by ``'adjoint'``).
    seed: optimizer seed (both engines).
    verbose: print progress (the pymoo table, or the adjoint loss every few
        steps).
    progress: show a single progress bar instead (sets ``verbose=False``).
    verify_n_orders: harmonic truncation at which the final design's reported
        ``achieved``/``F`` are computed (``int``/tuple).  Default ``None`` keeps
        the optimization truncation, but a **convergence check** always runs: the
        final design is re-simulated at a higher truncation and, if the metric is
        still moving, a warning is issued (raise ``n_orders``/set this to
        confirm).  Optimizing at a modest ``n_orders`` and verifying higher is
        the recommended, cheap pattern.
    restarts: **adjoint only** -- run this many random-seeded restarts
        (``seed, seed+1, ...``) and keep the best verified design.  Strongly
        recommended for steering/deflection objectives, whose landscape is
        multi-modal (single-seed results vary a lot).
    adjoint_options: forwarded to the adjoint engine -- ``steps``,
        ``learning_rate``, ``min_feature`` (meters), ``beta``, ``init``.  All
        defaulted; see :func:`ikarus.inverse.adjoint.adjoint_optimize`.

    Whatever the engine, the result is the same :class:`OptimizeResult`:
    ``result.rcwa`` is the optimized, ready-to-simulate structure and
    ``result.report()`` summarizes it.
    """
    import warnings
    if isinstance(targets, Target):
        targets = [targets]
    if algorithm == "auto":
        algorithm = _auto_algorithm(atom, targets)
        if algorithm == "adjoint" and (pop, n_gen) != (100, 60):
            warnings.warn(
                "optimize() picked the adjoint engine for this problem, so the "
                "GA settings pop/n_gen are ignored (adjoint uses steps=/"
                "learning_rate=). Pass algorithm='ga' to force the GA.",
                stacklevel=2)

    if restarts > 1 and algorithm != "adjoint":
        warnings.warn(f"restarts only applies to the adjoint engine; ignored "
                      f"for '{algorithm}' (its population already searches "
                      f"broadly).", stacklevel=2)
        restarts = 1

    if algorithm == "adjoint":
        from .adjoint import adjoint_optimize
        result = None
        for i in range(max(1, restarts)):
            r = adjoint_optimize(atom, targets, n_orders=n_orders, seed=seed + i,
                                 verbose=verbose, progress=progress,
                                 **adjoint_options)
            if result is None or _total_loss(r.F) < _total_loss(result.F):
                result = r
        _verify_convergence(result, verify_n_orders)
        return result

    if adjoint_options:
        raise TypeError(f"unexpected keyword(s) for the GA path: "
                        f"{sorted(adjoint_options)} (adjoint-only options)")

    _require_pymoo()
    from pymoo.optimize import minimize as pymoo_minimize

    problem = _build_problem(atom, targets, n_orders)
    algo = _make_algorithm(algorithm, pop, len(targets))

    # Only pass a callback when one exists -- pymoo's default is a no-op Callback,
    # and explicitly handing it ``callback=None`` makes it crash.
    from pymoo.core.callback import Callback

    bar = None
    if progress:
        from .._progress import counter
        bar = counter(n_gen, desc="optimize")
        verbose = False

    ga_history: list = []
    single = len(targets) == 1

    class _TrackCallback(Callback):
        def notify(self, algorithm):  # called once per generation
            if single:                # best-so-far fitness -> result.plot()
                ga_history.append(float(algorithm.opt[0].F[0]))
            if bar is not None:
                bar.update(1)

    res = pymoo_minimize(problem, algo, ("n_gen", n_gen), seed=seed,
                         verbose=verbose, save_history=False,
                         callback=_TrackCallback())
    if bar is not None:
        bar.close()
    result = OptimizeResult(atom, targets, n_orders, res.X, res.F,
                            ga_history if single else None, algorithm=algorithm)
    _verify_convergence(result, verify_n_orders)
    return result


def _total_loss(F) -> float:
    return float(np.sum(np.ravel(np.asarray(F))))


def _bump_orders(n):
    """A higher truncation for the convergence check (keep a 0 axis at 0)."""
    if isinstance(n, (tuple, list)):
        return tuple(0 if int(m) == 0 else max(int(m) + 4, int(np.ceil(1.5 * m)))
                     for m in n)
    return max(int(n) + 4, int(np.ceil(1.5 * n)))


def _design_objective_at(result, n_orders):
    """Evaluate the packaged design's objective(s) at a given truncation
    (single-objective returns a scalar loss)."""
    rcwa = result.atom.build(result.params, n_orders)
    per_wl = {}
    for wl in sorted({w for t in result.targets for w in t.wavelengths}):
        rcwa.set_source(wavelength=wl, theta=0.0,
                        polarization=result.atom.polarization,
                        linear_pol_angle=result.atom.pol_angle)
        per_wl[wl] = rcwa.simulate()[2]
    return [t.objective(per_wl) for t in result.targets]


def _verify_convergence(result, verify_n_orders):
    """Re-evaluate the final design at a higher truncation.  If
    ``verify_n_orders`` is given, report ``achieved``/``F`` there; always warn
    when the metric is still moving with ``n_orders`` (the packaged number is
    honest for its truncation but may not be converged -- exactly what an
    energy-balance check cannot catch)."""
    if result.multi:
        return                       # Pareto-front verification is out of scope
    import warnings
    opt_orders = result.n_orders
    t = result.targets[0]

    if verify_n_orders is not None:
        F_v = _design_objective_at(result, verify_n_orders)
        result.F = np.asarray(F_v[0])
        result.n_orders = verify_n_orders      # .achieved/.rcwa now agree here
        opt_orders = verify_n_orders

    check_orders = _bump_orders(opt_orders)
    if check_orders == opt_orders:
        return
    a_here = t.achieved(_total_loss(result.F))
    a_high = t.achieved(_total_loss(_design_objective_at(result, check_orders)))
    if abs(a_high - a_here) > 0.02:
        warnings.warn(
            f"the optimized design may not be converged in n_orders: "
            f"{t.achieved_label} = {a_here:.3f} at n_orders={opt_orders} but "
            f"{a_high:.3f} at n_orders={check_orders}. The reported value is "
            f"honest for its truncation; raise n_orders (or pass "
            f"verify_n_orders=) and re-check before trusting the headline "
            f"number.", stacklevel=3)
