"""Adjoint (gradient-based) inverse design -- same API, different engine.

``optimize(atom, target)`` dispatches here (instead of the GA) when the problem
suits gradients: pixel-map and/or continuous DOFs, scalarizable objectives.
The user experience is unchanged -- same ``MetaAtom`` / ``Target`` inputs, same
:class:`~ikarus.inverse.optimize.OptimizeResult` out -- but the engine is
reverse-mode differentiation through :mod:`ikarus.grad` (the adjoint method):
the gradient with respect to *every* pixel costs about one extra forward solve,
so freeform topologies scale to thousands of DOFs where a GA cannot follow.

Pixel maps are optimized with the standard relax-and-project scheme
(:mod:`ikarus.grad.topology`): continuous densities, a conic minimum-feature
filter, and a beta-ramped tanh projection toward binary.  The final design is
hard-thresholded and re-evaluated with the **NumPy core** -- the reported
objective is exactly what ``result.rcwa.simulate()`` reproduces.

Requires the optional ``[grad]`` extra (JAX + optax).
"""

from __future__ import annotations

import numpy as np

from ..core.materials import default_library
from ..core.fourier import HarmonicGrid
from ..core.source import Source
from .dof import Free, MetaAtom, Pixels
from .targets import Target


class AdjointIncompatible(ValueError):
    """The problem has DOFs or metrics the gradient engine does not support."""


def _require_grad():
    try:
        import jax  # noqa: F401
        import optax  # noqa: F401
    except ModuleNotFoundError as exc:
        raise ImportError(
            "algorithm='adjoint' needs JAX and optax -- install with "
            "pip install \"ikarus-rcwa[grad]\"") from exc


def check_compatible(atom, targets) -> None:
    """Raise :class:`AdjointIncompatible` if this problem needs the GA."""
    if not isinstance(atom, MetaAtom):
        raise AdjointIncompatible(
            "adjoint optimization currently supports MetaAtom only "
            "(Structure stays on the GA path)")
    if atom.pattern is None:
        raise AdjointIncompatible("call add_pattern(...) first")
    topo = atom.pattern["topology"]
    from ..shapes.parametric import Shape
    if isinstance(topo, Shape):
        raise AdjointIncompatible(
            "parametric-shape DOFs are not differentiable (binary "
            "rasterization); use the GA for Shape parameters")
    if isinstance(topo, Pixels) and len(atom.pattern["materials"]) != 2:
        raise AdjointIncompatible("pixel-map DOFs need exactly two materials")
    for spec in ([atom.cover, atom.substrate] + list(atom.pattern["materials"])):
        if default_library.is_anisotropic(spec):
            raise AdjointIncompatible(
                "anisotropic materials are not yet supported by ikarus.grad")
    if atom.polarization in ("RCP", "LCP"):
        for t in targets:
            if t.metric not in ("R", "T"):
                raise AdjointIncompatible(
                    "circular-polarization coefficient metrics (co/cross) "
                    "are not yet differentiable; use the GA")


# ---------------------------------------------------------------------------
# Differentiable Target evaluation (mirrors targets.Target semantics)
# ---------------------------------------------------------------------------
def _order_flat_index(grid: HarmonicGrid, order) -> int:
    p, q = grid.index_arrays()
    hit = np.where((p == order[0]) & (q == order[1]))[0]
    if hit.size == 0:
        raise ValueError(f"order {order} not in the truncated set")
    return int(hit[0])


def _target_loss_grad(target: Target, sols: dict, grid: HarmonicGrid, pol):
    """Differentiable twin of ``Target.objective`` on GradSolutions per lambda."""
    import jax.numpy as jnp

    i0 = grid.zero_order_index()

    def metric_value(sol):
        m = target.metric
        if m in ("R", "T"):
            orders = sol.R_orders if m == "R" else sol.T_orders
            if target.order in (None, "total"):
                return jnp.sum(orders)
            return orders[_order_flat_index(grid, target.order)]
        # zero-order co-pol coefficient / phase (linear polarization)
        if m in ("r_co", "r_phase"):
            fx, fy, fz = sol.rx[i0], sol.ry[i0], sol.rz[i0]
            eff = sol.R_orders[i0]
        elif m in ("t_co", "t_phase"):
            fx, fy, fz = sol.tx[i0], sol.ty[i0], sol.tz[i0]
            eff = sol.T_orders[i0]
        else:
            raise AdjointIncompatible(f"metric {m!r} is not differentiable yet")
        proj = fx * np.conj(pol[0]) + fy * np.conj(pol[1]) + fz * np.conj(pol[2])
        if m.endswith("phase"):
            return jnp.angle(proj)
        return jnp.sqrt(jnp.maximum(eff, 0.0))       # |coeff| == sqrt(efficiency)

    def loss_at(sol):
        val = metric_value(sol)
        if target.mode == "match":
            if target.metric.endswith("phase"):
                return jnp.abs(jnp.angle(jnp.exp(1j * (val - target.value))))
            return jnp.abs(val - target.value)
        score = jnp.abs(val)
        return (1.0 - score) if target.mode == "max" else score

    losses = jnp.stack([loss_at(sols[wl]) for wl in target.wavelengths])
    agg = jnp.max(losses) if target.worst_case else jnp.mean(losses)
    return target.weight * agg


# ---------------------------------------------------------------------------
# The optimizer
# ---------------------------------------------------------------------------
def adjoint_optimize(atom, targets, n_orders: int = 8, steps: int = 150,
                     learning_rate: float = 0.05, min_feature: float = None,
                     beta: tuple = (8.0, 256.0), seed: int = 0,
                     verbose: bool = True, progress: bool = False):
    """Gradient-based drop-in for the GA path of :func:`ikarus.inverse.optimize`.

    Same inputs and result type as the GA; see :func:`optimize` for the shared
    parameters.  Adjoint-specific knobs (all defaulted -- plug and play):

    ``steps``          optimizer iterations (each ~1.5 forward solves per
                       wavelength).
    ``learning_rate``  Adam step on the normalized parameters.
    ``min_feature``    minimum feature size in meters for pixel maps (conic
                       filter radius = half this).  Default: two pixels.
    ``beta``           ``(beta_min, beta_max)`` binarization ramp.
    """
    _require_grad()
    import jax
    import jax.numpy as jnp
    import optax
    from ..grad import solve, tangent_fields_for
    from ..grad.topology import (beta_schedule, conic_filter, conic_kernel_fft,
                                 tanh_projection)

    if isinstance(targets, Target):
        targets = [targets]
    check_compatible(atom, targets)

    # ---- static problem data -------------------------------------------
    wavelengths = sorted({wl for t in targets for wl in t.wavelengths})
    grid = HarmonicGrid(n_orders, n_orders)
    topo = atom.pattern["topology"]
    mats = atom.pattern["materials"]
    is_pixels = isinstance(topo, Pixels)
    nx, ny = (topo.nx, topo.ny) if is_pixels else np.asarray(topo).shape

    per_wl = {}
    for wl in wavelengths:
        eps_cover = complex(default_library.permittivity(atom.cover, wl))
        eps_sub = complex(default_library.permittivity(atom.substrate, wl))
        eps_mats = [complex(default_library.permittivity(m, wl)) for m in mats]
        per_wl[wl] = (eps_cover, eps_sub, eps_mats)

    src = Source(wavelength=wavelengths[0], theta=0.0, phi=0.0,
                 polarization=atom.polarization,
                 linear_pol_angle=atom.pol_angle)
    src.n_incident = np.sqrt(per_wl[wavelengths[0]][0]).real
    pol = np.asarray(src.polarization_vector())

    # ---- parameter vector: [pixels | height01 | period01] ----------------
    height_free = isinstance(atom.pattern["height"], Free)
    period_free = isinstance(atom.period, Free)
    n_pix = topo.n_free if is_pixels else 0

    def scalars(theta):
        k = n_pix
        h = atom.pattern["height"]
        if height_free:
            height = h.low + theta[k] * (h.high - h.low)
            k += 1
        else:
            height = float(h)
        if period_free:
            period = atom.period.low + theta[k] * (atom.period.high - atom.period.low)
        else:
            period = float(atom.period)
        return height, period

    period0 = atom.period.low if period_free else float(atom.period)
    if min_feature is None:
        radius_px = 2.0
    else:
        radius_px = 0.5 * min_feature / (period0 / nx)
    kfft = jnp.asarray(conic_kernel_fft((nx, ny), radius_px)) if is_pixels else None
    index_map = topo._index if is_pixels else None
    fixed_topo = None if is_pixels else np.asarray(topo).astype(int)

    def density(theta, beta_now):
        """theta -> filtered+projected density grid in [0, 1]."""
        rho = theta[:n_pix][index_map]
        rho = conic_filter(rho, kfft)
        return tanh_projection(rho, beta_now)

    def eps_grids_for(theta, beta_now):
        height, period = scalars(theta)
        grids = {}
        for wl in wavelengths:
            _, _, eps_mats = per_wl[wl]
            if is_pixels:
                rho = density(theta, beta_now)
                grids[wl] = eps_mats[0] + rho * (eps_mats[1] - eps_mats[0])
            else:
                values = jnp.asarray(eps_mats)
                grids[wl] = values[fixed_topo]
        return grids, height, period

    def loss_fn(theta, beta_now, tfields):
        grids, height, period = eps_grids_for(theta, beta_now)
        total = 0.0
        sols = {}
        for wl in wavelengths:
            eps_cover, eps_sub, _ = per_wl[wl]
            sols[wl] = solve([grids[wl]], [height], eps_cover, eps_sub, grid,
                             0.0, 0.0, period, period, wl, (pol[0], pol[1]),
                             factorization="auto", tangent_fields=tfields)
        for t in targets:
            total = total + _target_loss_grad(t, sols, grid, pol)
        return total

    value_and_grad = jax.jit(jax.value_and_grad(loss_fn))

    # ---- optimize --------------------------------------------------------
    rng = np.random.default_rng(seed)
    theta0 = []
    if n_pix:
        theta0.append(np.full(n_pix, 0.5) + 0.01 * rng.standard_normal(n_pix))
    if height_free:
        theta0.append([0.5])
    if period_free:
        theta0.append([0.5])
    theta = jnp.asarray(np.clip(np.concatenate(theta0), 0.0, 1.0))
    if theta.size == 0:
        raise AdjointIncompatible("no free DOFs to optimize")

    opt = optax.adam(learning_rate)
    opt_state = opt.init(theta)
    betas = beta_schedule(steps, *beta) if is_pixels else np.ones(steps)
    history = []

    bar = None
    if progress:
        from .._progress import counter
        bar = counter(steps, desc="adjoint")

    for it in range(steps):
        # Tangent fields are constants per iteration, computed OUTSIDE the
        # traced loss from the current (eagerly evaluated) density.
        grids, _, _ = eps_grids_for(theta, betas[it])
        tfields = tangent_fields_for([np.asarray(grids[wavelengths[0]])])
        val, g = value_and_grad(theta, betas[it], tfields)
        updates, opt_state = opt.update(g, opt_state)
        theta = jnp.clip(theta + updates, 0.0, 1.0)
        history.append(float(val))
        if bar is not None:
            bar.update(1)
        elif verbose and (it % max(1, steps // 10) == 0 or it == steps - 1):
            print(f"  adjoint step {it:4d}/{steps}  loss = {float(val):.5f}")
    if bar is not None:
        bar.close()

    # ---- binarize, re-evaluate with the NumPy core, package --------------
    params = {}
    if is_pixels:
        rho_free = np.asarray(theta[:n_pix])
        filtered = np.asarray(conic_filter(jnp.asarray(rho_free[index_map]), kfft))
        bits_full = (filtered >= 0.5)
        # collapse back to independent DOFs (orbit-consistent by construction)
        bits = np.zeros(n_pix, dtype=int)
        bits[index_map.ravel()] = bits_full.ravel().astype(int)
        for k in range(n_pix):
            params[f"px{k}"] = int(bits[k])
    height, period = (float(x) for x in scalars(theta))
    if height_free:
        params["height"] = height
    if period_free:
        params["period"] = period

    rcwa = atom.build(params, n_orders)
    results = {}
    for wl in wavelengths:
        rcwa.set_source(wavelength=wl, theta=0.0, polarization=atom.polarization,
                        linear_pol_angle=atom.pol_angle)
        results[wl] = rcwa.simulate()[2]
    F = [t.objective(results) for t in targets]

    from .optimize import OptimizeResult
    return OptimizeResult(atom, targets, n_orders, params,
                          np.asarray(F if len(F) > 1 else F[0]), history)
