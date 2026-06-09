"""Automatic harmonic-order convergence testing.

RCWA accuracy is controlled by the number of retained Fourier harmonics
``n_orders``.  Too few orders under-resolves the field; too many wastes time (the
eigensolve is ``O(N^3)`` in the harmonic count).  These utilities sweep
``n_orders`` and pick the smallest value at which the result has stabilized.

Convergence metric: the change in the specular (0th-order) transmittance between
successive order counts, combined with the energy-balance defect ``|R+T-1|`` for
lossless checks.
"""

from __future__ import annotations

import warnings

import numpy as np


def _metric(rcwa) -> tuple[float, float]:
    """Return ``(specular_T, energy_defect)`` at the current ``n_orders``."""
    sol = rcwa._solve()
    i0 = sol.grid.zero_order_index()
    return float(sol.T_orders[i0]), abs(sol.R_total + sol.T_total - 1.0)


def auto_converge_orders(
    rcwa,
    mode: str = "once",
    tol: float = 1e-4,
    max_orders: int = 200,
    start: int = 5,
    step: int = 4,
    verbose: bool = False,
) -> tuple[int, int]:
    """Find and set the smallest converged ``n_orders`` on ``rcwa``.

    ``mode='once'`` caches the result so subsequent calls are no-ops;
    ``mode='always'`` re-converges every time.  Returns the chosen
    ``(n_orders_x, n_orders_y)``.  The sweep is isotropic (equal x/y order); for
    strongly 1-D structures set ``n_orders`` manually instead.
    """
    if mode == "once" and getattr(rcwa, "_converged", False):
        return rcwa.n_orders

    # Respect the user's intent: a starting n_orders of (M, 0) converges as a 1-D
    # (x-only) sweep; (M, M) converges isotropically in 2-D.  (Inferring 2-D from
    # the topology shape is wrong -- a 1-D grating is stored as an (Nx, 2) map.)
    original = rcwa.n_orders
    is_2d = original[1] > 0

    prev_T = None
    chosen = original
    M = start
    while M <= max_orders:
        rcwa.n_orders = (M, M) if is_2d else (M, 0)
        try:
            spec_T, defect = _metric(rcwa)
        except np.linalg.LinAlgError:
            break
        if verbose:
            print(f"[converge] n_orders={M:>3}  T0={spec_T:.6f}  |R+T-1|={defect:.2e}")
        if prev_T is not None:
            rel = abs(spec_T - prev_T) / max(abs(spec_T), 1e-12)
            if rel < tol and defect < max(tol, 1e-3):
                chosen = M
                break
        prev_T = spec_T
        chosen = M
        M += step
    else:
        warnings.warn(
            f"Convergence not reached by max_orders={max_orders}; using "
            f"n_orders={chosen}. Increase max_orders or check the structure.",
            RuntimeWarning, stacklevel=2,
        )

    rcwa.n_orders = (chosen, chosen) if is_2d else (chosen, 0)
    rcwa._converged = True
    rcwa._converged_value = rcwa.n_orders
    if verbose:
        print(f"[converge] selected n_orders={rcwa.n_orders}")
    return rcwa.n_orders


def convergence_curve(rcwa, orders, metric: str = "T0") -> tuple[np.ndarray, np.ndarray]:
    """Evaluate a convergence metric over a list of harmonic-order counts.

    Returns ``(orders, values)`` for plotting ``n_orders`` vs. the metric, where
    ``metric`` is ``'T0'`` (specular transmittance), ``'R'``/``'T'`` (totals) or
    ``'energy'`` (``|R+T-1|``).  Restores the original ``n_orders`` afterwards.
    """
    original = rcwa.n_orders
    orders = list(orders)
    values = []
    for M in orders:
        rcwa.n_orders = (M, M) if original[1] > 0 else (M, 0)
        sol = rcwa._solve()
        i0 = sol.grid.zero_order_index()
        if metric == "T0":
            values.append(float(sol.T_orders[i0]))
        elif metric == "R":
            values.append(sol.R_total)
        elif metric == "T":
            values.append(sol.T_total)
        else:  # energy defect
            values.append(abs(sol.R_total + sol.T_total - 1.0))
    rcwa.n_orders = original
    return np.array(orders), np.array(values)
