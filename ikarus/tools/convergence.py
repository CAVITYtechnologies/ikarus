"""Automatic harmonic-order convergence testing.

RCWA accuracy is controlled by the number of retained Fourier harmonics
``n_orders``.  Too few orders under-resolves the field; too many wastes time (the
eigensolve is ``O(N^3)`` in the harmonic count).  These utilities sweep
``n_orders`` and pick the smallest value at which the result has stabilized.

**Convergence metric:** the change in the **complex zeroth-order coefficients**
(reflected and transmitted, magnitude *and* phase) between successive order
counts.  This is deliberately *not* the energy balance: a lossless structure
conserves energy at every truncation while its ``R`` and **phase** are still
drifting, so ``|R+T-1|`` is unreliable as a convergence test (here it is used
only to flag numerical breakdown).  Tracking the complex coefficients catches the
phase drift that bites phase-sensitive designs (metalenses, metamirrors).
"""

from __future__ import annotations

import warnings

import numpy as np


def _zeroth_order_coeffs(sol) -> np.ndarray:
    """Complex zeroth-order tangential amplitudes ``[rx, ry, tx, ty]``.

    These capture magnitude **and** phase and are polarization-agnostic (raw
    field amplitudes), so their stability across ``n_orders`` is a faithful
    convergence signal -- unlike the energy balance, which is blind to phase.
    """
    i0 = sol.grid.zero_order_index()
    return np.array([sol.rx[i0], sol.ry[i0], sol.tx[i0], sol.ty[i0]], dtype=complex)


def _coeff_delta(a: np.ndarray, b: np.ndarray) -> float:
    """Largest change in any complex coefficient.  Amplitudes are ``O(1)``, so an
    absolute tolerance reads directly: ``1e-3`` is ~0.06 deg of phase at unit
    amplitude; ``1e-4`` is ~0.006 deg."""
    return float(np.max(np.abs(a - b)))


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

    Convergence is declared when the complex zeroth-order coefficients (``R`` and
    ``T`` -- magnitude *and* phase) stop changing by more than ``tol`` between
    successive truncations.  ``mode='once'`` caches the result so later calls are
    no-ops; ``mode='always'`` re-converges every time.  Returns the chosen
    ``(n_orders_x, n_orders_y)``.  The sweep is isotropic (equal x/y order) for a
    2-D starting point and x-only for a 1-D one (``n_orders=(M, 0)``).
    """
    if mode == "once" and getattr(rcwa, "_converged", False):
        return rcwa.n_orders

    original = rcwa.n_orders
    is_2d = original[1] > 0

    prev = None
    chosen = start
    breakdown = False
    M = start
    while M <= max_orders:
        rcwa.n_orders = (M, M) if is_2d else (M, 0)
        try:
            sol = rcwa._solve()
        except np.linalg.LinAlgError:
            break
        coeff = _zeroth_order_coeffs(sol)
        balance = sol.R_total + sol.T_total
        if verbose:
            r2 = abs(coeff[0]) ** 2 + abs(coeff[1]) ** 2
            print(f"[converge] n_orders={M:>3}  |r0|^2~{r2:.4f}  R+T={balance:.4f}")
        if prev is not None and _coeff_delta(coeff, prev) < tol:
            chosen = M
            breakdown = balance > 1.01
            break
        prev = coeff
        chosen = M
        M += step
    else:
        warnings.warn(
            f"Convergence not reached by max_orders={max_orders} (using "
            f"n_orders={chosen}); the zeroth-order R/T were still changing. "
            "Raise max_orders, increase resolution, or check the structure.",
            RuntimeWarning, stacklevel=2,
        )

    rcwa.n_orders = (chosen, chosen) if is_2d else (chosen, 0)
    rcwa._converged = True
    rcwa._converged_value = rcwa.n_orders
    if breakdown:
        warnings.warn(
            f"Energy balance R+T>1.01 at n_orders={rcwa.n_orders}: possible "
            "numerical breakdown -- try raising `resolution`.",
            RuntimeWarning, stacklevel=2,
        )
    if verbose:
        print(f"[converge] selected n_orders={rcwa.n_orders}")
    return rcwa.n_orders


def check_convergence(rcwa, baseline=None, tol: float = 1e-3, step: int = 4
                      ) -> tuple[bool, float]:
    """Probe whether the current ``n_orders`` is converged.

    Re-solves at a higher truncation (``+step`` per active axis) and compares the
    complex zeroth-order coefficients against ``baseline`` (or a fresh solve).
    Returns ``(is_converged, delta)`` and emits a ``RuntimeWarning`` when not
    converged.  Restores the original ``n_orders`` and last solution, so it is
    side-effect-free apart from the warning.
    """
    original = rcwa.n_orders
    if baseline is None:
        baseline = rcwa._solve()
    base_coeff = _zeroth_order_coeffs(baseline)
    Mx, My = original
    rcwa.n_orders = (Mx + step, My + step) if My > 0 else (Mx + step, 0)
    try:
        delta = _coeff_delta(_zeroth_order_coeffs(rcwa._solve()), base_coeff)
    finally:
        rcwa.n_orders = original
        rcwa._last_solution = baseline
    converged = delta < tol
    if not converged:
        warnings.warn(
            f"Result may not be converged: the zeroth-order R/T coefficients moved "
            f"by {delta:.2e} (> {tol:.0e}) when n_orders rose by {step}. Raise "
            f"n_orders, or use simulate(auto_converge='once') to pick it automatically. "
            "(Energy balance does NOT detect this.)",
            RuntimeWarning, stacklevel=3,
        )
    return converged, delta


def convergence_curve(rcwa, orders, metric: str = "T0") -> tuple[np.ndarray, np.ndarray]:
    """Evaluate a convergence metric over a list of harmonic-order counts.

    Returns ``(orders, values)`` for plotting ``n_orders`` vs. the metric:
    ``'T0'`` (specular transmittance), ``'R'``/``'T'`` (totals), ``'R_phase'``/
    ``'T_phase'`` (zeroth-order phase, **degrees** -- the one to watch for
    phase-sensitive design), or ``'energy'`` (``|R+T-1|``).  Restores the original
    ``n_orders`` afterwards.
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
        elif metric == "R_phase":
            values.append(np.degrees(np.angle(rcwa._linear_coeff(sol, "ref", i0))))
        elif metric == "T_phase":
            values.append(np.degrees(np.angle(rcwa._linear_coeff(sol, "trn", i0))))
        else:  # energy defect
            values.append(abs(sol.R_total + sol.T_total - 1.0))
    rcwa.n_orders = original
    return np.array(orders), np.array(values)
