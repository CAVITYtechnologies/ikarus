"""Shared input checks for the public constructors.

Ikarus works in SI metres throughout.  The commonest mistake -- and by far the
commonest one in code written by an assistant -- is passing nanometres
(``wavelength=1550``) where metres (``wavelength=1550e-9``) are meant.  The
number is positive, so every existing check passes, the solver runs happily,
and it returns a confident, plausible, completely wrong answer.  That is the
dangerous failure: a crash gets noticed, a wrong reflectance gets published.

These helpers make that mistake audible.
"""

from __future__ import annotations

import warnings

import numpy as np

# A length at or above this is almost certainly a unit slip.  THz and mm-wave
# metasurfaces legitimately reach a few millimetres, so this *warns* rather
# than raising -- a heuristic should never be the thing that stops a valid run.
# ponytail: one flat threshold; if mm-wave users find it noisy, key it off the
# period/wavelength ratio instead, which is unit-free.
SUSPICIOUS_LENGTH_M = 1e-3


def warn_if_not_metres(value, name: str) -> None:
    """Warn when ``value`` looks like nanometres passed as metres.

    ``inf`` is normal (semi-infinite layers) and never warns.
    """
    if value is None or not np.isfinite(value) or value < SUSPICIOUS_LENGTH_M:
        return
    warnings.warn(
        f"{name} = {value:g} m, which is {value * 1e9:.3g} nm. Ikarus works in "
        f"metres -- did you mean {name}={value:g}e-9? Lengths above 1 mm are "
        f"unusual outside THz/mm-wave work; if you meant it, ignore this.",
        stacklevel=3,
    )


# The solver eigendecomposes a 2P x 2P matrix, P = (2Mx+1)(2My+1) harmonics, so
# wall time scales like P^3 -- O(M^6) for a 2-D structure. P = 1089 is (16, 16),
# already minutes; the sizes above that are hours to never. A human feels this
# coming from experience; an assistant writing `n_orders=(50, 50)` does not.
# ponytail: flat harmonic-count threshold, not a time estimate -- a real timing
# model would need per-machine calibration for no extra decision value.
SLOW_HARMONIC_COUNT = 1089


def warn_if_expensive(grid_size: int, n_orders) -> None:
    """Warn before a truncation that will take minutes to hours to solve."""
    if grid_size <= SLOW_HARMONIC_COUNT:
        return
    warnings.warn(
        f"n_orders={n_orders} keeps {grid_size} harmonics, so the solver "
        f"eigendecomposes a {2 * grid_size}x{2 * grid_size} matrix per layer. "
        f"Cost grows like this count cubed (O(M^6) in 2-D), so expect minutes "
        f"to hours. If you want accuracy, raise n_orders gradually and watch "
        f"the result converge; for a 1-D grating use n_orders=(M, 0), which is "
        f"linear rather than quadratic in M.",
        stacklevel=3,
    )


def check_count(value, name: str, minimum: int, why: str) -> None:
    """Reject counts below ``minimum``, per axis.

    ``n_orders`` and ``resolution`` accept a scalar or an ``(x, y)`` pair, and
    the two have different floors: ``n_orders=(81, 0)`` is a perfectly good 1-D
    grating (no harmonics in y), whereas a resolution of 0 pixels is never
    meaningful.
    """
    pair = value if isinstance(value, (tuple, list)) else (value,)
    for v in pair:
        if int(v) < minimum:
            raise ValueError(
                f"{name} must be >= {minimum} on every axis, got {value!r}. {why}"
            )
