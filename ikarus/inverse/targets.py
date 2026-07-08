"""Optimization targets (figures of merit) for inverse design.

A :class:`Target` declares *what you want* from a metaatom -- a metric, a goal
(match a value / maximize / minimize), and the wavelength(s) it applies over.
Each target reduces to a single non-negative objective (lower = better) that the
optimizer minimizes; pass several targets to :func:`ikarus.inverse.optimize` for
multi-objective (Pareto) optimization.

Metrics
-------
* ``'R'`` / ``'T'``       -- diffraction efficiency into ``order`` (default the
  specular ``(0, 0)``; pass ``order=None`` for the total).
* ``'r_co'`` / ``'t_co'`` -- complex zero-order reflection/transmission coefficient.
* ``'r_phase'`` / ``'t_phase'`` -- their phase (radians; matched modulo 2*pi).

Wavelengths
-----------
``at=1550e-9`` (single), ``at=[1064e-9, 1550e-9]`` (discrete), or
``band=(1064e-9, 1550e-9[, n])`` (a sampled continuous range).  Multiple
wavelengths are aggregated by the mean, or the worst case if ``worst_case=True``.
"""

from __future__ import annotations

import numpy as np


def _resolve_wavelengths(at, band):
    if band is not None:
        lo, hi = float(band[0]), float(band[1])
        n = int(band[2]) if len(band) > 2 else 8
        return list(np.linspace(lo, hi, n))
    if at is None:
        raise ValueError("a Target needs at=<wavelength(s)> or band=(lo, hi[, n])")
    return [float(w) for w in np.atleast_1d(at)]


def _co_cross(coeff, which):
    """Pick co/cross from a coefficient that is scalar (linear) or dict (circular)."""
    if isinstance(coeff, dict):
        return coeff["co"] if which.endswith("co") else coeff["cross"]
    # linear polarization: the returned scalar is the co-pol coefficient.
    if which.endswith("cross"):
        return 0.0
    return coeff


class Target:
    """A single figure of merit.  Build with :meth:`match`, :meth:`maximize` or
    :meth:`minimize`."""

    def __init__(self, metric, mode, value=None, at=None, band=None,
                 order=(0, 0), weight=1.0, worst_case=False, name=None):
        self.metric = metric
        self.mode = mode
        self.value = value
        self.order = order
        self.weight = float(weight)
        self.worst_case = bool(worst_case)
        self.wavelengths = _resolve_wavelengths(at, band)
        self.name = name or f"{mode}({metric})"

    # -- constructors ------------------------------------------------------
    @classmethod
    def match(cls, metric, value, at=None, band=None, order=(0, 0), **kw):
        """Drive ``metric`` toward ``value`` (e.g. ``match('r_co', 1)`` = perfect,
        in-phase reflection)."""
        return cls(metric, "match", value, at, band, order, **kw)

    @classmethod
    def maximize(cls, metric, at=None, band=None, order=(0, 0), **kw):
        """Maximize ``metric`` (e.g. ``maximize('R', order=(1, 0))`` = steer power
        into the +1 reflected order)."""
        return cls(metric, "max", None, at, band, order, **kw)

    @classmethod
    def minimize(cls, metric, at=None, band=None, order=(0, 0), **kw):
        """Minimize ``metric`` (e.g. ``minimize('R', band=(...))`` = AR coating)."""
        return cls(metric, "min", None, at, band, order, **kw)

    # -- evaluation --------------------------------------------------------
    def objective(self, results: dict) -> float:
        """Aggregate objective (>= 0, minimize) given ``{wavelength: result}``."""
        losses = [self._loss(results[wl]) for wl in self.wavelengths]
        agg = max(losses) if self.worst_case else float(np.mean(losses))
        return self.weight * agg

    def value_at(self, result):
        """The raw metric value (for reporting)."""
        return self._extract(result)

    # -- reporting in metric units ------------------------------------------
    def achieved(self, objective: float) -> float:
        """Translate an aggregated *objective* (the loss the optimizer
        minimized) back into **metric units** -- what the user actually asked
        for.  For ``mode='max'`` the loss is ``weight * (1 - score)``, so the
        achieved score is ``1 - objective/weight``; for ``'min'`` the loss *is*
        the (weighted) metric; for ``'match'`` it is the residual
        ``|value - target|``.  Exact for the mean aggregation and for the worst
        case (both commute with the affine map)."""
        x = float(objective) / self.weight
        return 1.0 - x if self.mode == "max" else x

    @property
    def achieved_label(self) -> str:
        """How to caption :meth:`achieved` (e.g. ``'R'`` or ``'|r_phase - target|'``)."""
        if self.mode == "match":
            return f"|{self.metric} - target|"
        if self.metric in ("r_co", "t_co", "r_cross", "t_cross"):
            return f"|{self.metric}|"
        return self.metric

    def _loss(self, result) -> float:
        val = self._extract(result)
        if self.mode == "match":
            if self.metric in ("r_phase", "t_phase"):
                return float(abs(np.angle(np.exp(1j * (val - self.value)))))
            return float(abs(val - self.value))
        score = float(abs(val))  # efficiency (real) or amplitude (|complex|)
        return (1.0 - score) if self.mode == "max" else score

    def _extract(self, result):
        m = self.metric
        if m == "R":
            return self._order_efficiency(result.R_orders, result)
        if m == "T":
            return self._order_efficiency(result.T_orders, result)
        if m in ("r_co", "r_cross"):
            return _co_cross(result.R, m)
        if m in ("t_co", "t_cross"):
            return _co_cross(result.T, m)
        if m == "r_phase":
            return float(np.angle(_co_cross(result.R, "r_co")))
        if m == "t_phase":
            return float(np.angle(_co_cross(result.T, "t_co")))
        raise ValueError(f"unknown metric {m!r}")

    def _order_efficiency(self, orders, result):
        if self.order in (None, "total"):
            return float(np.sum(orders))
        return float(orders[result.order_index(*self.order)])

    def __repr__(self):
        wl = f"{self.wavelengths[0]*1e9:.0f}nm" if len(self.wavelengths) == 1 \
            else f"{len(self.wavelengths)} wavelengths"
        tgt = f"->{self.value}" if self.mode == "match" else ""
        return f"Target[{self.mode} {self.metric}{tgt} @ {wl}]"
