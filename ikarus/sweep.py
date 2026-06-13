"""Declarative parameter sweeps with a single progress bar.

A :class:`Sweep` runs one :class:`~ikarus.RCWA` over a grid of **source**
parameters (wavelength, incidence angle, polarization) without a hand-written
``for`` loop, and reports one progress bar for the whole sweep::

    from ikarus import Sweep
    res = Sweep(rcwa).over(wavelength=np.linspace(400e-9, 700e-9, 200)).run()
    res.R_total          # array aligned to the sweep axis

Structural sweeps (height, period, topology) rebuild the structure and are
therefore genuine loops -- write them with :func:`ikarus.progress` instead.
"""

from __future__ import annotations

import itertools

import numpy as np

from ._progress import progress as _progress


class SweepResult:
    """The outcome of :meth:`Sweep.run` -- per-point results plus vectorized
    access to the common scalar metrics, shaped like the sweep grid."""

    def __init__(self, axes: dict, results: np.ndarray):
        self.axes = axes            # {name: 1-D array of swept values}
        self.results = results      # object ndarray of SimulationResult
        self.shape = results.shape

    def _map(self, fn) -> np.ndarray:
        out = np.empty(self.shape, dtype=float)
        for idx, res in np.ndenumerate(self.results):
            out[idx] = fn(res)
        return out

    @property
    def R_total(self) -> np.ndarray:
        return self._map(lambda r: r.R_total)

    @property
    def T_total(self) -> np.ndarray:
        return self._map(lambda r: r.T_total)

    @property
    def energy_balance(self) -> np.ndarray:
        return self._map(lambda r: r.energy_balance)

    def order(self, p: int, q: int, which: str = "T") -> np.ndarray:
        """Efficiency of diffraction order ``(p, q)`` across the sweep
        (``which='T'`` transmitted, ``'R'`` reflected)."""
        attr = which.upper() + "_orders"
        return self._map(lambda r: getattr(r, attr)[r.order_index(p, q)])

    def __repr__(self) -> str:
        ax = ", ".join(f"{k}[{len(v)}]" for k, v in self.axes.items())
        return f"SweepResult({ax})"


class Sweep:
    """Sweep a configured :class:`~ikarus.RCWA` over one or more source axes.

    The structure and any non-swept source fields are taken from ``rcwa`` as
    configured; :meth:`over` declares the axes, :meth:`run` executes them.
    """

    def __init__(self, rcwa):
        self.rcwa = rcwa
        self._axes: dict = {}

    def over(self, **axes) -> "Sweep":
        """Declare swept source parameters, e.g. ``over(wavelength=..., theta=...)``.

        Each keyword is a :meth:`~ikarus.RCWA.set_source` field mapped to a 1-D
        sequence of values.  Multiple keywords form a Cartesian grid (axis order
        = keyword order).
        """
        for name, values in axes.items():
            self._axes[name] = np.asarray(values)
        return self

    def run(self, progress: bool = True, desc: str = "sweep") -> SweepResult:
        """Run the sweep, returning a :class:`SweepResult`.  ``progress`` toggles
        the bar (one bar for the whole sweep)."""
        if not self._axes:
            raise ValueError("call .over(param=values) before .run()")
        names = list(self._axes)
        grids = [self._axes[n] for n in names]
        shape = tuple(len(g) for g in grids)
        results = np.empty(shape, dtype=object)

        combos = list(itertools.product(*(range(len(g)) for g in grids)))
        for idx in _progress(combos, enable=progress, desc=desc, total=len(combos)):
            point = {names[d]: _native(grids[d][i]) for d, i in enumerate(idx)}
            self.rcwa.set_source(**point)
            results[idx] = self.rcwa.simulate()[2]
        return SweepResult(self._axes, results)


def _native(value):
    """Convert a numpy scalar to a plain Python scalar for ``set_source``."""
    return value.item() if isinstance(value, np.generic) else value
