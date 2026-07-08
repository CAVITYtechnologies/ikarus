"""Degree-of-freedom definitions for inverse design.

A :class:`MetaAtom` is a normal Ikarus structure in which selected quantities are
marked as *free* degrees of freedom: continuous parameters via :func:`free` and a
binary pixel map via :func:`pixels`.  Given a parameter assignment (produced by an
optimizer) it builds a concrete :class:`~ikarus.RCWA` object.

All lengths are SI (meters).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..core.rcwa import RCWA
from ..shapes.parametric import Shape


@dataclass
class Free:
    """A continuous degree of freedom bounded to ``[low, high]`` (SI units)."""

    low: float
    high: float


def free(low: float, high: float) -> Free:
    """Mark a continuous parameter (e.g. a height or period) as a free DOF."""
    return Free(float(low), float(high))


class Pixels:
    """A binary pixel-map DOF, optionally constrained by a point symmetry.

    ``symmetry`` reduces the number of independent pixels (and enforces the
    corresponding structural symmetry): ``None``, ``'mirror_x'``, ``'mirror_y'``,
    ``'mirror_xy'``, ``'c2'``, ``'c4'`` or ``'c4v'`` (the last two require a square
    grid).
    """

    def __init__(self, nx: int, ny: int, symmetry: str | None = None):
        self.nx, self.ny = int(nx), int(ny)
        self.symmetry = symmetry
        self._index = _symmetry_index_map(self.nx, self.ny, symmetry)
        self.n_free = int(self._index.max()) + 1

    def expand(self, bits) -> np.ndarray:
        """Expand ``n_free`` independent bits to the full ``(nx, ny)`` 0/1 grid."""
        bits = np.asarray(bits).astype(int)
        return bits[self._index]


def pixels(nx: int, ny: int, symmetry: str | None = None) -> Pixels:
    """Mark the patterned-layer topology as a free binary pixel map."""
    return Pixels(nx, ny, symmetry)


def _generators(symmetry, nx, ny):
    """Generator maps ``(i, j) -> (i', j')`` for the chosen symmetry group."""
    mx = lambda i, j: (i, ny - 1 - j)            # mirror across the vertical axis
    my = lambda i, j: (nx - 1 - i, j)            # mirror across the horizontal axis
    c2 = lambda i, j: (nx - 1 - i, ny - 1 - j)   # 180 deg rotation
    r90 = lambda i, j: (j, nx - 1 - i)           # 90 deg rotation (square only)
    diag = lambda i, j: (j, i)                    # diagonal mirror (square only)
    table = {
        None: [], "none": [],
        "mirror_x": [mx], "mirror_y": [my], "mirror_xy": [mx, my],
        "c2": [c2], "c4": [r90], "c4v": [r90, diag],
    }
    if symmetry not in table:
        raise ValueError(f"unknown symmetry {symmetry!r}")
    if symmetry in ("c4", "c4v") and nx != ny:
        raise ValueError(f"symmetry {symmetry!r} requires a square pixel grid")
    return table[symmetry]


def _symmetry_index_map(nx: int, ny: int, symmetry) -> np.ndarray:
    """Map each pixel to an independent-DOF index (shared within a symmetry orbit)."""
    if symmetry in (None, "none"):
        return np.arange(nx * ny).reshape(nx, ny)
    gens = _generators(symmetry, nx, ny)
    idx = -np.ones((nx, ny), dtype=int)
    next_id = 0
    for i in range(nx):
        for j in range(ny):
            if idx[i, j] >= 0:
                continue
            orbit, frontier = {(i, j)}, [(i, j)]
            while frontier:
                a, b = frontier.pop()
                for g in gens:
                    c = g(a, b)
                    if c not in orbit:
                        orbit.add(c)
                        frontier.append(c)
            for a, b in orbit:
                idx[a, b] = next_id
            next_id += 1
    return idx


class MetaAtom:
    """A parameterized 3-region metaatom: cover / patterned layer / substrate.

    ``period`` may be a fixed float (square cell), a fixed ``(period_x,
    period_y)`` tuple (rectangular cell), or a :func:`free` range (square, a
    single DOF).  The pattern ``height`` may be a fixed float or :func:`free`;
    the ``topology`` a fixed integer array or a :func:`pixels` map.
    :meth:`build` turns a parameter dict into an :class:`~ikarus.RCWA`.
    """

    def __init__(self, period, cover, substrate,
                 polarization: str = "linear", pol_angle: float = 0.0):
        self.period = self._validate_period(period)
        self.cover = cover
        self.substrate = substrate
        self.polarization = polarization
        self.pol_angle = pol_angle
        self.pattern: dict | None = None

    @staticmethod
    def _validate_period(period):
        """Accept a positive number, a ``free(...)`` range, or a fixed
        ``(period_x, period_y)`` tuple; reject anything else with a clear
        message (the alternative -- a raw ``float()`` TypeError three calls
        deep in the optimizer -- is impossible to trace back)."""
        if isinstance(period, Free):
            return period
        if isinstance(period, (tuple, list)):
            if len(period) != 2:
                raise ValueError(
                    "period tuple must be (period_x, period_y) for a "
                    f"rectangular cell, got length {len(period)}")
            if any(isinstance(p, Free) for p in period):
                raise ValueError(
                    "a free rectangular period is not supported; use a single "
                    "free(lo, hi) for a square cell, or a fixed "
                    "(period_x, period_y) tuple")
            px, py = float(period[0]), float(period[1])
            if px <= 0 or py <= 0:
                raise ValueError("periods must be positive (meters)")
            return (px, py)
        if isinstance(period, (int, float)) and not isinstance(period, bool):
            if period <= 0:
                raise ValueError("period must be positive (meters)")
            return float(period)
        raise TypeError(
            "period must be a positive number, a free(lo, hi) range, or a "
            f"fixed (period_x, period_y) tuple; got {type(period).__name__}")

    def period_xy(self, params: dict) -> tuple[float, float]:
        """Resolve ``period`` to ``(period_x, period_y)`` for a parameter dict."""
        p = self.period
        if isinstance(p, Free):
            v = float(params["period"])
            return v, v
        if isinstance(p, tuple):
            return p
        return p, p

    def add_pattern(self, topology, materials, height) -> "MetaAtom":
        """Add the single patterned layer (Si-on-substrate metaatom)."""
        self.pattern = {"topology": topology, "materials": list(materials),
                        "height": height}
        return self

    # -- DOF enumeration ---------------------------------------------------
    def variables(self) -> dict:
        """Return ``{name: ('real', (lo, hi)) | ('binary',)}`` for the optimizer."""
        if self.pattern is None:
            raise ValueError("call add_pattern(...) before optimizing")
        v: dict = {}
        if isinstance(self.period, Free):
            v["period"] = ("real", (self.period.low, self.period.high))
        h = self.pattern["height"]
        if isinstance(h, Free):
            v["height"] = ("real", (h.low, h.high))
        topo = self.pattern["topology"]
        if isinstance(topo, Pixels):
            for k in range(topo.n_free):
                v[f"px{k}"] = ("binary",)
        elif isinstance(topo, Shape):
            for name, (lo, hi) in topo.free_parameters().items():
                v[f"shape__{name}"] = ("real", (lo, hi))
        return v

    @property
    def n_dof(self) -> int:
        return len(self.variables())

    # -- build a concrete RCWA ---------------------------------------------
    def _resolution(self):
        topo = self.pattern["topology"]
        if isinstance(topo, Pixels):
            return (topo.nx, topo.ny)
        if isinstance(topo, Shape):
            return topo.grid_shape
        return np.asarray(topo).shape

    def build(self, params: dict, n_orders: int) -> RCWA:
        """Construct the RCWA structure for a parameter assignment (no source set)."""
        px, py = self.period_xy(params)
        # n_orders may be an int (isotropic truncation) or an (Mx, My) tuple --
        # RCWA's own coercion handles both.
        rcwa = RCWA(period_x=px, period_y=py,
                    resolution=self._resolution(), n_orders=n_orders)
        rcwa.add_uniform_layer(np.inf, self.cover)

        topo = self.pattern["topology"]
        if isinstance(topo, Pixels):
            bits = np.array([params[f"px{k}"] for k in range(topo.n_free)])
            grid = topo.expand(bits)
        elif isinstance(topo, Shape):
            overrides = {name: params[f"shape__{name}"]
                         for name in topo.free_parameters()}
            grid = topo.to_grid(self._resolution(), overrides)
        else:
            grid = np.asarray(topo).astype(int)

        h = self.pattern["height"]
        height = params["height"] if isinstance(h, Free) else float(h)
        rcwa.add_layer(height, grid, self.pattern["materials"])
        rcwa.add_uniform_layer(np.inf, self.substrate)
        return rcwa
