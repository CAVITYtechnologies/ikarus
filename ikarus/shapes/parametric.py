"""Parametric shape classes for topology generation and inverse design.

Unlike the functional primitives in :mod:`ikarus.shapes.primitives` (which return
a fixed integer array), a :class:`Shape` carries *named parameters* and a
rotation ``angle``.  Any parameter may be a plain number **or** a ``free(lo, hi)``
range (from :mod:`ikarus.inverse`); when the shape is used as a metaatom topology,
the free parameters automatically become optimization degrees of freedom.

All coordinates are fractional unit-cell units in ``[0, 1)`` -- a shape is
resolution-independent and only rasterized to a grid on demand.

Example
-------
>>> from ikarus.shapes import Cross
>>> Cross(arm_length=0.7, arm_width=0.2, angle=30).to_grid((128, 128)).shape
(128, 128)

For inverse design, leave parameters free::

    from ikarus.inverse import free, MetaAtom
    from ikarus.shapes import Cross

    atom = MetaAtom(period=400e-9, cover="Air", substrate="SiO2")
    atom.add_pattern(topology=Cross(arm_length=free(0.3, 0.9),
                                    arm_width=free(0.1, 0.4),
                                    angle=free(0, 90)),
                     materials=["Air", "Si"], height=free(0.2e-6, 0.6e-6))
"""

from __future__ import annotations

import numpy as np


def _is_free(value) -> bool:
    """True for a ``free(lo, hi)`` range (duck-typed: has ``low`` and ``high``).

    Kept structural so this module never imports :mod:`ikarus.inverse`.
    """
    return (hasattr(value, "low") and hasattr(value, "high")
            and not isinstance(value, (str, bytes)))


def _coords(grid_shape):
    nx, ny = grid_shape
    x = (np.arange(nx) + 0.5) / nx
    y = (np.arange(ny) + 0.5) / ny
    return np.meshgrid(x, y, indexing="ij")


class Shape:
    """Base class for a parametric topology shape.

    Subclasses declare their geometric parameters in ``_PARAMS`` (a tuple of
    ``(name, default)`` pairs) and implement :meth:`_mask`, which receives
    coordinates already centered on ``center`` and rotated by ``angle``.
    """

    #: ordered ``(name, default)`` geometric parameters; overridden per subclass.
    _PARAMS: tuple = ()

    def __init__(self, *, center=(0.5, 0.5), angle=0.0, grid_shape=(128, 128),
                 value=1, background=0, **params):
        defaults = dict(self._PARAMS)
        unknown = set(params) - set(defaults)
        if unknown:
            raise TypeError(
                f"{type(self).__name__} got unknown parameter(s) {sorted(unknown)}; "
                f"valid parameters are {sorted(defaults) + ['angle', 'center']}"
            )
        self.center = (float(center[0]), float(center[1]))
        self.angle = angle
        self.grid_shape = (int(grid_shape[0]), int(grid_shape[1]))
        self.value = int(value)
        self.background = int(background)
        self._values = {**defaults, **params}

    # -- the free-DOF protocol (consumed by ikarus.inverse) ----------------
    def _named(self):
        yield "angle", self.angle
        for name, _ in self._PARAMS:
            yield name, self._values[name]

    def free_parameters(self) -> dict:
        """``{name: (low, high)}`` for every parameter left free (incl. ``angle``)."""
        return {name: (float(v.low), float(v.high))
                for name, v in self._named() if _is_free(v)}

    def resolved(self, overrides: dict | None = None) -> "Shape":
        """Return a concrete copy with every free parameter replaced by a value.

        ``overrides`` maps a free parameter name to its chosen value.
        """
        overrides = overrides or {}

        def pick(name, v):
            if _is_free(v):
                if name not in overrides:
                    raise ValueError(
                        f"{type(self).__name__}: free parameter {name!r} needs a "
                        f"value (this shape has unresolved degrees of freedom)"
                    )
                return float(overrides[name])
            return v

        vals = {name: pick(name, self._values[name]) for name, _ in self._PARAMS}
        angle = pick("angle", self.angle)
        return type(self)(center=self.center, angle=angle, grid_shape=self.grid_shape,
                          value=self.value, background=self.background, **vals)

    # -- rasterization -----------------------------------------------------
    def to_grid(self, grid_shape=None, overrides: dict | None = None) -> np.ndarray:
        """Rasterize to an integer ``(Nx, Ny)`` array.

        Free parameters must be supplied via ``overrides``.
        """
        shape = self.resolved(overrides)
        gs = (int(grid_shape[0]), int(grid_shape[1])) if grid_shape else shape.grid_shape
        xx, yy = _coords(gs)
        cx, cy = shape.center
        th = np.deg2rad(shape.angle)
        dx, dy = xx - cx, yy - cy
        u = dx * np.cos(th) + dy * np.sin(th)      # centered + rotated coords
        v = -dx * np.sin(th) + dy * np.cos(th)
        mask = shape._mask(u, v, shape._values)
        out = np.full(gs, shape.background, dtype=int)
        out[mask] = shape.value
        return out

    @property
    def img(self) -> np.ndarray:
        """Rendered integer grid (the Topology-Species-compatible attribute)."""
        return self.to_grid()

    def _mask(self, u, v, p) -> np.ndarray:  # pragma: no cover - abstract
        raise NotImplementedError

    def __repr__(self) -> str:
        parts = [f"{n}={self._values[n]!r}" for n, _ in self._PARAMS]
        parts.append(f"angle={self.angle!r}")
        return f"{type(self).__name__}({', '.join(parts)})"


class Circle(Shape):
    """A filled circle of fractional ``radius``."""
    _PARAMS = (("radius", 0.25),)

    def _mask(self, u, v, p):
        return u ** 2 + v ** 2 <= p["radius"] ** 2


class Ellipse(Shape):
    """A filled ellipse with semi-axes ``rx``, ``ry`` (rotated by ``angle``)."""
    _PARAMS = (("rx", 0.3), ("ry", 0.18))

    def _mask(self, u, v, p):
        return (u / p["rx"]) ** 2 + (v / p["ry"]) ** 2 <= 1.0


class Rectangle(Shape):
    """A filled rectangle of fractional ``width`` x ``height`` (rotated by ``angle``)."""
    _PARAMS = (("width", 0.5), ("height", 0.3))

    def _mask(self, u, v, p):
        return (np.abs(u) <= p["width"] / 2) & (np.abs(v) <= p["height"] / 2)


class Ring(Shape):
    """An annulus between ``inner_radius`` and ``outer_radius``."""
    _PARAMS = (("inner_radius", 0.15), ("outer_radius", 0.3))

    def _mask(self, u, v, p):
        r = np.hypot(u, v)
        return (r >= p["inner_radius"]) & (r <= p["outer_radius"])


class Cross(Shape):
    """A plus/cross of two bars: ``arm_length`` long, ``arm_width`` wide."""
    _PARAMS = (("arm_length", 0.6), ("arm_width", 0.2))

    def _mask(self, u, v, p):
        al, aw = p["arm_length"], p["arm_width"]
        horiz = (np.abs(u) <= al / 2) & (np.abs(v) <= aw / 2)
        vert = (np.abs(u) <= aw / 2) & (np.abs(v) <= al / 2)
        return horiz | vert


class SplitRing(Shape):
    """A split-ring resonator: a ``Ring`` with an angular gap of ``gap_angle`` (deg).

    The gap opens along the shape's local +x axis, so ``angle`` rotates the gap
    around the ring -- the canonical chiral/anisotropic meta-atom parameter.
    """
    _PARAMS = (("inner_radius", 0.18), ("outer_radius", 0.32), ("gap_angle", 60.0))

    def _mask(self, u, v, p):
        r = np.hypot(u, v)
        in_ring = (r >= p["inner_radius"]) & (r <= p["outer_radius"])
        ang = np.degrees(np.arctan2(v, u)) % 360.0          # 0 deg = local +x
        half = p["gap_angle"] / 2.0
        in_gap = (ang <= half) | (ang >= 360.0 - half)
        return in_ring & ~in_gap


__all__ = ["Shape", "Circle", "Ellipse", "Rectangle", "Ring", "Cross", "SplitRing"]
