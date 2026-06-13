"""Geometric primitives that generate integer topology maps.

Coordinates are expressed in *fractional* unit-cell units in ``[0, 1)`` so that a
shape definition is independent of the pixel resolution.  Pixel centers are
sampled, i.e. pixel ``(i, j)`` sits at ``((i + 0.5)/Nx, (j + 0.5)/Ny)``.
"""

from __future__ import annotations

import numpy as np


def _coords(grid_shape: tuple[int, int]):
    nx, ny = grid_shape
    x = (np.arange(nx) + 0.5) / nx
    y = (np.arange(ny) + 0.5) / ny
    return np.meshgrid(x, y, indexing="ij")


def circle(center=(0.5, 0.5), radius=0.25, grid_shape=(32, 32), value=1,
           background=0) -> np.ndarray:
    """Filled circle (in fractional coordinates)."""
    return ellipse(center=center, radii=(radius, radius), grid_shape=grid_shape,
                   value=value, background=background)


def ellipse(center=(0.5, 0.5), radii=(0.25, 0.15), grid_shape=(32, 32),
            angle=0.0, value=1, background=0) -> np.ndarray:
    """Filled (optionally rotated) ellipse.  ``angle`` in degrees."""
    xx, yy = _coords(grid_shape)
    cx, cy = center
    rx, ry = radii
    th = np.deg2rad(angle)
    dx, dy = xx - cx, yy - cy
    xr = dx * np.cos(th) + dy * np.sin(th)
    yr = -dx * np.sin(th) + dy * np.cos(th)
    mask = (xr / rx) ** 2 + (yr / ry) ** 2 <= 1.0
    out = np.full(grid_shape, background, dtype=int)
    out[mask] = value
    return out


def rectangle(center=(0.5, 0.5), size=(0.5, 0.5), grid_shape=(32, 32), value=1,
              background=0) -> np.ndarray:
    """Axis-aligned filled rectangle of fractional ``size = (width, height)``."""
    xx, yy = _coords(grid_shape)
    cx, cy = center
    w, h = size
    mask = (np.abs(xx - cx) <= w / 2) & (np.abs(yy - cy) <= h / 2)
    out = np.full(grid_shape, background, dtype=int)
    out[mask] = value
    return out


def ring(center=(0.5, 0.5), inner_radius=0.15, outer_radius=0.25,
         grid_shape=(32, 32), value=1, background=0) -> np.ndarray:
    """Annulus between ``inner_radius`` and ``outer_radius``."""
    xx, yy = _coords(grid_shape)
    cx, cy = center
    r = np.hypot(xx - cx, yy - cy)
    mask = (r >= inner_radius) & (r <= outer_radius)
    out = np.full(grid_shape, background, dtype=int)
    out[mask] = value
    return out


def cross(center=(0.5, 0.5), arm_length=0.4, arm_width=0.12, grid_shape=(32, 32),
          value=1, background=0) -> np.ndarray:
    """A plus/cross shape (two overlapping rectangles)."""
    horiz = rectangle(center, (arm_length, arm_width), grid_shape, value, background)
    vert = rectangle(center, (arm_width, arm_length), grid_shape, value, background)
    out = np.full(grid_shape, background, dtype=int)
    out[(horiz == value) | (vert == value)] = value
    return out


def polygon(vertices, grid_shape=(32, 32), value=1, background=0) -> np.ndarray:
    """Filled simple polygon from fractional ``vertices`` (list of (x, y)).

    Uses the even-odd ray-casting rule, fully vectorized over pixels.
    """
    verts = np.asarray(vertices, dtype=float)
    xx, yy = _coords(grid_shape)
    px, py = xx.ravel(), yy.ravel()
    inside = np.zeros(px.shape, dtype=bool)
    n = len(verts)
    j = n - 1
    for i in range(n):
        xi, yi = verts[i]
        xj, yj = verts[j]
        cond = ((yi > py) != (yj > py)) & (
            px < (xj - xi) * (py - yi) / (yj - yi + 1e-300) + xi
        )
        inside ^= cond
        j = i
    out = np.full(px.shape, background, dtype=int)
    out[inside] = value
    return out.reshape(grid_shape)


def rotate(topology, angle, order=0) -> np.ndarray:
    """Rotate an integer topology map by ``angle`` degrees (CCW) about its center.

    Uses periodic wrapping, so the rotated map still tiles the unit cell.
    ``order=0`` (nearest-neighbour) keeps the result integer-valued; raise it for
    smoother boundaries on coarse grids.  For parametric shapes prefer the native
    ``angle`` argument of :mod:`ikarus.shapes.parametric` (no resampling).
    """
    from scipy.ndimage import rotate as _ndrotate
    arr = np.asarray(topology)
    out = _ndrotate(arr.astype(float), angle, reshape=False, order=order,
                    mode="grid-wrap")
    return np.rint(out).astype(int)


def combine(*maps, mode="overlay") -> np.ndarray:
    """Combine several topology maps.

    ``'overlay'`` (default) lets later non-background pixels win; ``'max'`` takes
    the elementwise maximum index.
    """
    if not maps:
        raise ValueError("combine() needs at least one map")
    out = np.array(maps[0])
    for m in maps[1:]:
        m = np.asarray(m)
        if mode == "max":
            out = np.maximum(out, m)
        else:
            out = np.where(m != 0, m, out)
    return out
