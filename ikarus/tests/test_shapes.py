"""Tests for the shape-primitive topology generators."""

import numpy as np

from ikarus import shapes


def test_circle_area_fraction():
    topo = shapes.circle(center=(0.5, 0.5), radius=0.25, grid_shape=(200, 200))
    frac = topo.mean()
    assert abs(frac - np.pi * 0.25 ** 2) < 5e-3


def test_rectangle_area_fraction():
    topo = shapes.rectangle(center=(0.5, 0.5), size=(0.4, 0.6), grid_shape=(128, 128))
    assert abs(topo.mean() - 0.4 * 0.6) < 5e-3


def test_ring_is_annulus():
    topo = shapes.ring(inner_radius=0.2, outer_radius=0.3, grid_shape=(128, 128))
    expected = np.pi * (0.3 ** 2 - 0.2 ** 2)
    assert abs(topo.mean() - expected) < 5e-3
    assert topo[64, 64] == 0  # hole at center


def test_values_and_background():
    topo = shapes.circle(radius=0.3, grid_shape=(64, 64), value=2, background=1)
    assert set(np.unique(topo)) <= {1, 2}


def test_polygon_matches_square():
    poly = shapes.polygon([(0.25, 0.25), (0.75, 0.25), (0.75, 0.75), (0.25, 0.75)],
                           grid_shape=(128, 128))
    assert abs(poly.mean() - 0.25) < 5e-3


def test_combine_overlay():
    a = shapes.rectangle(center=(0.3, 0.5), size=(0.2, 0.2), grid_shape=(64, 64), value=1)
    b = shapes.rectangle(center=(0.7, 0.5), size=(0.2, 0.2), grid_shape=(64, 64), value=2)
    c = shapes.combine(a, b)
    assert set(np.unique(c)) == {0, 1, 2}
