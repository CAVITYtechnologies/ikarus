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


def test_rotate_preserves_fill_and_changes_map():
    sq = shapes.rectangle(center=(0.5, 0.5), size=(0.5, 0.25), grid_shape=(128, 128))
    rot = shapes.rotate(sq, 30)
    assert set(np.unique(rot)) <= {0, 1}
    assert not np.array_equal(sq, rot)
    # area is approximately preserved by an in-plane rotation
    assert abs(rot.mean() - sq.mean()) < 0.02


# -- parametric shape classes ----------------------------------------------

def test_parametric_circle_matches_functional():
    from ikarus.shapes import Circle
    a = Circle(radius=0.25, grid_shape=(200, 200)).to_grid()
    b = shapes.circle(radius=0.25, grid_shape=(200, 200))
    assert abs(a.mean() - b.mean()) < 1e-3


def test_parametric_rotation_via_angle():
    from ikarus.shapes import Rectangle
    flat = Rectangle(width=0.6, height=0.2, angle=0, grid_shape=(128, 128)).to_grid()
    tilt = Rectangle(width=0.6, height=0.2, angle=45, grid_shape=(128, 128)).to_grid()
    assert not np.array_equal(flat, tilt)
    assert abs(flat.mean() - tilt.mean()) < 0.02  # area preserved


def test_img_property():
    from ikarus.shapes import Cross
    c = Cross(arm_length=0.6, arm_width=0.2)
    assert c.img.shape == (128, 128)
    assert set(np.unique(c.img)) <= {0, 1}


def test_free_parameters_and_resolve():
    from ikarus.shapes import Cross
    from ikarus.inverse import free
    c = Cross(arm_length=free(0.3, 0.9), arm_width=0.2, angle=free(0, 90))
    fp = c.free_parameters()
    assert set(fp) == {"arm_length", "angle"}
    assert fp["arm_length"] == (0.3, 0.9)
    # a free shape cannot rasterize without values
    try:
        c.to_grid((32, 32))
        assert False, "expected ValueError for unresolved free params"
    except ValueError:
        pass
    grid = c.to_grid((64, 64), {"arm_length": 0.7, "angle": 10.0})
    assert set(np.unique(grid)) <= {0, 1}


def test_unknown_parameter_raises():
    from ikarus.shapes import Circle
    try:
        Circle(radius=0.3, bogus=1.0)
        assert False, "expected TypeError for unknown parameter"
    except TypeError:
        pass
