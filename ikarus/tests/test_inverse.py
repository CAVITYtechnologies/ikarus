"""Tests for the inverse-design degree-of-freedom plumbing.

These cover ``MetaAtom.variables()`` and ``MetaAtom.build()`` -- the wiring
between free parameters and a concrete RCWA structure -- without invoking the
optimizer, so they need no optional dependencies (pymoo).
"""

import numpy as np
import pytest

from ikarus.inverse import MetaAtom, free, pixels
from ikarus.shapes import Cross


def test_period_validation_and_rectangular_cells():
    """period accepts a scalar, a free range, or a fixed (px, py) tuple; bad
    values fail CLEARLY at construction (not with a raw float() TypeError deep
    in the optimizer -- the sanity-check papercut)."""
    # fixed rectangular cell resolves to distinct period_x / period_y
    atom = MetaAtom(period=(800e-9, 300e-9), cover="Air", substrate="SiO2")
    atom.add_pattern(pixels(8, 4), ["Air", "Si"], height=200e-9)
    rcwa = atom.build({f"px{k}": 0 for k in range(32)}, n_orders=2)
    assert (rcwa.period_x, rcwa.period_y) == (800e-9, 300e-9)
    # a square tuple (what the reporter tried by analogy with RCWA) now works
    assert MetaAtom(period=(5e-7, 5e-7), cover="Air",
                    substrate="Si").period_xy({}) == (5e-7, 5e-7)
    # clear errors for malformed periods
    with pytest.raises(ValueError, match="period_x, period_y"):
        MetaAtom(period=(1e-6, 2e-6, 3e-6), cover="Air", substrate="Si")
    with pytest.raises(ValueError, match="free rectangular"):
        MetaAtom(period=(free(1e-6, 2e-6), 5e-7), cover="Air", substrate="Si")
    with pytest.raises(ValueError, match="positive"):
        MetaAtom(period=-1e-6, cover="Air", substrate="Si")
    with pytest.raises(TypeError, match="period must be"):
        MetaAtom(period="nope", cover="Air", substrate="Si")


def test_unknown_material_error_hints_workaround():
    from ikarus import default_library
    with pytest.raises(KeyError, match="constant complex index"):
        default_library.permittivity("Unobtainium", 1550e-9)


def test_silver_is_bundled_and_metallic():
    """Ag (Johnson & Christy 1972) ships in the default library and behaves like
    a low-loss NIR mirror."""
    from ikarus import default_library, RCWA
    assert "Ag" in default_library.available()
    nk = default_library.get("Ag", 1550e-9)
    assert nk.real < 0.5 and nk.imag > 8.0          # small n, large k
    rc = RCWA(period_x=500e-9, period_y=500e-9, n_orders=0)
    rc.add_uniform_layer(np.inf, "Air")
    rc.add_uniform_layer(np.inf, "Ag")
    rc.set_source(wavelength=1550e-9, theta=0, polarization="linear")
    assert rc.simulate()[2].R_total > 0.98          # silver mirror


def test_pixels_dof_enumeration_and_build():
    atom = MetaAtom(period=500e-9, cover="Air", substrate="SiO2")
    atom.add_pattern(topology=pixels(6, 6, symmetry="c4v"),
                     materials=["Air", "Si"], height=free(0.2e-6, 0.6e-6))
    v = atom.variables()
    # one real DOF (height) + n_free binary pixel DOFs
    n_free = pixels(6, 6, symmetry="c4v").n_free
    assert v["height"][0] == "real"
    assert sum(1 for k in v if k.startswith("px")) == n_free

    params = {"height": 0.3e-6}
    params.update({f"px{k}": (k % 2) for k in range(n_free)})
    rcwa = atom.build(params, n_orders=5)
    assert len(rcwa.layers) == 3
    assert rcwa.layers[1].height == 0.3e-6


def test_shape_dof_enumeration():
    atom = MetaAtom(period=500e-9, cover="Air", substrate="SiO2")
    atom.add_pattern(
        topology=Cross(arm_length=free(0.3, 0.9), arm_width=free(0.1, 0.4),
                       angle=free(0, 90), grid_shape=(64, 64)),
        materials=["Air", "Si"], height=0.3e-6)   # fixed height this time
    v = atom.variables()
    assert set(v) == {"shape__arm_length", "shape__arm_width", "shape__angle"}
    assert v["shape__arm_length"] == ("real", (0.3, 0.9))
    assert "height" not in v          # height is fixed, not a DOF


def test_shape_build_renders_topology():
    atom = MetaAtom(period=500e-9, cover="Air", substrate="SiO2")
    atom.add_pattern(
        topology=Cross(arm_length=free(0.3, 0.9), arm_width=free(0.1, 0.4),
                       angle=free(0, 90), grid_shape=(72, 72)),
        materials=["Air", "Si"], height=free(0.2e-6, 0.6e-6))
    params = {"height": 0.4e-6, "shape__arm_length": 0.7,
              "shape__arm_width": 0.25, "shape__angle": 30.0}
    rcwa = atom.build(params, n_orders=5)
    layer = rcwa.layers[1]
    assert not layer.is_uniform
    # the rendered cross must contain both materials
    assert set(np.unique(layer.topology)) == {0, 1}
    assert layer.height == 0.4e-6


def test_shape_used_directly_in_add_layer():
    from ikarus import RCWA
    # factorization="li" so the strict energy bound is meaningful even at this very
    # low order: the separable rule conserves energy exactly at any truncation,
    # whereas the default normal-vector method (more accurate for oblique shapes
    # like this rotated cross) has a small finite-order energy imbalance that only
    # vanishes as n_orders grows.  This test is about shape ergonomics, not the rule.
    rcwa = RCWA(period_x=500e-9, period_y=500e-9, resolution=(64, 64),
                n_orders=(6, 6), factorization="li")
    rcwa.add_uniform_layer(np.inf, "Air")
    rcwa.add_layer(200e-9, Cross(arm_length=0.6, arm_width=0.2, angle=15),
                   ["Air", "TiO2"])
    rcwa.add_uniform_layer(np.inf, "SiO2")
    rcwa.set_source(wavelength=600e-9, theta=0, polarization="linear")
    _, _, res = rcwa.simulate()
    assert 0.0 <= res.energy_balance <= 1.0001
