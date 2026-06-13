"""Tests for the inverse-design degree-of-freedom plumbing.

These cover ``MetaAtom.variables()`` and ``MetaAtom.build()`` -- the wiring
between free parameters and a concrete RCWA structure -- without invoking the
optimizer, so they need no optional dependencies (pymoo).
"""

import numpy as np

from ikarus.inverse import MetaAtom, free, pixels
from ikarus.shapes import Cross


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
    rcwa = RCWA(period_x=500e-9, period_y=500e-9, resolution=(64, 64), n_orders=(6, 6))
    rcwa.add_uniform_layer(np.inf, "Air")
    rcwa.add_layer(200e-9, Cross(arm_length=0.6, arm_width=0.2, angle=15),
                   ["Air", "TiO2"])
    rcwa.add_uniform_layer(np.inf, "SiO2")
    rcwa.set_source(wavelength=600e-9, theta=0, polarization="linear")
    _, _, res = rcwa.simulate()
    assert 0.0 <= res.energy_balance <= 1.0001
