"""Tests for the multi-layer Structure inverse-design construct."""

import numpy as np
import pytest

from ikarus.inverse import Structure, MetaAtom, free, Target
from ikarus.shapes import Circle, Cross


class TwoLayer(Structure):
    cover, substrate, resolution = "Air", "SiO2", 72
    period = free(0.3e-6, 0.9e-6)
    h1 = free(0.1e-6, 0.4e-6)
    h2 = 0.20e-6                       # fixed
    radius = free(0.10, 0.45)
    arm_len = free(0.30, 0.90)

    def define(self, p):
        self.add_layer(p.h1, Circle(radius=p.radius), ["Si", "Air"])
        self.add_layer(p.h2, Cross(arm_length=p.arm_len, arm_width=0.2), ["Air", "Si"])


class MothEye(Structure):
    cover, substrate, resolution = "Air", "Si", 64
    N = 8
    period = free(150e-9, 240e-9)
    height = free(200e-9, 1000e-9)
    r_base = free(0.15, 0.5)
    gamma = free(0.5, 3.0)

    def define(self, p):
        for i in range(p.N):
            r = p.r_base * ((i + 0.5) / p.N) ** p.gamma
            self.add_layer(p.height / p.N, Circle(radius=r), ["Air", "Si"])


def test_variables_lists_only_free_params():
    v = TwoLayer().variables()
    assert set(v) == {"period", "h1", "radius", "arm_len"}   # h2 is fixed -> not a DOF
    assert v["period"] == ("real", (0.3e-6, 0.9e-6))


def test_build_assembles_full_stack_and_simulates():
    rcwa = TwoLayer().build(
        {"period": 0.5e-6, "h1": 0.2e-6, "radius": 0.3, "arm_len": 0.6}, n_orders=6)
    assert len(rcwa.layers) == 4                              # cover + 2 interior + substrate
    assert rcwa.layers[1].height == 0.2e-6
    assert rcwa.layers[2].height == 0.20e-6                   # the fixed h2
    rcwa.set_source(wavelength=1550e-9, theta=0, polarization="linear")
    assert 0.0 <= rcwa.simulate()[2].energy_balance <= 1.0001


def test_shared_derived_params_build_many_layers():
    """The moth-eye: 4 DOF drive N derived slices -- impossible with a MetaAtom."""
    m = MothEye()
    assert set(m.variables()) == {"period", "height", "r_base", "gamma"}
    rcwa = m.build({"period": 180e-9, "height": 600e-9, "r_base": 0.45, "gamma": 1.2}, n_orders=6)
    assert len(rcwa.layers) == m.N + 2                        # cover + N slices + substrate
    # radii grow with depth (tip up): first interior slice smaller than the last
    grids = [lay.topology for lay in rcwa.layers[1:-1]]
    assert grids[0].mean() < grids[-1].mean()


def test_missing_period_raises():
    class NoPeriod(Structure):
        h = free(0.1e-6, 0.4e-6)

        def define(self, p):
            self.add_layer(p.h, Circle(radius=0.3), ["Air", "Si"])

    with pytest.raises(ValueError):
        NoPeriod().variables()


def test_instance_overrides_fixed_param():
    m = MothEye(N=12)
    rcwa = m.build({"period": 180e-9, "height": 600e-9, "r_base": 0.45, "gamma": 1.2}, n_orders=5)
    assert len(rcwa.layers) == 12 + 2


# -- optimizer integration (needs pymoo) ------------------------------------

def test_optimize_default_path_runs():
    """Regression: optimize() with progress=False (the default) must not crash."""
    pytest.importorskip("pymoo")
    from ikarus.inverse import optimize
    atom = MetaAtom(period=400e-9, cover="Air", substrate="SiO2")
    atom.add_pattern(topology=Circle(radius=free(0.2, 0.45)),
                     materials=["Air", "Si"], height=free(0.2e-6, 0.6e-6))
    best = optimize(atom, Target.maximize("T", at=1550e-9),
                    n_orders=5, pop=5, n_gen=2, seed=0, verbose=False)
    assert best.params is not None


def test_optimize_structure_end_to_end():
    pytest.importorskip("pymoo")
    from ikarus.inverse import optimize
    best = optimize(TwoLayer(), Target.minimize("R", at=1550e-9),
                    n_orders=5, pop=5, n_gen=2, seed=0, verbose=False)
    assert set(best.params) == {"period", "h1", "radius", "arm_len"}
    assert type(best.rcwa).__name__ == "RCWA"
    assert len(best.rcwa.layers) == 4
