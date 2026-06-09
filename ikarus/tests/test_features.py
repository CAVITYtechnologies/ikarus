"""Tests for circular polarization, field reconstruction, convergence and I/O."""

import numpy as np
import pytest

from ikarus import RCWA, shapes


def _slab(pol="linear", **src):
    rcwa = RCWA(period_x=500e-9, period_y=500e-9, resolution=16, n_orders=3)
    rcwa.add_uniform_layer(np.inf, 1.0)
    rcwa.add_uniform_layer(200e-9, 3.5)
    rcwa.add_uniform_layer(np.inf, 1.45)
    rcwa.set_source(wavelength=1550e-9, theta=0, polarization=pol, **src)
    return rcwa


# -- circular polarization --------------------------------------------------
def test_circular_handedness_preserved_in_transmission():
    rcwa = _slab(pol="RCP")
    T, R, res = rcwa.simulate()
    # Achiral slab: transmission keeps handedness, reflection flips it.
    assert abs(T["co"]) > 0.9
    assert abs(T["cross"]) < 1e-6
    assert abs(R["co"]) < 1e-6
    assert abs(R["cross"]) > 0.0


def test_circular_energy_conservation():
    T, R, res = _slab(pol="LCP").simulate()
    total = sum(abs(v) ** 2 for v in (T["co"], T["cross"], R["co"], R["cross"]))
    assert abs(total - res.energy_balance) < 1e-6
    assert abs(res.energy_balance - 1.0) < 1e-6


# -- field reconstruction ---------------------------------------------------
def test_field_continuity_at_interfaces():
    rcwa = _slab(linear_pol_angle=0.0)
    rcwa.simulate()
    eps = 1e-12
    maps = rcwa.get_fields(z_positions=[-eps, eps, 200e-9 - eps, 200e-9 + eps],
                           plane="xy", nx=4, ny=4)
    vals = [m.E[0, 0, 1] for m in maps.values()]  # Ey at the 4 z
    assert abs(vals[0] - vals[1]) < 1e-4  # continuous across cover/slab
    assert abs(vals[2] - vals[3]) < 1e-4  # continuous across slab/substrate


def test_uniform_slab_field_is_xy_invariant():
    rcwa = _slab(linear_pol_angle=0.0)
    rcwa.simulate()
    fm = rcwa.get_fields(z_positions=[100e-9], plane="xy", nx=8, ny=8)
    m = list(fm.values())[0]
    assert np.std(np.abs(m.E[..., 1])) < 1e-10


def test_cross_section_shape():
    rcwa = _slab()
    rcwa.simulate()
    fm = rcwa.get_fields(plane="xz", nx=32)["xz"]
    assert fm.E.shape[-1] == 3
    assert fm.intensity.ndim == 2


# -- convergence ------------------------------------------------------------
def test_auto_converge_once_caches():
    topo = np.zeros((128, 2), dtype=int)
    topo[64:, :] = 1
    rcwa = RCWA(period_x=800e-9, period_y=800e-9, resolution=(256, 2), n_orders=(5, 0))
    rcwa.add_uniform_layer(np.inf, 1.0)
    rcwa.add_layer(300e-9, topo, [1.0, 2.5])
    rcwa.add_uniform_layer(np.inf, 1.0)
    rcwa.set_source(wavelength=633e-9, theta=0, polarization="linear")
    rcwa.simulate(auto_converge="once", converge_tol=1e-3, max_orders=40)
    chosen = rcwa.n_orders
    assert chosen[0] > 5  # converged to a higher order
    assert rcwa._converged
    # Second call must reuse the cache (no change).
    rcwa.simulate(auto_converge="once")
    assert rcwa.n_orders == chosen


# -- HDF5 I/O ---------------------------------------------------------------
def test_hdf5_roundtrip(tmp_path):
    rcwa = _slab()
    _, _, res = rcwa.simulate()
    path = tmp_path / "r.h5"
    rcwa.save_results(path, include=["T", "R", "metadata"], result=res)
    loaded = RCWA.load_results(path)
    assert abs(float(loaded["R_total"]) - res.R_total) < 1e-12
    assert abs(float(loaded["T_total"]) - res.T_total) < 1e-12
    assert loaded["metadata"]["period_x"] == rcwa.period_x


def test_energy_warning_on_gain(recwarn):
    # A material with gain (negative k) should trip the energy-balance warning.
    rcwa = RCWA(period_x=500e-9, period_y=500e-9, resolution=8, n_orders=3)
    rcwa.add_uniform_layer(np.inf, 1.0)
    rcwa.add_uniform_layer(100e-9, 3.5 - 0.3j)  # gain under physics convention
    rcwa.add_uniform_layer(np.inf, 1.0)
    rcwa.set_source(wavelength=600e-9, theta=0, polarization="linear")
    rcwa.simulate()
    assert any("Energy balance" in str(w.message) for w in recwarn.list)
