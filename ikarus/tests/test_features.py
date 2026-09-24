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


# --- masks must be rasterized onto an integer multiple of their own size -----
# A mask is resampled onto the FFT grid by nearest neighbour. At a non-integer
# ratio that jitters every pixel boundary, and the geometry error moves with
# n_orders while the energy balance stays clean -- the worst kind of wrong.

def _freeform_1d(mask, M, resolution=None):
    import numpy as np
    from ikarus import RCWA
    topo = np.repeat(np.asarray(mask)[:, None], 2, axis=1)
    r = RCWA(period_x=3100e-9, period_y=3100e-9,
             resolution=(resolution or len(mask), 2), n_orders=(M, 0))
    r.add_uniform_layer(np.inf, "Air")
    r.add_layer(500e-9, topo, ["Air", "Si"])
    r.add_uniform_layer(np.inf, "SiO2")
    r.set_source(wavelength=1550e-9, theta=0)
    _, _, res = r.simulate()
    return res.T_orders[res.order_index(1, 0)]


def test_fft_grid_is_an_integer_multiple_of_the_mask():
    import numpy as np
    from ikarus import RCWA
    topo = np.zeros((62, 2), int); topo[:31] = 1
    r = RCWA(period_x=3100e-9, period_y=3100e-9, resolution=(62, 2), n_orders=(50, 0))
    r.add_uniform_layer(np.inf, "Air")
    r.add_layer(500e-9, topo, ["Air", "Si"])
    r.add_uniform_layer(np.inf, "SiO2")
    nx, _ = r._fft_sampling()
    assert nx >= 4 * 50 + 1          # still anti-aliased
    assert nx % 62 == 0              # and exact for this mask


def test_coarse_mask_converges_instead_of_wandering():
    """The regression this guards: a 62-px mask swept to n_orders=210 used to
    wander 44% against its own exact geometry. It must now converge."""
    import numpy as np
    rng = np.random.default_rng(3)
    mask = (rng.random(62) > 0.5).astype(int)
    truth = _freeform_1d(np.repeat(mask, 32), 210)     # identical geometry, 1984 px
    for M in (50, 120, 210):
        got = _freeform_1d(mask, M)
        assert abs(got - truth) / truth < 0.02, (
            f"n_orders={M}: {got:.5f} vs exact {truth:.5f}")


# --- exit angles are measured inside the substrate ---------------------------

def test_theta_out_trn_in_converts_and_flags_total_internal_reflection():
    import numpy as np
    from ikarus import RCWA
    r = RCWA(period_x=4.0e-6, period_y=4.0e-6, resolution=(256, 4), n_orders=(10, 0))
    r.add_uniform_layer(np.inf, "Air")
    topo = np.zeros((256, 4), int); topo[:128, :] = 1
    r.add_layer(600e-9, topo, ["Air", "Si"])
    r.add_uniform_layer(np.inf, "SiO2")
    r.set_source(wavelength=1550e-9, theta=0)
    _, _, res = r.simulate()

    n_sub = float(np.sqrt(res.solution.eps_trn).real.flat[0])
    air = res.theta_out_trn_in()
    i2 = res.order_index(2, 0)
    # Snell out of the substrate, and steeper in air than in glass
    assert np.isclose(air[i2], np.degrees(np.arcsin(
        n_sub * np.sin(np.radians(res.theta_out_trn[i2])))))
    assert air[i2] > res.theta_out_trn[i2]

    # order +3 propagates in the substrate but is past the critical angle: it
    # never leaves the chip, and a bare arcsin would not have told you.
    i3 = res.order_index(3, 0)
    assert not np.isnan(res.theta_out_trn[i3])
    assert np.isnan(air[i3])


# --- the shipped example must build what it says it builds -------------------

def test_metasurface_example_builds_silicon_pillars_not_air_holes():
    """The materials list was ["Si", "Air"], which makes the circle an air hole
    in a silicon film -- the opposite structure, and a 0.73 vs 0.98 difference
    in zero-order transmission with a clean energy balance either way."""
    import numpy as np
    from ikarus import RCWA, shapes
    pillar = shapes.circle(center=(0.5, 0.5), radius=0.32, grid_shape=(64, 64))
    r = RCWA(period_x=500e-9, period_y=500e-9, resolution=(64, 64), n_orders=(6, 6))
    r.add_uniform_layer(np.inf, "Air")
    r.add_layer(220e-9, pillar, ["Air", "Si"])     # same call the example makes
    r.add_uniform_layer(np.inf, "SiO2")
    r.set_source(wavelength=1550e-9, theta=0)
    r.simulate()
    eps = r.get_fields(plane="xy", nx=64, ny=64)   # mid-layer permittivity
    # the pillar centre must be silicon, the corner must be air
    topo = r.layers[1].topology
    assert topo[32, 32] == 1 and topo[0, 0] == 0
    mats = r.layers[1].materials
    assert mats[topo[32, 32]] == "Si", "circle centre must be the pillar material"
    assert mats[topo[0, 0]] == "Air", "background must be air"


def test_simulate_tuple_is_complex_amplitude_not_power():
    """The guide claimed T/R were "plain floats in [0,1]". They are the
    zero-order complex AMPLITUDE coefficients, and abs(.)**2 is the power --
    a caller doing `power = T` overstates transmission."""
    import numpy as np
    from ikarus import RCWA
    r = RCWA(period_x=200e-9, period_y=200e-9, resolution=8, n_orders=0)
    r.add_uniform_layer(np.inf, "Air")
    r.add_uniform_layer(np.inf, "SiO2")
    r.set_source(wavelength=729e-9, theta=0, polarization="linear")
    T, R, res = r.simulate()
    assert not isinstance(T, float)                 # it is complex, not a float
    assert T is res.T and R is res.R                # same objects as on the result
    assert np.isclose(abs(T) ** 2, res.T_total)     # power is the squared modulus
    assert not np.isclose(abs(T), res.T_total)      # ...and differs from the amplitude
