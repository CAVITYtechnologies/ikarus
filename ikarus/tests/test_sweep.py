"""Tests for the Sweep helper and the progress utilities."""

import numpy as np

from ikarus import RCWA, Sweep, progress
from ikarus._progress import counter, _FallbackCounter, _fallback_iter


def _thin_film():
    rcwa = RCWA(period_x=400e-9, period_y=400e-9, n_orders=0)
    rcwa.add_uniform_layer(np.inf, "Air")
    rcwa.add_uniform_layer(120e-9, "TiO2")
    rcwa.add_uniform_layer(np.inf, "SiO2")
    rcwa.set_source(wavelength=500e-9, theta=0, polarization="linear")
    return rcwa


def test_sweep_1d_shapes_and_energy():
    rcwa = _thin_film()
    wl = np.linspace(400e-9, 700e-9, 11)
    res = Sweep(rcwa).over(wavelength=wl).run(progress=False)
    assert res.R_total.shape == (11,)
    assert res.T_total.shape == (11,)
    # TiO2/glass is lossless here
    assert np.all(np.abs(res.energy_balance - 1.0) < 1e-6)
    # axis is retained
    assert np.allclose(res.axes["wavelength"], wl)


def test_sweep_2d_grid_order():
    rcwa = _thin_film()
    res = Sweep(rcwa).over(theta=[0, 15, 30],
                           wavelength=np.linspace(450e-9, 650e-9, 5)).run(progress=False)
    assert res.R_total.shape == (3, 5)
    assert res.results.shape == (3, 5)


def test_sweep_order_extraction():
    period = 900e-9
    rcwa = RCWA(period_x=period, period_y=period, resolution=(256, 2), n_orders=(15, 0))
    topo = np.zeros((128, 2), dtype=int)
    topo[64:, :] = 1
    rcwa.add_uniform_layer(np.inf, "Air")
    rcwa.add_layer(300e-9, topo, ["TiO2", "Air"])
    rcwa.add_uniform_layer(np.inf, "SiO2")
    rcwa.set_source(wavelength=650e-9, theta=0, polarization="linear", linear_pol_angle=0)
    res = Sweep(rcwa).over(wavelength=np.linspace(500e-9, 750e-9, 6)).run(progress=False)
    t1 = res.order(1, 0, which="T")
    assert t1.shape == (6,)
    assert np.all(t1 >= 0)


def test_sweep_requires_axes():
    rcwa = _thin_film()
    try:
        Sweep(rcwa).run(progress=False)
        assert False, "expected ValueError when no axes declared"
    except ValueError:
        pass


def test_progress_disabled_is_passthrough():
    items = [1, 2, 3]
    assert list(progress(items, enable=False)) is items or list(progress(items, enable=False)) == items


def test_progress_enabled_yields_all_items():
    out = list(progress(range(5), enable=True, desc="t"))
    assert out == [0, 1, 2, 3, 4]


def test_counter_and_fallback_do_not_error():
    c = counter(3, enable=True, desc="t")
    c.update()
    c.update(2)
    c.close()
    # exercise the no-tqdm fallback explicitly
    fb = _FallbackCounter(2, desc="t")
    fb.update()
    fb.close()
    assert list(_fallback_iter(range(3), 3, "t")) == [0, 1, 2]
