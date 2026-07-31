"""The optional accelerator (GPU/Metal) forward path must be a drop-in.

Core guarantee: ``device != "cpu"`` routes the forward solve through the JAX mirror
but returns the *same* ``SimulationResult`` as the NumPy core.  We prove that here by
forcing the JAX path onto the CPU (``device="cpu-jax"``) and comparing to the core --
so the parity check runs anywhere JAX is installed, no GPU required.  On a real
accelerator the *same* JAX graph runs; only the device placement changes.
"""

import numpy as np
import pytest

pytest.importorskip("jax")

from ikarus import RCWA, shapes


def _grating(device, M=16, fac="auto"):
    topo = np.zeros((512, 2), dtype=int)
    topo[:256, :] = 1
    rc = RCWA(period_x=400e-9, period_y=400e-9, resolution=(512, 2),
              n_orders=(M, 0), factorization=fac, device=device)
    rc.add_uniform_layer(np.inf, "Air")
    rc.add_layer(300e-9, topo, ["Air", 3.5])
    rc.add_uniform_layer(np.inf, "Air")
    rc.set_source(wavelength=700e-9, theta=0, polarization="linear", linear_pol_angle=90)
    return rc.simulate()[2]


def _cylinder(device, M=10, theta=20.0):
    mask = shapes.circle(center=(0.5, 0.5), radius=0.30, grid_shape=(96, 96))
    rc = RCWA(period_x=400e-9, period_y=400e-9, resolution=(96, 96),
              n_orders=(M, M), device=device)
    rc.add_uniform_layer(np.inf, "Air")
    rc.add_layer(200e-9, mask.astype(int), [1.0, 3.5])
    rc.add_uniform_layer(np.inf, "Air")
    rc.set_source(wavelength=700e-9, theta=theta, polarization="linear", linear_pol_angle=0)
    return rc.simulate()[2]


@pytest.mark.parametrize("fac", ["auto", "li", "laurent"])
def test_jax_cpu_matches_core_1d_grating(fac):
    c, j = _grating("cpu", fac=fac), _grating("cpu-jax", fac=fac)
    assert abs(c.R_total - j.R_total) < 1e-10
    assert abs(c.T_total - j.T_total) < 1e-10
    assert np.max(np.abs(np.sort(c.R_orders) - np.sort(j.R_orders))) < 1e-10
    assert abs(c.R_phase - j.R_phase) < 1e-7            # complex coeff, hence phase


def test_jax_cpu_matches_core_2d_cylinder_oblique():
    c, j = _cylinder("cpu"), _cylinder("cpu-jax")
    assert abs(c.R_total - j.R_total) < 1e-10
    assert abs(c.T_total - j.T_total) < 1e-10
    assert np.max(np.abs(np.sort(c.R_orders) - np.sort(j.R_orders))) < 1e-10
    assert abs(c.R_phase - j.R_phase) < 1e-7
    # exit angles are reconstructed on the device path too, and must agree
    i0c, i0j = c.order_index(0, 0), j.order_index(0, 0)
    assert abs(c.theta_out_trn[i0c] - j.theta_out_trn[i0j]) < 1e-6


def test_auto_device_runs_everywhere():
    """device='auto' picks whatever is present (CPU at worst) and still matches."""
    c, a = _grating("cpu"), _grating("auto")
    assert abs(c.R_total - a.R_total) < 1e-9


def test_default_device_is_cpu():
    assert RCWA(period_x=1e-6, period_y=1e-6).device == "cpu"


def test_unknown_device_raises():
    with pytest.raises(ValueError, match="device"):
        RCWA(period_x=1e-6, period_y=1e-6, device="quantum")


def test_anisotropic_rejected_on_device():
    rc = RCWA(period_x=400e-9, period_y=400e-9, resolution=(64, 64),
              n_orders=(6, 6), device="cpu-jax")
    rc.add_uniform_layer(np.inf, "Air")
    rc.add_layer(200e-9, np.zeros((64, 64), dtype=int), [(2.0, 2.4, 2.0)])  # tensor material
    rc.add_uniform_layer(np.inf, "Air")
    rc.set_source(wavelength=700e-9, theta=0, polarization="linear")
    with pytest.raises(NotImplementedError, match="anisotropic"):
        rc.simulate()


def test_get_fields_rejected_on_device():
    rc = RCWA(period_x=400e-9, period_y=400e-9, resolution=(64, 4),
              n_orders=(8, 0), device="cpu-jax")
    rc.add_uniform_layer(np.inf, "Air")
    rc.add_layer(200e-9, np.zeros((64, 4), dtype=int), ["Air"])
    rc.add_uniform_layer(np.inf, "Air")
    rc.set_source(wavelength=700e-9, theta=0, polarization="linear")
    rc.simulate()
    with pytest.raises(NotImplementedError, match="field reconstruction is CPU-only"):
        rc.get_fields()
