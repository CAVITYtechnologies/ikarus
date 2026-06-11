# Angular response

**Goal.** Sweep the incidence angle, build an angle–wavelength dispersion map, and
recognize Rayleigh–Wood anomalies.

## Incidence-angle sweep

`theta` is the polar angle from +z (degrees); `phi` is the azimuth. As before,
reuse one `RCWA` and vary only the source.

```python
import numpy as np
from ikarus import RCWA, shapes

period, N = 500e-9, 96
disk = shapes.circle(radius=0.3, grid_shape=(N, N))
rcwa = RCWA(period_x=period, period_y=period, resolution=(N, N), n_orders=(10, 10))
rcwa.add_uniform_layer(np.inf, "Air")
rcwa.add_layer(180e-9, disk, ["Air", "TiO2"])
rcwa.add_uniform_layer(np.inf, "SiO2")

thetas = np.linspace(0, 60, 61)
R = np.empty_like(thetas)
for i, th in enumerate(thetas):
    rcwa.set_source(wavelength=600e-9, theta=th, phi=0, polarization="linear", linear_pol_angle=90)
    R[i] = rcwa.simulate()[2].R_total      # TM (p-pol); watch for the Brewster dip
```

!!! info "Conical (off-plane) incidence"
    A non-zero `phi` tilts the plane of incidence out of the x–z plane (conical
    mounting). The solver handles it natively — the full 2-D harmonic set is always
    in play once `theta > 0`.

## Angle–wavelength dispersion map

The most informative diagnostic for a periodic structure is a 2-D map of
reflectance over (wavelength, angle): resonances appear as dispersive bands and
diffraction onsets as sharp lines.

```python
import numpy as np
from ikarus import RCWA, shapes

period, N = 500e-9, 96
disk = shapes.circle(radius=0.3, grid_shape=(N, N))
rcwa = RCWA(period_x=period, period_y=period, resolution=(N, N), n_orders=(10, 10))
rcwa.add_uniform_layer(np.inf, "Air")
rcwa.add_layer(180e-9, disk, ["Air", "TiO2"])
rcwa.add_uniform_layer(np.inf, "SiO2")

wavelengths = np.linspace(450e-9, 750e-9, 80)
thetas = np.linspace(0, 50, 60)
Rmap = np.empty((thetas.size, wavelengths.size))
for j, th in enumerate(thetas):
    for i, wl in enumerate(wavelengths):
        rcwa.set_source(wavelength=wl, theta=th, polarization="linear")
        Rmap[j, i] = rcwa.simulate()[2].R_total

import matplotlib.pyplot as plt
plt.pcolormesh(wavelengths * 1e9, thetas, Rmap, shading="auto", cmap="magma")
plt.xlabel("wavelength (nm)"); plt.ylabel("incidence angle (deg)")
plt.colorbar(label="Reflectance"); plt.savefig("dispersion.png", dpi=150)
```

## Rayleigh–Wood anomalies

A diffraction order cuts on/off when its in-plane wavevector reaches the light line.
For the first order into the cover (index \(n_c\)) at azimuth \(\phi=0\):

\[
n_c\,\sin\theta + \frac{\lambda}{\Lambda} = n_c
\quad\Rightarrow\quad
\lambda_{\text{RW}} = \Lambda\,n_c\,(1 - \sin\theta).
\]

These appear as bright cusps in the map. Ikarus regularizes orders sitting exactly
on the light line with a tiny imaginary loss, so the solve stays well-posed across
the anomaly — but expect the energy defect to rise locally and the convergence to
slow there.

```python
# Track the number of propagating reflected orders as theta grows:
for th in (0, 20, 40):
    rcwa.set_source(wavelength=600e-9, theta=th, polarization="linear")
    _, _, res = rcwa.simulate()
    n_prop = int(np.sum(np.isfinite(res.theta_out_ref) & (res.R_orders > 1e-6)))
    print(f"theta={th:2d} deg: {n_prop} propagating reflected orders")
```

## Expected results

- A smooth specular response at small angles, with TM showing a Brewster-like dip.
- Diffraction onsets (Wood anomalies) trace bright curves across the dispersion
  map; the order count changes as you cross them.

## Best practices

- Re-check convergence at the **largest** angle of your sweep — oblique incidence
  populates more orders and can need a higher `n_orders`.
- Near a Wood anomaly, expect a slightly larger energy defect; refine `n_orders`
  if it matters for your metric.
- Mask `NaN` exit angles (evanescent orders) before plotting angle data.
