# Parameter sweeps

**Goal.** Run efficient sweeps over wavelength, geometry and incidence; perform a
convergence study; and build a 2-D response map.

## The sweep pattern

Reuse one `RCWA` object and change only what varies. `set_source` retains
unspecified fields, so a wavelength or angle sweep is a one-liner inside the loop.

```python
import numpy as np
from ikarus import RCWA, shapes

period, N = 450e-9, 96
disk = shapes.circle(radius=0.3, grid_shape=(N, N))
rcwa = RCWA(period_x=period, period_y=period, resolution=(N, N), n_orders=(9, 9))
rcwa.add_uniform_layer(np.inf, "Air")
rcwa.add_layer(200e-9, disk, ["Air", "Si3N4"])
rcwa.add_uniform_layer(np.inf, "SiO2")

wavelengths = np.linspace(400e-9, 800e-9, 81)
R = np.empty_like(wavelengths)
for i, wl in enumerate(wavelengths):
    rcwa.set_source(wavelength=wl, theta=0, polarization="linear")
    R[i] = rcwa.simulate()[2].R_total
```

## Convergence study

Before trusting a sweep, confirm the harmonic count is sufficient. Use
[`convergence_curve`](../api/tools.md#convergence_curve)
(it restores your original `n_orders` afterward):

```python
from ikarus.tools.convergence import convergence_curve

rcwa.set_source(wavelength=600e-9)
orders, defect = convergence_curve(rcwa, range(4, 21, 2), metric="energy")
harmonics = (2 * orders + 1) ** 2
for h, d in zip(harmonics, defect):
    print(f"{h:5d} harmonics: |R+T-1| = {d:.2e}")
```

Or let the solver pick `n_orders` automatically and cache it for the sweep:

```python
rcwa.simulate(auto_converge="once", verbose=True)   # sets rcwa.n_orders once
print("using n_orders =", rcwa.n_orders)
```

!!! warning "TM / high-contrast convergence"
    Metals and high-index TM problems converge slowly (Ikarus uses Laurent's rule
    — see [Theory → Limitations](../theory.md#limitations-of-rcwa)). Always run the
    convergence study at your *worst-case* wavelength, not a benign one.

## A 2-D response map (wavelength × geometry)

Sweep two parameters to build a design map — here reflectance vs. wavelength and
layer thickness:

```python
import numpy as np
from ikarus import RCWA, shapes

period, N = 450e-9, 96
disk = shapes.circle(radius=0.3, grid_shape=(N, N))
wavelengths = np.linspace(400e-9, 800e-9, 60)
heights = np.linspace(100e-9, 400e-9, 40)

Rmap = np.empty((heights.size, wavelengths.size))
for j, h in enumerate(heights):
    rcwa = RCWA(period_x=period, period_y=period, resolution=(N, N), n_orders=(9, 9))
    rcwa.add_uniform_layer(np.inf, "Air")
    rcwa.add_layer(h, disk, ["Air", "Si3N4"])
    rcwa.add_uniform_layer(np.inf, "SiO2")
    for i, wl in enumerate(wavelengths):
        rcwa.set_source(wavelength=wl, theta=0, polarization="linear")
        Rmap[j, i] = rcwa.simulate()[2].R_total

import matplotlib.pyplot as plt
plt.pcolormesh(wavelengths * 1e9, heights * 1e9, Rmap, shading="auto", cmap="inferno")
plt.xlabel("wavelength (nm)"); plt.ylabel("pillar height (nm)")
plt.colorbar(label="Reflectance"); plt.savefig("Rmap.png", dpi=150)
```

## Going faster

- **Pin BLAS to one thread** before importing NumPy for many small solves (see
  [Performance](../performance.md)); the small matrices in a sweep do not benefit
  from threaded BLAS and often slow down.
- **Parallelize across processes** — sweeps are embarrassingly parallel; see
  [Advanced → Batch simulations](../advanced.md#batch-simulations).
- **Only changing the source?** Rebuild nothing — keep the same `RCWA` and call
  `set_source`. Only changing geometry forces a new layer/eigensolve.

## Expected results

- A converged sweep has a monotonically shrinking energy defect; once
  `|R+T-1|` is below your tolerance (e.g. 10⁻⁴), more orders only cost time.
- The 2-D map reveals the resonance bands (where `R` peaks) and the broadband
  anti-reflection valleys.
