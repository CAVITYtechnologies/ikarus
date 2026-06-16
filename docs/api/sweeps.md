# Sweeps & Progress

Two small helpers remove the boilerplate (and the suspense) from parameter
sweeps and long optimizations. The division of labour:

| Tool | Use it for |
|---|---|
| [`Sweep`](#sweep) | sweeping **source** parameters (wavelength, angle, polarization) with no hand-written loop |
| [`progress`](#progress) | a single progress bar over **any** loop you write yourself (e.g. structural sweeps that rebuild the geometry) |
| [`optimize(progress=True)`](inverse.md#optimize) | a bar over a genetic-algorithm's generations |

!!! note "Optional dependency"
    For a rich bar, install **tqdm**: `pip install "ikarus-rcwa[progress]"`. Without
    it, a minimal built-in fallback bar is used — so `progress=True` always works.

!!! tip "Why not a bar on `simulate()`?"
    A single solve is *atomic* — a couple of eigendecompositions with no divisible
    sub-steps — so a bar there would only be a spinner. Progress is meaningful at
    the **loop** level: the sweep, or the optimizer's generations.

## `Sweep`

```python
from ikarus import Sweep
```

Sweep one configured [`RCWA`](rcwa.md) over a grid of **source** parameters. The
structure and any non-swept source fields are taken from the `rcwa` as you
configured it.

### `Sweep(rcwa)`

Bind a sweep to a configured solver.

### `.over(**axes) -> Sweep`

Declare the swept source parameters — each keyword is a
[`set_source`](rcwa.md#defining-the-source) field mapped to a 1-D sequence.
Multiple keywords form a **Cartesian grid** (axis order = keyword order).

### `.run(progress=True, desc="sweep") -> SweepResult`

Execute the sweep, with one progress bar for the whole thing.

```python
import numpy as np
from ikarus import RCWA, Sweep

rcwa = RCWA(period_x=400e-9, period_y=400e-9, n_orders=0)   # 0 = specular only; fine here — every layer is uniform
rcwa.add_uniform_layer(np.inf, "Air")
rcwa.add_uniform_layer(120e-9, "TiO2")
rcwa.add_uniform_layer(np.inf, "SiO2")
rcwa.set_source(wavelength=500e-9, theta=0, polarization="linear")

# 1-D wavelength sweep — one bar, no for-loop:
res = Sweep(rcwa).over(wavelength=np.linspace(400e-9, 700e-9, 200)).run()

# 2-D (angle x wavelength) grid — still one bar:
res = Sweep(rcwa).over(theta=np.linspace(0, 60, 31),
                       wavelength=np.linspace(400e-9, 700e-9, 100)).run()
res.R_total.shape   # (31, 100)
```

!!! warning "`n_orders=0` only works for *uniform* (thin-film) stacks"
    This example uses `n_orders=0` because **every layer is uniform** — a thin
    film has no diffraction, so the single specular order is exact (and instant).
    A **patterned** layer (a meta-atom, grating, photonic crystal…) needs
    `n_orders > 0`: with zero harmonics Ikarus keeps only the *average*
    permittivity, collapsing your pattern into an effective-medium slab — so every
    point of the sweep comes out **nearly identical**. Use `n_orders` ≈ 8–12 for
    patterned layers and confirm with a
    [convergence study](../tutorials/parameter-sweeps.md#convergence-study).

## `SweepResult`

The object returned by `.run()` — per-point results plus vectorized metrics
shaped like the sweep grid.

| Member | Description |
|---|---|
| `R_total`, `T_total`, `energy_balance` | float arrays shaped like the grid |
| `order(p, q, which="T")` | efficiency of diffraction order `(p, q)` across the grid (`which='R'` for reflected) |
| `results` | object array of [`SimulationResult`](rcwa.md#simulationresult) — full per-point access |
| `axes` | `{name: values}` of the swept axes |
| `shape` | the grid shape |

```python
import matplotlib.pyplot as plt
plt.plot(res.axes["wavelength"] * 1e9, res.R_total * 100)   # 1-D
plt.pcolormesh(res.axes["wavelength"] * 1e9, res.axes["theta"], res.R_total)  # 2-D
```

## `progress`

```python
from ikarus import progress
```

### `progress(iterable, enable=True, desc=None, total=None)`

Wrap any loop in a progress bar — a no-op pass-through when `enable` is false
(your on/off toggle). This is the escape hatch for **custom** sweeps, especially
**structural** ones that rebuild the geometry (and therefore can't be a `Sweep`):

```python
import numpy as np
from ikarus import RCWA, shapes, progress

heights = np.linspace(100e-9, 400e-9, 40)
show_bar = True

R = []
for h in progress(heights, desc="height", enable=show_bar):   # one bar over the loop
    rcwa = RCWA(period_x=450e-9, period_y=450e-9, resolution=(96, 96), n_orders=(9, 9))
    rcwa.add_uniform_layer(np.inf, "Air")
    rcwa.add_layer(h, shapes.circle(radius=0.3, grid_shape=(96, 96)), ["Air", "Si3N4"])
    rcwa.add_uniform_layer(np.inf, "SiO2")
    rcwa.set_source(wavelength=600e-9, theta=0, polarization="linear")
    R.append(rcwa.simulate()[2].R_total)
```

### Combining the two for a 2-D structural map

Structural axis outside (a real loop, with `progress`), source axis inside (a
`Sweep`):

```python
Rmap = []
for h in progress(heights, desc="height"):
    rcwa = build(h)
    Rmap.append(Sweep(rcwa).over(wavelength=wavelengths).run(progress=False).R_total)
Rmap = np.array(Rmap)        # (len(heights), len(wavelengths))
```

## Optimization progress

`optimize(..., progress=True)` shows one bar over the GA generations (and
silences the per-generation table). See [Inverse Design](inverse.md#optimize).

```python
best = optimize(atom, target, pop=40, n_gen=30, progress=True)
```
