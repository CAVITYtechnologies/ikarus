# Quick Start

This page walks through a complete simulation from scratch, explains every input
and output, and states the expected result. If you have not installed Ikarus yet,
see [Installation](installation.md).

!!! note "Units"
    Ikarus is **SI throughout**: all lengths (periods, heights, wavelengths,
    field coordinates) are in **meters**. Angles are in **degrees**.

## A minimal working example

A 200 nm crystalline-silicon film between air and a glass substrate, at normal
incidence:

```python
import numpy as np
from ikarus import RCWA

# 1. Create the solver: unit-cell periods, real-space resolution, harmonic orders.
rcwa = RCWA(period_x=500e-9, period_y=500e-9, resolution=64, n_orders=8)

# 2. Build the stack, top (cover) to bottom (substrate).
rcwa.add_uniform_layer(height=np.inf, material="Air")    # semi-infinite cover
rcwa.add_uniform_layer(height=200e-9, material="Si")     # finite interior layer
rcwa.add_uniform_layer(height=np.inf, material="SiO2")   # semi-infinite substrate

# 3. Define the illumination.
rcwa.set_source(wavelength=633e-9, theta=0, phi=0,
                polarization="linear", linear_pol_angle=0.0)

# 4. Solve.
T, R, result = rcwa.simulate()

print(f"R = {result.R_total:.4f}")
print(f"T = {result.T_total:.4f}")
print(f"R + T = {result.energy_balance:.6f}   (== 1 for a lossless stack)")
```

## Explanation of the inputs

### Constructing `RCWA`

```python
RCWA(period_x, period_y, resolution=32, n_orders=25,
     dtype=np.complex128, materials=None, convergence_tol=1e-6)
```

| Argument | Meaning |
|---|---|
| `period_x`, `period_y` | Unit-cell periods (m). For a 1-D grating, make the invariant direction's period arbitrary and use `n_orders=(M, 0)`. |
| `resolution` | Real-space grid used to build the Fourier (convolution) matrices. An `int` (square) or `(Nx, Ny)`. Internally raised to at least `4*n_orders + 1` to avoid aliasing. |
| `n_orders` | Maximum **positive** harmonic order kept per axis, `M` (an `int`, or `(Mx, My)`). The retained set runs `-M … +M`, so the count per axis is `2M+1` and the total is \(P=(2M_x+1)(2M_y+1)\). |
| `materials` | A custom [`MaterialLibrary`](api/materials-layers.md); defaults to the shared built-in `default_library`. |

### Building the stack

The stack is an ordered list of layers, **cover first, substrate last**:

- The **cover** (incidence region) and **substrate** (transmission region) must be
  **uniform and semi-infinite** — pass `height=np.inf`.
- Every **interior** layer must have a **finite** thickness.
- `add_uniform_layer(height, material)` adds a single-material layer.
- `add_layer(height, topology, materials)` adds a *patterned* layer (see
  [Tutorials → Metasurfaces](tutorials/metasurfaces.md)).

A material specifier can be a database name (`"Si"`), a number (a constant
refractive index, e.g. `1.5`), or a [`Material`](api/materials-layers.md) object.

### Defining the source

```python
set_source(wavelength, theta=0, phi=0,
           polarization="linear", linear_pol_angle=0.0)
```

| Argument | Meaning |
|---|---|
| `wavelength` | Vacuum wavelength (m). |
| `theta` | Polar angle from the +z axis (deg). `0` = normal incidence. |
| `phi` | Azimuth in the xy-plane from +x (deg). |
| `polarization` | `"linear"`, `"RCP"` or `"LCP"`. |
| `linear_pol_angle` | For linear polarization: angle (deg) from TE. `0` = TE/s, `90` = TM/p. At normal incidence, TE is along +y and TM along +x. |

`set_source` is **stateful**: after the first call, subsequent calls update only
the fields you pass and keep the rest — convenient for sweeps:

```python
for wl in np.linspace(450e-9, 750e-9, 31):
    rcwa.set_source(wavelength=wl)   # theta/pol retained
    _, _, res = rcwa.simulate()
```

## Understanding the outputs

`simulate()` returns a tuple `(T, R, result)`:

- **`T`, `R`** — the complex **zero-order** coefficients (convenience handles).
  For linear polarization each is a complex scalar with `abs(coeff)**2` equal to
  the specular efficiency. For circular polarization each is a dict
  `{"co": ..., "cross": ...}` (same/opposite handedness).
- **`result`** — a rich [`SimulationResult`](api/rcwa.md#simulationresult) with the
  full per-order data:

| Field | Meaning |
|---|---|
| `R_total`, `T_total` | Total reflected / transmitted power (sum over propagating orders). |
| `R_orders`, `T_orders` | Per-order efficiencies, aligned with `orders`. |
| `orders` | `(p, q)` integer arrays labelling each harmonic. |
| `theta_out_ref/trn`, `phi_out_ref/trn` | Exit angles per order (deg); `NaN` for evanescent orders. |
| `energy_balance` | `R_total + T_total` (`== 1` for a lossless stack). |
| `R_phase`, `T_phase` | Phase (rad) of the zero-order coefficient. |
| `solution` | The underlying `FieldSolution` (used for field reconstruction). |

Index a specific order with `order_index`:

```python
i00 = result.order_index(0, 0)        # specular
print(result.T_orders[i00], result.theta_out_trn[i00])
```

## Expected result

The example above (200 nm Si on SiO₂, λ = 633 nm, normal incidence) is a
thin-film stack: only the specular order propagates, so `R_total` and `T_total`
are the Fresnel/transfer-matrix reflectance and transmittance of the three-layer
system. Crystalline silicon is mildly absorbing at 633 nm, so you should see

- `R ≈ 0.35`, `T ≈ 0.40`, and `R + T ≈ 0.75 < 1` (the remainder is **absorbed**),
- `energy_balance` close to `0.75`, **not** 1, because Si has a non-zero
  extinction coefficient at this wavelength.

For a *lossless* stack (e.g. swap `"Si"` for `"SiO2"` or a constant index) you
should instead get `energy_balance == 1` to ~10⁻⁹.

!!! tip "Sanity check: energy balance"
    `result.energy_balance` is your first-line diagnostic. For a passive,
    lossless structure it must equal 1. A value above ~1.01 signals incomplete
    convergence (raise `n_orders`); a value far above 1 signals numerical
    breakdown (see [Troubleshooting](troubleshooting.md)). Ikarus emits a
    `RuntimeWarning` automatically in both cases.

## Next steps

- [Theory](theory.md) — the RCWA/Fourier-modal method behind the solver.
- [Core Concepts](core-concepts.md) — geometry, materials, layers, sources, results.
- [Tutorials](tutorials/index.md) — task-oriented walkthroughs.
- [API Reference](api/index.md) — every public class and function.
