# Quick Start

*Your first flight.* In the next five minutes you will build a layered
structure, shine a plane wave at it, and read off reflectance, transmittance
and phase — and you'll understand every line you typed.

!!! info "The two house rules"
    1. **Everything is SI.** Lengths in **meters** (yes, `200e-9`, not `200`),
       angles in **degrees**.
    2. **Light flies top-down.** The first layer is the sky (cover), the last
       is the ground (substrate), and both are semi-infinite.

## The whole program

A 200 nm crystalline-silicon film between air and glass, hit at normal
incidence with a red HeNe laser:

```python
import numpy as np
from ikarus import RCWA

# 1. The arena: unit-cell periods, sampling grid, harmonic orders.
rcwa = RCWA(period_x=500e-9, period_y=500e-9, resolution=64, n_orders=8)

# 2. The stack, top to bottom.
rcwa.add_uniform_layer(height=np.inf, material="Air")    # semi-infinite cover
rcwa.add_uniform_layer(height=200e-9, material="Si")     # the film
rcwa.add_uniform_layer(height=np.inf, material="SiO2")   # semi-infinite substrate

# 3. The light.
rcwa.set_source(wavelength=633e-9, theta=0, phi=0,
                polarization="linear", linear_pol_angle=0.0)

# 4. Fly.
T, R, result = rcwa.simulate()

print(f"R     = {result.R_total:.4f}")
print(f"T     = {result.T_total:.4f}")
print(f"R + T = {result.energy_balance:.6f}")
```

## What did I just type?

### Step 1 — the arena

```python
RCWA(period_x, period_y, resolution=32, n_orders=25,
     dtype=np.complex128, materials=None, convergence_tol=1e-6)
```

| Knob | What it really does |
|---|---|
| `period_x`, `period_y` | The unit cell of your periodic world (m). Even a plain thin film needs one — any value works, since nothing varies across it. |
| `n_orders` | **The** accuracy dial. `M` keeps Fourier harmonics \(-M..+M\) per axis, \(P=(2M_x+1)(2M_y+1)\) total. More harmonics = sharper features resolved = steeply more cost. |
| `resolution` | The real-space pixel grid your geometry is sampled on. Only needs to *draw* the structure — Ikarus silently raises it to ≥ `4*n_orders + 1` to keep the Fourier algebra alias-free. |
| `materials` | A custom [`MaterialLibrary`](api/materials-layers.md); default is the shared built-in one. |

### Step 2 — the stack

Layers are listed **cover first, substrate last**, and the rules are strict
(Ikarus will refuse to fly otherwise):

- First and last layer: **uniform** and **semi-infinite** (`height=np.inf`).
- Every interior layer: **finite** thickness.
- `add_uniform_layer(height, material)` → one material fills the cell.
- `add_layer(height, topology, materials)` → a *patterned* layer
  ([Lesson 3](tutorials/metasurfaces.md) is all about these).

A material can be a database name (`"Si"`), a bare number (a constant index —
`1.5` is instant glass), or a [`Material`](api/materials-layers.md#material)
object.

### Step 3 — the light

```python
set_source(wavelength, theta=0, phi=0, polarization="linear", linear_pol_angle=0.0)
```

| Knob | Meaning |
|---|---|
| `wavelength` | Vacuum wavelength (m). |
| `theta` | Tilt from straight-down (+z axis), degrees. `0` = normal incidence. |
| `phi` | Compass direction of the tilt in the xy-plane, from +x, degrees. |
| `polarization` | `"linear"`, `"RCP"` or `"LCP"`. |
| `linear_pol_angle` | Angle from TE, degrees: `0` = TE/s, `90` = TM/p. At normal incidence: 0 = E along +y, 90 = E along +x. |

`set_source` has a memory: after the first call, it only updates what you pass
and keeps the rest. That makes sweeps delightfully terse:

```python
for wl in np.linspace(450e-9, 750e-9, 31):
    rcwa.set_source(wavelength=wl)        # theta & polarization carried over
    _, _, res = rcwa.simulate()
```

### Step 4 — the landing

`simulate()` hands back a tuple `(T, R, result)`:

- **`T`, `R`** — complex **zero-order** (specular) amplitudes;
  `abs(coeff)**2` is the specular efficiency. (For circular polarization each
  becomes a `{"co", "cross"}` dict — see [Lesson 5](tutorials/polarization.md).)
- **`result`** — the full flight report, a
  [`SimulationResult`](api/rcwa.md#simulationresult):

| Field | Contents |
|---|---|
| `R_total`, `T_total` | Total reflected / transmitted power (all propagating orders). |
| `R_orders`, `T_orders` | Per-order efficiencies, aligned with `orders`. |
| `orders` | `(p, q)` integer labels for each harmonic. |
| `theta_out_*`, `phi_out_*` | Exit angles per order (deg); `NaN` = evanescent. |
| `energy_balance` | `R_total + T_total` — must be 1 for a lossless stack. |
| `R_phase`, `T_phase` | Phase (rad) of the zero-order coefficient. |
| `solution` | The raw modal solution, for [field maps](api/fields-viz.md). |

Pick out a specific diffraction order with `order_index`:

```python
i00 = result.order_index(0, 0)            # the specular order
print(result.T_orders[i00], result.theta_out_trn[i00])
```

## What numbers should I see?

This particular stack is a thin-film problem (nothing is patterned), so the
answer is the exact Fresnel/transfer-matrix result. Silicon mildly absorbs at
633 nm, so expect roughly:

- `R ≈ 0.35`, `T ≈ 0.40`
- `R + T ≈ 0.75` — **less than 1**, and that's physics, not a bug: the missing
  25 % was absorbed in the silicon.

Swap `"Si"` for `"SiO2"` (lossless) and `energy_balance` snaps to 1 within
~10⁻⁹.

!!! tip "The pilot's instrument: energy balance"
    `result.energy_balance` is your altimeter. Lossless structure → must read
    1. Reads ~1.01+? Unconverged — raise `n_orders`. Reads 10⁸? You flew too
    close to the sun — see [Troubleshooting](troubleshooting.md). Ikarus emits
    a `RuntimeWarning` in both cases, so you won't miss it.

## Where to next?

<div class="grid cards" markdown>

-   :material-sine-wave: **[How RCWA Works](theory.md)** — the physics, told
    properly (and entertainingly).

-   :material-school: **[Flight School](tutorials/index.md)** — six lessons
    from spectra to oblique incidence.

-   :material-cube-outline: **[Core Concepts](core-concepts.md)** — the cast
    of objects you'll compose.

-   :material-api: **[API Reference](api/index.md)** — every class, argument
    and return value.

</div>
