# RCWA & Results

The user-facing façade. It collects the layer stack, the illumination and the
numerical settings, then orchestrates the stateless solver into diffraction
efficiencies, complex coefficients and real-space fields. If you only learn two
objects in Ikarus, learn these.

## `RCWA`

```python
RCWA(period_x, period_y, resolution=32, n_orders=25,
     dtype=np.complex128, materials=None, convergence_tol=1e-6,
     factorization="li")
```

**Purpose.** Create a solver bound to one unit cell.

**Parameters**

| Name | Type | Default | Description |
|---|---|---|---|
| `period_x`, `period_y` | `float` | — | Unit-cell periods (m); must be positive. |
| `resolution` | `int` or `(int, int)` | `32` | Real-space sampling grid for the convolution matrices. Raised internally to ≥ `4*n_orders + 1`. |
| `n_orders` | `int` or `(int, int)` | `25` | Max positive harmonic order `M` per axis. Total harmonics \(P=(2M_x+1)(2M_y+1)\). |
| `dtype` | numpy dtype | `complex128` | Working precision. |
| `materials` | `MaterialLibrary` | `None` | Custom library; defaults to the shared `default_library`. |
| `convergence_tol` | `float` | `1e-6` | Default tolerance for `auto_converge`. |
| `factorization` | `str` | `"li"` | Fourier factorization rule: `"li"` (Li's inverse rule — fast TM/high-contrast convergence) or `"laurent"` (the classic direct rule). See [Factorization](#factorization) below. |

**Raises.** `ValueError` if a period is non-positive, or if `factorization` is not `"li"`/`"laurent"`.

### Factorization { #factorization }

Patterned layers represent products like \(D = \varepsilon E\) of truncated Fourier
series. The rule used to factorize them sets the convergence rate:

- **`"li"` (default)** — Li's **inverse rule**, applied in its two-step (separable)
  form for crossed gratings. The component of \(E\) that is *discontinuous* across a
  boundary is factorized with \(⟦1/\varepsilon⟧^{-1}\) instead of \(⟦\varepsilon⟧\),
  which restores fast convergence for **TM / high-index-contrast** structures
  (these settle by \(n_\text{orders}\approx 10\text{–}15\) instead of drifting).
  It is **fully automatic** for any topology and any number of materials — it acts
  on the rendered \(\varepsilon(x,y)\) grid, so you never write anything per shape.
- **`"laurent"`** — the classic **direct rule** (\(⟦\varepsilon⟧\) everywhere).
  Kept for comparison and reproducibility.

!!! warning "Energy balance is **not** a convergence test"
    For high-contrast TM, the direct rule can give `energy_balance` ≈ 1 while `R`
    and the phase are still far from converged. Always confirm they have stopped
    moving with `n_orders` — the inverse rule is what makes that happen quickly.

At the **same** `n_orders` the two rules cost the same (the eigensolve dominates;
uniform layers short-circuit to identical work). For the **same accuracy** Li is far
cheaper because it converges at much lower `n_orders`.

### Properties

| Property | Description |
|---|---|
| `n_orders` | `(Mx, My)` tuple; **settable** (resets the convergence cache). |
| `shapes` | The shape-primitive library: `rcwa.shapes.circle(...)`. |
| `last_solution` | The most recent `FieldSolution`, or `None`. |
| `layers` | The list of `Layer` objects. |
| `source` | The current `Source`, or `None`. |

### Building the stack

#### `add_uniform_layer(height, material, name="") -> Layer` { #add_uniform_layer }

Append a uniform (single-material) layer. Use `height=np.inf` for the
semi-infinite cover (first) and substrate (last).

```python
rcwa.add_uniform_layer(np.inf, "Air")       # cover
rcwa.add_uniform_layer(220e-9, "TiO2")      # interior
rcwa.add_uniform_layer(np.inf, "SiO2")      # substrate
```

#### `add_layer(height, topology, materials, resolution=None, name="") -> Layer` { #add_layer }

Append a patterned layer. `topology` is an integer `(Nx, Ny)` array; `materials`
is the list it indexes into.

```python
import numpy as np
topo = np.zeros((128, 128), dtype=int)
topo[32:96, 32:96] = 1                       # a square of material[1]
rcwa.add_layer(200e-9, topo, ["Air", "Si"]) # 0 -> Air, 1 -> Si
```

### Defining the source

#### `set_source(**kwargs) -> Source`

Create or update the illumination. The **first** call must include
`wavelength`. Later calls update only what you pass and retain the rest —
the backbone of every sweep. Accepts every [`Source`](source.md) field:
`wavelength`, `theta`, `phi`, `polarization`, `linear_pol_angle`.

```python
rcwa.set_source(wavelength=600e-9, theta=0, polarization="linear", linear_pol_angle=90)
rcwa.set_source(theta=15)   # keeps wavelength + polarization
```

### Solving

#### `simulate(auto_converge="never", converge_tol=None, max_orders=200, verbose=False, check_convergence=False) -> (T, R, result)`

Run a simulation; return the zero-order coefficients plus the full
[`SimulationResult`](#simulationresult).

| `auto_converge` | Behaviour |
|---|---|
| `"never"` (default) | Use the current `n_orders`. |
| `"once"` | Find and **cache** a converged `n_orders` (later calls reuse it). |
| `"always"` | Re-converge on every call. |

Convergence is judged on the **complex zero-order R/T coefficients (magnitude
*and* phase)**, never the energy balance — see the box below.

`check_convergence=True` re-solves once at a higher `n_orders` and **warns** if
the zero-order R/T are still moving — a cheap safety net for a single solve.
Leave it off inside tight sweep/optimization loops (it doubles that solve).

```python
T, R, result = rcwa.simulate()
T, R, result = rcwa.simulate(auto_converge="once", verbose=True)   # reliable, automatic
T, R, result = rcwa.simulate(check_convergence=True)               # warn me if under-resolved
```

!!! warning "Energy balance is not a convergence test"
    A lossless structure conserves energy (`R+T≈1`) at **every** `n_orders`, even
    while `R` and the **phase** are still drifting — the trap high-contrast TM
    sets, and it has cost real optimization runs. Always converge the *coefficients*
    (`auto_converge`, `check_convergence`, or watch `R`/`R_phase` vs `n_orders`),
    not the energy.

**Returns.** `(T, R, result)` — `T`/`R` are the zero-order coefficients
(complex scalar for linear polarization, `{"co", "cross"}` for circular);
`result` is a `SimulationResult`.

!!! warning "Energy-balance warnings"
    `simulate` emits a `RuntimeWarning` if `R+T` exceeds `1.01` (likely
    unconverged) or `1.5` (numerical breakdown — reduce `n_orders` or raise
    `resolution`). See [Troubleshooting](../troubleshooting.md).

### Field extraction

#### `get_fields(z_positions=None, plane="xy", nx=64, ny=64, x_position=0.0, y_position=0.0) -> dict` { #get_fields }

Reconstruct real-space `E`/`H` fields from the last simulation (solving first
if needed). Returns a dict of [`FieldMap`](fields-viz.md#fieldmap).

- `plane="xy"` → one map per depth `z` in `z_positions` (meters; `z=0` at the
  cover/first-layer interface, increasing into the stack).
- `plane="xz"` / `"yz"` → a single cross-section sweeping the whole stack at
  fixed `y_position` / `x_position`.

```python
xy = rcwa.get_fields(z_positions=[80e-9], plane="xy", nx=128, ny=128)
xz = rcwa.get_fields(plane="xz", nx=160, y_position=rcwa.period_y / 2)["xz"]
```

Each map carries the structure permittivity on its grid (`FieldMap.eps`) so
plots can overlay material outlines.

### Visualization

#### `visualize_structure(plane="xz", layer_index=None, **kwargs)`

Plot the layer stack (`plane="xz"`) or a patterned layer's topology
(`plane="xy"`, with `layer_index`). Requires matplotlib. See
[Fields & Visualization](fields-viz.md).

#### `visualize_fields(field_map=None, component="intensity", **kwargs)`

Plot a reconstructed field map (computes an `xz` map if none is supplied).

### Result I/O

#### `save_results(path, include=("T", "R", "metadata"), result=None)`

Write the most recent (or supplied) result to HDF5. `include` may also contain
`"fields"`. Requires h5py.

#### `load_results(path)` *(staticmethod)*

Load an Ikarus HDF5 result file into a nested dict.

---

## `SimulationResult` { #simulationresult }

The full flight report, returned as the third element of `simulate()`.

### Attributes

| Attribute | Type | Description |
|---|---|---|
| `T`, `R` | complex or dict | Zero-order coefficient(s); `abs(.)**2` = specular efficiency. |
| `T_total`, `R_total` | `float` | Total transmitted / reflected power. |
| `T_orders`, `R_orders` | `ndarray` | Per-order efficiencies, aligned with `orders`. |
| `orders` | `(ndarray, ndarray)` | `(p, q)` integer order labels. |
| `theta_out_trn`, `phi_out_trn` | `ndarray` | Transmitted exit angles (deg); `NaN` if evanescent. |
| `theta_out_ref`, `phi_out_ref` | `ndarray` | Reflected exit angles (deg). |
| `energy_balance` | `float` | `R_total + T_total`. |
| `solution` | `FieldSolution` | Underlying modal solution (for field reconstruction). |

### Properties

| Property | Description |
|---|---|
| `R_phase`, `T_phase` | Phase (rad) of the zero-order coefficient — a `float` (linear) or `{"co","cross"}` dict (circular). |

### Methods

#### `order_index(p, q) -> int`

Flat index of harmonic `(p, q)` into the `*_orders` / angle arrays. Raises
`KeyError` if the order is outside the truncated set.

```python
i = result.order_index(0, 0)
print("specular T:", result.T_orders[i], "at", result.theta_out_trn[i], "deg")

ip1 = result.order_index(1, 0)          # +1 reflected order of a grating
print("R(+1):", result.R_orders[ip1])
```

### Best practices

- `energy_balance` first, always: a lossless structure must read 1 within
  convergence error.
- Specular metasurface design → `T`/`R` (+ `T_phase`/`R_phase`). Grating order
  steering → `T_orders`/`R_orders` with `order_index`.
- Exit-angle arrays carry `NaN` for evanescent orders — `np.isfinite(...)`
  before plotting.
