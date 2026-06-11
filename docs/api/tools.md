# Tools

*The ground crew:* convergence automation, HDF5 result I/O, and the
material-import CLI.

## Convergence

```python
from ikarus.tools import convergence
```

RCWA accuracy hangs on `n_orders` — too few under-resolves, too many wastes
\(\mathcal{O}(P^3)\) time. These helpers sweep the order count and pick (or
chart) the point where the answer stops moving. Metric: the change in
**specular (0th-order) transmittance** between successive counts, combined
with the energy defect \(|R+T-1|\) for lossless sanity.

### `auto_converge_orders(rcwa, mode="once", tol=1e-4, max_orders=200, start=5, step=4, verbose=False) -> (int, int)`

Find and **set** the smallest converged `n_orders` on `rcwa`.

| Argument | Default | Description |
|---|---|---|
| `mode` | `"once"` | `"once"` caches (later calls are no-ops); `"always"` re-converges every call. |
| `tol` | `1e-4` | Relative tolerance on specular transmittance. |
| `max_orders` | `200` | Sweep ceiling. |
| `start`, `step` | `5`, `4` | First order count and increment. |
| `verbose` | `False` | Print each trial. |

The sweep respects your **dimensionality intent**: starting from
`n_orders=(M, 0)` it converges as a 1-D (x-only) problem; from `(M, M)`,
isotropically in 2-D. Also reachable as
`RCWA.simulate(auto_converge=...)`.

```python
from ikarus.tools.convergence import auto_converge_orders
auto_converge_orders(rcwa, tol=1e-4, verbose=True)
print("converged n_orders =", rcwa.n_orders)

# or inline:
T, R, res = rcwa.simulate(auto_converge="once", verbose=True)
```

### `convergence_curve(rcwa, orders, metric="T0") -> (ndarray, ndarray)` { #convergence_curve }

Evaluate a metric over a list of order counts — and politely restore the
original `n_orders` afterward. `metric` ∈ `"T0"` (specular transmittance),
`"R"`, `"T"` (totals), `"energy"` (\(|R+T-1|\)).

```python
import numpy as np
from ikarus.tools.convergence import convergence_curve
orders, vals = convergence_curve(rcwa, range(4, 21, 2), metric="energy")
# plot vals vs (2*orders+1)**2 to see the convergence rate
```

## HDF5 I/O

```python
from ikarus.tools import io
```

Results land in a self-describing HDF5 layout: totals, complex zero-order
coefficients, per-order efficiencies and exit angles, geometry/source metadata
and (optionally) field maps. Readable by any HDF5 viewer — `h5dump`, `h5py`,
HDFView.

!!! note "Optional dependency"
    Needs **h5py**: `pip install "ikarus-rcwa[io]"`.

### `save_results(rcwa, path, include=("T", "R", "metadata"), result=None)`

Write the most recent (or supplied `result`) to `path`. `include` may contain
`"T"`, `"R"`, `"metadata"` and `"fields"` (the last reconstructs an `xz`
cross-section). Also available as `RCWA.save_results(...)`.

### `load_results(path) -> dict`

Load an Ikarus HDF5 file into a nested dict. Also `RCWA.load_results(...)`
(staticmethod).

```python
rcwa.save_results("run.h5", include=["T", "R", "metadata", "fields"])

data = RCWA.load_results("run.h5")
print(data["R_total"], data["metadata"]["source"]["wavelength"])
```

Stored datasets: `R_total`, `T_total`, `R_orders`, `T_orders`, `order_p`,
`order_q`, the four exit-angle arrays, `energy_balance`, plus the JSON
`metadata` attribute (periods, `n_orders`, `resolution`, layers, source).

## Material import CLI { #material-import-cli }

A console script for importing tabulated `n, k` data into the material
database — measured something on the ellipsometer? Two minutes later it's a
named material:

```bash
ikarus-add-material my_material.csv --name MyMaterial --comment "from ellipsometry"
# equivalently:
python -m ikarus.tools.add_material my_material.csv --name MyMaterial
```

Input: whitespace- or comma-delimited columns `wavelength_nm  n  [k]`
(`k` optional → 0; `#` lines ignored). The data is sorted, written as JSON into
the package database, and immediately available by name to `RCWA`.

| Option | Description |
|---|---|
| `path` | CSV/text file. |
| `--name` | **(required)** name to store the material under. |
| `--comment` | optional description. |
| `--db` | target database directory (defaults to the package's). |

In-process equivalent:
[`MaterialLibrary.add_from_file`](materials-layers.md#materiallibrary).
