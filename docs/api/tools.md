# Tools

Utilities that surround the solver: convergence testing, HDF5 result I/O and the
material-import CLI.

## Convergence

```python
from ikarus.tools import convergence
```

RCWA accuracy is controlled by the harmonic count `n_orders`. Too few
under-resolves the field; too many wastes time (the eigensolve is
\(\mathcal{O}(P^3)\)). These helpers sweep `n_orders` and pick (or report) where the
result stabilizes. The metric is the change in the **specular (0th-order)
transmittance** between successive order counts, combined with the energy-balance
defect \(|R+T-1|\) for lossless checks.

### `auto_converge_orders(rcwa, mode="once", tol=1e-4, max_orders=200, start=5, step=4, verbose=False) -> (int, int)`

Find and **set** the smallest converged `n_orders` on `rcwa`.

| Argument | Default | Description |
|---|---|---|
| `mode` | `"once"` | `"once"` caches the result (subsequent calls are no-ops); `"always"` re-converges every call. |
| `tol` | `1e-4` | Relative tolerance on the specular transmittance. |
| `max_orders` | `200` | Upper bound on the sweep. |
| `start`, `step` | `5`, `4` | First order count and increment. |
| `verbose` | `False` | Print each trial. |

The sweep respects your **dimensionality intent**: a starting `n_orders=(M, 0)`
converges as a 1-D (x-only) problem, `(M, M)` isotropically in 2-D. This is also
reachable through `RCWA.simulate(auto_converge=...)`.

```python
from ikarus.tools.convergence import auto_converge_orders
auto_converge_orders(rcwa, tol=1e-4, verbose=True)
print("converged n_orders =", rcwa.n_orders)

# or, inline:
T, R, res = rcwa.simulate(auto_converge="once", verbose=True)
```

### `convergence_curve(rcwa, orders, metric="T0") -> (ndarray, ndarray)` { #convergence_curve }

Evaluate a metric over a list of order counts (restoring the original `n_orders`
afterward). `metric` ∈ `"T0"` (specular transmittance), `"R"`, `"T"` (totals),
`"energy"` (\(|R+T-1|\)).

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

Results are stored in a self-describing HDF5 layout: scalar totals, complex
zero-order coefficients, per-order efficiencies and exit angles, the geometry/source
metadata and (optionally) reconstructed field maps. Files are readable by any HDF5
viewer (`h5dump`, `h5py`, HDFView).

!!! note "Optional dependency"
    HDF5 I/O needs **h5py**: `pip install "ikarus-rcwa[io]"`.

### `save_results(rcwa, path, include=("T", "R", "metadata"), result=None)`

Write the most recent (or supplied `result`) to `path`. `include` may contain
`"T"`, `"R"`, `"metadata"` and `"fields"` (the last reconstructs an `xz`
cross-section). Also available as `RCWA.save_results(...)`.

### `load_results(path) -> dict`

Load an Ikarus HDF5 result file into a nested dict. Also `RCWA.load_results(...)`
(staticmethod).

```python
rcwa.save_results("run.h5", include=["T", "R", "metadata", "fields"])

data = RCWA.load_results("run.h5")
print(data["R_total"], data["metadata"]["source"]["wavelength"])
```

The stored datasets include `R_total`, `T_total`, `R_orders`, `T_orders`,
`order_p`, `order_q`, the four exit-angle arrays, `energy_balance`, and the
JSON `metadata` attribute (periods, `n_orders`, `resolution`, layer descriptions
and the source).

## Material import CLI

A console script is installed with the package to import tabulated `n, k` data into
the material database.

```bash
ikarus-add-material my_material.csv --name MyMaterial --comment "from ellipsometry"
# equivalently:
python -m ikarus.tools.add_material my_material.csv --name MyMaterial
```

The input is whitespace- or comma-delimited with columns `wavelength_nm  n  [k]`
(`k` optional → 0; `#` lines ignored). The data is sorted and written as JSON into
the package database, then available by name to `RCWA`.

| Option | Description |
|---|---|
| `path` | CSV/text file. |
| `--name` | **(required)** name to store the material under. |
| `--comment` | optional description. |
| `--db` | target database directory (defaults to the package's). |

See also [`MaterialLibrary.add_from_file`](materials-layers.md#materiallibrary)
for the in-process equivalent.
