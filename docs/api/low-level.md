# Low-level API

*The engine room.* Most users never come down here — the [`RCWA`](rcwa.md)
façade runs the machinery for you — but everything below deck is public,
documented, and genuinely useful for custom workflows, teaching, or auditing
the physics.

## Fourier machinery — `ikarus.core.fourier`

### `HarmonicGrid`

```python
from ikarus import HarmonicGrid
HarmonicGrid(n_orders_x, n_orders_y)
```

Bookkeeping for the truncated harmonic set; orders run
`-n_orders … +n_orders` inclusive.

| Member | Description |
|---|---|
| `num_x`, `num_y` | Count per axis, `2*n_orders + 1`. |
| `size` | Total harmonics \(P = num_x \cdot num_y\). |
| `orders_x`, `orders_y` | Integer order arrays per axis. |
| `index_arrays() -> (p, q)` | Flattened `(p, q)` order labels (row-major: `p` slow, `q` fast). |
| `zero_order_index() -> int` | Flat index of the specular `(0, 0)` harmonic. |

### `convolution_matrix(cell, grid) -> ndarray`

The convolution (Toeplitz) matrix of a periodic cell function sampled on an
`(Nx, Ny)` grid, for the harmonics in `grid` — how geometry enters the
algebra. Returns a complex `(P, P)` matrix. Raises `ValueError` if the
sampling can't resolve the required difference orders (needs ≥
`4*n_orders + 1` samples per axis).

```python
import numpy as np
from ikarus import HarmonicGrid
from ikarus.core.fourier import convolution_matrix

grid = HarmonicGrid(5, 5)
eps_cell = np.where(disk_mask, 6.0, 1.0)        # (Nx, Ny) permittivity
EPS = convolution_matrix(eps_cell, grid)        # (P, P)
```

### `reciprocal_vectors(period_x, period_y) -> (float, float)`

The reciprocal-lattice spacings \((2\pi/\Lambda_x,\ 2\pi/\Lambda_y)\).

## Solver — `ikarus.core.solver`

The stateless heart. Public names:

| Name | Description |
|---|---|
| `solve_stack(...)` | Solve a full stack → `FieldSolution`. |
| `FieldSolution` | The modal solution: mode matrices, S-matrices, per-order efficiencies/coefficients, wavevector matrices. |
| `SMatrix` | A scattering matrix with four `2P×2P` blocks (`S11`…`S22`); `SMatrix.identity(dim)`. |
| `redheffer_star(a, b) -> SMatrix` | The star product — computed via `solve`, never an explicit inverse. |
| `uniform_modes(eps, Kx, Ky)` | Analytic eigenmodes `(W, V, Kz)` of a homogeneous medium, with the consistent forward-branch selection. |
| `layer_modes(...)`, `layer_smatrix(...)` | Per-layer eigenmodes and scattering matrix. |
| `wavevector_matrices(grid, kx0, ky0, period_x, period_y, wavelength)` | Diagonal normalized `Kx, Ky`. |

`solve_stack` is exactly what `RCWA._solve()` calls (wavevectors normalized by
\(k_0\)):

```python
solve_stack(eps_grids, heights, eps_ref, eps_trn, grid,
            kx0, ky0, period_x, period_y, wavelength, polarization_xy)
```

| Argument | Description |
|---|---|
| `eps_grids` | List of `(Nx, Ny)` permittivity grids, one per **interior** layer. |
| `heights` | Interior layer thicknesses (m). |
| `eps_ref`, `eps_trn` | Cover and substrate scalar permittivities. |
| `grid` | A `HarmonicGrid`. |
| `kx0`, `ky0` | Incident in-plane wavevector (normalized). |
| `period_x`, `period_y`, `wavelength` | Geometry / wavelength (m). |
| `polarization_xy` | The `(p_x, p_y)` transverse polarization components. |

```python
import numpy as np
from ikarus import HarmonicGrid
from ikarus.core.solver import solve_stack

grid = HarmonicGrid(8, 8)
sol = solve_stack(
    eps_grids=[eps_cell],            # one patterned layer
    heights=[200e-9],
    eps_ref=1.0, eps_trn=2.13,       # Air / SiO2
    grid=grid,
    kx0=0.0, ky0=0.0,
    period_x=500e-9, period_y=500e-9, wavelength=600e-9,
    polarization_xy=(1.0, 0.0),
)
i0 = grid.zero_order_index()
print("specular T:", sol.T_orders[i0], "  R+T:", sol.R_total + sol.T_total)
```

!!! tip "When to come down here"
    (a) Reusing one eigendecomposition across many cascades, (b) building a
    non-standard excitation, (c) validating the façade. For daily flying, the
    [`RCWA`](rcwa.md) object is clearer and safer — it validates the stack and
    packages the results.

## Numerical fine print

- **Branch selection.** `uniform_modes` and the patterned-layer modes share one
  forward/decaying branch rule (`_forward_branch`) — evanescent magnetic-mode
  signs stay consistent across gap, regions and layers. Without this, every
  grating result is silently wrong
  ([the war story](../theory.md#branch-selection-and-stability)).
- **No explicit inverses.** The Redheffer star and other hot paths use
  `scipy.linalg.solve` right-division — faster *and* better conditioned.
- **Anomaly regularization.** Orders sitting exactly on a Rayleigh–Wood
  anomaly get a vanishing imaginary loss so the algebra stays non-singular.
