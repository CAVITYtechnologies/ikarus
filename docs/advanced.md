# Aerobatics

*Advanced maneuvers for licensed pilots:* batch fleets, optimization loops,
machine-learning pipelines, clusters, and custom everything.

## Batch simulations { #batch-simulations }

Every Ikarus solve is independent — sweeps and populations are embarrassingly
parallel. The right pattern on one machine is **process-level parallelism with
single-threaded BLAS in each worker** (otherwise your workers and BLAS fight
over the same cores and everyone loses):

```python
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
from concurrent.futures import ProcessPoolExecutor
from ikarus import RCWA, shapes


def reflectance_at(wavelength):
    period, N = 450e-9, 96
    disk = shapes.circle(radius=0.3, grid_shape=(N, N))
    rcwa = RCWA(period_x=period, period_y=period, resolution=(N, N), n_orders=(9, 9))
    rcwa.add_uniform_layer(np.inf, "Air")
    rcwa.add_layer(200e-9, disk, ["Air", "Si3N4"])
    rcwa.add_uniform_layer(np.inf, "SiO2")
    rcwa.set_source(wavelength=wavelength, theta=0, polarization="linear")
    return rcwa.simulate()[2].R_total


if __name__ == "__main__":
    wls = np.linspace(400e-9, 800e-9, 200)
    with ProcessPoolExecutor() as pool:
        R = np.array(list(pool.map(reflectance_at, wls)))
```

!!! tip "Right-size the cargo"
    Each task should be a whole solve (or a batch of them) — pickling overhead
    eats tiny jobs alive. If single solves are very fast, chunk dozens of
    wavelengths per task.

## Optimization workflows

The built-in [inverse module](api/inverse.md) covers the common cases:

- **Single objective** — one `Target`, the mixed-variable GA runs automatically.
- **Multi-objective** — a list of `Target`s switches to NSGA-III;
  `OptimizeResult.report()` summarizes the Pareto front.
- **Bring-your-own optimizer** — `MetaAtom` is also a clean parameterization
  layer for *any* black-box optimizer (scipy, Optuna, CMA-ES…):

```python
from ikarus.inverse import MetaAtom, free, pixels

atom = MetaAtom(period=180e-9, cover="Air", substrate="SiO2")
atom.add_pattern(topology=pixels(8, 8, "c4v"), materials=["Air", "Si3N4"],
                 height=free(40e-9, 200e-9))

def objective(params):                       # feed this to anything
    rcwa = atom.build(params, n_orders=6)
    rcwa.set_source(wavelength=450e-9, theta=0, polarization="linear")
    return rcwa.simulate()[2].R_total        # minimize reflection

print(atom.variables())                      # the labelled search space
```

Inside any tight loop, the [BLAS-pinning trick](performance.md#blas-threading)
is worth roughly an order of magnitude. Really.

## Integration with machine learning

Ikarus is a fast black-box forward model — prime surrogate-training material:

- **Dataset generation.** Sample random parameters/topologies, solve, store
  `(geometry → spectrum)` pairs. Parallelize as above.
- **Surrogate-in-the-loop.** Train a network on the dataset, optimize against
  the surrogate, verify winners with the true solve.
- **Population methods.** GA/NSGA-III already treat the solver as a black box;
  swapping in CMA-ES, Bayesian optimization or RL is just a different driver
  around `MetaAtom.build` + `simulate`.

```python
import numpy as np
from ikarus.inverse import MetaAtom, free, pixels

atom = MetaAtom(period=180e-9, cover="Air", substrate="SiO2")
atom.add_pattern(topology=pixels(10, 10, "c4v"), materials=["Air", "Si3N4"],
                 height=free(40e-9, 220e-9))
spec = atom.variables()

def random_sample(rng):
    params = {name: (rng.uniform(*kind[1]) if kind[0] == "real"
                     else int(rng.integers(2)))
              for name, kind in spec.items()}
    rcwa = atom.build(params, n_orders=6)
    spectrum = []
    for wl in np.linspace(400e-9, 700e-9, 16):
        rcwa.set_source(wavelength=wl, theta=0, polarization="linear")
        spectrum.append(rcwa.simulate()[2].R_total)
    return params, np.array(spectrum)
```

!!! note "No autodiff on board"
    Ikarus provides no analytic or AD gradients — it targets gradient-free and
    surrogate workflows. Gradient-based topology optimization needs a
    differentiable RCWA (JAX/autograd-based) instead.

## High-performance computing

- **One node, many cores:** process pool + single-thread BLAS. Near-linear
  scaling across a sweep until memory-bound.
- **Clusters:** independent tasks → a plain job array (SLURM `--array`, any
  scheduler) over parameter chunks. No MPI, no communication, no drama.
- **Memory budget:** the big objects are \(2P \times 2P\) complex matrices per
  layer, \(P=(2M_x+1)(2M_y+1)\) — see
  [Memory scaling](performance.md#memory-scaling). Lower `n_orders` or the
  number of live solves if you hit the ceiling.
- **Large single solves** (\(M \gtrsim 20\)): the opposite regime — let BLAS
  thread, parallelize coarser.

## Custom materials { #custom-materials }

Three escalating levels of commitment
(full API: [Layers & Materials](api/materials-layers.md)):

```python
from ikarus import RCWA, MaterialLibrary, Material

# 1. Commitment-free: a constant index inline.
rcwa.add_uniform_layer(100e-9, 1.46)

# 2. Session-scoped: tabulated n,k from CSV into a library.
lib = MaterialLibrary()
lib.add_from_file("ito_nk.csv", name="ITO")        # columns: lambda_nm n [k]
rcwa = RCWA(period_x=1e-6, period_y=1e-6, materials=lib)
rcwa.add_uniform_layer(50e-9, "ITO")

# 3. A physical model: Lorentz oscillators from JSON.
mat = Material.from_file("my_lorentz.json")
lib.register(mat)
```

For a permanent, name-addressable entry across sessions, use the
[`ikarus-add-material` CLI](api/tools.md#material-import-cli) or
`add_from_file(..., persist=True)`.

## Custom geometries { #custom-geometries }

A topology is just an integer NumPy array — the
[shape primitives](api/shapes.md) are a convenience, not a cage:

```python
import numpy as np
from ikarus import shapes

# Analytic mask: a super-ellipse.
N = 160
x = (np.arange(N) + 0.5) / N - 0.5
X, Y = np.meshgrid(x, x, indexing="ij")
topo = ((np.abs(X) ** 4 + np.abs(Y) ** 4) < 0.3 ** 4).astype(int)

# From an image (binarized) — e.g. a published inverse design:
# topo = (imageio.imread("design.png").mean(-1) > 127).astype(int)

# Three materials by composing shapes:
ring = shapes.ring(inner_radius=0.25, outer_radius=0.4, grid_shape=(N, N), value=1)
core = shapes.circle(radius=0.18, grid_shape=(N, N), value=2)
topo3 = shapes.combine(ring, core)        # indices 0,1,2
```

Slanted or curved sidewalls? Slice them into several thin layers of
progressively varying cross-section — RCWA is uniform within a layer, and the
staircase converges as the slices thin
([Theory → Limitations](theory.md#limitations-of-rcwa)).
