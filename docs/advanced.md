# Advanced Usage

## Batch simulations

Wavelength, angle and geometry sweeps are **embarrassingly parallel** — every
solve is independent. The serial pattern (reuse one `RCWA`, change the source) is
covered in [Parameter sweeps](tutorials/parameter-sweeps.md); here is how to
distribute the work.

Because each solve is CPU- and BLAS-bound, the right pattern is **process-level
parallelism with single-threaded BLAS in each worker** (otherwise the workers and
BLAS fight over cores):

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

!!! tip "Worker granularity"
    Make each task a *whole* solve (or a small batch of them), not a fraction —
    the per-task overhead of pickling a tiny job dominates otherwise. For very fast
    solves, chunk the wavelengths so each worker does dozens.

## Optimization workflows

The built-in [`ikarus.inverse`](api/inverse.md) module covers gradient-free
inverse design (mixed-variable GA / NSGA-III). Typical workflows:

- **Single objective** (e.g. broadband AR, a beam deflector): one `Target`, the
  GA runs automatically.
- **Multi-objective** (e.g. high transmittance *and* a phase target): pass a list
  of `Target`s; `optimize` switches to NSGA-III and `OptimizeResult.report()`
  summarizes the Pareto front.
- **Custom figures of merit**: if a metric is not built in, write your own loop
  over `atom.build(params, n_orders)` and drive any optimizer (scipy,
  Optuna, CMA-ES, …) — `MetaAtom.variables()` gives you the search space.

```python
# A hand-rolled objective using your own optimizer:
from ikarus.inverse import MetaAtom, free, pixels

atom = MetaAtom(period=180e-9, cover="Air", substrate="SiO2")
atom.add_pattern(topology=pixels(8, 8, "c4v"), materials=["Air", "Si3N4"],
                 height=free(40e-9, 200e-9))

def objective(params):
    rcwa = atom.build(params, n_orders=6)
    rcwa.set_source(wavelength=450e-9, theta=0, polarization="linear")
    return rcwa.simulate()[2].R_total      # minimize reflection
```

See [Performance](performance.md) for why single-threaded BLAS matters inside
these loops.

## Integration with machine learning

Ikarus is a fast, differentiable-free **forward model**, which makes it a natural
data generator and evaluator for ML-driven photonics:

- **Dataset generation.** Sample random `MetaAtom` parameters (or topologies),
  solve, and store `(geometry → spectrum)` pairs to train a surrogate. Parallelize
  as in [Batch simulations](#batch-simulations).
- **Surrogate-in-the-loop.** Train a neural network on the dataset, optimize on the
  surrogate, then verify candidates with the true Ikarus solve.
- **Population optimizers.** The GA/NSGA-III backend already treats the solver as a
  black box; swap in CMA-ES, Bayesian optimization or RL by reusing
  `MetaAtom.build` + `simulate`.

```python
import numpy as np
from ikarus.inverse import MetaAtom, free, pixels

atom = MetaAtom(period=180e-9, cover="Air", substrate="SiO2")
atom.add_pattern(topology=pixels(10, 10, "c4v"), materials=["Air", "Si3N4"],
                 height=free(40e-9, 220e-9))
spec = atom.variables()       # the labelled search space for sampling

def random_sample(rng):
    params = {}
    for name, kind in spec.items():
        params[name] = rng.uniform(*kind[1]) if kind[0] == "real" else int(rng.integers(2))
    rcwa = atom.build(params, n_orders=6)
    spectrum = []
    for wl in np.linspace(400e-9, 700e-9, 16):
        rcwa.set_source(wavelength=wl, theta=0, polarization="linear")
        spectrum.append(rcwa.simulate()[2].R_total)
    return params, np.array(spectrum)
```

!!! note "No autodiff"
    Ikarus does not provide analytic/AD gradients. For gradient-based topology
    optimization you would need a differentiable RCWA (e.g. a JAX/autograd
    implementation); Ikarus targets gradient-free and surrogate workflows.

## High-performance computing

- **Single node, many cores:** process pool + single-thread BLAS (above). This
  scales near-linearly across a sweep until memory-bound.
- **Clusters:** because tasks are independent, a job array (SLURM `--array`, or any
  scheduler) over wavelength/angle/parameter chunks is the simplest and most
  robust approach — no inter-process communication needed.
- **Memory:** the dominant object is the set of `2P×2P` complex matrices per layer,
  \(P=(2M_x+1)(2M_y+1)\). Budget memory by `n_orders` (see
  [Performance → Memory scaling](performance.md#memory-scaling)); reduce `n_orders`
  or the number of simultaneously-live solves if you hit limits.
- **Large single solves** (\(M \gtrsim 20\)) *do* benefit from multi-threaded BLAS —
  there, let BLAS use the cores and parallelize at a coarser grain.

## Custom materials

Three ways to add a material (full API: [Layers & Materials](api/materials-layers.md)):

```python
from ikarus import RCWA, MaterialLibrary, Material

# 1. Inline constant index (quick tests):
rcwa.add_uniform_layer(100e-9, 1.46)

# 2. Import tabulated n,k from CSV into a library:
lib = MaterialLibrary()
lib.add_from_file("ito_nk.csv", name="ITO")            # columns: lambda_nm n [k]
rcwa = RCWA(period_x=1e-6, period_y=1e-6, materials=lib)
rcwa.add_uniform_layer(50e-9, "ITO")

# 3. A Lorentz-model material from JSON, persisted to the database:
mat = Material.from_file("my_lorentz.json")
lib.register(mat)
```

For a permanent, name-addressable material across sessions, use the
[`ikarus-add-material` CLI](api/tools.md#material-import-cli) or
`add_from_file(..., persist=True)`.

## Custom geometries

Beyond the [shape primitives](api/shapes.md), a topology is just an integer
`numpy.ndarray` — build it however you like:

```python
import numpy as np
from ikarus import shapes

# From an analytic mask (a super-ellipse):
N = 160
x = (np.arange(N) + 0.5) / N - 0.5
X, Y = np.meshgrid(x, x, indexing="ij")
topo = ((np.abs(X) ** 4 + np.abs(Y) ** 4) < 0.3 ** 4).astype(int)

# From an image (binarized) -- e.g. an inverse-designed pattern:
# topo = (imageio.imread("design.png").mean(-1) > 127).astype(int)

# Multi-material by composing shapes:
ring = shapes.ring(inner_radius=0.25, outer_radius=0.4, grid_shape=(N, N), value=1)
core = shapes.circle(radius=0.18, grid_shape=(N, N), value=2)
topo3 = shapes.combine(ring, core)        # indices 0,1,2 -> three materials
```

Slanted or curved sidewalls are handled by **slicing** the feature into several
thin patterned layers of progressively different cross-section (RCWA is uniform
within a layer — see [Theory → Limitations](theory.md#limitations-of-rcwa)).
