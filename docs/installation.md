# Installation

## Requirements

| Requirement | Version | Notes |
|---|---|---|
| Python | ≥ 3.9 | CPython tested on 3.9–3.12 |
| NumPy | ≥ 1.21 | required |
| SciPy | ≥ 1.7 | required (linear algebra, interpolation) |
| matplotlib | ≥ 3.4 | optional — visualization only |
| h5py | ≥ 3.0 | optional — HDF5 result I/O only |
| pymoo | ≥ 0.6 | optional — inverse design only |

Only NumPy and SciPy are hard dependencies. The optional packages are imported
lazily, so the core solver works without them and you only pay for what you use.

## pip installation

The package is published on PyPI as **`ikarus-rcwa`** (the import name is
`ikarus`):

```bash
pip install ikarus-rcwa
```

### Optional extras

Install the optional feature groups with the bracket syntax:

```bash
pip install "ikarus-rcwa[viz]"      # + matplotlib  (plotting)
pip install "ikarus-rcwa[io]"       # + h5py        (HDF5 export/import)
pip install "ikarus-rcwa[inverse]"  # + pymoo       (gradient-free inverse design)
pip install "ikarus-rcwa[all]"      # everything above
```

| Extra | Pulls in | Enables |
|---|---|---|
| `viz` | matplotlib | [`visualize_structure`](api/rcwa.md), [`visualize_fields`](api/fields-viz.md), `ikarus.visualization` |
| `io` | h5py | [`save_results` / `load_results`](api/tools.md) |
| `inverse` | pymoo | [`ikarus.inverse`](api/inverse.md) (`optimize`) |
| `all` | matplotlib + h5py + pymoo | all optional features |
| `dev` | + pytest | running the test suite |

## Source installation

```bash
git clone https://github.com/CAVITYtechnologies/ikarus.git
cd ikarus
pip install -e ".[all]"     # editable install with every optional feature
```

An editable (`-e`) install lets you modify the source and have the changes picked
up immediately — convenient for development or for tweaking the material
database.

## Verifying the installation

```python
import ikarus
print(ikarus.__version__)

from ikarus import RCWA, default_library
print("materials:", default_library.available())
# -> ['Air', 'Au', 'GaN', 'GaP', 'Si', 'Si3N4', 'SiO2', 'TiO2', 'aSi']
```

Run the bundled validation tests (requires the `dev` extra):

```bash
pytest ikarus/tests -q
```

## GPU support

Ikarus is **CPU-only**: the linear algebra runs on NumPy/SciPy (LAPACK/BLAS).
There is currently no CUDA/JAX/PyTorch backend.

In practice the dominant cost is a dense eigendecomposition whose size is the
harmonic count \(P=(2M_x+1)(2M_y+1)\) (see [Performance](performance.md)). Two
things matter more than a GPU for typical metaatom problems:

- **BLAS threading.** For the small-to-moderate matrices in a tight sweep or
  optimization loop, *single-threaded* BLAS is often faster than letting it
  oversubscribe cores. Set the thread count before importing NumPy:

  ```python
  import os
  for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
            "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
      os.environ.setdefault(v, "1")
  import numpy as np  # noqa: E402
  ```

- **Process-level parallelism.** Sweeps over wavelength/angle and inverse-design
  populations are embarrassingly parallel; distribute them across processes (see
  [Advanced Usage → Batch simulations](advanced.md#batch-simulations)).

## Dependency explanations

- **NumPy** — array storage and FFTs (the Fourier coefficients of each layer's
  permittivity are obtained with `numpy.fft.fft2`).
- **SciPy** — `scipy.linalg.eig` for the layer eigenmodes (called with
  `overwrite_a=True, check_finite=False` for speed), `scipy.linalg.solve` for the
  Redheffer star product (it never forms an explicit inverse), and
  `scipy.interpolate.interp1d` for material dispersion.
- **matplotlib** *(optional)* — all plotting helpers in `ikarus.visualization`.
- **h5py** *(optional)* — the self-describing HDF5 result format.
- **pymoo** *(optional)* — the mixed-variable genetic algorithm and NSGA-III used
  by `ikarus.inverse.optimize`.

## Troubleshooting installation

??? failure "`ImportError: inverse.optimize needs pymoo`"
    The inverse-design module is an optional extra. Install it with
    `pip install "ikarus-rcwa[inverse]"` or `pip install pymoo`.

??? failure "`ImportError: HDF5 I/O requires the 'h5py' package`"
    Install the I/O extra: `pip install "ikarus-rcwa[io]"` or `pip install h5py`.

??? failure "Plotting calls raise `ModuleNotFoundError: matplotlib`"
    Install the visualization extra: `pip install "ikarus-rcwa[viz]"`.

??? failure "`pip install ikarus` installs the wrong package"
    The PyPI distribution name is **`ikarus-rcwa`**, not `ikarus` (which is an
    unrelated project). The *import* name is still `import ikarus`.

??? failure "Slow first import / slow sweeps on many-core machines"
    This is usually BLAS thread oversubscription. Pin the BLAS thread count to 1
    as shown in [GPU support](#gpu-support) above; for large single solves
    (\(M \gtrsim 20\)) you may instead want multiple threads — benchmark both.
