# Installation

*Strapping on the wings.* The core needs exactly two things — NumPy and SciPy —
and everything else is optional, lazily imported, and pay-as-you-go.

## The one-liner

The package lives on PyPI as **`ikarus-rcwa`** (the import name is plain
`ikarus` — that name was already taken on PyPI by a stranger):

```bash
pip install ikarus-rcwa
```

That's a working solver. For the full experience:

```bash
pip install "ikarus-rcwa[all]"     # + plotting, HDF5, inverse design
```

## Choose your loadout

=== "pip (recommended)"

    ```bash
    pip install ikarus-rcwa             # core solver only
    pip install "ikarus-rcwa[viz]"      # + matplotlib  (plots)
    pip install "ikarus-rcwa[io]"       # + h5py        (HDF5 export/import)
    pip install "ikarus-rcwa[inverse]"  # + pymoo       (inverse design)
    pip install "ikarus-rcwa[progress]" # + tqdm        (rich progress bars)
    pip install "ikarus-rcwa[all]"      # everything above
    ```

=== "From source"

    ```bash
    git clone https://github.com/CAVITYtechnologies/ikarus.git
    cd ikarus
    pip install -e ".[all]"     # editable: your edits take effect immediately
    ```

    An editable install is the right choice if you want to hack on the solver
    or tweak the shipped material database.

| Extra | Pulls in | Unlocks |
|---|---|---|
| `viz` | matplotlib ≥ 3.4 | structure & field plots, `ikarus.visualization` |
| `io` | h5py ≥ 3.0 | [`save_results` / `load_results`](api/tools.md) |
| `inverse` | pymoo ≥ 0.6 | [`ikarus.inverse`](api/inverse.md) — `optimize()` |
| `progress` | tqdm ≥ 4 | rich [`Sweep` / `progress`](api/sweeps.md) bars (a fallback works without it) |
| `all` | all four | everything |
| `dev` | + pytest ≥ 7.0 | running the test suite |

## Requirements

| Requirement | Version | Role |
|---|---|---|
| Python | ≥ 3.9 | tested on 3.9–3.12 |
| NumPy | ≥ 1.21 | arrays + the FFTs that build the Fourier matrices |
| SciPy | ≥ 1.7 | the eigensolver, `solve`-based linear algebra, dispersion splines |

## Pre-flight check

```python
import ikarus
print(ikarus.__version__)

from ikarus import default_library
print("materials:", default_library.available())
# -> ['Air', 'Au', 'GaN', 'GaP', 'Si', 'Si3N4', 'SiO2', 'TiO2', 'aSi']
```

And if you installed the `dev` extra, run the physics validation:

```bash
pytest ikarus/tests -q
```

If the Fresnel tests pass, your install reproduces analytic electromagnetics
to machine precision. Good wings.

## GPU support { #gpu-support }

Short version: **there isn't one, and for typical metaatom work you won't miss
it.** Ikarus is pure NumPy/SciPy (LAPACK/BLAS). The dominant cost is a dense
eigendecomposition whose size is set by the harmonic count, and two CPU-side
tricks matter more than CUDA in practice:

1. **Tame BLAS threading.** For the small matrices in a tight sweep or
   optimization loop, *single-threaded* BLAS is often dramatically faster than
   letting it oversubscribe your cores. Set this **before importing NumPy**:

    ```python
    import os
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
              "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ.setdefault(v, "1")
    import numpy as np  # noqa: E402
    ```

2. **Parallelize across processes.** Wavelength sweeps and GA populations are
   embarrassingly parallel — see
   [Aerobatics → Batch simulations](advanced.md#batch-simulations).

The full story lives in [Need for Speed](performance.md).

## Troubleshooting installation

??? failure "`ImportError: inverse.optimize needs pymoo`"
    The inverse-design module is an optional extra:
    `pip install "ikarus-rcwa[inverse]"` (or `pip install pymoo`).

??? failure "`ImportError: HDF5 I/O requires the 'h5py' package`"
    `pip install "ikarus-rcwa[io]"` (or `pip install h5py`).

??? failure "Plot calls raise `ModuleNotFoundError: matplotlib`"
    `pip install "ikarus-rcwa[viz]"`.

??? failure "`pip install ikarus` installed something weird"
    That's an unrelated project squatting the short name. The distribution is
    **`ikarus-rcwa`**; the import is still `import ikarus`.

??? failure "Everything is mysteriously slow on a big machine"
    Almost always BLAS thread oversubscription — pin the thread count to 1 as
    shown [above](#gpu-support). For very large single solves the opposite can
    hold; [benchmark both](performance.md#blas-threading).

---

Wings on? [Take your first flight →](quickstart.md)
