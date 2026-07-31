# Ikarus GPU device path — validation & benchmark on the A6000

**Purpose.** A new *opt-in* `device=` argument on `RCWA` routes the forward solve
through Ikarus's JAX solver, so it runs on a GPU. It was implemented and proven
correct **on a MacBook (CPU only)**. This note is for the person/agent on the
**NVIDIA RTX A6000** machine: confirm it runs on CUDA, confirm it's *correct*, and
benchmark CPU vs GPU. (The Ikarus Skill should be installed there — this doc is the
task-specific brief on top of it.)

---

## What was built (already done, on the MacBook)

- `RCWA(..., device="cpu")` — **the default — is the unchanged NumPy core.** Nothing
  about existing behavior moved.
- `device` ∈ `{"gpu"/"cuda", "metal", "auto", "cpu-jax"}` routes the forward solve
  through `ikarus.grad.solve` (the JAX mirror of the core), placed on that device.
  The API and the returned `SimulationResult` are **identical** — only the device
  changes. One keyword, nothing else.
- **Proven:** forcing the JAX path onto the CPU (`device="cpu-jax"`) reproduces the
  NumPy core to **~1e-13** — R/T totals, per-order efficiencies, complex phase, and
  exit angles — on a 1-D grating and an oblique 2-D cylinder, for all three
  factorization rules. See `ikarus/tests/test_device.py` (9 tests, all green).
- **Scope of the accelerated path:** forward efficiencies + phase + exit angles,
  **isotropic** materials, every factorization. *Not yet* on the accelerator:
  anisotropic (tensor) layers and `get_fields()` real-space reconstruction — both
  raise a clear error telling you to use `device="cpu"`.

### Why CPU parity de-risks CUDA
JAX runs the *same* computation graph regardless of device; only op placement
changes. So JAX-CPU == NumPy-core (shown) implies the CUDA run executes the identical
graph, with cuSolver's `eig` agreeing with LAPACK's to ~1e-10. The A6000 therefore
only needs to confirm it (a) *runs* on CUDA and (b) is *faster* — not re-derive
correctness from scratch.

### Apple Metal: tested, not viable (context, not a task)
On the MacBook (M4 Pro, `jax-metal` 0.6.2) the Metal GPU is detected, but **both
float64 and complex `eig` fail** with `UNIMPLEMENTED: default_memory_space is not
supported` — exactly the two ops RCWA needs. So `device="metal"` now **warns and
falls back to JAX-CPU** by design. The A6000/CUDA is the real acceleration target.

---

## Your job on the A6000

### 1. Install
```bash
git clone -b feature/gpu-device https://github.com/CAVITYtechnologies/ikarus
cd ikarus
pip install -e .
pip install -U "jax[cuda12]"      # CUDA-enabled jaxlib; match your CUDA toolkit
python -c "import jax; print(jax.devices())"     # must list a CudaDevice
```
If `jax.devices()` shows a `CudaDevice`, JAX (and therefore `device="cuda"`) will use
it. If it shows only CPU, the CUDA jax install didn't take — fix that first.

### 2. Confirm correctness on CUDA
```bash
python - <<'PY'
import numpy as np
from ikarus import RCWA, shapes
def cyl(dev, M=12):
    m = shapes.circle(center=(.5,.5), radius=.30, grid_shape=(96,96))
    rc = RCWA(period_x=400e-9, period_y=400e-9, resolution=(96,96),
              n_orders=(M,M), device=dev)
    rc.add_uniform_layer(np.inf,"Air"); rc.add_layer(200e-9, m.astype(int), [1.0,3.5])
    rc.add_uniform_layer(np.inf,"Air")
    rc.set_source(wavelength=700e-9, theta=20, polarization="linear")
    return rc.simulate()[2]
c, g = cyl("cpu"), cyl("cuda")
d = abs(c.R_total - g.R_total)
print(f"R_total cpu={c.R_total:.10f}  cuda={g.R_total:.10f}  |diff|={d:.2e}")
assert d < 1e-8, "CUDA PARITY FAILED — report this"
print("CUDA parity OK")
PY
```
A `|diff| < 1e-8` here is the green light: the GPU path is numerically correct.

### 3. Benchmark CPU vs GPU
```bash
python scripts/benchmark_device.py cuda
```
It sweeps `n_orders` on a 2-D pillar metasurface and prints, per size, the CPU time,
the CUDA time, the speedup, and the R-agreement. Expected shape of the result: **CPU
wins at small M** (kernel-launch + transfer + JAX tracing overhead dominate), **GPU
wins at large M** where the `2*(2M+1)²` complex eig dominates. Read off:
- the **crossover M** (where GPU first beats CPU), and
- the **speedup at M = 26** (the large end).

---

## What to report back

1. Does `jax.devices()` show the A6000? (paste it)
2. Does the CUDA parity check pass (`|diff| < 1e-8`)? Any error — especially around
   complex `eig` on CUDA — paste it verbatim.
3. The benchmark table: the crossover M and the speedup at the large end.

## Known caveats / future work
- The adapter currently calls the JAX solver **un-jitted**, so every `simulate()`
  includes JAX *tracing* overhead. That penalizes small solves; the big-M numbers are
  the honest GPU signal. Wrapping the forward solve in `jax.jit` (keyed on shapes) is
  the obvious next speedup once the win is confirmed.
- Accelerated path is **isotropic, forward-only** for now (anisotropy & `get_fields`
  → `device="cpu"`).
- If CUDA parity passes and the benchmark shows a real win, the follow-up is to merge
  `feature/gpu-device` → `main`, then decide the default device-selection ergonomics
  (`"auto"`, docs, the performance-envelope page).
