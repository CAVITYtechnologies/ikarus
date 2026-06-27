<!--
  Ikarus — condensed reference for AI assistants / LLMs.
  This file SHIPS INSIDE the installed package and is the single source of truth.
  Read it from any chat with:  python -c "import ikarus; print(ikarus.ai_guide())"
  Keep it terse, accurate, and in sync with the code — it is the authority other
  sessions rely on. Full human docs: https://cavitytechnologies.github.io/ikarus/
-->

# Ikarus — Reference for AI Assistants

You are working with **Ikarus**, a 2-D **RCWA** (Rigorous Coupled-Wave Analysis /
Fourier Modal Method) solver for periodic photonics. This page is a dense,
authoritative cheat-sheet so you can use it like an expert without rediscovering
its API or its sharp edges. When in doubt, prefer these facts over guesses.

- **PyPI:** `ikarus-rcwa` · **import:** `import ikarus` (the names differ on purpose).
- **Docs:** <https://cavitytechnologies.github.io/ikarus/> · **GitHub:** `CAVITYtechnologies/ikarus`.
- **Extras:** `pip install "ikarus-rcwa[inverse]"` (pymoo, for `optimize`),
  `[io]` (h5py, for save/load), `[progress]` (tqdm), `[all]`.

## What it is for — and what it is not

Frequency-domain electromagnetics of structures **periodic in x and y** and
**layered along z**: gratings, metasurfaces, photonic-crystal slabs, thin-film
stacks, Bragg mirrors, metamirrors. You get per-order efficiencies, complex
amplitudes, phase, exit angles and real-space fields.

**Not** for: isolated (non-periodic) scatterers, continuously-varying-in-z shapes
that resist layer slicing, time-domain/pulse physics, or full anisotropy. Those
are not RCWA's domain.

## Conventions — the mistakes everyone makes

1. **`simulate()` returns `(T, R, result)` — transmission FIRST.** The single most
   common bug is unpacking it as `(R, T, ...)`.
2. **SI units, always.** Meters for length, degrees for angles. `1550e-9`, not `1550`.
3. **Time/sign convention is physics `exp(-iωt)`:** absorbers have `k > 0`,
   `Im(ε) > 0`. If you feed gain-signed (negative-k) data the energy balance
   exceeds 1 — that is the tell.
4. **`n_orders` ≠ `resolution`.** `n_orders` is the number of Fourier harmonics
   that resolve the *field* — the accuracy/cost dial. `resolution` only *draws*
   the geometry (real-space pixels) and is auto-raised to ≥ `4*M+1`. You almost
   always tune only `n_orders`.
5. **Stack order is cover → … → substrate** (sky to ground). The first and last
   layers must be **uniform and semi-infinite** (`height=np.inf`); the cover's
   index sets the incident wavevector. Interior layers are finite.
6. **Cost is `O(M⁶)` in 2-D** (`M` = orders per axis). Doubling `M` ≈ 64× the time.
   A 1-D grating should use `n_orders=(M, 0)` with an `(Nx, 2)` topology — linear,
   not quadratic. A uniform thin-film stack runs at `n_orders=0` (instant).
7. **Energy balance is the universal smoke test.** `R_total + T_total` (`result.energy_balance`):
   `≈1` lossless & converged; `<1` absorption (physics); slightly `>1`
   unconverged → raise `n_orders`; wildly `>1` numerical breakdown → *lower*
   `n_orders` or raise `resolution`. Note the default factorization (`"auto"`, the
   normal-vector method) is not strictly energy-conserving at *finite* order — a
   small `>1` imbalance on curved/oblique structures is normal and shrinks to 0 as
   `n_orders` grows; it is a *more* honest convergence signal than the classic
   inverse rule, which can report `R_total+T_total==1` while still being a percent
   off the true answer on curved boundaries.
8. **Factorization is automatic — don't set it.** The default `factorization="auto"`
   applies the normal-vector (Fast Fourier Factorization) method to every patterned
   layer: full accuracy on curved/oblique high-contrast boundaries, reducing exactly
   to the classic inverse rule on axis-aligned geometry. Users never need to choose.
   `"li"`, `"laurent"`, `"normal"` remain as explicit overrides for benchmarking.

## Minimal forward simulation

```python
import numpy as np
from ikarus import RCWA, shapes

rcwa = RCWA(period_x=500e-9, period_y=500e-9, resolution=(96, 96), n_orders=(8, 8))
rcwa.add_uniform_layer(np.inf, "Air")                                   # cover
rcwa.add_layer(220e-9, shapes.circle(radius=0.3, grid_shape=(96, 96)),
               ["Air", "Si"])                                           # patterned
rcwa.add_uniform_layer(np.inf, "SiO2")                                  # substrate

rcwa.set_source(wavelength=600e-9, theta=0, polarization="linear")
T, R, result = rcwa.simulate()                                         # T FIRST
print(result.R_total, result.T_total, result.energy_balance)
print(result.R_phase)                                                  # zero-order phase (rad)
```

## The API surface

**`RCWA(period_x, period_y, resolution=32, n_orders=25, materials=None, convergence_tol=1e-6)`**
- `.add_uniform_layer(height, material, name="")` — `height=np.inf` for cover/substrate.
- `.add_layer(height, topology, materials, resolution=None, name="")` — `topology`
  is an integer array, a parametric `Shape`, or any object exposing an `.img` array.
  `materials[i]` dresses topology index `i`.
- `.set_source(**kwargs)` — **remembers** unspecified fields between calls (so a
  sweep mutates one parameter at a time). First call needs `wavelength`.
- `.simulate(auto_converge="never"|"once"|"always") -> (T, R, result)`.
- `.get_fields(plane="xy"|"xz"|"yz", ...)` → field maps; `.visualize_structure(plane="xz", layer_index=None, savefig=...)`.
- `.save_results(path, include=("T","R","metadata"[, "fields"]), result=None)` and
  `RCWA.load_results(path)` (needs `[io]`).

**`Source`** (via `set_source`): `wavelength`, `theta=0` (from +z, normal),
`phi=0` (azimuth from +x), `polarization="linear"|"RCP"|"LCP"`,
`linear_pol_angle=0` (degrees from TE: `0`=TE/s, `90`=TM/p).

**`SimulationResult`** (the 3rd return value):
- `T`, `R` — **zero-order complex amplitudes** (a scalar for linear pol, or a
  `{"co","cross"}` dict for circular). `T_total`, `R_total` — summed over all
  propagating orders.
- `T_orders`, `R_orders` — per-order efficiencies; `orders` = `(p, q)` integer
  arrays; `order_index(p, q)` → the array index of one order.
- `T_phase`, `R_phase` — zero-order phase in radians (`np.angle` of the coeff).
- `theta_out_ref/trn`, `phi_out_ref/trn` — exit angles in degrees (`NaN` = evanescent).
- `energy_balance` — `R_total + T_total`. `solution` — raw modal solution for fields.

**Materials:** a shared `default_library` ships **Air, Au, GaN, GaP, Si, Si₃N₄,
SiO₂, TiO₂, aSi**. Anywhere a material is wanted you may pass a name (`"Si"`), a
bare number (`1.5`, `2.4+0.01j` = constant index), a JSON path, or a `Material`.
`default_library.get("Si", 1.55e-6)` returns the complex index;
`default_library.available()` lists names.

**Shapes** (`from ikarus import shapes`): functions in fractional `[0,1)` unit-cell
coords — `circle, rectangle, ring, ellipse, cross, polygon`, all taking
`grid_shape=(Nx,Ny)`; `combine`, `rotate`. Parametric classes (`ikarus.shapes`):
`Circle, Ellipse, Rectangle, Ring, Cross, SplitRing` — any parameter may be
`free(lo,hi)`, plus a rotation `angle`; `.to_grid(shape)` rasterizes.

## Sweeps and progress bars

```python
from ikarus import Sweep, progress

sw = Sweep(rcwa).over(wavelength=np.linspace(1.4e-6, 1.7e-6, 31)).run(progress=True)
sw.R_total           # ndarray shaped like the axes; also T_total, energy_balance
sw.order(0, 0, "R")  # per-order efficiency across the grid; sw.axes, sw.results
```

`Sweep.over(**axes)` sweeps **source** parameters only (`wavelength`/`theta`/
`polarization`/…); geometry changes rebuild, so keep those in a manual loop wrapped
in `progress(iterable, enable=True, desc=...)`. **Reuse one `RCWA`** across a sweep —
only `set_source` changes; structural edits force a fresh eigensolve.

## Inverse design (`from ikarus.inverse import …`)

Gradient-free GA (single objective) / NSGA-III (multi-objective Pareto), via pymoo
(`[inverse]` extra). Anything exposing `variables()` + `build(params, n_orders)` is
optimizable — `MetaAtom`, `Structure`, or your own class.

**`MetaAtom(period, cover, substrate, polarization="linear", pol_angle=0.0)`**
then `.add_pattern(topology, materials, height)`:
- `period`, `height` — a float (fixed) or `free(lo, hi)` (a DOF).
- `topology` — a fixed integer array, `pixels(nx, ny, symmetry=...)` (freeform
  binary map), or a parametric `Shape` (its `free(...)` params become DOFs).
- `pixels` symmetry: `None`, `'mirror_x'`, `'mirror_y'`, `'mirror_xy'`, `'c2'`,
  `'c4'`, `'c4v'` (`c4`/`c4v` need a square grid). Symmetry cuts the DOF count.

**`Target`** — the figure of merit. Metrics:
- `'R'` / `'T'` — efficiency into `order` (default specular `(0,0)`; `order=None` → total).
- `'r_co'` / `'t_co'` — the **complex** zero-order coefficient (for linear pol, the co-pol one).
- `'r_phase'` / `'t_phase'` — its phase in radians (matched modulo 2π).

Constructors (all take `at=`/`band=`, `order=(0,0)`, `weight=`, `worst_case=`):
- `Target.maximize(metric, at=1550e-9)` · `Target.minimize(metric, band=(300e-9,600e-9,6))`
- `Target.match(metric, value, at=...)` — drive `metric` to `value`.
- Wavelengths: `at=1550e-9`, `at=[1064e-9, 1550e-9]`, or `band=(lo, hi[, n])`;
  multiple are aggregated by the **mean**, or the **worst case** if `worst_case=True`.

**`optimize(atom, targets, n_orders=8, algorithm="auto", pop=100, n_gen=60, seed=0,
verbose=True, progress=False) -> OptimizeResult`**. One `Target` → GA; a list →
NSGA-III. Result: `.params` (best dict), `.metaatom`/`.rcwa` (a ready-to-simulate
`RCWA`), `.report()`, `.X`, `.F`, `.history`.

**`Structure`** — multi-layer / shared-parameter inverse design. Subclass it,
declare params as **class attributes** (`free(...)` = DOF, plain value = fixed;
a `period` attribute is **required**), and implement `define(self, p)` calling
`self.add_layer(height, topology, materials)` (`p` is the resolved-parameter
namespace). `cover`, `substrate`, `resolution`, `polarization`, `pol_angle` are
configuration, not DOFs. It plugs into `optimize()` unchanged.

```python
from ikarus.inverse import Structure, free
from ikarus.shapes import Circle

class ARStack(Structure):
    cover, substrate, resolution = "Air", "SiO2", 64
    period = free(0.20e-6, 0.40e-6)
    h1 = free(0.05e-6, 0.30e-6)
    def define(self, p):
        self.add_layer(p.h1, Circle(radius=0.3), ["Air", "Si3N4"])
```

**Which construct?** Forward sweep already answers it → use `Sweep`, skip the
optimizer. One patterned layer with a few meaningful knobs → `MetaAtom` + a
parametric `Shape`. Freeform topology, no shape prior → `MetaAtom` + `pixels`.
Multiple layers or parameters shared/derived across layers → `Structure`.

**Idiom — phase-controlled high-reflectivity mirror.** To maximize R *and* hit a
target reflected phase with one objective, match the complex coefficient to a
unit-modulus target: `Target.match("r_co", value=np.exp(1j*phi), at=lam)` drives
`|r| → 1` (R → 100%) and `arg(r) → phi` simultaneously — no R-vs-phase weighting
to tune.

## Performance and accuracy

- **Pin BLAS to one thread for GA/sweep loops** (many small solves): set
  `OMP_NUM_THREADS`/`OPENBLAS_NUM_THREADS`/`MKL_NUM_THREADS`/`VECLIB_MAXIMUM_THREADS`
  to `"1"` **before** importing numpy — often ~10× faster on many-core machines.
  For one big solve (`M ≳ 20`) do the opposite: let BLAS thread.
- **Converge `n_orders` at the worst-case wavelength/polarization.** Prefer
  `simulate(auto_converge="once")` — it raises `n_orders` until the **complex
  zero-order R/T (magnitude and phase)** stop changing, and caches the result.
  For a one-off solve, `simulate(check_convergence=True)` re-solves once higher and
  **warns** if it's still moving. **Do not trust `R+T≈1` as convergence** — a
  lossless structure conserves energy at every `n_orders` while R/phase still
  drift (the classic high-contrast-TM trap that has cost real optimization runs).
- **Fourier factorization:** the default `factorization="li"` applies Li's
  inverse rule (two-step for crossed gratings), giving fast TM / high-contrast
  convergence — high-contrast TM gratings converge by `n_orders≈10–15` instead of
  drifting. It works automatically for **any** topology and any number of
  materials (it acts on the rendered `ε(x,y)` grid). Pass
  `factorization="laurent"` to force the old direct rule for comparison. Caveat:
  **R+T≈1 does not prove convergence** for high-contrast TM — always check that
  R/phase have stopped moving with `n_orders`.

## Known gaps (do not promise these)

- **Anisotropic (3×3 tensor) materials** — isotropic only.
- **Smooth-boundary (off-diagonal) normal-vector factorization** — the Li rule is
  the two-step *diagonal* form (exact for axis-aligned/pixelated masks); the full
  normal-vector method for sub-pixel-accurate *curved* boundaries is not yet in.
- **No GPU** (pure NumPy/SciPy CPU) and **no analytic/AD gradients** (inverse design is gradient-free).

## Read more

Tutorials ("Flight School"), `Core Concepts`, `FAQ`, `Need for Speed`
(performance) and the full per-object API all live at
<https://cavitytechnologies.github.io/ikarus/>.
