# Gradient-based (adjoint) inverse design for Ikarus — scoping brief

> Handoff note for the Ikarus dev chat. Summarizes a scoping discussion about
> whether to add gradient-based / adjoint optimization alongside the current
> gradient-free GA / NSGA-III inverse design. Written 2026-07-07.

## Motivation

Adjoint / gradient-based topology optimization has produced the striking
freeform-photonics results we keep seeing in the literature, and it has been on
the wishlist for years. This started as a question about CuPy for GPU speed, but
the more valuable prize turned out to be **differentiable RCWA → adjoint
optimization**, not raw GPU throughput.

**Goal, stated plainly:** implement gradient-based optimization and *demonstrate*
that it is faster/better than the current GA setup on a representative design,
while still supporting **many objectives** the way the GA path does today.
If we can't show it's actually better, it's not worth shipping.

## Where the current inverse design stands

- Backend is **pure NumPy + SciPy, CPU only**. No GPU, no autodiff.
- Optimization is **gradient-free**: GA (single objective) / NSGA-III
  (multi-objective Pareto) via pymoo.
- The core solve is a **general non-Hermitian eigendecomposition**,
  `scipy.linalg.eig(P @ Q)` in `ikarus/core/solver.py` (~line 243). Linear
  algebra is already funneled through a thin seam (`_inv`, `_rdiv`, `_solve`,
  the single `_sla.eig` call) — a clean insertion point for a backend swap.

## Why CuPy alone is the wrong tool (context from the earlier discussion)

- Not a drop-in `numpy → cupy` swap: CuPy has **no general `eig`** (cuSOLVER
  lacks it), and RCWA is built around exactly that operation. We also depend on
  `scipy.linalg`, not just NumPy.
- GPUs help *one big solve* (large `n_orders`), but hurt the *many small solves*
  in GA/sweep loops (kernel-launch latency). Our current guidance even says to
  pin BLAS to 1 thread for those loops.
- Crucially, **CuPy gives GPU but not autodiff.** It would not unlock adjoint
  optimization — the thing we actually want. So the backend of interest is
  **JAX or PyTorch** (GPU *and* gradients), which is also what FMMax — the
  package we validated v0.9.0 against — uses.

## Can gradient-based do what GA does today? Capability-by-capability

| Capability | Gradient-based verdict |
|---|---|
| **Continuous DOFs** (height, period) | ✅ Strictly better — its home turf, far fewer forward solves. |
| **Binary pixel maps** (freeform topology) | ✅ **State of the art.** This is the win. Adjoint scales to 10⁴–10⁶ pixels; GA chokes at a few hundred binary DOFs. |
| **Many objectives** | ✅ As a loss (weighted sum, or min-max / worst-case). ⚠️ But one run → one point, **not** a free Pareto front. |
| **Discrete material from a set** | ⚠️ The genuinely awkward case. OK for 2–3 materials via relaxation; messy for large catalogs. GA keeps the edge here. |

### Binary pixel maps — the mechanism (relax → project)

1. Each pixel is a continuous density ρ ∈ [0,1] instead of {0,1}.
2. Interpolate permittivity: ε(ρ) = ε_lo + ρ·(ε_hi − ε_lo).
3. Optimize continuously with gradients.
4. Push ρ → binary with a **tanh/Heaviside projection** whose sharpness β ramps
   over iterations, plus a spatial filter enforcing a **minimum feature size**.

The engine is the **adjoint method**: the gradient w.r.t. *all* pixels costs
~one extra forward solve, independent of DOF count. That is why it scales where
GA cannot — and the min-feature-size filter is exactly the kind of fab
constraint the metamirror/PreFab feasibility work cares about. Only real caveat:
the final hard-threshold to binary can nudge performance, managed by the
β-continuation schedule.

### Many objectives — the one real difference

- Prefer a **min-max / worst-case** formulation ("hit R at 1064 **and** 1550").
  This maps directly onto the existing `Target(..., worst_case=True)` idea.
- A single gradient run gives **one Pareto point**. Tracing the whole front
  costs multiple runs (swept weights / ε-constraint), whereas NSGA-III returns
  the full front in one shot. This is the capability we'd be trading away — so
  **keep the GA/NSGA-III path** for when the full tradeoff surface is the goal.

### Discrete materials — deprioritized

Confirmed as "nice to have, not important." In our real studies (metamirror,
bispectral, LER, crossed-cylinder) materials are effectively fixed
(aSi/SiO₂/cSi) and the DOFs are geometry + topology. So the discrete-material
weakness is largely hypothetical for us; leave it to the GA path.

## The two catches to go in with eyes open

1. **Requires a differentiable backend.** All of this needs gradients through the
   solver — a JAX/Torch re-implementation of the core, **including
   differentiating through the non-Hermitian `eig`**. Eigendecomposition
   gradients are well-defined but numerically delicate near mode degeneracies (a
   known differentiable-RCWA subtlety; FMMax handles it). The GA path needs none
   of this and works on today's NumPy code.
2. **Local vs. global.** Gradients find a *local* optimum. In high-DOF topology
   opt that's usually fine (multi-start from a few seeds); for low-dimensional
   rugged landscapes GA can be more robust.

## Recommended shape of the work (not GA *or* gradients — both)

- **Add** a differentiable backend + adjoint optimizer; **keep** GA/NSGA-III.
  They compose: gradient for continuous + high-DOF topology; GA for discrete
  material choice and full Pareto-front exploration; optionally **GA to seed,
  gradient to refine**.
- Preserve the existing `MetaAtom` / `Structure` / `Target` API surface so the
  new optimizer is a backend/algorithm choice, not a new mental model.

## Success criteria (the bar for shipping)

1. On a representative design (e.g. a freeform metamirror pixel map), adjoint
   optimization reaches **equal-or-better FOM in less wall-clock time** than the
   current GA.
2. It handles **freeform topology at a DOF count GA cannot reach** (10³–10⁴+
   pixels) with a min-feature-size fab constraint.
3. It still supports **multiple objectives** (min-max / worst-case) matching how
   the GA multi-objective path is used today.
4. Verified against a known reference (FMMax adjoint result, or a published
   topology-optimized device).

## Open questions for the dev chat

- **JAX vs. PyTorch** for the differentiable core? (JAX is closest to FMMax and
  our validation reference; Torch has broader tooling.)
- Full rewrite of `core/solver.py` in the new framework, or a dual-backend
  abstraction behind the existing linalg seam?
- Differentiate through our own `eig`, or restructure to an eigensolve the
  framework already differentiates cleanly?
- Which existing study is the **benchmark case** for the "faster/better than GA"
  demonstration?

## Phase 0 result (2026-07-07): GO

The de-risk spike (`adjoint_phase0_spike.py` in this directory, run on the
validated 1-D high-contrast TM grating, `M=12`, 512 pixel DOFs) passed every
gate:

| Gate | Result |
|---|---|
| Forward: JAX mirror vs NumPy core | **5e-15 / 1e-14** (binary / gray density grids) |
| `d(R)/d(rho_i)` vs central finite differences | worst rel. err **1.8e-6** (at a near-zero-gradient pixel; others 1e-7–1e-9) |
| `d(R)/d(height)` vs finite differences | rel. err **2.2e-10** |
| Adjoint cost | full 512-DOF gradient = **1.87x** one forward solve |

The custom differentiable non-Hermitian `eig` (Boeddeker 2019 eq. 4.77 VJP with
scale-aware Lorentzian broadening, mirroring FMMax `eig.py`) works through the
full pipeline: the engineering-conjugation bridge, li mixed convolution
(batched `inv` + FFT), forward-branch selection and the Redheffer cascade all
differentiate cleanly under `jax.grad` + `jax.jit`.

## Phase A/B results (2026-07-07)

- **Phase A** (`ikarus/grad`, commit c4b36ed): differentiable solve pinned to
  the NumPy core at <=1e-12 across the geometry zoo; 2-D cylinder M=8 with
  9216 pixel DOFs: full gradient = **1.44x** one forward solve (jitted).
- **Phase B** (commit 26b2eee): `optimize(atom, target)` auto-dispatches to the
  adjoint engine with zero UX change; quarter-wave AR height converges to the
  analytic optimum exactly; relax-and-project pixel maps with min-feature
  filter; final designs re-verified by the NumPy core.
- **Gradient oracle cross-check (success criterion 4):** on a 48x48 gray
  density map (2304 pixels, laurent/FFT, identical parallelogramic harmonic
  set), ikarus.grad vs FMMax's independent autodiff: forward R agrees to
  **2e-8**, the full 2304-pixel gradient to **rel-L2 3.9e-7**.
- Phase C benchmark results (B1-B3) are recorded below the run script
  `adjoint_benchmarks.py`.

**Open questions answered:** JAX backend (FMMax is the gradient oracle; optax
ecosystem; functional style matches the stateless core). A **separate optional
differentiable core** (`ikarus-rcwa[grad]` extra) rather than a dual-backend
abstraction — the NumPy core stays canonical and default. Custom eig VJP rather
than framework-default `eig` differentiation. Benchmarks: freeform 1550 nm
aSi/SiO2 reflector pixel map first, then the bispectral 1064/1550 min-max as
the many-objectives demo. For the `auto` factorization under topology
optimization, plan is `stop_gradient` on the tangent field (second-order effect
on the FOM), falling back to `li` if noisy. Hardware: development/validation on
macOS (CPU JAX); large design runs later on a Windows machine via WSL2 GPU JAX.


## Phase C results (2026-07-07): benchmarks, findings, ship-bar verdict

All numbers are **verified**: hard-binarized designs re-evaluated with the
NumPy core at a higher truncation than the optimizer used.

| Benchmark | Adjoint | GA | Note |
|---|---|---|---|
| Toy 1-D deflector, 40 DOFs (docs figure) | 0.23, ~1 s (best of 2 starts) | **0.41**, ~2 s (780 solves) | GA legitimately wins small discrete spaces |
| Freeform deflector, 2,048 DOFs, M=8 opt / M=12 verify | **0.357**, 572 s (best of 3 starts) | 0.182, 1161 s (2x the budget) | gradients win at scale: 2x the FOM in half the time |
| Bispectral min-max (1064+1550), 210 DOFs + free height | **worst-case R = 0.61** (0.623/0.611), 96 s; M=12 re-check identical | (adjoint-only demo) | the many-objectives mode of this brief, working |
| Gradient oracle (2,304-pixel gradient vs FMMax autodiff) | rel-L2 **3.9e-7** | -- | success criterion 4 |

**Ship-bar verdict:** criteria 2 (freeform DOF counts with a min-feature fab
filter), 3 (min-max multi-wavelength) and 4 (external gradient verification)
met cleanly. Criterion 1 ("equal-or-better FOM in less wall-clock") holds **at
freeform scale** -- and measurably *fails at toy scale*, where a GA can nearly
enumerate the space. That crossover is now the docs' honestly-told story
(Lesson 7 figure) and the reason both engines ship.

### Findings the benchmarks forced (all fixed/documented)

1. **Optimizers exploit unconverged forward models.** At M=6, the GA "won"
   with unphysical fitness (R = 1.015) that collapsed to R = 0.10 on converged
   re-evaluation -- the bispectral-study failure mode reproduced in a
   controlled experiment. Protocol everywhere now: optimize on a faithful
   model, verify higher. (The adjoint's min-feature filter incidentally
   shields it from artifact-mining.)
2. **Hard min-max starves gradients** -> smoothed maximum (logsumexp) in the
   adjoint Target loss. B3 went from stalled (0.02) to 0.61 worst-case.
3. **Deflection landscapes are flat at uniform gray** -> `init="random"`
   option + a-few-seeds practice. B2 went from 0.09 to 0.45 (@M8).
4. **Maximize-total-R has a trivial uniform-slab attractor** (R = 0.52 for
   aSi/SiO2 at 1550): a poor objective for gradient demos and a genuine local
   optimum in production use -- documented in the lesson's pilot habits.
5. **The scale crossover** (~10^2 DOFs): below, GA competitive-or-better;
   above, adjoint dominates. Auto-selection currently always prefers adjoint
   for pixel problems; a DOF-count threshold is a possible future refinement.
