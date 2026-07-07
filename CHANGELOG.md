# Changelog

All notable changes to Ikarus are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/), and the project aims to follow
semantic versioning.

## 0.10.0

### Added
- **Adjoint (gradient-based) inverse design.** `optimize(atom, target)` now
  automatically uses reverse-mode differentiation through a new differentiable
  JAX solver when the problem suits gradients — freeform `pixels(...)` maps and
  free heights/periods with a single (possibly multi-wavelength / worst-case)
  target. The gradient with respect to *every* pixel costs about one extra
  forward solve, so freeform topology scales to thousands of DOFs where a GA
  cannot follow. Pixel maps use relax-and-project (continuous densities, conic
  **minimum-feature filter** via `min_feature=<meters>`, sharpness-ramped
  binarization); the final design is hard-thresholded and re-evaluated with the
  standard NumPy solver, so the reported objective is exactly what
  `result.rcwa.simulate()` reproduces. The GA / NSGA-III path is unchanged and
  still chosen for parametric shapes, discrete/anisotropic materials and full
  Pareto fronts; `algorithm="adjoint"`/`"ga"` force an engine. **The user
  experience is unchanged** — same `MetaAtom`/`Target` in, same
  `OptimizeResult` out.
- **`ikarus.grad`** — the differentiable solver itself (optional `[grad]`
  extra: JAX + optax): a line-for-line mirror of the forward solve (all
  factorizations including normal-vector, multi-layer, oblique incidence,
  complex amplitudes for phase objectives) with a custom-VJP non-Hermitian
  eigendecomposition (Boeddeker 2019 + scale-aware Lorentzian broadening).
  Pinned to the NumPy core at ~1e-13; gradients validated against finite
  differences and against FMMax's independent autodiff (relative agreement
  ~4e-7 over a 2304-pixel gradient). Supports `jax.jit`; normal-vector tangent
  fields are precomputed per iteration (`ikarus.grad.tangent_fields_for`).

### Notes
- Anisotropic (tensor) layers and `Structure` stacks stay on the GA path for
  now; parametric-`Shape` DOFs are inherently non-differentiable (binary
  rasterization) and always use the GA.

## 0.9.1

### Changed
- **`plot_field` cross-sections are now drawn the way the physics happens:** for
  `xz`/`yz` maps the stack is vertical — cover on top, substrate at the bottom,
  light entering from the top (`z` increases downward). Previously `z` ran
  horizontally, which read as light arriving from the left. `xy` slices are
  unchanged.
- `plot_field` axes now auto-scale to nm/µm (no more `1e-7` offset notation on
  nanophotonic length scales).

### Docs
- Reworked home page (centered layout, a structure→spectrum→field hero) and the
  Physics chapter (pipeline schematic, the factorization convergence race, the
  normal-vector tangent field). The convergence example figure now tracks
  coefficient stability (`|R − R_ref|` and phase), not the energy defect that the
  docs themselves warn is not a convergence test. Corrected lingering
  `factorization="li"`-is-default references (the default has been `"auto"` since
  0.8.0) across the AI guide, performance page and citations.

## 0.9.0

### Added
- **Anisotropic (birefringent) materials.** Anywhere Ikarus accepts a material you
  may now pass a `(n_x, n_y, n_z)` tuple (diagonal tensor; each component is any
  scalar spec, so dispersive anisotropy works), an `AnisotropicMaterial(n_x, n_y,
  n_z, angle=deg)` (in-plane principal-axis rotation → `eps_xy` off-diagonals), or
  the `uniaxial(n_o, n_e, axis="z"|"x"|"y"|angle_deg)` shorthand — in **uniform and
  patterned layers**, at every factorization (`auto` uses the rotated normal-vector
  construction of Liu & Fan 2012 eq. 45 for anisotropic patterned layers).
  Wave plates, c-plates, and patterned birefringence just work.
- Validation: uniform anisotropic layers (including rotated plates with
  off-diagonal tensor components) match analytic Fresnel results and FMMax
  *exactly*; patterned birefringent pillars agree with FMMax's anisotropic
  eigensolve to ~2×10⁻⁵ (Laurent-vs-FFT, formulation-identical) and to a few
  10⁻⁴ at convergence for the normal-vector formulations.

### Scope / limitations
- The permittivity tensor is `[[exx, exy, 0], [eyx, eyy, 0], [0, 0, ezz]]`:
  any in-plane optic-axis orientation plus a distinct z response. Tilted optic
  axes (`exz`/`eyz`) and magneto-optic gyrotropy (`exy != eyx`) are not
  supported. The cover and substrate must remain isotropic (clear `ValueError`).
- `(n, n, n)` is detected and reduces *exactly* to the scalar (isotropic) path.

### CI
- New `tests.yml` workflow runs the full pytest suite on every push/PR
  (Python 3.10 + 3.12), and publishing to PyPI now requires a green suite.

## 0.8.0

### Added
- **Normal-vector factorization (Fast Fourier Factorization), on by default.** A new
  `factorization="auto"` (the new default) applies the inverse rule along the *true
  local boundary normal* of every patterned layer, instead of the fixed x/y axes.
  This restores correct, fast convergence on **curved and oblique** high-contrast
  boundaries (cylinders, rings, ellipses, rotated shapes) — geometries where the
  separable inverse rule is not just slow but *biased* (e.g. a high-index ring sat
  ~2 % off the true reflectance no matter how high `n_orders` went). Validated
  against FMMax's `Formulation.NORMAL` to ≤2×10⁻³ across the geometry zoo. It is
  fully automatic — **users never choose a factorization** — and reduces *exactly*
  to `"li"` on axis-aligned geometry.
- `factorization="normal"` exposes the method explicitly (what `"auto"` resolves to
  for patterned layers); `"li"` and `"laurent"` remain for benchmarking.

### Changed
- **Default `factorization` `"li"` → `"auto"`** (the normal-vector method). On
  axis-aligned/1-D structures results are unchanged (it reduces exactly to `"li"`);
  on curved/oblique structures they are now correct rather than biased.

### Notes
- The normal-vector tensor formulation is not strictly energy-conserving at *finite*
  `n_orders`: `energy_balance` can deviate slightly from 1 on curved/oblique layers
  and shrinks toward 0 as it converges. This is expected, and a more honest
  convergence signal than the separable rule — which can report `R+T==1` exactly
  while still biased on a curved boundary. As always, converge `R`/phase, not energy.

## 0.7.0

### Added
- **Phase/R-aware convergence checking.** `auto_converge` now decides convergence
  from the **complex zeroth-order R/T coefficients (magnitude *and* phase)** rather
  than the specular transmittance alone — so phase-sensitive designs (metalenses,
  metamirrors) actually converge.
- **`simulate(check_convergence=True)`** — a single-solve safety net: re-solves once
  at a higher `n_orders` and warns if the zeroth-order R/T are still moving. Off by
  default (skip it inside tight sweep/optimization loops).
- `convergence_curve(..., metric="R_phase"|"T_phase")` for plotting phase vs. `n_orders`.

### Fixed
- `auto_converge` no longer fails to converge **absorbing** structures. The old
  criterion required `|R+T−1| < tol`, which is never true when `R+T<1` legitimately
  (absorptance); convergence is now judged on coefficient stability instead.

### Changed
- `RCWA(convergence_tol=...)` default `1e-6 → 1e-4` (an absolute tolerance on the
  complex coefficient now; `1e-4` ≈ 0.006° of phase — `1e-6` was needlessly tight).

### Docs
- Reinforced throughout: **energy balance (`R+T≈1`) is not a convergence test** —
  converge the coefficients/phase, not the energy.

## 0.6.0

### Added
- **Li's inverse-rule Fourier factorization**, in its two-step (separable) form
  for crossed gratings. Patterned layers now factorize the discontinuous E-field
  with \(⟦1/\varepsilon⟧^{-1}\) along the boundary normal, restoring fast
  convergence for **TM / high-index-contrast** structures: cases that previously
  drifted (or never settled) now converge at `n_orders ≈ 10–15`.
  - Works **automatically** for any topology (pixel maps, parametric shapes,
    height-maps, any `.img` object) and any number of materials — it acts on the
    rendered `ε(x, y)` grid, so there is nothing to write per geometry.
  - New `factorization=` parameter on `RCWA(...)` (and `solve_stack`):
    `"li"` (default) or `"laurent"` (the classic direct rule, kept for comparison).

### Changed
- **`factorization="li"` is now the default.** Existing patterned-layer
  simulations converge to the **same** limit, but values at a fixed *low*
  `n_orders` may differ from 0.5.x (they are now closer to the truth). Uniform /
  thin-film stacks and TE responses are unaffected. To reproduce old numbers
  exactly, pass `factorization="laurent"`.
- Per-solve cost at a given `n_orders` is unchanged (the eigensolve dominates;
  the operator build is vectorized). For a given *accuracy* solves are typically
  far cheaper, since fewer harmonics are needed.

### Docs
- New [RCWA → Factorization](https://cavitytechnologies.github.io/ikarus/api/rcwa/#factorization)
  reference; theory, performance, FAQ and the gratings lesson updated. Emphasis
  added throughout: **`R + T ≈ 1` does not prove convergence** — always check that
  `R`/phase have stopped moving with `n_orders`.

## 0.5.0
- Shipped `ikarus.ai_guide()` and a bundled condensed reference for AI assistants.

## 0.4.0
- `ikarus.inverse.Structure` for multi-layer / shared-parameter inverse design;
  flight-school tutorial overhaul.

## Earlier
- 0.2.0 documentation site; 0.1.0 initial RCWA core (eigenmodes, S-matrix cascade,
  materials, fields, HDF5 I/O, parametric shapes, sweeps, gradient-free inverse
  design).
