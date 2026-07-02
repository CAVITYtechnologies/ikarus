# Changelog

All notable changes to Ikarus are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/), and the project aims to follow
semantic versioning.

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
