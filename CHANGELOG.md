# Changelog

All notable changes to Ikarus are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/), and the project aims to follow
semantic versioning.

## Unreleased

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
