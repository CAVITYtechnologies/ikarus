# FAQ

### 1. What problems is Ikarus for?

Frequency-domain electromagnetics of structures that are **periodic in two
transverse directions** and **layered along the propagation axis**: gratings,
metasurfaces, photonic-crystal slabs, thin-film stacks, distributed Bragg
reflectors. It returns diffraction efficiencies, complex coefficients, phase, exit
angles and real-space fields.

### 2. What is *not* a good fit?

Isolated/finite scatterers (no periodicity), strongly 3-D topographies that vary
continuously along \(z\) without a sensible layer slicing, time-domain/pulse
problems, and anything needing full anisotropy — none of those are RCWA's domain
(see [Theory → Limitations](theory.md#limitations-of-rcwa)).

### 3. What units does Ikarus use?

SI throughout: **meters** for all lengths (periods, heights, wavelengths,
coordinates) and **degrees** for angles.

### 4. What is the time/sign convention?

The physics convention \(\exp(-i\omega t)\). Absorbing media have \(k>0\) and
\(\mathrm{Im}(\varepsilon) > 0\). If you import data with the opposite sign of
\(k\), the energy balance will exceed 1.

### 5. How do I choose `n_orders`?

Start at 8–12 for dielectric metasurfaces, more for metals/TM, and run a
[convergence study](tutorials/parameter-sweeps.md#convergence-study) — or let
`simulate(auto_converge="once")` pick it. `n_orders = M` keeps harmonics
\(-M..+M\) per axis. Cost scales like \(M^6\) in 2-D, so do not over-resolve.

### 6. What is the difference between `resolution` and `n_orders`?

`resolution` is the real-space grid used to build the Fourier matrices; `n_orders`
is the number of retained harmonics (the accuracy/cost knob). `resolution` only
needs to represent the geometry and is auto-raised to ≥ `4*n_orders+1`.

### 7. Why is `R + T` not exactly 1?

Three cases: (a) the structure **absorbs** (a material with \(k>0\)) — then `R+T<1`
and the deficit is absorptance; (b) the result is **not converged** — `R+T`
slightly exceeds 1, so raise `n_orders`; (c) **numerical breakdown** at very high
`n_orders` — `R+T` blows up, so reduce `n_orders` or raise `resolution`. Ikarus
warns automatically for (b) and (c).

### 8. How do I get the specular order only?

`result.T_orders[result.order_index(0, 0)]` (and `R_orders`). `T_total`/`R_total`
sum over *all* propagating orders; for a thin film they coincide because only the
specular order exists.

### 9. How do I set up a 1-D grating?

Use an `(Nx, 2)` topology and `n_orders=(M, 0)` so only x-orders are expanded — see
[Gratings](tutorials/gratings.md). This is far cheaper than a crossed (2-D)
grating.

### 10. What does `linear_pol_angle` mean exactly?

The angle (degrees) from TE: `0` = TE/s, `90` = TM/p. At normal incidence TE is
along +y and TM along +x, so it is the physical E-field angle in the xy-plane.

### 11. How is circular polarization reported?

For `RCP`/`LCP`, `T` and `R` are dicts `{"co", "cross"}` — complex amplitudes of
the same/opposite handedness, normalized so \(|co|^2 + |cross|^2\) equals the
zero-order efficiency. Square the magnitude for power; use `np.angle` for phase.

### 12. How do I add a custom material?

Inline as a number (constant index), via
[`MaterialLibrary.add_from_file`](api/materials-layers.md#materiallibrary) from a
CSV (`λ_nm n [k]`), via a Lorentz-model JSON, or with the
[`ikarus-add-material`](api/tools.md#material-import-cli) CLI for a permanent entry.

### 13. My material is only tabulated over part of my band — what happens?

Tabulated data is cubic-spline interpolated and **extrapolated by clamping** to the
nearest endpoint. Out-of-range requests return the boundary value (no divergence),
which may be inaccurate — check the material's `wavelength_nm` coverage.

### 14. Can Ikarus compute the phase of the transmitted/reflected field?

Yes: `result.T_phase` / `result.R_phase` give the zero-order phase (radians). Note
it is in the \(\exp(-i\omega t)\) convention, so a comparison against an
engineering-convention tool may differ by a sign and/or a constant offset (often
\(-\pi\)); compare the *dispersion*, not the absolute value.

### 15. Why does my phase differ from another RCWA code by a constant?

Convention differences (time sign, reference plane). In a validated cross-check
against grcwa, the Ikarus phase matched after removing a constant \(\approx -\pi\)
offset; the *shape* of the dispersion agreed to ~20 mrad.

### 16. Does Ikarus run on the GPU?

No — it is CPU-only (NumPy/SciPy). For speed, pin BLAS to one thread and
parallelize sweeps across processes (see [Performance](performance.md)).

### 17. Does Ikarus provide gradients for optimization?

No analytic/automatic-differentiation gradients. The built-in
[inverse design](api/inverse.md) is **gradient-free** (genetic algorithms); for
gradient-based topology optimization you would need a differentiable RCWA.

### 18. Does it support anisotropic materials?

Not yet — isotropic permittivity only. Full \(3\times 3\) tensors are a roadmap
item.

### 19. How do I reconstruct and plot the near field?

`rcwa.get_fields(plane="xz"|"yz"|"xy", ...)` returns
[`FieldMap`](api/fields-viz.md#fieldmap)s; plot with
`ikarus.visualization.plot_field`. The maps carry the structure outline for
overlays. See [Metasurfaces](tutorials/metasurfaces.md#near-field-maps).

### 20. How do I save and reload results?

`rcwa.save_results("run.h5", include=[...])` writes self-describing HDF5;
`RCWA.load_results("run.h5")` reads it back into a dict. Needs h5py
(`pip install "ikarus-rcwa[io]"`).

### 21. Can I sweep efficiently without rebuilding everything?

Yes — reuse one `RCWA` and call `set_source(...)`, which retains unspecified
fields. Only changing geometry forces a new eigensolve. See
[Parameter sweeps](tutorials/parameter-sweeps.md).

### 22. Why is the PyPI name `ikarus-rcwa` but the import `ikarus`?

The distribution name on PyPI is `ikarus-rcwa` (the name `ikarus` was taken); the
Python import name is `import ikarus`.

### 23. Which optional dependencies do I actually need?

None for the core solver. Add `matplotlib` (`[viz]`) for plotting, `h5py`
(`[io]`) for HDF5, `pymoo` (`[inverse]`) for inverse design, or `[all]` for
everything.

### 24. How do I cite Ikarus?

See [Citation](citation.md).
