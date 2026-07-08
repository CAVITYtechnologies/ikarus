# FAQ

*Two dozen questions, answered straight.* Click to expand.

## Scope & philosophy

??? question "1 · What problems is Ikarus actually for?"
    Frequency-domain electromagnetics of structures **periodic in two
    transverse directions** and **layered along z**: gratings, metasurfaces,
    photonic-crystal slabs, thin-film stacks, Bragg mirrors. You get
    per-order efficiencies, complex amplitudes, phase, exit angles and
    real-space fields.

??? question "2 · And what is *not* a good fit?"
    Isolated (non-periodic) scatterers, continuously-varying-in-z topographies
    that resist layer slicing, and time-domain/pulse physics.
    Not Ikarus's fault — not RCWA's domain
    ([Theory → Limitations](theory.md#limitations-of-rcwa)).

??? question "3 · What units does Ikarus use?"
    SI, uncompromisingly: **meters** for every length, **degrees** for every
    angle. `200e-9`, not `200`.

??? question "4 · What's the time/sign convention?"
    Physics \(\exp(-i\omega t)\): absorbers have \(k>0\),
    \(\mathrm{Im}(\varepsilon) > 0\). Feed it gain-signed data and the energy
    balance exceeds 1 — that's your smoke detector.

## Numerics

??? question "5 · How do I choose `n_orders`?"
    Start 8–12 for dielectric metasurfaces, more for metals/TM, then run the
    [convergence ritual](tutorials/parameter-sweeps.md#convergence-study) — or
    let `simulate(auto_converge="once")` decide. Cost is \(\mathcal{O}(M^6)\)
    in 2-D, so don't dial it up "to be safe". The default **normal-vector
    factorization** keeps even high-contrast TM and curved boundaries converging at
    modest `M` ([RCWA → Factorization](api/rcwa.md#factorization)).

??? question "6 · `resolution` vs. `n_orders` — what's the difference?"
    `resolution` *draws* the geometry (real-space pixels); `n_orders` *resolves*
    the field (Fourier harmonics — the accuracy dial). `resolution` is
    auto-raised to ≥ `4*n_orders+1`, so usually you only think about `n_orders`.

??? question "7 · Why isn't R + T exactly 1?"
    Three cases. **(a)** Absorbing material → `R+T < 1`, the rest is
    absorptance — physics. **(b)** Slightly above 1 → unconverged; raise
    `n_orders`. **(c)** Wildly above 1 → numerical breakdown; *lower*
    `n_orders` or raise `resolution`. Ikarus warns on (b) and (c)
    automatically.

    **But beware the converse:** `R+T ≈ 1` does **not** prove convergence. With the
    separable rules a lossless structure conserves energy at *every* `n_orders`, even
    while `R` and the phase are still drifting — exactly the trap high-contrast TM
    sets. The default [normal-vector factorization](api/rcwa.md#factorization)
    converges fast *and* makes `energy_balance` itself a more honest signal (it
    deviates slightly until converged on curved layers). Always confirm `R`/phase
    have stopped moving with `n_orders`.

??? question "8 · How do I get just the specular order?"
    `result.T_orders[result.order_index(0, 0)]` (same for `R_orders`).
    The totals sum over *all* propagating orders.

??? question "9 · How do I set up a 1-D grating properly?"
    `(Nx, 2)` topology + `n_orders=(M, 0)` —
    [Lesson 2](tutorials/gratings.md). 1-D physics at a 1-D price; a 2-D
    expansion of a 1-D grating is the classic accidental slowdown.

## Polarization & phase

??? question "10 · What exactly does `linear_pol_angle` mean?"
    Degrees from TE: `0` = TE/s, `90` = TM/p. At normal incidence it's the
    literal E-field angle in the xy-plane (0 → +y, 90 → +x).

??? question "11 · How is circular polarization reported?"
    As `{"co", "cross"}` dicts of complex amplitudes (same/opposite
    handedness), normalized so \(|co|^2 + |cross|^2\) is the zero-order
    efficiency. Square magnitudes for power; `np.angle` for phase.
    [Lesson 5](tutorials/polarization.md) has the guided tour.

??? question "12 · Can I get the transmitted/reflected phase?"
    `result.T_phase` / `result.R_phase` (radians, zero order). Mind the
    convention when comparing across tools — see the next question.

??? question "13 · My phase disagrees with another RCWA code by a constant!"
    Working as intended. Time-sign and reference-plane conventions differ
    between tools; expect a sign flip and/or constant offset. In our grcwa
    cross-check the offset was a constant ≈ −π while the *dispersion* agreed to
    ~21 mrad. Compare dispersions, not absolutes.

## Materials

??? question "14 · How do I add a custom material?"
    Four ways, by commitment level: a bare number inline (constant index),
    [`MaterialLibrary.add_from_file`](api/materials-layers.md#materiallibrary)
    from CSV (`λ_nm n [k]`), a Lorentz-model JSON, or the
    [`ikarus-add-material` CLI](api/tools.md#material-import-cli) for a
    permanent entry.

??? question "15 · What if my band extends past the tabulated data?"
    Interpolation is cubic-spline inside the table; outside, values are
    **clamped** to the nearest endpoint — no divergence, but also no accuracy.
    Check the table's coverage before sweeping wide.

??? question "16 · Which materials ship in the box?"
    Air, **Ag**, Au, GaN, GaP, Si, Si₃N₄, SiO₂, TiO₂, aSi —
    `default_library.available()` to confirm; `default_library.get("Si",
    1.55e-6)` for the index.

## Capabilities

??? question "17 · Does Ikarus run on the GPU?"
    No — pure NumPy/SciPy CPU. Before mourning: pin BLAS to one thread and
    parallelize across processes ([Need for Speed](performance.md)); for
    typical metaatom sizes that's worth more than a GPU port.

??? question "18 · Are there gradients for optimization?"
    Yes — **adjoint** gradients. With the `[grad]` extra installed
    (`pip install "ikarus-rcwa[grad]"`), the same `optimize(atom, target)` call
    automatically uses gradient-based (adjoint) optimization when the problem
    suits it: freeform pixel maps (thousands of DOFs — the gradient with
    respect to *every* pixel costs about one extra solve), free heights and
    periods, with a minimum-feature-size fab filter built in. Parametric-shape
    DOFs, discrete material choices and full Pareto fronts stay on the
    GA / NSGA-III path — same call, the engine is picked for you
    ([Inverse Design](api/inverse.md)). The differentiable solver itself is
    exposed as `ikarus.grad.solve` for custom objectives.

??? question "19 · Anisotropic materials?"
    Yes. Pass `(n_x, n_y, n_z)` anywhere a material goes, or use
    `ikarus.uniaxial(n_o, n_e, axis=...)` for wave plates and c-plates at any
    in-plane optic-axis angle — in uniform *and* patterned layers, with
    dispersive components. Scope: the in-plane tensor plus a distinct z
    response (`eps_xz`/`eps_yz` tilted-axis media and magneto-optic gyrotropy
    are not supported). The cover and substrate must stay isotropic.
    See [Materials → Anisotropic](api/materials-layers.md#anisotropic).

??? question "20 · How do I look at the near field?"
    `rcwa.get_fields(plane="xz"|"yz"|"xy", ...)` →
    [`FieldMap`](api/fields-viz.md#fieldmap)s → `plot_field(...)`, with the
    structure outline overlaid automatically.
    [Lesson 3](tutorials/metasurfaces.md#near-field-maps) demonstrates.

## Workflow

??? question "21 · How do I save and reload results?"
    `rcwa.save_results("run.h5", include=[...])` → self-describing HDF5;
    `RCWA.load_results("run.h5")` → dict. Needs the `io` extra (h5py).

??? question "22 · What's the fast way to sweep?"
    One `RCWA`, many `set_source(...)` calls — it retains unspecified fields.
    Only geometry changes force a new eigensolve.
    [Lesson 4](tutorials/parameter-sweeps.md) is the masterclass.

??? question "23 · Why is the PyPI name `ikarus-rcwa` but the import `ikarus`?"
    A stranger got to the short name on PyPI first. `pip install ikarus-rcwa`,
    then `import ikarus`. The myth survives the namespace collision.

??? question "24 · How do I cite Ikarus?"
    BibTeX and guidance on the [Citation page](citation.md).
