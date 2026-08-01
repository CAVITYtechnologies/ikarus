# What Ikarus Is (and Isn't)

*Or: the honest edges of the tool.*

Ikarus does one thing and tries to do it exactly right: **rigorous, high-precision
electromagnetic simulation of periodic photonic structures.** This page states
plainly what that covers, what it deliberately doesn't, and the design choices
behind those boundaries — so you can tell in two minutes whether Ikarus fits your
problem.

## What it's built for

Frequency-domain Maxwell solutions for structures **periodic in x and y** and
**layered along z**:

- diffraction gratings (1-D and 2-D), metasurfaces and meta-atoms,
- photonic-crystal slabs, thin-film stacks, Bragg mirrors, metamirrors.

You get per-order diffraction efficiencies, complex reflection/transmission
coefficients and phase, exit angles, real-space `E`/`H` fields, and gradient- or
GA-based [inverse design](tutorials/inverse-design.md).

## What it deliberately is: the faithful CPU reference

Ikarus optimizes for **correctness and simplicity over raw speed** — on purpose.

- **Faithful by default.** The normal-vector / Fast-Fourier-Factorization method is
  the default, so high-index-contrast TM structures converge to the *right* answer
  where direct-rule solvers quietly don't — [proven against independent
  codes](validation.md).
- **Honest about convergence.** Energy balance is treated as a smoke test, not a
  convergence proof (it isn't one — see [Core Concepts](core-concepts.md)); Ikarus
  judges convergence on the complex coefficients and warns when you're not there.
- **Simple and portable.** `pip install ikarus-rcwa`, pure NumPy/SciPy — no GPU
  drivers, no CUDA toolkit, no OS-specific setup. It runs identically on Windows,
  macOS and Linux, and gives the same numbers on each. Nothing to configure, nothing
  to break.
- **Reproducible.** A CPU reference implementation you can trust as ground truth and
  cite ([Citation](citation.md)).

!!! note "Why CPU-only is a choice, not a gap"
    The only practical GPU route for a solver like this is JAX, whose GPU builds
    exist only on Linux (or Windows-via-WSL2) with a CUDA toolkit. That means driver
    versions, subsystems, and setup that can silently go wrong — exactly the friction
    Ikarus exists to avoid. We'd rather be **the solver that always just works** than
    one that's fast only after an afternoon of environment surgery. RCWA on CPU is
    already fast for the metasurface/grating regime; if you ever hit the ceiling, the
    lever is `n_orders`, not a GPU (see [Need for Speed](performance.md)).

## What it's not

Being honest about the edges is part of being a reference:

- **Not for non-periodic problems.** Isolated scatterers, single nanoparticles,
  aperiodic or finite devices, waveguide bends — those are FDTD/FEM/BEM territory,
  not RCWA's.
- **Not for continuously-graded-in-z or time-domain physics.** Ikarus slices the
  stack into uniform-in-z layers; smoothly graded profiles must be approximated by
  slicing, and pulses/transients are out of scope (it's frequency-domain).
- **Anisotropy is partial by design.** Supported: a full **in-plane** permittivity
  tensor plus a distinct `εzz`. *Not* supported: a **tilted optic axis**
  (`εxz`/`εyz`) or **magneto-optic gyrotropy** (`εxy ≠ εyx`); the cover and substrate
  must be isotropic. You'll get a clear `ValueError`, never a silently wrong answer.
- **CPU-only.** No GPU/accelerator backend — see the note above; it's deliberate.
- **2-D cost grows as `O(M⁶)`.** Ideal for the metasurface/grating/thin-film regime.
  Very large supercells or very high truncations get expensive — budget with the cost
  table in [Need for Speed](performance.md).

## The trade-off, stated plainly

Every solver picks a point on the *fast ↔ faithful ↔ simple* triangle. Ikarus picks
**faithful and simple**, and accepts "CPU-fast" rather than "GPU-fast." If your work
needs the *right* answer for periodic photonics with zero setup friction — and a
reference you can validate other tools against — that's exactly what Ikarus is for.
If you need GPU-scale time-domain or non-periodic simulation, reach for a different
tool; we'll happily tell you which.

*Not sure whether your problem fits? [Ask us](citation.md) — it's the kind of
question we like.*
