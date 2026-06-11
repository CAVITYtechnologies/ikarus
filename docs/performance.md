# Performance

## Computational complexity

The cost of a single solve is dominated by dense linear algebra over the Fourier
harmonics. Let

\[
P = (2M_x + 1)(2M_y + 1)
\]

be the harmonic count for `n_orders = (Mx, My)`. Each interior layer requires an
**eigendecomposition** of a \(2P \times 2P\) matrix and the scattering-matrix
algebra works on \(2P \times 2P\) blocks, so per layer

\[
\text{time} \sim \mathcal{O}(P^3), \qquad \text{memory} \sim \mathcal{O}(P^2).
\]

A stack of \(N\) interior layers scales linearly in \(N\). The takeaways:

- **`n_orders` is the cost driver.** Doubling \(M\) in 2-D multiplies \(P\) by ~4
  and the time by ~64. Choose the smallest converged \(M\) (see
  [convergence](tutorials/parameter-sweeps.md#convergence-study)).
- **1-D gratings are cheap.** `n_orders=(M, 0)` gives \(P = 2M+1\) (linear in \(M\)),
  not quadratic — keep gratings genuinely 1-D.
- **Uniform layers and the cover/substrate are nearly free** — their modes are
  analytic (`uniform_modes`), so thin-film stacks (`n_orders=0`) are instant.

## Indicative single-solve timings

Approximate wall time for one 2-D solve of a patterned layer (single thread;
exact numbers depend on the CPU and BLAS). These are the orders of magnitude to
expect, not a benchmark guarantee:

| `n_orders` \(M\) | harmonics \(P\) | matrix \(2P\) | time (indicative) |
|---:|---:|---:|---|
| 5 | 121 | 242 | ~0.1 s |
| 8 | 289 | 578 | ~0.4 s |
| 9 | 361 | 722 | ~1 s |
| 12 | 625 | 1250 | ~4 s |
| 13 | 729 | 1458 | ~8 s |
| 15 | 961 | 1922 | ~20 s |

The cubic trend is clear: from \(M=9\) to \(M=13\), \(P\) grows ~2× and time ~8×.

!!! info "Relative to other codes"
    In a head-to-head wavelength sweep and convergence study against the
    independent [grcwa](https://github.com/weiliangjinca/grcwa) package (both fed
    identical material data), Ikarus agreed on \(R\), \(T\) and phase to ~10⁻³ and
    ran **~1.5–1.7× faster per solve** (e.g. 729 harmonics: ~8 s vs ~13 s). The
    speedup comes from exploiting the identity gap modes (\(W_0 = I\)), diagonal
    structure in homogeneous regions, and using `scipy.linalg.solve`
    (right-division) instead of explicit inverses.

## Memory scaling

The largest objects are the \(2P \times 2P\) complex128 mode/scattering matrices —
several are live per interior layer. One such matrix occupies \((2P)^2 \times 16\)
bytes:

| `n_orders` \(M\) | \(2P\) | one \(2P\times 2P\) matrix |
|---:|---:|---:|
| 5 | 242 | ~0.9 MB |
| 8 | 578 | ~5 MB |
| 10 | 882 | ~12 MB |
| 12 | 1250 | ~25 MB |
| 15 | 1922 | ~59 MB |
| 20 | 3362 | ~181 MB |

Budget several of these per layer. For deep stacks or large \(M\), reduce
`n_orders`, cut the number of simultaneously-live solves, or process layers/sweeps
in chunks.

## BLAS threading

This is the single most impactful knob for typical metaatom work.

- **Tight loops of small solves** (sweeps, inverse-design populations): the
  \(2P\)-sized matrices are small enough that threaded BLAS spends more time
  spawning/synchronizing threads than computing. **Pin BLAS to one thread** and
  parallelize at the process level instead:

  ```python
  import os
  for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
            "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
      os.environ.setdefault(v, "1")
  import numpy as np  # noqa: E402  -- must come after the env vars
  ```

  In testing, this made an inverse-design run roughly an order of magnitude
  faster on a many-core machine.

- **Large single solves** (\(M \gtrsim 20\)): here a single eigendecomposition is
  big enough to use the cores well — let BLAS thread and parallelize at a coarser
  grain.

When in doubt, benchmark both on your problem size.

## Convergence recommendations

| Structure | Typical `n_orders` | Notes |
|---|---|---|
| Thin films (uniform) | `0` | exact, instant |
| Low-contrast dielectric metasurfaces | 8–12 | converges quickly, TE and TM alike |
| High-contrast / high-index dielectric | 12–18 | check TM specifically |
| Metals (TM / p-pol) | 18–30+ | slow — Laurent's rule; see below |
| 1-D gratings | `(15–30, 0)` | cheap because 1-D |

Always run a [convergence study](tutorials/parameter-sweeps.md#convergence-study)
at your **worst-case** wavelength and polarization, and watch the energy defect
\(|R+T-1|\) for lossless structures.

!!! warning "Laurent vs. Li factorization"
    Ikarus factors the permittivity with **Laurent's rule** (direct convolution).
    The accelerated **Li inverse rule** — which dramatically speeds TM convergence
    at sharp high-contrast/metallic interfaces — is **not yet implemented**. For
    such structures expect to need a larger \(M\) than a Li-rule code would. This is
    a known roadmap item (see [Theory → Limitations](theory.md#limitations-of-rcwa)).

## Accuracy considerations

- **Validated to machine precision** against analytic Fresnel/transfer-matrix
  solutions (~10⁻¹⁵) at any angle and polarization; energy conserved to ~10⁻⁹ for
  lossless gratings.
- The real-space `resolution` must resolve the geometry; Ikarus auto-raises it to
  ≥ `4*M + 1` to avoid aliasing the convolution matrix, so under-sampling shows up
  as a `ValueError` rather than a silent error.
- Near **Rayleigh–Wood anomalies** the solution is regularized with a tiny
  imaginary loss; the energy defect rises locally and convergence slows — refine
  `n_orders` if the metric there matters.

## Checklist for fast, trustworthy runs

1. Use the smallest converged `n_orders`; verify with a convergence study.
2. Keep 1-D gratings 1-D (`(M, 0)`); use `n_orders=0` for thin films.
3. Pin BLAS to one thread for sweeps/optimization; parallelize across processes.
4. Reuse one `RCWA` and change only the source when geometry is fixed.
5. Watch `energy_balance`; an unexplained excess means raise `n_orders` (lossless)
   or check material `k`-signs (absorbing).
