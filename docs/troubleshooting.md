# Troubleshooting

Common errors, what they mean, and how to fix them. Each entry quotes the message
(or symptom) you will see.

## Stack construction

??? failure "`ValueError: need at least a cover and a substrate layer`"
    A stack needs **at least two** layers. Add a semi-infinite cover and substrate:
    ```python
    rcwa.add_uniform_layer(np.inf, "Air")    # cover
    rcwa.add_uniform_layer(np.inf, "SiO2")   # substrate
    ```

??? failure "`ValueError: the cover/substrate layer must be uniform and semi-infinite (height=inf)`"
    The **first** and **last** layers are the incidence and transmission
    half-spaces; they must be `add_uniform_layer(np.inf, ...)`. Patterned or
    finite-thickness boundary layers are not allowed.

??? failure "`ValueError: interior layers must have finite thickness`"
    Only the cover and substrate may be `np.inf`. Give every interior layer a real
    thickness in meters.

??? failure "`ValueError: topology references N materials but only M provided`"
    The integer topology contains an index without a matching entry in `materials`.
    The list must be at least `topology.max() + 1` long (index `0` → `materials[0]`).

??? failure "`ValueError: patterned layer requires a 'materials' list`"
    `add_layer(height, topology, materials)` needs the `materials` list the topology
    indexes into.

## Source

??? failure "`ValueError: first set_source call requires 'wavelength'`"
    The first `set_source` must include `wavelength`. Later calls can omit it (it is
    retained).

??? failure "`ValueError: polarization must be one of ('linear', 'RCP', 'LCP')`"
    Use one of those exact strings. For a specific linear orientation use
    `polarization="linear"` with `linear_pol_angle=...`.

??? failure "`ValueError: call set_source(...) before simulating`"
    Define the illumination before `simulate()`.

## Convergence & energy balance

??? warning "`RuntimeWarning: Energy balance R+T=... exceeds 1.01`"
    Usually **incomplete convergence** for a lossless structure — raise `n_orders`
    (or use `simulate(auto_converge="once")`). If a material legitimately has gain,
    check the sign of its `k` (Ikarus expects \(k>0\) for loss).

??? warning "`RuntimeWarning: Energy balance R+T=... is far above 1`"
    **Numerical breakdown** at very high `n_orders` — the eigenmode/scattering
    algebra loses conditioning for high-contrast structures. **Reduce** `n_orders`
    or **increase** `resolution`; the optimum is the smallest converged order count,
    not the largest.

??? failure "`ValueError: Cell resolution too coarse for the requested harmonic orders`"
    The real-space grid cannot resolve the difference orders the convolution matrix
    needs (≥ `4*n_orders + 1` samples per axis). Increase `resolution`, or lower
    `n_orders`. The façade auto-raises sampling, so this typically appears only when
    calling `convolution_matrix` directly.

??? failure "`numpy.linalg.LinAlgError: ... singular matrix` during a solve"
    Often an order sitting exactly on a Rayleigh–Wood anomaly (the light line) for a
    perfectly lossless structure. Ikarus regularizes these with a tiny imaginary
    loss; if it still occurs, nudge the wavelength/angle/period slightly off the
    exact anomaly or add a small material loss. The auto-convergence loop catches
    `LinAlgError` and stops the sweep.

## Optional dependencies

??? failure "`ImportError: inverse.optimize needs pymoo`"
    `pip install "ikarus-rcwa[inverse]"` (or `pip install pymoo`).

??? failure "`ImportError: HDF5 I/O requires the 'h5py' package`"
    `pip install "ikarus-rcwa[io]"` (or `pip install h5py`).

??? failure "`ModuleNotFoundError: No module named 'matplotlib'` on a plot call"
    `pip install "ikarus-rcwa[viz]"` (or `pip install matplotlib`).

## Results & fields

??? failure "`KeyError: order (p, q) not in truncated set`"
    `order_index(p, q)` was asked for an order outside `-M..+M`. Increase `n_orders`
    or request an order within range.

??? failure "Exit angles are `NaN`"
    Not an error — `NaN` marks an **evanescent** order (it does not propagate).
    Mask with `np.isfinite(result.theta_out_trn)` before plotting.

??? failure "Field maps look uniform / empty"
    Make sure you ran a solve first (`simulate()` or any solve) before
    `get_fields`; reconstruct a patterned layer's depth (`z` inside the interior)
    rather than deep in the cover/substrate; and increase `nx`/`ny` for detail.

## Performance

??? failure "Sweeps/optimization are far slower than expected on a many-core machine"
    BLAS thread oversubscription. Pin BLAS to one thread **before importing NumPy**:
    ```python
    import os
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
              "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ.setdefault(v, "1")
    import numpy as np
    ```
    See [Performance → BLAS threading](performance.md#blas-threading).

??? failure "A single large solve is slow"
    The eigensolve is \(\mathcal{O}(P^3)\) in the harmonic count. Confirm you are not
    over-resolving `n_orders` (run a convergence study), keep 1-D gratings 1-D, and
    for genuinely large \(M\) let BLAS use multiple threads.

## Still stuck?

Open an issue with a minimal reproducer at
[github.com/CAVITYtechnologies/ikarus/issues](https://github.com/CAVITYtechnologies/ikarus/issues).
Include the structure definition, `n_orders`, `resolution`, the source, and the
full warning/traceback.
