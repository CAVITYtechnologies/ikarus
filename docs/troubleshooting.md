# Troubleshooting

*Splashdown procedures.* Every pilot ditches occasionally — what matters is
how fast you're back in the air. Find your error below; each entry quotes the
exact message (or symptom), explains the cause, and gives the fix.

## Stack construction

??? failure "`ValueError: need at least a cover and a substrate layer`"
    Every flight needs a sky and a ground:
    ```python
    rcwa.add_uniform_layer(np.inf, "Air")    # cover
    rcwa.add_uniform_layer(np.inf, "SiO2")   # substrate
    ```

??? failure "`ValueError: the cover/substrate layer must be uniform and semi-infinite (height=inf)`"
    The first and last layers are the incidence and transmission half-spaces —
    `add_uniform_layer(np.inf, ...)`, no patterns, no finite thickness.

??? failure "`ValueError: interior layers must have finite thickness`"
    Only the two outer layers may be `np.inf`. Give everything in between a
    real thickness in meters.

??? failure "`ValueError: topology references N materials but only M provided`"
    Your integer map contains an index with no matching entry. The `materials`
    list must be at least `topology.max() + 1` long (index `0` →
    `materials[0]`).

??? failure "`ValueError: patterned layer requires a 'materials' list`"
    `add_layer(height, topology, materials)` — the third argument isn't
    optional.

## Source

??? failure "`ValueError: first set_source call requires 'wavelength'`"
    The first call must name a wavelength; afterwards it's retained and you can
    update fields one at a time.

??? failure "`ValueError: polarization must be one of ('linear', 'RCP', 'LCP')`"
    Exactly those strings. A specific linear orientation is
    `polarization="linear"` + `linear_pol_angle=...`.

??? failure "`ValueError: call set_source(...) before simulating`"
    No light, no flight. Define the source first.

## Energy balance & convergence

??? warning "`RuntimeWarning: Energy balance R+T=... exceeds 1.01`"
    For a lossless structure: **unconverged** — raise `n_orders` or use
    `simulate(auto_converge="once")`. If a material is *supposed* to be
    lossy, check its `k` sign (Ikarus expects \(k>0\) for absorption; gain-sign
    data manufactures free energy).

??? warning "`RuntimeWarning: Energy balance R+T=... is far above 1`"
    You flew too close to the sun: at very high `n_orders` the eigenmode
    algebra loses conditioning for high-contrast structures. Counterintuitive
    fix: **reduce** `n_orders` (or raise `resolution`). The goal is the
    smallest *converged* order count, not the largest survivable one.

??? failure "`ValueError: Cell resolution too coarse for the requested harmonic orders`"
    The convolution matrix needs ≥ `4*n_orders + 1` real-space samples per
    axis. Raise `resolution` or lower `n_orders`. (The `RCWA` façade auto-raises
    sampling, so you'll usually meet this only when calling
    `convolution_matrix` directly.)

??? failure "`numpy.linalg.LinAlgError: ... singular matrix`"
    Usually an order parked *exactly* on a Rayleigh–Wood anomaly (the light
    line) in a perfectly lossless structure. Ikarus regularizes these with a
    vanishing imaginary loss; if it still bites, nudge wavelength/angle/period
    off the exact anomaly or add a whisper of material loss. The
    auto-convergence loop catches `LinAlgError` and stops the sweep gracefully.

## Optional dependencies

??? failure "`ImportError: inverse.optimize needs pymoo`"
    `pip install "ikarus-rcwa[inverse]"`

??? failure "`ImportError: HDF5 I/O requires the 'h5py' package`"
    `pip install "ikarus-rcwa[io]"`

??? failure "`ModuleNotFoundError: No module named 'matplotlib'`"
    `pip install "ikarus-rcwa[viz]"`

## Results & fields

??? failure "`KeyError: order (p, q) not in truncated set`"
    `order_index(p, q)` asked for a harmonic outside \(-M..+M\). Increase
    `n_orders` or request an order inside the kept set.

??? failure "Exit angles are `NaN`"
    Not an error — `NaN` marks an **evanescent** order (a ghost lane carrying
    no far-field power). Mask with `np.isfinite(...)` before plotting.

??? failure "Field maps look empty or uniform"
    Three usual suspects: no solve ran yet (call `simulate()` first); you
    reconstructed at a depth inside the cover/substrate instead of the
    patterned layer (check the z convention: 0 = cover/first-layer interface,
    increasing downward); or `nx`/`ny` is too coarse to show the detail.

## Performance

??? failure "Sweeps/optimization crawl on a many-core machine"
    BLAS thread oversubscription — the classic. Pin one thread **before
    importing NumPy**:
    ```python
    import os
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
              "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ.setdefault(v, "1")
    import numpy as np
    ```
    Full reasoning: [Need for Speed → BLAS](performance.md#blas-threading).

??? failure "One big solve is slow"
    The eigensolve is \(\mathcal{O}(P^3)\). Confirm `n_orders` isn't
    over-dialed (run the
    [convergence ritual](tutorials/parameter-sweeps.md#convergence-study)),
    keep 1-D problems 1-D, and for genuinely large \(M\) let BLAS use its
    threads.

## Still in the water?

Open an issue with a minimal reproducer:
[github.com/CAVITYtechnologies/ikarus/issues](https://github.com/CAVITYtechnologies/ikarus/issues).
Include the structure definition, `n_orders`, `resolution`, the source, and the
full warning or traceback — rescue is much faster with coordinates.
