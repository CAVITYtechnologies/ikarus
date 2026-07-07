# Flight School

*Eight lessons from first taxi to confident aerobatics.* Each one is
self-contained, copy-paste runnable, states what you should expect to see, and
ends with the habits that separate good pilots from lucky ones. Prerequisite:
the [Quick Start](../quickstart.md).

<div class="grid cards" markdown>

-   :material-chart-bell-curve:{ .lg .middle } **Lesson 1 · Spectra 101**

    ---

    Sweep wavelength, read totals vs. the specular order, and meet
    absorption — the honest reason R+T < 1.

    [:octicons-arrow-right-24: Start](reflection-transmission.md)

-   :material-view-week:{ .lg .middle } **Lesson 2 · Splitting Light**

    ---

    1-D gratings: diffraction orders as exit lanes, exit angles checked
    against the grating equation.

    [:octicons-arrow-right-24: Start](gratings.md)

-   :material-dots-grid:{ .lg .middle } **Lesson 3 · Sculpting Wavefronts**

    ---

    Metasurfaces: build meta-atoms from shapes, harvest 2π of phase, peek at
    the near field.

    [:octicons-arrow-right-24: Start](metasurfaces.md)

-   :material-tune-variant:{ .lg .middle } **Lesson 4 · Sweeping Gracefully**

    ---

    Efficient parameter sweeps, the convergence ritual, and 2-D design maps.

    [:octicons-arrow-right-24: Start](parameter-sweeps.md)

-   :material-rotate-orbit:{ .lg .middle } **Lesson 5 · Twisting Light**

    ---

    Linear angles, circular co/cross decomposition, and measuring chirality.

    [:octicons-arrow-right-24: Start](polarization.md)

-   :material-angle-acute:{ .lg .middle } **Lesson 6 · Coming in at an Angle**

    ---

    Oblique incidence, dispersion maps, and spotting Rayleigh–Wood
    anomalies in the wild.

    [:octicons-arrow-right-24: Start](angular-response.md)

-   :material-dna:{ .lg .middle } **Lesson 7 · Inverse Design**

    ---

    Inverse design: state a goal and let the optimizer design the meta-atom —
    parametric shapes via a genetic algorithm, or freeform pixel maps via
    adjoint gradients, chosen automatically.

    [:octicons-arrow-right-24: Start](inverse-design.md)

-   :material-layers-triple:{ .lg .middle } **Lesson 8 · Stacking the Deck**

    ---

    Multi-layer inverse design: optimize a whole stack at once with
    `Structure` — including a moth-eye driven by shared parameters.

    [:octicons-arrow-right-24: Start](structures.md)

</div>

After graduation: [Aerobatics](../advanced.md) (batch runs, ML pipelines,
custom everything), [The Hangar](../examples-gallery.md) (complete worked
examples) and [Need for Speed](../performance.md) (making it all fast).

!!! tip "Shipped demo scripts"
    The package carries runnable examples too — try
    `python -m ikarus.examples.feature_tour` for the full guided airshow.
