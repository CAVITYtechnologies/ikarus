# API Reference

*The cockpit manual.* Reference pages are organized by responsibility — fun is
rationed here in favor of precision, because this is where you come mid-flight
with a question.

<div class="grid cards" markdown>

-   :material-movie-open:{ .lg .middle } **[RCWA & Results](rcwa.md)**

    ---

    The `RCWA` façade and the `SimulationResult` it returns — the two objects
    you'll touch every day.

-   :material-lightbulb-on:{ .lg .middle } **[Source](source.md)**

    ---

    Plane-wave illumination: wavelength, angles, polarization.

-   :material-palette:{ .lg .middle } **[Layers & Materials](materials-layers.md)**

    ---

    `Layer`, `Material`, `MaterialLibrary` and the shipped database.

-   :material-shape:{ .lg .middle } **[Shapes](shapes.md)**

    ---

    Topology primitives: circle, ellipse, rectangle, ring, cross, polygon.

-   :material-dna:{ .lg .middle } **[Inverse Design](inverse.md)**

    ---

    `MetaAtom`, `free`, `pixels`, `Target`, `optimize` — declare, then evolve.

-   :material-map:{ .lg .middle } **[Fields & Visualization](fields-viz.md)**

    ---

    `FieldMap`, `reconstruct`, and the matplotlib plotting helpers.

-   :material-wrench:{ .lg .middle } **[Tools](tools.md)**

    ---

    Convergence automation, HDF5 I/O, the material-import CLI.

-   :material-engine:{ .lg .middle } **[Low-level](low-level.md)**

    ---

    `HarmonicGrid`, `convolution_matrix`, `solve_stack` — the engine room.

</div>

## Top-level exports

```python
from ikarus import (
    RCWA,             # the main façade
    SimulationResult, # rich result object
    Source,           # plane-wave illumination
    Layer,            # a single stack layer
    Material,         # one optical material
    MaterialLibrary,  # registry of materials
    default_library,  # the shared built-in library (a MaterialLibrary instance)
    HarmonicGrid,     # Fourier-order bookkeeping
    shapes,           # topology primitives subpackage
)
```

Optional subpackages are imported explicitly:

```python
from ikarus.inverse import MetaAtom, free, pixels, Target, optimize  # needs pymoo
from ikarus.visualization import plot_field, plot_stack, plot_topology  # needs matplotlib
from ikarus.tools import convergence, io
```

!!! note "House conventions (memorize once)"
    - **Units:** SI — meters for lengths, degrees for angles.
    - **Time convention:** \(\exp(-i\omega t)\); absorbers have \(k>0\).
    - **`n_orders = M`** keeps harmonics \(-M..+M\) per axis (count \(2M+1\),
      total \(P=(2M_x+1)(2M_y+1)\)).
    - A *material specifier* is a database name, a number (constant index), a
      JSON path, or a `Material`.
