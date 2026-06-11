# API Reference

The public API is re-exported from the top-level `ikarus` package. The reference
is organized by responsibility:

| Page | Contents |
|---|---|
| [RCWA & Results](rcwa.md) | `RCWA` façade, `SimulationResult` |
| [Source](source.md) | `Source` plane-wave illumination |
| [Layers & Materials](materials-layers.md) | `Layer`, `Material`, `MaterialLibrary`, `default_library` |
| [Shapes](shapes.md) | topology primitives (`circle`, `ring`, `polygon`, …) |
| [Inverse Design](inverse.md) | `MetaAtom`, `free`, `pixels`, `Target`, `optimize` |
| [Fields & Visualization](fields-viz.md) | `FieldMap`, `reconstruct`, plotting helpers |
| [Tools](tools.md) | convergence utilities, HDF5 I/O, the material CLI |
| [Low-level](low-level.md) | `HarmonicGrid`, `convolution_matrix`, the solver internals |

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

The optional subpackages are imported explicitly:

```python
from ikarus.inverse import MetaAtom, free, pixels, Target, optimize  # needs pymoo
from ikarus.visualization import plot_field, plot_stack, plot_topology  # needs matplotlib
from ikarus.tools import convergence, io
```

!!! note "Conventions used throughout"
    - **Units:** SI (meters) for all lengths; degrees for angles.
    - **Time convention:** \(\exp(-i\omega t)\); absorbing media have \(k>0\).
    - **`n_orders = M`** keeps harmonics \(-M..+M\) per axis (count \(2M+1\)).
    - A *material specifier* is a database name, a number (constant index), a JSON
      path or a `Material`.
