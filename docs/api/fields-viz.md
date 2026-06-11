# Fields & Visualization

Real-space field reconstruction (`ikarus.core.fields`) and the matplotlib plotting
helpers (`ikarus.visualization`).

!!! note "Optional dependency"
    Plotting needs **matplotlib**: `pip install "ikarus-rcwa[viz]"`. Field
    *reconstruction* itself only needs NumPy/SciPy.

## Field reconstruction

### `FieldMap`

A dataclass holding real-space vector fields on a sampling grid.

| Attribute | Description |
|---|---|
| `E`, `H` | Complex arrays of shape `(..., 3)` → `(Ex, Ey, Ez)` / `(Hx, Hy, Hz)`. |
| `coords` | Dict mapping axis name (`"x"`, `"y"`, `"z"`) → 1-D coordinate array (m). |
| `z` | The depth (m) for an `xy` slice, else `None`. |
| `eps` | Real part of the permittivity on the same grid (set by `RCWA.get_fields`, used for structure overlays). |
| `intensity` *(property)* | \(|E|^2\) summed over components. |

### `RCWA.get_fields(...)`

The high-level entry point — see
[`RCWA.get_fields`](rcwa.md#get_fields).
It returns a `dict` of `FieldMap`s keyed by depth label (`xy`) or plane name
(`xz`/`yz`).

```python
# xy slices at three depths inside the structure:
maps = rcwa.get_fields(z_positions=[0, 100e-9, 200e-9], plane="xy", nx=128, ny=128)
for label, fm in maps.items():
    print(label, fm.intensity.max())

# an xz cross-section through the cell center:
xz = rcwa.get_fields(plane="xz", nx=200, y_position=rcwa.period_y / 2)["xz"]
```

!!! info "z convention"
    `z = 0` is the cover / first-interior-layer interface and `z` increases into
    the stack toward the substrate. Negative `z` is inside the cover; `z` beyond
    the total interior thickness is inside the substrate.

### `reconstruct(...)` *(low-level)*

```python
from ikarus.core.fields import reconstruct
reconstruct(sol, z_positions, nx=64, ny=64, plane="xy",
            x_position=0.0, y_position=0.0) -> dict
```

Operates directly on a `FieldSolution` (`result.solution`). `RCWA.get_fields`
wraps this and additionally attaches the structure permittivity for overlays.

## Plotting

```python
from ikarus.visualization import plot_field, plot_stack, plot_topology, plot_field_xy
```

### `plot_field(field_map, component="intensity", ax=None, savefig=None, cmap=None, overlay=True, overlay_color="white", overlay_alpha=0.45)`

Plot a 2-D `FieldMap` cross-section. `component` is `"intensity"`, a field
component (`"Ex"`…`"Hz"`) for magnitude, or `"<comp>phase"` (e.g. `"Eyphase"`) for
phase. If `overlay` is true and the map carries `eps`, material boundaries are
drawn as semi-transparent contours.

```python
ax = plot_field(xz, component="intensity")
ax.figure.savefig("field.png", dpi=150)

plot_field(xz, component="Eyphase", cmap="twilight")  # phase map
```

### `plot_field_xy(field_dict, component="intensity", savefig=None)`

Plot one or more `xy` maps (the dict from `get_fields`) side by side.

### `plot_stack(rcwa, ax=None, savefig=None, finite_frac=0.25)`

Draw the layer stack as an xz cross-section, color-coded by material. The
semi-infinite cover/substrate are drawn with a finite visual height
(`finite_frac` of the interior thickness).

### `plot_topology(rcwa, layer_index, wavelength=None, ax=None, savefig=None)`

Show a patterned layer's permittivity (real part) over the unit cell.

### Via the façade

`RCWA.visualize_structure(plane=..., layer_index=...)` and
`RCWA.visualize_fields(...)` are convenience wrappers around `plot_stack` /
`plot_topology` / `plot_field`.

```python
rcwa.visualize_structure(plane="xz")                       # stack
rcwa.visualize_structure(plane="xy", layer_index=1,        # topology
                         savefig="topology.png")
```

## Full example

```python
import numpy as np
from ikarus import RCWA, shapes
from ikarus.visualization import plot_field

period = 700e-9
cross = shapes.cross(arm_length=0.75, arm_width=0.25, grid_shape=(96, 96))

rcwa = RCWA(period_x=period, period_y=period, resolution=(96, 96), n_orders=(8, 8))
rcwa.add_uniform_layer(np.inf, "Air")
rcwa.add_layer(160e-9, cross, ["Air", "TiO2"])
rcwa.add_uniform_layer(np.inf, "SiO2")
rcwa.set_source(wavelength=600e-9, theta=0, polarization="linear")
rcwa.simulate()

xz = rcwa.get_fields(plane="xz", nx=120, y_position=period / 2)["xz"]
ax = plot_field(xz, component="intensity")     # |E|^2 with structure outline
ax.figure.savefig("cross_field_xz.png", dpi=150, bbox_inches="tight")
```

### Best practices

- Call `simulate()` (or any solve) before `get_fields`; otherwise it solves
  implicitly with the current settings.
- For publication figures, reconstruct on a finer `nx`/`ny` than the solver
  `resolution` — reconstruction is independent of the Fourier sampling.
- Keep `overlay=True` to verify the field is registered with the geometry.
