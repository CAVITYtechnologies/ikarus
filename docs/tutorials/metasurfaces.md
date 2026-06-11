# Metasurface simulation

**Goal.** Build a 2-D patterned metasurface from shape primitives, simulate it,
extract the transmission phase, and visualize the near field.

## A dielectric nanopillar metasurface

A square lattice of TiO₂ cylinders on glass — a canonical building block for
phase-gradient metasurfaces.

```python
import numpy as np
from ikarus import RCWA, shapes

period = 420e-9
N = 128
pillar = shapes.circle(center=(0.5, 0.5), radius=0.32, grid_shape=(N, N))

rcwa = RCWA(period_x=period, period_y=period, resolution=(N, N), n_orders=(10, 10))
rcwa.add_uniform_layer(np.inf, "Air")
rcwa.add_layer(600e-9, pillar, ["Air", "TiO2"])   # 0 -> Air, 1 -> TiO2 cylinder
rcwa.add_uniform_layer(np.inf, "SiO2")
rcwa.set_source(wavelength=532e-9, theta=0, polarization="linear")

_, _, res = rcwa.simulate()
print(f"T={res.T_total:.3f}  R={res.R_total:.3f}  R+T={res.energy_balance:.5f}")
```

## Visualizing the structure

```python
# The layer stack (xz) and the unit-cell topology (xy):
rcwa.visualize_structure(plane="xz", savefig="stack.png")
rcwa.visualize_structure(plane="xy", layer_index=1, savefig="topology.png")
```

## Transmission phase — the metasurface design variable

A metalens or beam-deflector is designed by mapping a geometric parameter (here the
pillar radius) to the **transmission phase** of the specular order while keeping
transmittance high. Sweep the radius:

```python
radii = np.linspace(0.15, 0.45, 25)
T, phase = [], []
for r in radii:
    pillar = shapes.circle(radius=r, grid_shape=(N, N))
    rcwa = RCWA(period_x=period, period_y=period, resolution=(N, N), n_orders=(10, 10))
    rcwa.add_uniform_layer(np.inf, "Air")
    rcwa.add_layer(600e-9, pillar, ["Air", "TiO2"])
    rcwa.add_uniform_layer(np.inf, "SiO2")
    rcwa.set_source(wavelength=532e-9, theta=0, polarization="linear")
    _, _, res = rcwa.simulate()
    T.append(res.T_total)
    phase.append(res.T_phase)             # radians, zero-order

phase = np.unwrap(phase)
print(f"phase coverage: {np.degrees(phase.ptp()):.0f} deg "
      f"(need >= 360 for a full library)")
```

A good meta-atom library spans a full \(2\pi\) of phase with near-unity
transmittance.

## Near-field maps

Reconstruct the field to see the resonance inside the pillar:

```python
from ikarus.visualization import plot_field

rcwa.set_source(wavelength=532e-9)
rcwa.simulate()

# xz cross-section through the pillar center:
xz = rcwa.get_fields(plane="xz", nx=160, y_position=period / 2)["xz"]
ax = plot_field(xz, component="intensity")          # |E|^2 with outline overlay
ax.figure.savefig("pillar_field_xz.png", dpi=150, bbox_inches="tight")

# xy slice at mid-height:
xy = rcwa.get_fields(z_positions=[300e-9], plane="xy", nx=160, ny=160)
ax = plot_field(list(xy.values())[0], component="intensity")
ax.figure.savefig("pillar_field_xy.png", dpi=150, bbox_inches="tight")
```

## Inverse design

Rather than scanning a parameter by hand, you can let Ikarus design the meta-atom
for a target response — see [Inverse Design](../api/inverse.md) and the
[broadband AR-coating example](../examples-gallery.md#inverse-design-ar-coating).

## Expected results

- High transmittance (`T ≳ 0.9`) away from resonances, dropping at the Mie
  resonances of the pillar.
- A monotonic-ish phase ramp vs. radius spanning ≥ 360° for a suitable pillar
  height — the basis of phase-gradient design.

## Best practices

- Choose `period` **subwavelength** in the substrate (`period < λ/n_sub`) to keep
  the device 0-order (no stray diffraction).
- Use `n_orders` of 8–12 for dielectric pillars; verify with a
  [convergence study](parameter-sweeps.md#convergence-study).
- Reconstruct fields on a finer grid (`nx`, `ny`) than the solver `resolution` for
  crisp figures.
