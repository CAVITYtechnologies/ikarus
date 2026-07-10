# Lesson 3 · Sculpting Wavefronts

**Mission:** build a 2-D meta-atom from shape primitives, harvest the
transmission *phase* (the metasurface designer's currency), and look at the
near field with your own eyes.

## A dielectric nanopillar

A square lattice of TiO₂ cylinders on glass — the canonical building block of
visible-light metalenses:

```python
import numpy as np
from ikarus import RCWA, shapes

period = 420e-9
N = 128
pillar = shapes.circle(center=(0.5, 0.5), radius=0.32, grid_shape=(N, N))

rcwa = RCWA(period_x=period, period_y=period, resolution=(N, N), n_orders=(10, 10))
rcwa.add_uniform_layer(np.inf, "Air")
rcwa.add_layer(600e-9, pillar, ["Air", "TiO2"])   # 0 -> Air, 1 -> TiO2
rcwa.add_uniform_layer(np.inf, "SiO2")
rcwa.set_source(wavelength=532e-9, theta=0, polarization="linear")

_, _, res = rcwa.simulate()
print(f"T={res.T_total:.3f}  R={res.R_total:.3f}  R+T={res.energy_balance:.5f}")
```

See what you built:

```python
rcwa.visualize_structure(plane="xz", savefig="stack.png")          # the stack
rcwa.visualize_structure(plane="xy", layer_index=1, savefig="topology.png")
```

## Phase: the designer's currency

A metalens is a map from position to phase delay. You build it from a
**library** of pillars whose radius tunes the transmission phase of the
specular order — ideally covering a full \(2\pi\) while staying transparent:

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
    phase.append(res.T_phase)             # zero-order phase, radians

phase = np.unwrap(phase)
print(f"phase coverage: {np.degrees(np.ptp(phase)):.0f} deg "
      f"(a full library wants >= 360)")

# plot the phase library: transmittance and phase vs. radius (twin axes)
import matplotlib.pyplot as plt
fig, ax1 = plt.subplots(figsize=(7, 4))
ax2 = ax1.twinx()
ax1.plot(radii, np.array(T) * 100, "C1", lw=2)
ax2.plot(radii, np.degrees(phase), "C0", lw=2)
ax1.set_xlabel("pillar radius (fraction of period)")
ax1.set_ylabel("transmittance (%)", color="C1")
ax2.set_ylabel("transmission phase (deg)", color="C0")
fig.tight_layout(); fig.savefig("phase_library.png", dpi=150, bbox_inches="tight")
plt.show()
```

Each pillar is a tiny truncated waveguide; fatter pillar → higher effective
index → more phase delay. When the sweep covers \(2\pi\) with \(T \gtrsim 0.9\),
you have a complete set of "phase pixels" to tile any wavefront you like.

<figure markdown="span">
  ![Transmission phase vs pillar radius](../assets/metasurface_phase.png){ width="640" }
  <figcaption>Sweeping the pillar radius scans the transmission phase across a full 2π while transmittance stays high — the meta-atom phase library.</figcaption>
</figure>

## Near-field maps { #near-field-maps }

Numbers are good; *seeing* the Mie resonance inside the pillar is better:

```python
from ikarus.visualization import plot_field

rcwa.set_source(wavelength=532e-9)
rcwa.simulate()

# xz cross-section through the pillar center:
xz = rcwa.get_fields(plane="xz", nx=160, y_position=period / 2)["xz"]
ax = plot_field(xz, component="intensity")          # |E|² + structure outline
ax.figure.savefig("pillar_field_xz.png", dpi=150, bbox_inches="tight")

# xy slice at mid-height:
xy = rcwa.get_fields(z_positions=[300e-9], plane="xy", nx=160, ny=160)
ax = plot_field(list(xy.values())[0], component="intensity")
ax.figure.savefig("pillar_field_xy.png", dpi=150, bbox_inches="tight")
```

The plots come with the material outline overlaid automatically, so you can
check the field actually lives where you think it does.

<figure markdown="span">
  ![Near-field intensity in a nanopillar](../assets/metasurface_field.png){ width="640" }
  <figcaption>Reconstructed |E|² through the TiO₂ nanopillar (xz cross-section), with the material outline overlaid automatically.</figcaption>
</figure>

## Too lazy to sweep radii? Good.

Declaring the goal and letting a genetic algorithm sculpt the meta-atom is a
one-liner away — see [Inverse Design](../api/inverse.md) and the
[broadband AR coating](../examples-gallery.md#inverse-design-ar-coating) in
The Hangar.

## Expected results

- High transmittance (`T ≳ 0.9`) away from resonances, with dips at the
  pillar's Mie resonances.
- A phase ramp vs. radius spanning ≥ 360° for a well-chosen height — your
  metalens alphabet.

## Pilot habits

- Keep the period **subwavelength in the substrate** (`period < λ/n_sub`) so
  no stray diffraction lanes open — all the power stays in the specular order
  you're phase-engineering.
- `n_orders` 8–12 is the dielectric-pillar sweet spot; confirm with
  [Lesson 4's convergence ritual](parameter-sweeps.md#convergence-study).
- Reconstruct fields on a finer `nx`/`ny` than the solver `resolution` — the
  reconstruction grid is free to choose.

---

*Next:* [Lesson 4 · Sweeping Gracefully →](parameter-sweeps.md)
