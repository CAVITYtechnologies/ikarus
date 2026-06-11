# Polarization analysis

**Goal.** Sweep linear polarization angle, work with circular polarization and the
co/cross decomposition, and quantify a chiral response.

## Linear polarization angle

`linear_pol_angle` is measured from TE (0 = TE/s, 90 = TM/p). At normal incidence
TE is along +y and TM along +x, so the angle is the physical E-field orientation.

```python
import numpy as np
from ikarus import RCWA, shapes

period, N = 500e-9, 96
bar = shapes.rectangle(center=(0.5, 0.5), size=(0.7, 0.25), grid_shape=(N, N))  # anisotropic

rcwa = RCWA(period_x=period, period_y=period, resolution=(N, N), n_orders=(9, 9))
rcwa.add_uniform_layer(np.inf, "Air")
rcwa.add_layer(220e-9, bar, ["Air", "Si"])   # a Si nanobar: form-birefringent
rcwa.add_uniform_layer(np.inf, "SiO2")

angles = np.linspace(0, 180, 37)
T = []
for psi in angles:
    rcwa.set_source(wavelength=700e-9, theta=0, polarization="linear", linear_pol_angle=psi)
    T.append(rcwa.simulate()[2].T_total)

T = np.array(T)
print(f"T varies from {T.min():.3f} (across the bar) to {T.max():.3f} (along it)")
```

A shape with different x/y extents is **form-birefringent**: its response depends
on `linear_pol_angle`, with extrema at 0°/90°.

## Circular polarization and the co/cross decomposition

For `RCP`/`LCP` illumination, `T` and `R` are reported as a dict
`{"co", "cross"}` — the same-handedness (`co`) and opposite-handedness (`cross`)
amplitudes of the zero order, normalized so \(|co|^2 + |cross|^2\) is that order's
efficiency.

```python
rcwa.set_source(wavelength=700e-9, theta=0, polarization="RCP")
T, R, res = rcwa.simulate()
print(f"T_co={abs(T['co'])**2:.3f}  T_cross={abs(T['cross'])**2:.3f}")
print(f"R_co={abs(R['co'])**2:.3f}  R_cross={abs(R['cross'])**2:.3f}")
```

For an **achiral** structure at normal incidence, a mirror-symmetric meta-atom
converts handedness predictably; a structure with broken in-plane mirror symmetry
(e.g. an L- or gammadion shape) produces a handedness-dependent response — the
basis of chiral metasurfaces.

## Quantifying chirality — circular dichroism

Circular dichroism (CD) is the difference in transmittance between RCP and LCP:

```python
def total_T(pol):
    rcwa.set_source(wavelength=700e-9, theta=0, polarization=pol)
    return rcwa.simulate()[2].T_total

CD = total_T("RCP") - total_T("LCP")
print(f"circular dichroism CD = {CD:+.4f}")
```

A symmetric structure gives `CD ≈ 0`; a genuinely chiral one (often requiring
oblique incidence or a 3-D-like stacked geometry) gives `CD ≠ 0`.

## Targeting polarization in inverse design

The [inverse module](../api/inverse.md) exposes polarization metrics directly:
`Target.maximize("t_cross", ...)` drives a polarization converter,
`Target.match("t_phase", value, ...)` sets a phase. Set the metaatom polarization
with `MetaAtom(..., polarization="RCP")`.

## Expected results

- The nanobar's transmittance swings between its along-axis and across-axis values
  as `linear_pol_angle` rotates, period 180°.
- For an achiral meta-atom, `CD ≈ 0` to numerical precision; asymmetric shapes
  break this.

## Best practices

- Confirm the convention with a sanity case: a plain isotropic slab must be
  polarization-independent (`T` identical for all `linear_pol_angle`, `CD = 0`).
- Circular `co`/`cross` are **complex amplitudes** — square the magnitude for
  efficiencies, take `np.angle` for phase.
- See `python -m ikarus.examples.feature_tour` (step 6) for a runnable circular
  example.
