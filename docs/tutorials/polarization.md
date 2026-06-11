# Lesson 5 · Twisting Light

**Mission:** rotate linear polarization across an anisotropic meta-atom, speak
the co/cross language of circular light, and put a number on chirality.

## Rotating linear polarization

`linear_pol_angle` measures from TE: `0` = TE/s, `90` = TM/p. At normal
incidence that's simply the E-field direction in the xy-plane (0 → +y,
90 → +x). A nanobar — longer than it is wide — responds differently along its
two axes (**form birefringence**):

```python
import numpy as np
from ikarus import RCWA, shapes

period, N = 500e-9, 96
bar = shapes.rectangle(center=(0.5, 0.5), size=(0.7, 0.25), grid_shape=(N, N))

rcwa = RCWA(period_x=period, period_y=period, resolution=(N, N), n_orders=(9, 9))
rcwa.add_uniform_layer(np.inf, "Air")
rcwa.add_layer(220e-9, bar, ["Air", "Si"])   # a Si nanobar
rcwa.add_uniform_layer(np.inf, "SiO2")

angles = np.linspace(0, 180, 37)
T = []
for psi in angles:
    rcwa.set_source(wavelength=700e-9, theta=0, polarization="linear", linear_pol_angle=psi)
    T.append(rcwa.simulate()[2].T_total)

T = np.array(T)
print(f"T swings from {T.min():.3f} (across the bar) to {T.max():.3f} (along it)")
```

Expect a 180°-periodic oscillation with extrema at 0°/90° — the bar acting as
a tiny waveplate. This is exactly how geometric-phase metasurfaces are born.

## Circular light: the co/cross dialect

Under `RCP`/`LCP` illumination, `T` and `R` come back as dicts
`{"co", "cross"}` — the part that kept the incident handedness and the part
that flipped it. They're complex amplitudes, normalized so
\(|co|^2 + |cross|^2\) equals the zero order's efficiency:

```python
rcwa.set_source(wavelength=700e-9, theta=0, polarization="RCP")
T, R, res = rcwa.simulate()
print(f"T_co={abs(T['co'])**2:.3f}  T_cross={abs(T['cross'])**2:.3f}")
print(f"R_co={abs(R['co'])**2:.3f}  R_cross={abs(R['cross'])**2:.3f}")
```

The anisotropic bar converts some handedness (that's its waveplate nature). A
*symmetric* disk wouldn't. And a structure with genuinely broken mirror
symmetry — an L, a gammadion — treats the two handednesses *unequally*.

## Putting a number on chirality

Circular dichroism (CD): does the structure transmit RCP and LCP differently?

```python
def total_T(pol):
    rcwa.set_source(wavelength=700e-9, theta=0, polarization=pol)
    return rcwa.simulate()[2].T_total

CD = total_T("RCP") - total_T("LCP")
print(f"circular dichroism CD = {CD:+.4f}")
```

For the achiral bar: `CD ≈ 0` to numerical precision — a free correctness
check. Break the in-plane mirror symmetry (or go oblique / stack layers) and
CD wakes up.

## Designing for polarization

The [inverse module](../api/inverse.md) speaks this dialect natively:
`Target.maximize("t_cross", ...)` breeds a polarization converter,
`Target.match("t_phase", value, ...)` pins a phase, and
`MetaAtom(..., polarization="RCP")` sets the illumination for the whole
evolution.

## Expected results

- Nanobar transmittance oscillates with 180° period; extrema on its axes.
- Achiral structure → `CD ≈ 0`. If it isn't, *that's* a finding (or a typo in
  your topology).

## Pilot habits

- Calibrate with a boring case first: an isotropic slab must be blind to
  `linear_pol_angle` and have `CD = 0`.
- `co`/`cross` are **amplitudes** — square for power, `np.angle` for phase.
- Step 6 of `python -m ikarus.examples.feature_tour` is this lesson, live.

---

*Next:* [Lesson 6 · Coming in at an Angle →](angular-response.md)
