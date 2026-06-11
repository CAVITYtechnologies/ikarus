# Examples Gallery

Runnable, self-contained examples. The first group ships **inside the package**
under `ikarus/examples/` and can be run as modules; the rest are copy-paste
recipes. All use SI units.

## Bundled example scripts

| Script | Run | What it shows |
|---|---|---|
| Feature tour | `python -m ikarus.examples.feature_tour` | End-to-end TiO₂ cross metasurface: materials, structure plots, order-resolved efficiencies, field maps, a spectrum, circular polarization, HDF5 output. Writes to `ikarus_tour_output/`. |
| Grating diffraction | `python -m ikarus.examples.grating_diffraction` | 1-D TiO₂ binary grating; propagating orders and exit angles vs. wavelength. |
| Metasurface spectrum | `python -m ikarus.examples.metasurface_spectrum` | Reflection/transmission spectrum of a 2-D meta-atom. |
| Inverse metamirror | `python -m ikarus.examples.inverse_metamirror` | Gradient-free inverse design of a reflective meta-atom. |
| Fresnel validation | `python -m ikarus.examples.validation_fresnel` | Reproduces the analytic Fresnel result to machine precision. |

## Fresnel validation (correctness baseline)

The cleanest sanity check: a single interface must match the analytic Fresnel
coefficients.

```python
import numpy as np
from ikarus import RCWA

rcwa = RCWA(period_x=1e-6, period_y=1e-6, n_orders=0)   # specular only
rcwa.add_uniform_layer(np.inf, 1.0)      # air
rcwa.add_uniform_layer(np.inf, 1.5)      # glass (constant index)

rcwa.set_source(wavelength=600e-9, theta=0, polarization="linear")
_, _, res = rcwa.simulate()

R_fresnel = ((1.0 - 1.5) / (1.0 + 1.5)) ** 2
print(f"Ikarus R = {res.R_total:.12f}")
print(f"Fresnel  R = {R_fresnel:.12f}")
print(f"|diff| = {abs(res.R_total - R_fresnel):.2e}")   # ~1e-15
```

## Anti-reflection thin film

A quarter-wave MgF₂-like layer (constant index) minimizing reflection at 550 nm.

```python
import numpy as np
from ikarus import RCWA

n_film, lam0 = 1.23, 550e-9          # ideal AR index ~ sqrt(1.5)
d = lam0 / (4 * n_film)              # quarter-wave thickness

rcwa = RCWA(period_x=1e-6, period_y=1e-6, n_orders=0)
rcwa.add_uniform_layer(np.inf, 1.0)
rcwa.add_uniform_layer(d, n_film)
rcwa.add_uniform_layer(np.inf, 1.5)

for wl in (450e-9, 550e-9, 650e-9):
    rcwa.set_source(wavelength=wl, theta=0, polarization="linear")
    print(f"{wl*1e9:.0f} nm: R = {rcwa.simulate()[2].R_total:.4f}")
# minimum at 550 nm
```

## Guided-mode resonance filter

A 1-D grating waveguide showing a sharp reflection peak.

```python
import numpy as np
from ikarus import RCWA

period = 880e-9
rcwa = RCWA(period_x=period, period_y=period, resolution=(256, 2), n_orders=(25, 0))

topo = np.zeros((200, 2), dtype=int)
topo[:100, :] = 1                     # 50% duty cycle
rcwa.add_uniform_layer(np.inf, "Air")
rcwa.add_layer(180e-9, topo, ["Si3N4", "Air"])   # high-index grating layer
rcwa.add_uniform_layer(np.inf, "SiO2")

for wl in np.linspace(1.0e-6, 1.1e-6, 11):
    rcwa.set_source(wavelength=wl, theta=0, polarization="linear", linear_pol_angle=0)
    R = rcwa.simulate()[2].R_total
    print(f"{wl*1e9:.0f} nm: R = {R:.3f}")        # a narrow resonance appears in-band
```

## Inverse design: AR coating

A subwavelength Si₃N₄ moth-eye optimized for broadband anti-reflection on glass
(300–600 nm). This mirrors the `inverse_metamirror` example but for transmission.

```python
import os
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(v, "1")        # single-thread BLAS for the tight loop

import numpy as np
from ikarus.inverse import MetaAtom, free, pixels, Target, optimize

atom = MetaAtom(period=180e-9, cover="Air", substrate="SiO2")
atom.add_pattern(topology=pixels(8, 8, symmetry="c4v"),
                 materials=["Air", "Si3N4"], height=free(40e-9, 200e-9))

target = Target.minimize("R", band=(300e-9, 600e-9, 6), worst_case=True)
best = optimize(atom, target, n_orders=6, pop=16, n_gen=10, seed=0)
print(best.report())

coating = best.metaatom
wl = np.linspace(300e-9, 600e-9, 13)
R = []
for w in wl:
    coating.set_source(wavelength=w, theta=0, polarization="linear")
    R.append(coating.simulate()[2].R_total)
print("worst-case R:", f"{max(R)*100:.2f}%")     # ~1.5% vs ~3.8% bare glass
```

## Beam deflector (blazed metasurface)

Maximize power into the +1 reflected order.

```python
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
from ikarus.inverse import MetaAtom, free, pixels, Target, optimize

atom = MetaAtom(period=1.2e-6, cover="Air", substrate="SiO2")
atom.add_pattern(topology=pixels(16, 4, symmetry="mirror_y"),
                 materials=["Air", "Si"], height=free(0.2e-6, 0.6e-6))

best = optimize(atom, Target.maximize("R", order=(1, 0), at=1550e-9),
                n_orders=(12, 4), pop=40, n_gen=30)
print(best.report())
```

## Where to go next

- The [Tutorials](tutorials/index.md) explain each workflow step by step.
- The [API Reference](api/index.md) documents every option used above.
- [Performance](performance.md) covers making large sweeps fast.
