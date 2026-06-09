# Ikarus

**High-precision 2-D RCWA simulation for periodic photonic structures.**

Ikarus is a rigorous coupled-wave analysis (RCWA / Fourier modal method) solver
for 2-D periodic photonic structures — metasurfaces, gratings and photonic
crystals. It uses a numerically stable scattering-matrix formulation, supports
full vectorial (linear and circular) polarization, arbitrary pixel-map
topologies, a built-in dispersive material database, real-space field
reconstruction, automatic convergence testing and HDF5 I/O.

```python
import numpy as np
from ikarus import RCWA

rcwa = RCWA(period_x=1e-6, period_y=1e-6, resolution=64, n_orders=15)
rcwa.add_uniform_layer(height=np.inf, material='Air')   # semi-infinite cover
rcwa.add_uniform_layer(height=200e-9, material='Si')
rcwa.add_uniform_layer(height=np.inf, material='SiO2')  # semi-infinite substrate
rcwa.set_source(wavelength=1550e-9, theta=0, polarization='linear')

T, R, result = rcwa.simulate()
print(f"R = {result.R_total:.4f}, T = {result.T_total:.4f}, R+T = {result.energy_balance:.6f}")
```

## Why "high-precision"

Ikarus reproduces the analytic Fresnel/transfer-matrix solution for stratified
media to **machine precision (~1e-15)** at any incidence angle and polarization,
and conserves energy to ~1e-9 for diffraction gratings. The patterned-layer
machinery is independently validated against a direct mode-matching reference and
the effective-medium limit. See `ikarus/tests/validation/`.

## Features

| Capability | Status |
|---|---|
| 2-D periodic structures (crossed gratings, metasurfaces) | ✅ |
| Pixel-map topologies + shape primitives (circle, ring, polygon, …) | ✅ |
| Linear polarization (any angle), oblique incidence | ✅ |
| Circular polarization with co/cross-pol decomposition | ✅ |
| All diffraction orders with exit angles | ✅ |
| Dispersive material database (Si, SiO₂, TiO₂, GaN, GaP, aSi, Au, Si₃N₄, …) | ✅ |
| Custom materials from CSV (`n, k`) or Lorentz model | ✅ |
| Real-space field reconstruction (xy / xz / yz planes) | ✅ |
| Structure & field visualization (matplotlib) | ✅ |
| Automatic convergence testing (`never` / `once` / `always`) | ✅ |
| HDF5 export / import of results | ✅ |
| Numerically stable S-matrix cascade (no transfer-matrix overflow) | ✅ |
| Anisotropic (3×3 tensor) materials | ⛔ not yet (isotropic only) |
| Li inverse-rule factorization (faster TM convergence) | ⛔ Laurent rule only |

## Installation

```bash
pip install -e .            # core (numpy, scipy)
pip install -e ".[all]"     # + matplotlib (viz) and h5py (HDF5 I/O)
```

## Usage

### Metasurface with a pixel-map topology

```python
import numpy as np
from ikarus import RCWA

rcwa = RCWA(period_x=400e-9, period_y=400e-9, resolution=64, n_orders=10)
topology = rcwa.shapes.ring(center=(0.5, 0.5), inner_radius=0.15,
                            outer_radius=0.3, grid_shape=(64, 64))
rcwa.add_uniform_layer(np.inf, 'Air')
rcwa.add_layer(height=120e-9, topology=topology, materials=['TiO2', 'Air'])
rcwa.add_uniform_layer(np.inf, 'SiO2')
rcwa.set_source(wavelength=532e-9, theta=0, polarization='linear')
T, R, result = rcwa.simulate()
```

The integer `topology` array indexes into `materials`; pixel `(i, j)` covers
`period_x/Nx × period_y/Ny` of the cell.

### Circular polarization (co / cross)

```python
rcwa.set_source(wavelength=1550e-9, theta=30, polarization='RCP')
T, R, result = rcwa.simulate()
print(f"co-pol T:    {abs(T['co']):.4f}")
print(f"cross-pol T: {abs(T['cross']):.4f}")
```

`|T['co']|**2 + |T['cross']|**2` equals the zero-order transmittance.

### Order-resolved output and exit angles

```python
T, R, result = rcwa.simulate()
p, q = result.orders
for i in range(len(p)):
    if result.T_orders[i] > 1e-3:
        print(f"order ({p[i]:+d},{q[i]:+d}): T={result.T_orders[i]:.4f} "
              f"at theta_out={result.theta_out_trn[i]:.1f} deg")
```

### Field extraction and visualization

```python
rcwa.simulate()
fields = rcwa.get_fields(z_positions=[50e-9, 100e-9], plane='xy')   # one map per z
xz = rcwa.get_fields(plane='xz')['xz']                              # cross-section

rcwa.visualize_structure(plane='xz', savefig='stack.png')
rcwa.visualize_structure(plane='xy', layer_index=1, savefig='topology.png')
from ikarus.visualization import plot_field
plot_field(xz, component='intensity', savefig='field.png')           # |E|^2
plot_field(xz, component='Ey', savefig='ey_mag.png')                 # |Ey|
plot_field(xz, component='Eyphase', savefig='ey_phase.png')         # arg(Ey)
```

`z = 0` is the cover / first-layer interface, increasing into the stack. Field
plots **overlay the structure outline** semi-transparently by default (the
topology for `xy` slices, the layer interfaces for `xz`/`yz`); pass
`overlay=False` to disable.

### Reflection / transmission phase

The complex zero-order coefficients carry phase (key for phase-gradient
metasurface design):

```python
T, R, result = rcwa.simulate()
print(result.T_phase, result.R_phase)   # radians; dict {'co','cross'} for circular pol
```

### Convergence control

```python
T, R, result = rcwa.simulate(auto_converge='once', converge_tol=1e-4, max_orders=80)
print("converged n_orders:", rcwa.n_orders)
```

`'once'` finds and caches the optimal order count (reused across a sweep);
`'always'` re-converges every call; `'never'` (default) uses the current
`n_orders`.

### Materials

```python
from ikarus.core.materials import default_library as lib
lib.get('Si', wavelength=1550e-9)          # -> (3.479+0j)
lib.available()                            # list built-in materials
lib.add_from_file('my_nk.csv', name='MyMat', persist=True)
```

Or from the command line:

```bash
python -m ikarus.tools.add_material my_nk.csv --name MyMaterial
```

CSV columns are `wavelength_nm  n  [k]` (`k` optional, `#` comments ignored).

### Saving results

```python
rcwa.save_results('run.h5', include=['T', 'R', 'fields', 'metadata'], result=result)
data = RCWA.load_results('run.h5')
```

## Conventions

* Time convention **`exp(-i ω t)`**: absorbing materials have `k > 0`,
  `Im(ε) > 0`; `materials.get` returns `n + i k`.
* `theta` is measured from the +z axis (0 = normal incidence); `phi` is the
  azimuth from +x.
* Wavevectors are normalized by the vacuum wavenumber internally.
* The two outermost layers must be uniform and semi-infinite (`height=inf`).

## Architecture

```
ikarus/
├── core/            rcwa, layer, source, materials, fourier, solver, fields, polarization
├── shapes/          topology primitives (circle, ring, rectangle, polygon, …)
├── materials/       JSON dispersion database
├── tools/           add_material (CLI), convergence, io (HDF5)
├── visualization/   structure & field plotting
├── tests/           unit + validation (Fresnel, gratings, fields, …)
└── examples/        runnable scripts
```

The solver (`core/solver.py`) is a stateless scattering-matrix engine: per-layer
eigenmodes → layer S-matrices referenced to a vacuum gap → Redheffer cascade →
diffraction efficiencies and field amplitudes.

## Testing and examples

```bash
pytest ikarus/tests/                              # full suite (~1.4s)
python -m ikarus.examples.feature_tour            # guided tour: structure/field plots, spectrum, HDF5
python -m ikarus.examples.validation_fresnel      # machine-precision Fresnel check
python -m ikarus.examples.grating_diffraction     # order-resolved grating
python -m ikarus.examples.metasurface_spectrum    # resonant metasurface sweep
```

`feature_tour` is the best starting point — it exercises the material database,
structure/topology visualization, field reconstruction, a wavelength sweep,
circular polarization and HDF5 export, saving figures to `ikarus_tour_output/`.

## License

MIT — Copyright © 2026 CAVITY technologies UG. See [LICENSE](LICENSE).
Created by Liam Shelling Neto.
