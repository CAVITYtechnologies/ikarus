# Ikarus

**High-precision 2-D RCWA simulation for periodic photonic structures.**

Ikarus is a rigorous coupled-wave analysis (RCWA / Fourier Modal Method) solver
for **2-D periodic** photonic structures — metasurfaces, crossed gratings and
photonic crystal slabs. It is implemented in pure NumPy/SciPy, uses a numerically
stable **scattering-matrix (Redheffer) cascade**, and is validated against the
analytic Fresnel solution to machine precision.

```python
import numpy as np
from ikarus import RCWA

rcwa = RCWA(period_x=1e-6, period_y=1e-6, resolution=64, n_orders=15)
rcwa.add_uniform_layer(height=np.inf, material="Air")    # semi-infinite cover
rcwa.add_uniform_layer(height=200e-9, material="Si")     # a 200 nm Si film
rcwa.add_uniform_layer(height=np.inf, material="SiO2")   # semi-infinite substrate
rcwa.set_source(wavelength=1550e-9, theta=0, polarization="linear")

T, R, result = rcwa.simulate()
print(f"R = {result.R_total:.4f}  T = {result.T_total:.4f}  "
      f"R+T = {result.energy_balance:.6f}")
```

---

## Why Ikarus exists

Most research-grade RCWA codes fall into one of two camps: terse academic scripts
that are fast but hard to read and extend, or large frameworks with steep setup
costs. Ikarus aims for a third point in that space:

- **A readable, decomposed implementation.** The numerically heavy core
  (`ikarus.core.solver`) is *stateless* and separated from the user-facing
  [`RCWA`](api/rcwa.md) façade, the [materials](api/materials-layers.md) layer and
  the [Fourier machinery](api/low-level.md). Each piece is independently testable.
- **Correctness you can audit.** The package ships a validation suite
  (`ikarus/tests/validation/`) that checks the engine against the analytic
  Fresnel/transfer-matrix solution and an independent 1-D mode-matching reference.
- **An ergonomic API** that still exposes the full per-order, vectorial result —
  diffraction efficiencies, complex coefficients, exit angles, and real-space
  field maps.
- **Gradient-free inverse design** built in, so that the same metaatom definition
  can be optimized for a target response without leaving the package.

## Key features

| Capability | Status |
|---|---|
| 2-D periodic structures (crossed gratings, metasurfaces) | ✅ |
| Pixel-map topologies + shape primitives (circle, ring, polygon, …) | ✅ |
| Linear polarization (any angle), oblique incidence | ✅ |
| Circular polarization with co/cross-pol decomposition | ✅ |
| All diffraction orders with exit angles | ✅ |
| Dispersive material database (Si, SiO₂, TiO₂, GaN, GaP, aSi, Au, Si₃N₄, Air) | ✅ |
| Custom materials from CSV (`n, k`) or a Lorentz model | ✅ |
| Real-space field reconstruction (xy / xz / yz planes) | ✅ |
| Structure & field visualization (matplotlib) | ✅ |
| Automatic convergence testing (`never` / `once` / `always`) | ✅ |
| HDF5 export / import of results | ✅ |
| Numerically stable S-matrix cascade (no transfer-matrix overflow) | ✅ |
| Gradient-free inverse design (pixels + parameters, GA / NSGA-III via pymoo) | ✅ |
| Anisotropic (3×3 tensor) materials | ⛔ not yet (isotropic only) |
| Li inverse-rule factorization (faster TM convergence) | ⛔ Laurent rule only |
| GPU acceleration | ⛔ CPU (NumPy/SciPy) only |

## Main advantages

- **Validated to machine precision.** Reproduces the analytic Fresnel solution to
  ~10⁻¹⁵ at any incidence angle and polarization; conserves energy to ~10⁻⁹ for
  lossless diffraction gratings.
- **Numerically stable for thick/evanescent layers.** The scattering-matrix
  cascade avoids the exponential overflow that destroys transfer-matrix products.
- **Cross-validated and fast.** Head-to-head against the independent
  [grcwa](https://github.com/weiliangjinca/grcwa) package, Ikarus agrees on R/T
  and phase to ~10⁻³ while running ~1.5–1.7× faster per solve (see
  [Performance](performance.md)).

## Example results

A guided end-to-end demo (a TiO₂ "cross" antenna metasurface on glass) is shipped
as a runnable script and exercises every major feature — structure plots,
order-resolved efficiencies, field maps, a wavelength spectrum, circular
polarization and HDF5 output:

```bash
python -m ikarus.examples.feature_tour
# writes figures + results.h5 to ./ikarus_tour_output/
```

See the [Examples Gallery](examples-gallery.md) for the full catalogue.

## Quick-start example

A complete reflection/transmission calculation for a patterned layer:

```python
import numpy as np
from ikarus import RCWA, shapes

# A square lattice of TiO2 pillars (circles) in air, on a glass substrate.
period = 500e-9
disk = shapes.circle(center=(0.5, 0.5), radius=0.3, grid_shape=(128, 128))

rcwa = RCWA(period_x=period, period_y=period, resolution=(128, 128), n_orders=(10, 10))
rcwa.add_uniform_layer(np.inf, "Air")
rcwa.add_layer(220e-9, disk, ["Air", "TiO2"])   # topology 0 -> Air, 1 -> TiO2
rcwa.add_uniform_layer(np.inf, "SiO2")
rcwa.set_source(wavelength=600e-9, theta=0, polarization="linear")

T, R, result = rcwa.simulate()
print(f"Transmittance = {result.T_total:.3f}")
print(f"Specular (0,0) order efficiency = {result.T_orders[result.order_index(0, 0)]:.3f}")
```

Continue with the [Installation](installation.md) and [Quick Start](quickstart.md)
guides, or read the [Theory](theory.md) chapter for the method background.
