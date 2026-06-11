# Reflection & Transmission spectra

**Goal.** Compute reflectance \(R(\lambda)\) and transmittance \(T(\lambda)\) of a
layered structure, read off the specular order, and account for absorption.

## A thin-film stack

We start with a uniform stack — a 120 nm TiO₂ film on glass — and sweep the
visible band. Because the structure is unpatterned, only the specular order
exists and the result is the exact transfer-matrix spectrum.

```python
import numpy as np
from ikarus import RCWA

rcwa = RCWA(period_x=400e-9, period_y=400e-9, n_orders=0)  # 0 orders: specular only
rcwa.add_uniform_layer(np.inf, "Air")
rcwa.add_uniform_layer(120e-9, "TiO2")
rcwa.add_uniform_layer(np.inf, "SiO2")

wavelengths = np.linspace(400e-9, 700e-9, 61)
R, T = [], []
for wl in wavelengths:
    rcwa.set_source(wavelength=wl, theta=0, polarization="linear")
    _, _, res = rcwa.simulate()
    R.append(res.R_total)
    T.append(res.T_total)

R, T = np.array(R), np.array(T)
print(f"R+T spans [{(R+T).min():.6f}, {(R+T).max():.6f}]")  # ~1: TiO2 lossless here
```

!!! note "Why `n_orders=0`"
    For a **uniform** stack there is no diffraction, so a single harmonic (the
    specular order) is exact and instant. Use `n_orders=0` (or a small value) for
    thin-film calculations; reserve large `n_orders` for patterned layers.

## Plotting the spectrum

```python
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(wavelengths * 1e9, R * 100, label="Reflectance")
ax.plot(wavelengths * 1e9, T * 100, label="Transmittance")
ax.plot(wavelengths * 1e9, (R + T) * 100, "k--", lw=1, label="R + T")
ax.set_xlabel("wavelength (nm)")
ax.set_ylabel("efficiency (%)")
ax.legend(); ax.grid(alpha=0.3)
fig.savefig("thin_film_spectrum.png", dpi=150, bbox_inches="tight")
```

## Reading the specular order explicitly

For a patterned structure `R_total`/`T_total` sum over *all* propagating orders.
To isolate the **specular** (0,0) order use `order_index`:

```python
i00 = res.order_index(0, 0)
print("specular transmittance T(0,0) =", res.T_orders[i00])
print("specular reflectance  R(0,0) =", res.R_orders[i00])
```

## Absorption

`energy_balance = R_total + T_total`. For an **absorbing** material it is below 1,
and the absorptance is the remainder:

```python
rcwa = RCWA(period_x=400e-9, period_y=400e-9, n_orders=0)
rcwa.add_uniform_layer(np.inf, "Air")
rcwa.add_uniform_layer(50e-9, "Au")     # gold: strongly absorbing in the visible
rcwa.add_uniform_layer(np.inf, "SiO2")

rcwa.set_source(wavelength=550e-9, theta=0, polarization="linear")
_, _, res = rcwa.simulate()
A = 1.0 - res.energy_balance
print(f"R={res.R_total:.3f}  T={res.T_total:.3f}  A={A:.3f}")
```

## Expected results

- **TiO₂/glass:** smooth thin-film interference fringes; `R+T ≈ 1` to ~10⁻⁹
  (TiO₂ is essentially lossless across the visible in the shipped data).
- **Au film:** large absorptance `A`, with `R+T` well below 1 — *not* a sign of an
  error but of real ohmic loss.

## Best practices

- Use `n_orders=0` for thin films; it is exact and fast.
- Treat `energy_balance` as a built-in correctness check: for lossless materials it
  must be ≈ 1. If it drifts above 1 for a *patterned* layer, raise `n_orders`
  ([convergence study](parameter-sweeps.md#convergence-study)).
- Reuse one `RCWA` across the sweep and change only the wavelength via
  `set_source(wavelength=...)`.
