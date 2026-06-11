# Grating diffraction

**Goal.** Simulate a 1-D binary grating, enumerate its propagating diffraction
orders, and verify the exit angles against the grating equation.

## A 1-D binary grating

A 1-D grating is invariant along \(y\); store its topology as an `(Nx, 2)` map and
expand only x-orders with `n_orders=(M, 0)`.

```python
import numpy as np
from ikarus import RCWA

period = 900e-9
rcwa = RCWA(period_x=period, period_y=period, resolution=(256, 2), n_orders=(20, 0))

topo = np.zeros((128, 2), dtype=int)
topo[64:, :] = 1                      # 50% duty cycle
rcwa.add_uniform_layer(np.inf, "Air")
rcwa.add_layer(300e-9, topo, ["TiO2", "Air"])   # 0 -> TiO2, 1 -> Air
rcwa.add_uniform_layer(np.inf, "SiO2")

rcwa.set_source(wavelength=550e-9, theta=0, polarization="linear", linear_pol_angle=0.0)
_, _, res = rcwa.simulate()
print(f"R={res.R_total:.4f}  T={res.T_total:.4f}  R+T={res.energy_balance:.6f}")
```

!!! info "Polarization on a 1-D grating"
    With the grating grooves along \(y\), `linear_pol_angle=0` (TE, E along +y) is
    the fast-converging case; `90` (TM, E across the grooves) converges more slowly
    and may need a larger `n_orders`.

## Listing diffraction orders with exit angles

```python
p, q = res.orders
print("Transmitted orders (efficiency @ exit angle):")
for i in np.argsort(-res.T_orders):
    if res.T_orders[i] > 1e-4:
        print(f"  ({p[i]:+d},{q[i]:+d}): T={res.T_orders[i]:.4f} "
              f"@ theta={res.theta_out_trn[i]:5.1f} deg")
```

Only propagating orders have finite angles; evanescent ones carry `NaN`.

## Checking the grating equation

For normal incidence the transmitted order \(m\) leaves the substrate (index
\(n_t\)) at

\[
\sin\theta_{t,m} = \frac{m\,\lambda}{n_t\,\Lambda}.
\]

```python
from ikarus import default_library
n_t = default_library.get("SiO2", 550e-9).real

for m in (-1, 0, 1):
    i = res.order_index(m, 0)
    if np.isfinite(res.theta_out_trn[i]):
        predicted = np.degrees(np.arcsin(m * 550e-9 / (n_t * period)))
        print(f"order {m:+d}: Ikarus {res.theta_out_trn[i]:6.2f} deg, "
              f"grating eq {predicted:6.2f} deg")
```

The two should agree to plotting precision.

## Wavelength dependence of the order content

As \(\lambda\) crosses \(\Lambda/n\) the higher orders cut on/off (Rayleigh–Wood
anomalies). Sweep to watch orders appear:

```python
for wl in (450e-9, 550e-9, 650e-9, 750e-9):
    rcwa.set_source(wavelength=wl)
    _, _, res = rcwa.simulate()
    n_prop = int(np.sum(np.isfinite(res.theta_out_trn) & (res.T_orders > 1e-6)))
    print(f"lambda={wl*1e9:.0f} nm: {n_prop} propagating transmitted orders, "
          f"R+T={res.energy_balance:.6f}")
```

## Expected results

- **Energy conservation:** `R+T ≈ 1` to ~10⁻⁶ or better (TiO₂/air/SiO₂ are
  lossless here). Near an anomaly the order count changes and you may need a few
  more harmonics.
- **Exit angles** match the grating equation to numerical precision.

## Best practices

- Keep gratings genuinely 1-D: `(Nx, 2)` topology + `n_orders=(M, 0)`. A `(Nx, N)`
  map with `N>2` and a 2-D `n_orders` is a *crossed* grating and far more
  expensive.
- Run a [convergence study](parameter-sweeps.md#convergence-study) in TM — it is
  the slow case (Ikarus uses Laurent's rule; see [Theory →
  Limitations](../theory.md#limitations-of-rcwa)).
- See the runnable script `python -m ikarus.examples.grating_diffraction`.
