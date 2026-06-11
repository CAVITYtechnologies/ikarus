# Source

```python
from ikarus import Source
```

The plane wave doing the illuminating. Usually you create it indirectly through
[`RCWA.set_source`](rcwa.md#defining-the-source) — but it's a plain dataclass
you can build and inspect directly.

## `Source`

```python
Source(wavelength, theta=0.0, phi=0.0,
       polarization="linear", linear_pol_angle=0.0, n_incident=1.0)
```

**Parameters**

| Name | Type | Default | Description |
|---|---|---|---|
| `wavelength` | `float` | — | Vacuum wavelength (m); must be positive. |
| `theta` | `float` | `0.0` | Polar angle from `+z` (deg). `0` = normal incidence. |
| `phi` | `float` | `0.0` | Azimuth from `+x` in the xy-plane (deg). |
| `polarization` | `str` | `"linear"` | One of `"linear"`, `"RCP"`, `"LCP"`. |
| `linear_pol_angle` | `float` | `0.0` | For linear: angle from TE (deg). `0` = TE/s, `90` = TM/p. |
| `n_incident` | `complex` | `1.0` | Index of the cover medium; **set automatically** by the solver from the top layer. |

**Raises.** `ValueError` for an unknown polarization or non-positive wavelength.

!!! info "Angle and polarization conventions"
    `theta` is measured from `+z` (the wave travels in `-z`, into the stack).
    The transverse basis is \(\hat a_{TE} = \hat z \times \hat k\),
    \(\hat a_{TM} = \hat a_{TE} \times \hat k\). At **normal incidence** the
    split is degenerate, so Ikarus pins \(\hat a_{TE} = +\hat y\),
    \(\hat a_{TM} = +\hat x\) — `linear_pol_angle` becomes the literal E-field
    angle in the xy-plane.
    [Theory → Polarization](../theory.md#polarization-conventions) has the full
    picture.

### Properties

| Property | Description |
|---|---|
| `k0` | Vacuum wavenumber \(2\pi/\lambda\) (rad/m). |
| `theta_rad`, `phi_rad` | The angles in radians. |

### Methods

#### `incident_wavevector() -> ndarray`

Normalized (by `k0`) incident wavevector \((k_x, k_y, k_z)\) in the cover
medium.

#### `kx0_ky0() -> (complex, complex)`

The in-plane components \((k_{x0}, k_{y0})\) — continuous across all layers.

#### `polarization_vector() -> ndarray`

Complex unit polarization vector \((p_x, p_y, p_z)\), transverse to \(k\).

#### `te_tm_vectors() -> (ndarray, ndarray)`

The real TE and TM unit vectors for the current geometry.

#### `copy_with(**changes) -> Source`

A copy with selected fields overridden — the mechanism behind the stateful
`set_source` used in sweeps.

## Examples

```python
from ikarus import Source

# TM (p-polarized) wave at 30 degrees, 1550 nm.
src = Source(wavelength=1550e-9, theta=30, polarization="linear", linear_pol_angle=90)
print("k0 =", src.k0)
print("polarization vector =", src.polarization_vector())

# Right-circular at normal incidence.
rcp = Source(wavelength=633e-9, polarization="RCP")

# Derive a tweaked copy without mutating the original.
src_15 = src.copy_with(theta=15)
```

### Best practices

- Prefer `RCWA.set_source(**kwargs)` over constructing `Source` directly — it
  wires `n_incident` from the cover layer for you.
- Phase work across tools: Ikarus reports phase in the physics
  \(\exp(-i\omega t)\) convention; another code may differ by a sign or
  constant offset ([FAQ #13](../faq.md)).
