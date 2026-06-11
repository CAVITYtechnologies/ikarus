# Layers & Materials

```python
from ikarus import Layer, Material, MaterialLibrary, default_library
```

The stack's building blocks and their wardrobe. Layers carry geometry;
materials carry dispersion; the library translates whatever you hand it into
physics.

## `Layer` { #layer }

```python
Layer(height, material=None, topology=None, materials=None,
      resolution=None, name="")
```

A single stack layer. You normally create layers through
[`RCWA.add_uniform_layer`](rcwa.md#add_uniform_layer) /
[`add_layer`](rcwa.md#add_layer) rather than instantiating `Layer` directly.

| Name | Type | Description |
|---|---|---|
| `height` | `float` | Thickness (m); `np.inf` for the semi-infinite cover/substrate. |
| `material` | specifier | For a **uniform** layer: a single material. |
| `topology` | `ndarray` | For a **patterned** layer: an integer `(Nx, Ny)` map. |
| `materials` | `list` | Material list the topology indexes into. |
| `resolution` | `(int, int)` | Optional per-layer sampling (else the solver's global value). |
| `name` | `str` | Label used in plots and metadata. |

**Raises.** `ValueError` if neither `material` nor `topology` is given, if a
patterned layer lacks `materials`, or if the topology references more
materials than provided.

### Properties & methods

| Member | Description |
|---|---|
| `is_uniform` | `True` if the layer has no topology. |
| `is_semi_infinite` | `True` if `height` is infinite. |
| `permittivity_grid(wavelength, library, resolution) -> ndarray` | Sample \(\varepsilon(x,y)\) on an `(Nx, Ny)` grid (uniform → constant; patterned → nearest-neighbour resample, each index replaced by its complex permittivity). |
| `uniform_permittivity(wavelength, library) -> complex` | Scalar \(\varepsilon\) of a uniform layer (raises if patterned). |

---

## `Material` { #material }

```python
Material(name, wavelength_nm=None, n=None, k=None,
         lorentz=None, comment="")
```

One optical material. Construct via the classmethods, not the raw fields.

### Constructors

#### `Material.constant(value, name="const") -> Material`

A non-dispersive material with complex index `value`.

```python
glass = Material.constant(1.5, name="glass")
absorber = Material.constant(2.0 + 0.05j, name="absorber")
```

#### `Material.from_dict(data) -> Material`

Build from the JSON schema (tabulated `n,k` or a `lorentz` block).

#### `Material.from_file(path) -> Material`

Load a JSON material file.

### Evaluation

| Method | Description |
|---|---|
| `index(wavelength) -> complex` | Complex refractive index \(n+ik\) at `wavelength` (m). Scalar or array. |
| `permittivity(wavelength) -> complex` | \(\varepsilon = (n+ik)^2\). |

### JSON schema

Tabulated:

```json
{
  "name": "MyMat",
  "comment": "source / notes",
  "wavelength_nm": [400, 500, 600, 700],
  "n": [2.1, 2.05, 2.02, 2.00],
  "k": [0.0, 0.0, 0.0, 0.0]
}
```

Lorentz model
(\(\varepsilon(\omega) = \varepsilon_\infty + \sum_j f_j\,\omega_{0j}^2 / (\omega_{0j}^2 - \omega^2 - i\gamma_j\omega)\),
angular frequencies in rad/s):

```json
{
  "name": "MyLorentz",
  "lorentz": {
    "eps_inf": 1.0,
    "oscillators": [
      {"f": 0.5, "w0": 3.0e15, "gamma": 1.0e14}
    ]
  }
}
```

!!! note "Interpolation & extrapolation"
    Tabulated data uses a **cubic** spline (linear below four points) and
    extrapolates by **clamping** to the nearest tabulated endpoint — an
    out-of-range request returns the boundary value, never a divergence. Check
    coverage by evaluating `index` at your band edges.

---

## `MaterialLibrary` { #materiallibrary }

```python
MaterialLibrary(db_dir=<package material dir>)
```

A lazy-loading registry that resolves every *material specifier* the API
accepts: a database **name**, a **number** (constant index), a **JSON path**,
or a `Material` instance.

| Method | Description |
|---|---|
| `available() -> list[str]` | Names in the database directory + anything registered in memory. |
| `register(material)` | Add a `Material` to the in-memory cache. |
| `resolve(spec) -> Material` | Turn any accepted specifier into a `Material`. |
| `get(spec, wavelength) -> complex` | Complex index \(n+ik\) at `wavelength`. |
| `permittivity(spec, wavelength) -> complex` | \(\varepsilon\) at `wavelength`. |
| `add_from_file(path, name=None, persist=False) -> Material` | Import from CSV (`λ_nm, n[, k]`) or JSON; `persist=True` also writes into the database directory. |
| `save(material) -> Path` | Serialize a tabulated material to the database as JSON. |

### `default_library`

The module-level `MaterialLibrary` shared by every `RCWA` that doesn't bring
its own:

```python
from ikarus import default_library
default_library.available()
# ['Air', 'Au', 'GaN', 'GaP', 'Si', 'Si3N4', 'SiO2', 'TiO2', 'aSi']

default_library.get("Si", 1.55e-6)   # complex index at 1550 nm
```

## Examples

```python
from ikarus import RCWA, MaterialLibrary, Material
import numpy as np

# 1. Built-ins by name.
rcwa = RCWA(period_x=1e-6, period_y=1e-6, n_orders=10)
rcwa.add_uniform_layer(np.inf, "Air")
rcwa.add_uniform_layer(150e-9, "TiO2")
rcwa.add_uniform_layer(np.inf, "SiO2")

# 2. A constant index inline.
rcwa.add_uniform_layer(100e-9, 1.46)            # SiO2-ish, non-dispersive

# 3. A custom library with an imported material.
lib = MaterialLibrary()
lib.add_from_file("my_polymer.csv", name="Polymer", persist=False)
rcwa2 = RCWA(period_x=1e-6, period_y=1e-6, materials=lib)
rcwa2.add_uniform_layer(200e-9, "Polymer")
```

### Best practices

- Verify a material covers your band *before* sweeping — clamped extrapolation
  silently returns endpoint values.
- `persist=True` (or the [`ikarus-add-material` CLI](tools.md#material-import-cli))
  only when you want the material in future sessions; otherwise keep custom
  materials in-memory.
- Respect the \(k>0\) (absorbing) convention — gain-signed imports inflate the
  energy balance past 1.
