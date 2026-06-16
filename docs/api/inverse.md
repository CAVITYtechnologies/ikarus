# Inverse Design

```python
from ikarus.inverse import MetaAtom, Structure, free, pixels, Target, optimize
```

*Declare what you want; let evolution do the drafting.* Three steps: describe a
parameterized **design**, state one or more **targets**, call **optimize** — a
gradient-free mixed-variable GA (one objective) or NSGA-III (several).

!!! note "Optional dependency"
    Needs **pymoo**: `pip install "ikarus-rcwa[inverse]"`.

## Which construct should I use? { #which-construct }

There are a few ways to describe an inverse-design problem in Ikarus. Pick by how
your geometry is parameterized — this is the whole decision:

| Your design is… | Use | degrees of freedom |
|---|---|---|
| **one** patterned layer with a few meaningful knobs (radius, cross arms, rotation) | [`MetaAtom`](#metaatom) + a parametric [`Shape`](shapes.md#parametric-shapes) | a handful of reals |
| **one** patterned layer, freeform topology (no shape prior) | [`MetaAtom`](#metaatom) + [`pixels`](#degrees-of-freedom) | a binary grid |
| **several** layers, and/or **shared / derived** geometry, free heights & period | [`Structure`](#structure) | reals + binaries across the stack |
| something none of the above can express, or you want your own optimizer/objective | Ikarus as a **forward model** + any optimizer | whatever you write |

The first three feed the built-in [`optimize`](#optimize); the last is the
"bring-your-own-optimizer" pattern in
[Aerobatics → Optimization workflows](../advanced.md#optimization-workflows).

```mermaid
flowchart TD
    Q{How many<br/>patterned layers?} -->|one| S{Shape prior?}
    Q -->|"several / shared params"| ST["Structure"]
    S -->|"yes: radius, arms, …"| MA["MetaAtom + Shape"]
    S -->|"no: freeform"| PX["MetaAtom + pixels"]
    MA --> OPT["optimize(design, targets)"]
    PX --> OPT
    ST --> OPT
    OPT --> RES["OptimizeResult → .rcwa"]
```

!!! tip "It's one small contract"
    `optimize` only ever calls two methods on the design you hand it —
    `variables()` and `build(params, n_orders)`. `MetaAtom` and `Structure` are
    just two implementations of that contract, so even a fully custom design class
    works as long as it implements those two methods.

## Degrees of freedom

#### `free(low, high) -> Free`

Mark a continuous parameter (height or period) as a free DOF bounded to
`[low, high]` (SI units).

#### `pixels(nx, ny, symmetry=None) -> Pixels`

Mark the patterned-layer topology as a free binary pixel map. `symmetry`
shrinks the search space *and* enforces the physical symmetry:

| `symmetry` | Meaning | Constraint |
|---|---|---|
| `None` | all `nx*ny` pixels free | — |
| `"mirror_x"`, `"mirror_y"`, `"mirror_xy"` | reflection symmetry | — |
| `"c2"` | 180° rotation | — |
| `"c4"` | 90° rotation | square grid |
| `"c4v"` | 90° rotation + mirrors | square grid |

`Pixels.n_free` is the independent bit count (an 8×8 `c4v` grid → just 10
bits); `Pixels.expand(bits)` rebuilds the full `(nx, ny)` 0/1 grid.

#### Parametric shapes

A [`Shape`](shapes.md#parametric-shapes) (`Cross`, `SplitRing`, `Ellipse`, …)
used as a topology turns each of its `free(...)` parameters into a real DOF named
`shape__<param>` (e.g. `shape__arm_length`, `shape__angle`). This optimizes a
*physically interpretable* meta-atom — arm widths, radii, rotation — instead of a
pixel grid, over far fewer variables. See
[Lesson 7](../tutorials/inverse-design.md).

```python
from ikarus.shapes import Cross
from ikarus.inverse import free

topology = Cross(arm_length=free(0.3, 0.95), arm_width=free(0.1, 0.45),
                 angle=free(0, 90))   # 3 free DOF + a clean, manufacturable shape
```

## `MetaAtom`

```python
MetaAtom(period, cover, substrate, polarization="linear", pol_angle=0.0)
```

A parameterized 3-region metaatom: **cover / patterned layer / substrate**.
`period` and the pattern `height` may be fixed floats or `free(...)` ranges;
the topology may be a fixed array, a `pixels(...)` map, or a parametric
[`Shape`](shapes.md#parametric-shapes) with free parameters.

#### `add_pattern(topology, materials, height) -> MetaAtom`

Add the single patterned layer (`0 -> materials[0]`, etc.).

#### `variables() -> dict`

`{name: ('real', (lo, hi)) | ('binary',)}` for every free DOF (`period`,
`height`, `px0`, `px1`, …) — also your search space when bringing your own
optimizer.

#### `n_dof -> int`

Number of free degrees of freedom.

#### `build(params, n_orders) -> RCWA`

The concrete [`RCWA`](rcwa.md) for one parameter assignment (no source set).

```python
atom = MetaAtom(period=180e-9, cover="Air", substrate="SiO2")
atom.add_pattern(topology=pixels(8, 8, symmetry="c4v"),
                 materials=["Air", "Si3N4"],
                 height=free(40e-9, 200e-9))
print(atom.variables())
# {'height': ('real', (4e-08, 2e-07)), 'px0': ('binary',), ...}
```

## `Structure` { #structure }

```python
from ikarus.inverse import Structure
```

Where `MetaAtom` optimizes a **single** patterned layer, a `Structure` optimizes
an **entire stack** — several patterned layers, free heights, a free period, and
(the key capability) **shared / derived geometry**, where many layers are computed
from a few parameters.

You **subclass** it, **declare** each parameter as a class attribute —
`free(lo, hi)` for a degree of freedom, a plain value for a fixed parameter — and
implement **`define(self, p)`** to lay out the stack. `p` is a namespace whose
attributes are the *resolved* values of every declared parameter; the optimizer
picks the free ones, fixed ones pass through. The cover, substrate and period are
wrapped on for you.

| Reserved attribute | Meaning | Default |
|---|---|---|
| `cover`, `substrate` | semi-infinite end materials | `"Air"`, `"SiO2"` |
| `resolution` | real-space grid (int or `(nx, ny)`) | `96` |
| `polarization`, `pol_angle` | illumination for the build | `"linear"`, `0.0` |

A `period` parameter is **required** (free or fixed). Everything else you declare
becomes a parameter available in `p`; the `free(...)` ones become optimization DOF.

### Two patterned layers, optimized together

An air hole in a silicon layer *and* a cross in another — radii, arm length,
both heights and the period all free, all optimized at once:

```python
from ikarus.inverse import Structure, free, optimize, Target
from ikarus.shapes import Circle, Cross

class TwoLayer(Structure):
    cover, substrate, resolution = "Air", "SiO2", 96
    period  = free(0.3e-6, 0.9e-6)     # free
    h1      = free(0.1e-6, 0.4e-6)     # free
    h2      = 0.20e-6                  # fixed
    radius  = free(0.10, 0.45)         # free
    arm_len = free(0.30, 0.90)         # free

    def define(self, p):
        self.add_layer(p.h1, Circle(radius=p.radius), ["Si", "Air"])       # air hole in Si
        self.add_layer(p.h2, Cross(arm_length=p.arm_len, arm_width=0.2), ["Air", "Si"])

best = optimize(TwoLayer(), Target.minimize("R", at=1550e-9))
best.rcwa            # the optimized stack as a ready-to-simulate RCWA
```

### Shared / derived parameters (a moth-eye)

The thing a `MetaAtom` cannot do: drive **many** layers from a **few** parameters.
A graded moth-eye cone is a stack of slices whose radii are all functions of
`r_base` and `gamma` — so four DOF describe the whole cone:

```python
from ikarus.inverse import Structure, free
from ikarus.shapes import Circle

class MothEye(Structure):
    cover, substrate, resolution = "Air", "Si", 96
    N = 12                                 # fixed (available as p.N)
    period = free(150e-9, 240e-9)
    height = free(200e-9, 1000e-9)
    r_base = free(0.15, 0.5)
    gamma  = free(0.5, 3.0)

    def define(self, p):
        for i in range(p.N):
            r = p.r_base * ((i + 0.5) / p.N) ** p.gamma     # derived from shared DOF
            self.add_layer(p.height / p.N, Circle(radius=r), ["Air", "Si"])
```

[Lesson 8](../tutorials/structures.md) builds and optimizes this end-to-end.

### Methods

| Member | Description |
|---|---|
| `define(self, p)` | **you implement this** — add interior layers via `self.add_layer(...)` using the resolved parameters `p`. |
| `add_layer(height, topology, materials)` | add one interior layer (call from `define`). `topology` may be an array, a [`Shape`](shapes.md#parametric-shapes), or an `.img` object; a single-material list makes a uniform layer. |
| `variables() -> dict` | the free DOF (auto-discovered from the declared `free(...)` attributes). |
| `build(params, n_orders) -> RCWA` | resolve `params` and assemble the full stack (this is what `optimize` calls). |

## `Target`

One figure of merit. Build with a classmethod:

```python
Target.maximize(metric, at=None, band=None, order=(0, 0), **kw)
Target.minimize(metric, at=None, band=None, order=(0, 0), **kw)
Target.match(metric, value, at=None, band=None, order=(0, 0), **kw)
```

**Metrics**

| Metric | Meaning |
|---|---|
| `"R"`, `"T"` | Diffraction efficiency into `order` (default specular `(0,0)`; `order=None` → total). |
| `"r_co"`, `"t_co"` | Complex zero-order coefficient (co-pol). |
| `"r_cross"`, `"t_cross"` | Cross-pol coefficient (0 for linear polarization). |
| `"r_phase"`, `"t_phase"` | Phase (rad), matched modulo \(2\pi\). |

**Wavelengths** (pick one)

| Argument | Meaning |
|---|---|
| `at=1550e-9` | one wavelength |
| `at=[1064e-9, 1550e-9]` | a discrete set |
| `band=(lo, hi)` or `band=(lo, hi, n)` | a sampled range (`n` defaults to 8) |

**Options**

| Keyword | Default | Meaning |
|---|---|---|
| `order` | `(0, 0)` | diffraction order; `None`/`"total"` for the sum |
| `weight` | `1.0` | scales this target's contribution |
| `worst_case` | `False` | aggregate wavelengths by the worst point, not the mean |
| `name` | auto | label used in `report()` |

```python
# AR coating: minimize reflection across a band, robustly.
ar = Target.minimize("R", band=(300e-9, 600e-9, 6), worst_case=True)

# Beam steering: shove power into the +1 reflected order.
steer = Target.maximize("R", order=(1, 0), at=1550e-9)

# A metalens pixel: pin the transmission phase.
phase = Target.match("t_phase", value=1.57, at=1550e-9)
```

## `optimize`

```python
optimize(atom, targets, n_orders=8, algorithm="auto",
         pop=100, n_gen=60, seed=0, verbose=True) -> OptimizeResult
```

| Argument | Default | Description |
|---|---|---|
| `atom` | — | a `MetaAtom`. |
| `targets` | — | a `Target` or list (≥ 2 → multi-objective Pareto). |
| `n_orders` | `8` | harmonic truncation for every forward solve. |
| `algorithm` | `"auto"` | GA if one objective, NSGA-III if several; or `"ga"`, `"nsga2"`, `"nsga3"`. |
| `pop`, `n_gen` | `100`, `60` | population size and generations. |
| `seed` | `0` | RNG seed — runs are reproducible. |
| `verbose` | `True` | print the per-generation pymoo table. |
| `progress` | `False` | show one [progress bar](sweeps.md#optimization-progress) over the generations (sets `verbose=False`). |

### `OptimizeResult`

| Member | Description |
|---|---|
| `params` | Best parameter dict (first Pareto point if multi-objective). |
| `rcwa` | The optimized design as a ready-to-simulate `RCWA`. |
| `metaatom` | Alias of `rcwa` (kept for back-compat). |
| `report() -> str` | Human-readable summary (objective + parameters, or the Pareto front). |
| `X`, `F` | Raw best parameters and objective value(s). |
| `multi` | `True` for multi-objective runs. |

## Complete example — broadband AR coating

```python
import numpy as np
from ikarus.inverse import MetaAtom, free, pixels, Target, optimize

atom = MetaAtom(period=180e-9, cover="Air", substrate="SiO2")
atom.add_pattern(topology=pixels(8, 8, symmetry="c4v"),
                 materials=["Air", "Si3N4"], height=free(40e-9, 200e-9))

target = Target.minimize("R", band=(300e-9, 600e-9, 6), worst_case=True)
best = optimize(atom, target, n_orders=6, pop=16, n_gen=10, seed=0)
print(best.report())

coating = best.metaatom                       # a ready RCWA
coating.set_source(wavelength=450e-9, theta=0, polarization="linear")
print("R @ 450 nm:", coating.simulate()[2].R_total)
```

### Best practices

- **Pin BLAS to one thread** for these tight loops
  ([why it's worth ~10×](../performance.md#blas-threading)).
- Keep the metaatom **subwavelength** for effective-medium behavior — no
  parasitic diffraction lanes during evolution.
- `worst_case=True` for broadband robustness; a discrete `at=[...]` list when
  only specific lines matter.
- Exploit symmetry: 8×8 `c4v` is a 10-bit search, 8×8 free is 64 bits — a
  difference of *eighteen orders of magnitude* in search-space size.
- Start with small `pop`/`n_gen` to gauge runtime, then scale.
- Competing goals (high `T` *and* a phase)? Pass a list of targets and read
  the Pareto front from `report()`.
