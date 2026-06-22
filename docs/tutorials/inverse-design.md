# Lesson 7 · Designing the Shape Itself

**Mission:** stop choosing parameters by hand. State a goal — "maximum
transmission at 1300 nm" — and let a genetic algorithm evolve a meta-atom's
*own* shape parameters: arm lengths, widths, rotation, height.

Every previous lesson was **forward** design: you set the geometry, Ikarus told
you the optics. This one runs the arrow backwards.

## Two ways to hold a degree of freedom

Ikarus's inverse module ([`ikarus.inverse`](../api/inverse.md)) optimizes two
kinds of topology DOF:

| DOF | What it is | Best for |
|---|---|---|
| **Parametric shape** | a shape class (`Cross`, `SplitRing`, …) with named, bounded parameters | physically interpretable meta-atoms with a few meaningful knobs |
| **Pixel map** | a free binary grid (`pixels(nx, ny, symmetry=...)`) | freeform / topology optimization with no shape prior |

This lesson is about the first — the one your intuition can read. (For the pixel
route, see the [broadband AR coating](../examples-gallery.md#inverse-design-ar-coating).)

!!! note "Needs pymoo"
    `pip install "ikarus-rcwa[inverse]"`.

## Parametric shapes

A parametric [`Shape`](../api/shapes.md#parametric-shapes) carries named
parameters and a rotation `angle`. Used normally, it's just a tidy topology:

```python
from ikarus.shapes import Cross
topo = Cross(arm_length=0.7, arm_width=0.2, angle=30).to_grid((128, 128))
```

The trick: **any parameter may be a `free(lo, hi)` range instead of a number.**
Mark a few free, and they become the optimization variables — no pixel grids, no
loss of physical meaning.

```python
from ikarus.inverse import free
from ikarus.shapes import Cross

shape = Cross(arm_length=free(0.3, 0.95),   # free: the GA will choose
              arm_width=free(0.1, 0.45),    # free
              angle=free(0, 90))            # free: rotation is a knob too
shape.free_parameters()
# {'angle': (0.0, 90.0), 'arm_length': (0.3, 0.95), 'arm_width': (0.1, 0.45)}
```

## The three-step recipe

Declare the meta-atom, state the target, optimize:

```python
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")   # single-thread BLAS for the GA loop

from ikarus.inverse import MetaAtom, free, optimize, Target
from ikarus.shapes import Cross

# 1. a Si cross on glass whose arms, rotation and height are all free
atom = MetaAtom(period=700e-9, cover="Air", substrate="SiO2")
atom.add_pattern(topology=Cross(arm_length=free(0.3, 0.95),
                                arm_width=free(0.1, 0.45),
                                angle=free(0, 90), grid_shape=(96, 96)),
                 materials=["Air", "Si"],
                 height=free(0.3e-6, 0.9e-6))

# 2. what we want: maximum transmission at 1300 nm
target = Target.maximize("T", at=1300e-9)

# 3. evolve it
best = optimize(atom, target, n_orders=6, pop=16, n_gen=10, seed=0)
print(best.report())

design = best.metaatom          # a ready-to-simulate RCWA
```

The optimizer enumerates the free shape parameters automatically — they appear in
the report as `shape__arm_length`, `shape__angle`, and so on, alongside the free
`height`:

```text
Inverse-design result:
  objective = 0.016  (max(T))
    height = 7.89e-07
    shape__angle = 12.6
    shape__arm_length = 0.85
    shape__arm_width = 0.40
```

<figure markdown="span">
  ![Evolved Si cross meta-atom and its spectrum](../assets/lesson7_inverse_shape.png){ width="760" }
  <figcaption>The genetic algorithm chose the arm dimensions, rotation and height of a Si cross to maximize transmission at 1300&nbsp;nm (dotted line). Left: the evolved topology. Right: its full spectrum, computed by Ikarus.</figcaption>
</figure>

## Plot the result

Visualize the evolved meta-atom and sweep its spectrum:

```python
import numpy as np
import matplotlib.pyplot as plt

design = best.metaatom            # the optimized RCWA (also best.rcwa)

# the evolved topology (the patterned layer is layer 1)
design.visualize_structure(plane="xy", layer_index=1, savefig="evolved_atom.png")

# its transmission spectrum
wl = np.linspace(1.0e-6, 1.6e-6, 31)
T = []
for w in wl:
    design.set_source(wavelength=w, theta=0, polarization="linear")
    T.append(design.simulate()[2].T_total)

plt.figure(figsize=(7, 4))
plt.plot(wl * 1e9, np.array(T) * 100, lw=2)
plt.axvline(1300, color="0.6", ls=":")        # the optimization target
plt.xlabel("wavelength (nm)"); plt.ylabel("transmittance (%)")
plt.title("Spectrum of the evolved meta-atom"); plt.grid(alpha=0.3)
plt.tight_layout(); plt.savefig("evolved_spectrum.png", dpi=150, bbox_inches="tight")
plt.show()
```

## Why parametric beats pixels (sometimes)

- **Fewer variables.** A `Cross` has 4 knobs; a 16×16 pixel map has 256. The GA
  converges far faster on the smaller, smoother space.
- **Manufacturable by construction.** The result is a clean cross with a definite
  arm width — not a speckle pattern that a fab process may not resolve.
- **Physically legible.** You can read *why* it works (a longer arm red-shifts the
  resonance) and transfer the intuition.

Pixels win when you have **no** good shape prior and want the optimizer to invent
topology you wouldn't have guessed. Many real workflows do both: parametric to
get close, pixels to polish.

## Bringing your own shape class

`add_layer` and the inverse module accept **any object that exposes an `img`
array** (or a `to_grid()` method), so an external topology library drops straight
in:

```python
# any class with a binary `.img` numpy array works as a topology
rcwa.add_layer(200e-9, MyTopologySpecies(lx=0.4, ly=0.7), ["Air", "Si"])
```

To make a custom shape *optimizable*, subclass
[`ikarus.shapes.Shape`](../api/shapes.md#parametric-shapes): declare its
parameters in `_PARAMS` and implement `_mask`. It then inherits `free_parameters`,
rotation and the inverse-design plumbing for free.

## Expected results

- A converging objective (the GA's `f_min` drops each generation), ending in a
  high-transmission cross with a specific rotation.
- A spectrum with the target wavelength sitting in a transmission window — and,
  often, a sharp resonance nearby that the optimizer steered *away* from the
  target.

## Pilot habits

- **Pin BLAS to one thread** ([why](../performance.md#blas-threading)) — GA loops
  are many small solves.
- Start with **small `pop`/`n_gen`** to gauge runtime and sanity, then scale up.
- Bound parameters **physically** (an `arm_width` can't exceed the period) — tight,
  honest ranges make the search both faster and manufacturable.
- Free the **rotation**: `angle` is a cheap extra knob that unlocks
  polarization-dependent and chiral responses (try `SplitRing`).

---

*Next:* [Lesson 8 · Stacking the Deck →](structures.md) — optimize a whole
multi-layer stack at once.
