# Lesson 8 · Stacking the Deck

**Mission:** optimize a whole **multi-layer stack** at once — not just a single
meta-atom. You'll meet the [`Structure`](../api/inverse.md#structure) class, use it
to drive **many layers from a few shared parameters**, and turn a graded moth-eye
into a first-class `optimize()` problem.

[Lesson 7](inverse-design.md) optimized one patterned layer. But real devices
stack them: a hole in one layer, a cross in another; or a graded cone sliced into
a dozen sub-layers. A `MetaAtom` can't express that. A `Structure` can.

!!! note "Needs pymoo"
    `pip install "ikarus-rcwa[inverse]"`.

## The idea: declare, then `define`

You **subclass** `Structure`, **declare** each parameter as a class attribute
(`free(...)` for a degree of freedom, a plain value for fixed), and implement
**`define(self, p)`** to lay out the layers. Inside `define`, `p` carries the
*resolved* values — the optimizer's picks for the free ones — and you call
`self.add_layer(...)` to stack the structure. Cover, substrate and period are
added for you.

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
        self.add_layer(p.h1, Circle(radius=p.radius), ["Si", "Air"])        # air hole in Si
        self.add_layer(p.h2, Cross(arm_length=p.arm_len, arm_width=0.2), ["Air", "Si"])

best = optimize(TwoLayer(), Target.minimize("R", at=1550e-9))
print(best.report())
device = best.rcwa        # the optimized stack, ready to simulate / visualize
```

Five degrees of freedom — two in-plane shape params, two heights, and the period —
all optimized **simultaneously** across two patterned layers.

## The superpower: shared / derived parameters

Here's what no single-layer construct can do. A moth-eye is a graded cone, which
RCWA models as a **stack of slices** whose radii are all *functions* of a couple
of parameters. With `define()` you just write that relationship — **four DOF
describe the entire cone:**

```python
from ikarus.inverse import Structure, free, optimize, Target
from ikarus.shapes import Circle

class MothEye(Structure):
    cover, substrate, resolution = "Air", "Si", 96
    N = 12                                  # number of slices (fixed)
    period = free(150e-9, 240e-9)
    height = free(200e-9, 1000e-9)
    r_base = free(0.15, 0.5)
    gamma  = free(0.5, 3.0)

    def define(self, p):
        for i in range(p.N):
            r = p.r_base * ((i + 0.5) / p.N) ** p.gamma     # all slices from 2 DOF
            self.add_layer(p.height / p.N, Circle(radius=r), ["Air", "Si"])

# broadband anti-reflection on silicon, worst-case over the band
best = optimize(MothEye(),
                Target.minimize("R", band=(300e-9, 600e-9, 6), worst_case=True),
                n_orders=8, pop=12, n_gen=8, seed=0, progress=True)
print(best.report())
```

<figure markdown="span">
  ![Optimized moth-eye structure and spectrum](../assets/structure_motheye.png){ width="760" }
  <figcaption>A graded silicon moth-eye optimized as a single <code>Structure</code> — a stack of slices driven by 4 shared parameters. Left: the cone (xz). Right: reflectance vs. bare silicon. The whole thing runs through the built-in <code>optimize()</code>.</figcaption>
</figure>

This is the same moth-eye that, with only a single-layer `MetaAtom`, you'd have had
to hand-roll with an external optimizer. As a `Structure` it's a **first-class
`optimize()` problem** — so you get `Target` objectives, worst-case aggregation,
multi-objective NSGA-III, and the `OptimizeResult` niceties for free.

## How it plugs in

`optimize()` doesn't care whether you hand it a `MetaAtom` or a `Structure` — it
only calls two methods on the design:

- `variables()` → the free DOF (auto-discovered from your `free(...)` attributes),
- `build(params, n_orders)` → the assembled `RCWA`.

`Structure` implements both from your `define()`. (So could a class of your own —
see the [decision guide](../api/inverse.md#which-construct).)

## Expected results

- `best.report()` lists the optimized free parameters (period, heights, shape
  params); `best.rcwa` is the assembled, ready-to-simulate stack.
- For the moth-eye: a tall, gently-tapered silicon cone that suppresses reflection
  across the band (confirm the *real* breadth with a fine wavelength sweep — see
  [Lesson 4](parameter-sweeps.md) — because a few-point objective can leave bumps
  between its samples).

## Pilot habits

- **Always declare a `period`** (free or fixed) — `Structure` requires it.
- Keep DOF **physical and few**: shared/derived parameters (like the cone's
  `r_base`, `gamma`) converge far faster than one free radius per slice.
- Reuse the [convergence ritual](parameter-sweeps.md#convergence-study): optimize
  at a light `n_orders`, then confirm the winner at a higher one.
- Pin BLAS to one thread for the GA loop ([why](../performance.md#blas-threading)).

---

🎓 **Flight School complete — all eight lessons flown.** Where to next:
[Aerobatics](../advanced.md) · [The Hangar](../examples-gallery.md) ·
[Inverse Design API](../api/inverse.md)
