"""Adjoint-vs-GA benchmarks -- the ship-bar from the gradient proposal.

B1  Head-to-head on the same problem (freeform aSi reflector @1550, 32x32 c4v
    pixels): final FOM and wall-clock for both engines, same n_orders, same
    Target. The GA gets a demonstration-sized budget (pop 30 x 25 gen ~ 780
    solves, a few minutes) -- enough to climb visibly; this measures the
    engines' character, not a production GA campaign.
B2  The GA-impossible regime: 64x64 free pixels (4096 binary DOFs). Adjoint
    runs to convergence; the GA gets the *same wall-clock budget* and we report
    what each achieved.
B3  Many-objectives mode: worst-case (min-max) reflector at 1064 AND 1550 nm
    (the bispectral-study formulation) -- adjoint only, reporting per-lambda R.

Run:  python docs/dev/adjoint_benchmarks.py [b1|b2|b3]
Results are printed; the canonical numbers live in the proposal doc.

PROTOCOL NOTE (learned the hard way): both engines OPTIMIZE at M=8 -- a
faithful forward model for these structures -- and the verdict comes from a
converged re-evaluation at higher M. An earlier run optimized at M=6:
the GA then out-optimized the *model* (fitness 1.0153 > 1, unphysical) by
mining truncation artifacts in the unregularized pixel space, and its design
collapsed to R=0.10 at M=12, while the adjoint (shielded by its min-feature
filter) held 0.84. That is a statement about optimizing unconverged models --
the bispectral-study failure mode -- not about the GA, which climbs fine on a
faithful model. Keep the model honest and the engines compare on speed.
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import sys
import time

import numpy as np

from ikarus.inverse import MetaAtom, Target, optimize, pixels

WL = 1550e-9


def _reflector_atom(n_px, symmetry):
    atom = MetaAtom(period=900e-9, cover="Air", substrate="SiO2")
    atom.add_pattern(topology=pixels(n_px, n_px, symmetry=symmetry),
                     materials=["Air", "aSi"], height=500e-9)
    return atom


def _final_R(res, wavelengths=(WL,), n_orders=None):
    """Re-evaluate the *final binary design* with the NumPy core.

    ``n_orders`` above the optimization's own truncation is the honesty check:
    an optimizer can exploit numerical non-convergence of its forward model
    (fitness > physical bounds is the tell), so the verdict must come from a
    converged evaluation of both engines' designs.
    """
    rcwa = res.atom.build(res.params, n_orders or res.n_orders)
    out = []
    for wl in wavelengths:
        rcwa.set_source(wavelength=wl, theta=0.0, polarization="linear",
                        linear_pol_angle=0.0)
        out.append(rcwa.simulate()[2].R_total)
    return out


def b1():
    """The docs-figure toy: 1-D beam deflector, 40 binary DOFs, seconds.

    At this scale the GA legitimately wins (a few hundred evaluations genuinely
    search a 40-bit space); the adjoint's edge appears at freeform scale (B2).
    Honest scoring throughout: binarized designs re-evaluated at M=(20,0),
    twice the optimization truncation.
    """
    print("== B1: toy 1-D deflector, 40 DOFs (the docs-figure problem) ==")
    target = Target.maximize("R", at=WL, order=(1, 0))
    atom = MetaAtom(period=2000e-9, cover="Air", substrate="SiO2")
    atom.add_pattern(topology=pixels(40, 1), materials=["Air", "aSi"],
                     height=500e-9)

    def eff(res, M=(20, 0)):
        rc = res.atom.build(res.params, M)
        rc.set_source(wavelength=WL, theta=0.0, polarization="linear",
                      linear_pol_angle=0.0)
        r = rc.simulate()[2]
        return r.R_orders[r.order_index(1, 0)]

    t0 = time.perf_counter()
    res_a, best = None, -1.0
    for s_ in range(2):
        r = optimize(atom, target, n_orders=(10, 0), algorithm="adjoint",
                     steps=200, init="random", seed=s_, verbose=False)
        v = eff(r)
        if v > best:
            res_a, best = r, v
    t_adj = time.perf_counter() - t0
    print(f"   adjoint (best of 2 starts): R(+1) = {best:.4f}   "
          f"wall = {t_adj:.0f} s", flush=True)

    t0 = time.perf_counter()
    res_g = optimize(atom, target, n_orders=(10, 0), algorithm="ga",
                     pop=30, n_gen=25, verbose=False)
    t_ga = time.perf_counter() - t0
    print(f"   GA (780 solves): R(+1) = {eff(res_g):.4f}   "
          f"wall = {t_ga:.0f} s", flush=True)


def b2():
    """The GA-impossible regime, on a *discriminating* objective.

    A beam deflector -- steer reflected power into the (+1, 0) order -- is the
    canonical adjoint benchmark: random or greedy patterns score near zero, so
    the search actually has to find structure. 64x64 pixels with mirror_y
    symmetry = 2080 binary DOFs for BOTH engines; the GA gets the same
    wall-clock budget as the adjoint run.
    """
    print("== B2: beam deflector, 64x64/mirror_y = 2048 binary DOFs ==")
    target = Target.maximize("R", at=WL, order=(1, 0))
    atom = MetaAtom(period=2000e-9, cover="Air", substrate="SiO2")
    atom.add_pattern(topology=pixels(64, 64, symmetry="mirror_y"),
                     materials=["Air", "aSi"], height=500e-9)
    print(f"   binary DOFs: {atom.pattern['topology'].n_free}")

    def order_eff(res, M=8):
        rc = res.atom.build(res.params, M)
        rc.set_source(wavelength=WL, theta=0.0, polarization="linear",
                      linear_pol_angle=0.0)
        r = rc.simulate()[2]
        return r.R_orders[r.order_index(1, 0)]

    # Deflection landscapes are a plateau at uniform gray: standard practice is
    # random init + a few restarts (the GA gets random init by construction).
    t0 = time.perf_counter()
    res_a, eff_adj = None, -1.0
    for s in range(3):
        r = optimize(atom, target, n_orders=8, algorithm="adjoint",
                     steps=250, min_feature=100e-9, init="random", seed=s,
                     verbose=False)
        e = order_eff(r)
        if e > eff_adj:
            res_a, eff_adj = r, e
    t_adj = time.perf_counter() - t0
    print(f"   adjoint (best of 3 random starts): R(+1,0) = {eff_adj:.4f}   "
          f"wall = {t_adj:.0f} s total (min_feature = 100 nm)", flush=True)

    # GA gets the same wall-clock budget.
    t0 = time.perf_counter()
    optimize(atom, target, n_orders=8, algorithm="ga", pop=8, n_gen=1,
             verbose=False)
    per_eval = (time.perf_counter() - t0) / 16
    n_gen = max(2, int(t_adj / (per_eval * 40)))
    t0 = time.perf_counter()
    res_g = optimize(atom, target, n_orders=8, algorithm="ga",
                     pop=40, n_gen=n_gen, verbose=False)
    t_ga = time.perf_counter() - t0
    print(f"   GA (same budget): R(+1,0) = {order_eff(res_g):.4f}   "
          f"wall = {t_ga:.0f} s (pop 40 x {n_gen} gen)", flush=True)

    print(f"   converged re-check (M=12): adjoint R(+1,0) = "
          f"{order_eff(res_a, 12):.4f}   GA R(+1,0) = "
          f"{order_eff(res_g, 12):.4f}", flush=True)


def b3():
    print("== B3: min-max bispectral reflector (1064 AND 1550 nm) ==")
    from ikarus.inverse import free
    target = Target.maximize("R", at=[1064e-9, WL], order=None,
                             worst_case=True)
    atom = MetaAtom(period=900e-9, cover="Air", substrate="SiO2")
    atom.add_pattern(topology=pixels(40, 40, symmetry="c4v"),
                     materials=["Air", "aSi"],
                     height=free(300e-9, 800e-9))    # mixed pixel + continuous
    t0 = time.perf_counter()
    res = optimize(atom, target, n_orders=8, algorithm="adjoint",
                   steps=300, min_feature=80e-9, verbose=False)
    t_adj = time.perf_counter() - t0
    R1064, R1550 = _final_R(res, wavelengths=(1064e-9, WL))
    print(f"   adjoint: R(1064) = {R1064:.4f}  R(1550) = {R1550:.4f}  "
          f"worst = {min(R1064, R1550):.4f}   wall = {t_adj:.0f} s", flush=True)
    R1064c, R1550c = _final_R(res, wavelengths=(1064e-9, WL), n_orders=12)
    print(f"   converged re-check (M=12): R(1064) = {R1064c:.4f}  "
          f"R(1550) = {R1550c:.4f}", flush=True)


if __name__ == "__main__":
    which = sys.argv[1:] or ["b1", "b2", "b3"]
    for name in which:
        {"b1": b1, "b2": b2, "b3": b3}[name]()
    print("DONE", flush=True)
