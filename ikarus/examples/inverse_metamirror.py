"""Inverse-design a metamirror in a few lines (gradient-free).

Optimizes a Si-on-SiO2 metaatom -- a free binary pixel map (c4v-symmetric) plus a
free layer height -- to maximize reflection into the 0th order at 1550 nm, then
reports the design and plots the optimized topology.

Requires the inverse extra:  pip install ikarus-rcwa[inverse]
Run:  python -m ikarus.examples.inverse_metamirror
"""

import numpy as np

from ikarus.inverse import MetaAtom, free, pixels, Target, optimize


def main():
    # 1. define the metaatom and its degrees of freedom
    atom = MetaAtom(period=0.7e-6, cover="Air", substrate="SiO2")
    atom.add_pattern(topology=pixels(12, 12, symmetry="c4v"),
                     materials=["Air", "Si"], height=free(0.3e-6, 0.9e-6))

    # 2. say what you want, 3. optimize -- one line
    best = optimize(atom, Target.maximize("R", order=(0, 0), at=1550e-9),
                    n_orders=8, pop=40, n_gen=30)
    print(best.report())

    # the result is a ready-to-use RCWA structure
    rcwa = best.metaatom
    rcwa.set_source(wavelength=1550e-9, theta=0, polarization="linear")
    _, _, res = rcwa.simulate()
    print(f"\nMetamirror reflectance @1550 nm: R = {res.R_total:.4f}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        rcwa.visualize_structure(plane="xy", layer_index=1,
                                 savefig="metamirror_topology.png")
        print("Saved metamirror_topology.png")
    except Exception as exc:  # pragma: no cover
        print(f"(plot skipped: {exc})")


if __name__ == "__main__":
    main()
