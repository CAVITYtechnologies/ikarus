"""Validate Ikarus against analytic Fresnel for a uniform slab.

Sweeps incidence angle for a dielectric slab and compares the RCWA result to the
closed-form characteristic-matrix solution -- the agreement should be at the
level of machine precision.

Run:  python -m ikarus.examples.validation_fresnel
"""

import numpy as np

from ikarus import RCWA
from ikarus.tests.validation.fresnel_reference import fresnel_stack


def main():
    wl = 633e-9
    n_cover, n_film, n_sub = 1.0, 2.2, 1.5
    d_film = 250e-9
    angles = np.linspace(0, 80, 17)

    print(f"Air / n={n_film} ({d_film*1e9:.0f} nm) / glass slab @ {wl*1e9:.0f} nm")
    print(f"{'theta':>6} {'R (ikarus)':>12} {'R (fresnel)':>12} {'|dR|':>10}")
    max_err = 0.0
    for theta in angles:
        rcwa = RCWA(period_x=400e-9, period_y=400e-9, resolution=8, n_orders=3)
        rcwa.add_uniform_layer(np.inf, n_cover)
        rcwa.add_uniform_layer(d_film, n_film)
        rcwa.add_uniform_layer(np.inf, n_sub)
        rcwa.set_source(wavelength=wl, theta=theta, polarization="linear",
                        linear_pol_angle=0.0)  # s-pol
        _, _, res = rcwa.simulate()
        R_ref, _ = fresnel_stack([n_cover, n_film, n_sub], [d_film], wl, theta, "s")
        err = abs(res.R_total - R_ref)
        max_err = max(max_err, err)
        print(f"{theta:6.1f} {res.R_total:12.8f} {R_ref:12.8f} {err:10.1e}")

    print(f"\nMax |dR| over sweep: {max_err:.2e}  (target < 1e-8)")


if __name__ == "__main__":
    main()
