"""Diffraction efficiencies of a 1-D dielectric grating vs. wavelength.

Computes the order-resolved transmission of a binary TiO2/air grating on glass
and prints the propagating diffraction orders with their exit angles.

Run:  python -m ikarus.examples.grating_diffraction
"""

import numpy as np

from ikarus import RCWA


def main():
    period = 900e-9
    rcwa = RCWA(period_x=period, period_y=period, resolution=(256, 2), n_orders=(20, 0))

    topo = np.zeros((128, 2), dtype=int)
    topo[64:, :] = 1  # 50% duty cycle binary grating

    rcwa.add_uniform_layer(np.inf, "Air")
    rcwa.add_layer(300e-9, topo, ["TiO2", "Air"])
    rcwa.add_uniform_layer(np.inf, "SiO2")

    print(f"TiO2 binary grating, period {period*1e9:.0f} nm, normal incidence (TE)")
    for wl in [450e-9, 550e-9, 650e-9, 750e-9]:
        rcwa.set_source(wavelength=wl, theta=0, polarization="linear",
                        linear_pol_angle=0.0)
        _, _, res = rcwa.simulate()
        print(f"\n  lambda = {wl*1e9:.0f} nm   "
              f"R={res.R_total:.4f}  T={res.T_total:.4f}  R+T={res.energy_balance:.6f}")
        p, q = res.orders
        for i in np.argsort(-res.T_orders):
            if res.T_orders[i] > 1e-4:
                ang = res.theta_out_trn[i]
                print(f"    order ({p[i]:+d},{q[i]:+d}): T={res.T_orders[i]:.4f} "
                      f"at theta_out={ang:.1f} deg")


if __name__ == "__main__":
    main()
