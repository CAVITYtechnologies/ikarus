"""Resonant transmission spectrum of a dielectric-pillar metasurface.

A square lattice of silicon nano-pillars (Mie-resonant metasurface) on silica.
Sweeps wavelength, plots the zero-order transmission spectrum, and saves a field
map at the strongest resonance dip.

Run:  python -m ikarus.examples.metasurface_spectrum
"""

import numpy as np

from ikarus import RCWA, shapes


def main():
    period = 500e-9
    pillar = shapes.circle(center=(0.5, 0.5), radius=0.32, grid_shape=(64, 64))

    rcwa = RCWA(period_x=period, period_y=period, resolution=(64, 64), n_orders=(8, 8))
    rcwa.add_uniform_layer(np.inf, "Air")
    rcwa.add_layer(220e-9, pillar, ["Si", "Air"])
    rcwa.add_uniform_layer(np.inf, "SiO2")

    wavelengths = np.linspace(600e-9, 1000e-9, 41)
    T0 = []
    for wl in wavelengths:
        rcwa.set_source(wavelength=wl, theta=0, polarization="linear")
        _, _, res = rcwa.simulate()
        i0 = res.order_index(0, 0)
        T0.append(res.T_orders[i0])
    T0 = np.array(T0)

    dip = wavelengths[np.argmin(T0)]
    print(f"Si-pillar metasurface (period {period*1e9:.0f} nm, 220 nm tall)")
    print(f"Zero-order transmission minimum at {dip*1e9:.0f} nm (T0={T0.min():.3f})")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
        ax1.plot(wavelengths * 1e9, T0, "-o", ms=3)
        ax1.axvline(dip * 1e9, color="r", ls="--", lw=1)
        ax1.set_xlabel("wavelength (nm)"); ax1.set_ylabel("zero-order T")
        ax1.set_title("Metasurface transmission spectrum")

        rcwa.set_source(wavelength=dip)
        rcwa.simulate()
        fm = rcwa.get_fields(plane="xz", nx=80)["xz"]
        from ikarus.visualization import plot_field
        plot_field(fm, component="intensity", ax=ax2)
        fig.savefig("metasurface_spectrum.png", dpi=150, bbox_inches="tight")
        print("Saved metasurface_spectrum.png")
    except Exception as exc:  # pragma: no cover
        print(f"(plotting skipped: {exc})")


if __name__ == "__main__":
    main()
