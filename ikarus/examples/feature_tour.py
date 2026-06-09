"""A guided tour of Ikarus's main features.

Builds a TiO2 "cross" antenna metasurface on glass and walks through:
    1. the built-in material database
    2. visualizing the layer stack (xz) and the unit-cell topology (xy)
    3. a simulation: reflectance/transmittance + diffraction orders with angles
    4. real-space field maps (xz cross-section and xy slice)
    5. a wavelength spectrum (R and T vs lambda)
    6. circular-polarization co/cross response
    7. saving results to HDF5

Figures and the HDF5 file are written to an ``ikarus_tour_output/`` folder in the
current directory.

Run:  python -m ikarus.examples.feature_tour
"""

from pathlib import Path

import numpy as np

from ikarus import RCWA, shapes
from ikarus.core.materials import default_library as materials

try:
    import matplotlib
    import matplotlib.pyplot as plt
    from ikarus.visualization import plot_field
    HAVE_MPL = True
except ImportError:  # pragma: no cover
    HAVE_MPL = False


def banner(text):
    print("\n" + "=" * 64 + f"\n  {text}\n" + "=" * 64)


def main():
    out = Path.cwd() / "ikarus_tour_output"
    out.mkdir(exist_ok=True)
    saved = []

    # -- 1. materials ------------------------------------------------------
    banner("1. Built-in material database")
    print("Available:", ", ".join(materials.available()))
    for name in ["Si", "TiO2", "SiO2", "Au"]:
        nk = materials.get(name, 600e-9)
        print(f"  {name:5s} @ 600 nm:  n = {nk.real:.3f}   k = {nk.imag:.3f}")

    # -- 2. build a metasurface and visualize ------------------------------
    banner("2. Build a TiO2 'cross' metasurface and visualize it")
    period = 700e-9
    cross = shapes.cross(center=(0.5, 0.5), arm_length=0.75, arm_width=0.25,
                         grid_shape=(96, 96))

    rcwa = RCWA(period_x=period, period_y=period, resolution=(96, 96),
                n_orders=(8, 8))
    rcwa.add_uniform_layer(np.inf, "Air")
    # topology value 1 (the cross) -> TiO2; value 0 (background) -> Air.
    rcwa.add_layer(160e-9, cross, ["Air", "TiO2"])
    rcwa.add_uniform_layer(np.inf, "SiO2")
    rcwa.set_source(wavelength=600e-9, theta=0, polarization="linear",
                    linear_pol_angle=0)

    if HAVE_MPL:
        ax = rcwa.visualize_structure(plane="xz")
        ax.figure.savefig(out / "1_stack_xz.png", dpi=150, bbox_inches="tight")
        saved.append("1_stack_xz.png")
        ax = rcwa.visualize_structure(plane="xy", layer_index=1)
        ax.figure.savefig(out / "2_topology_xy.png", dpi=150, bbox_inches="tight")
        saved.append("2_topology_xy.png")
        print("Saved structure plots.")
    else:
        print("(matplotlib not installed -> skipping plots; pip install matplotlib)")

    # -- 3. simulate -------------------------------------------------------
    banner("3. Simulate at 600 nm -- efficiencies and diffraction orders")
    _, _, result = rcwa.simulate()
    print(f"R = {result.R_total:.4f}   T = {result.T_total:.4f}   "
          f"R+T = {result.energy_balance:.6f}   (A = {1 - result.energy_balance:.4f})")
    print("Propagating transmitted orders (order : efficiency @ exit angle):")
    p, q = result.orders
    for i in np.argsort(-result.T_orders):
        if result.T_orders[i] > 1e-3:
            print(f"   ({p[i]:+d},{q[i]:+d}) : {result.T_orders[i]:.4f}  "
                  f"@ theta={result.theta_out_trn[i]:5.1f} deg")

    # -- 4. fields ---------------------------------------------------------
    if HAVE_MPL:
        banner("4. Reconstruct and plot the electromagnetic fields")
        xz = rcwa.get_fields(plane="xz", nx=120, y_position=period / 2)["xz"]
        ax = plot_field(xz, component="intensity")
        ax.set_title("|E|^2  (xz cross-section through the cross)")
        ax.figure.savefig(out / "3_field_xz.png", dpi=150, bbox_inches="tight")
        saved.append("3_field_xz.png")

        xy = rcwa.get_fields(z_positions=[80e-9], plane="xy", nx=120, ny=120)
        fm = list(xy.values())[0]
        ax = plot_field(fm, component="intensity")
        ax.set_title("|E|^2  (xy slice inside the cross)")
        ax.figure.savefig(out / "4_field_xy.png", dpi=150, bbox_inches="tight")
        saved.append("4_field_xy.png")
        print("Saved field plots.")

    # -- 5. wavelength spectrum: efficiency AND phase ---------------------
    banner("5. Sweep wavelength -> efficiency and phase spectra")
    wavelengths = np.linspace(450e-9, 750e-9, 31)
    Rs, Ts, Tphase, Rphase = [], [], [], []
    for wl in wavelengths:
        rcwa.set_source(wavelength=wl)
        _, _, res = rcwa.simulate()
        Rs.append(res.R_total)
        Ts.append(res.T_total)
        # phase of the zero-order transmission/reflection coefficients (radians)
        Tphase.append(res.T_phase)
        Rphase.append(res.R_phase)
    print(f"Swept {len(wavelengths)} wavelengths from 450 to 750 nm "
          f"(T rises from {Ts[0]:.2f} to {Ts[-1]:.2f}).")
    print(f"Transmission phase swings {np.degrees(np.ptp(np.unwrap(Tphase))):.0f} deg "
          f"across the band (useful for phase-gradient metasurface design).")

    if HAVE_MPL:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 6.5), sharex=True)
        ax1.plot(wavelengths * 1e9, Rs, "-o", ms=3, label="Reflectance")
        ax1.plot(wavelengths * 1e9, Ts, "-o", ms=3, label="Transmittance")
        ax1.plot(wavelengths * 1e9, np.array(Rs) + np.array(Ts), "k--", lw=1,
                 label="R+T")
        ax1.set_ylabel("efficiency"); ax1.legend(); ax1.grid(alpha=0.3)
        ax1.set_title("TiO2 cross metasurface -- spectrum")

        ax2.plot(wavelengths * 1e9, np.degrees(np.unwrap(Tphase)), "-o", ms=3,
                 color="C2", label="transmission phase")
        ax2.plot(wavelengths * 1e9, np.degrees(np.unwrap(Rphase)), "-o", ms=3,
                 color="C3", label="reflection phase")
        ax2.set_xlabel("wavelength (nm)"); ax2.set_ylabel("phase (deg)")
        ax2.legend(); ax2.grid(alpha=0.3)
        fig.savefig(out / "5_spectrum.png", dpi=150, bbox_inches="tight")
        saved.append("5_spectrum.png")

    # -- 6. circular polarization -----------------------------------------
    banner("6. Circular polarization (co / cross decomposition)")
    rcwa.set_source(wavelength=600e-9, theta=20, polarization="RCP")
    T, R, res = rcwa.simulate()
    print(f"RCP @ 20 deg:  T_co={abs(T['co']):.4f}  T_cross={abs(T['cross']):.4f}"
          f"  |  R_co={abs(R['co']):.4f}  R_cross={abs(R['cross']):.4f}")

    # -- 7. HDF5 -----------------------------------------------------------
    banner("7. Save results to HDF5")
    try:
        h5 = out / "results.h5"
        rcwa.save_results(h5, include=["T", "R", "metadata"], result=res)
        print(f"Wrote {h5}")
    except Exception as exc:  # pragma: no cover
        print(f"(HDF5 skipped: {exc} -- install h5py with: pip install h5py)")

    # ---------------------------------------------------------------------
    print("\n" + "-" * 64)
    if saved:
        print(f"Done! {len(saved)} figures saved in:\n  {out}")
        for s in saved:
            print("   -", s)
        if HAVE_MPL and matplotlib.get_backend().lower() != "agg":
            print("\nShowing the figures (close the windows to exit)...")
            plt.show()
    else:
        print(f"Done! (numeric output only; results in {out})")


if __name__ == "__main__":
    main()
