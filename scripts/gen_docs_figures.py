"""Generate the figures embedded in the documentation.

Runs Ikarus end-to-end and writes PNGs into ``docs/assets/``. Re-run after any
change that should be reflected in the docs figures::

    python scripts/gen_docs_figures.py

The matrices here are small, so single-threaded BLAS is fastest.
"""

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

from pathlib import Path

import numpy as np
import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt

from ikarus import RCWA, shapes, default_library
from ikarus.visualization import plot_field

ASSETS = Path(__file__).resolve().parent.parent / "docs" / "assets"
ASSETS.mkdir(parents=True, exist_ok=True)

# -- shared style -----------------------------------------------------------
ORANGE = "#f4511e"
AMBER = "#ffb300"
DEEP = "#bf360c"
BLUE = "#1565c0"
plt.rcParams.update({
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
    "axes.facecolor": "white",
    "font.size": 11,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 130,
})


def save(fig, name):
    # 220 dpi: crisp on hi-dpi displays at the widths the docs use.  Names
    # ending in .svg are written as true vector graphics.
    fig.tight_layout()
    fig.savefig(ASSETS / name, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print("  wrote", name)


# -- 1. thin-film spectrum (Lesson 1) ---------------------------------------
def thin_film_spectrum():
    rcwa = RCWA(period_x=400e-9, period_y=400e-9, n_orders=0)
    rcwa.add_uniform_layer(np.inf, "Air")
    rcwa.add_uniform_layer(120e-9, "TiO2")
    rcwa.add_uniform_layer(np.inf, "SiO2")
    wl = np.linspace(400e-9, 700e-9, 151)
    R, T = [], []
    for w in wl:
        rcwa.set_source(wavelength=w, theta=0, polarization="linear")
        _, _, res = rcwa.simulate()
        R.append(res.R_total)
        T.append(res.T_total)
    R, T = np.array(R), np.array(T)
    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.plot(wl * 1e9, T * 100, color=ORANGE, lw=2.2, label="Transmittance")
    ax.plot(wl * 1e9, R * 100, color=BLUE, lw=2.2, label="Reflectance")
    ax.plot(wl * 1e9, (R + T) * 100, "--", color="0.5", lw=1.2, label="R + T")
    ax.set_xlabel("wavelength (nm)")
    ax.set_ylabel("efficiency (%)")
    ax.set_title("120 nm TiO₂ on glass — thin-film interference")
    ax.legend(frameon=False, ncol=3, loc="lower center")
    ax.set_ylim(0, 105)
    save(fig, "thin_film_spectrum.png")


# -- 2. grating diffraction orders (Lesson 2) -------------------------------
def grating_orders():
    period = 900e-9
    rcwa = RCWA(period_x=period, period_y=period, resolution=(256, 2),
                n_orders=(20, 0))
    topo = np.zeros((128, 2), dtype=int)
    topo[64:, :] = 1
    rcwa.add_uniform_layer(np.inf, "Air")
    rcwa.add_layer(300e-9, topo, ["TiO2", "Air"])
    rcwa.add_uniform_layer(np.inf, "SiO2")
    rcwa.set_source(wavelength=650e-9, theta=0, polarization="linear",
                    linear_pol_angle=0.0)
    _, _, res = rcwa.simulate()
    p, q = res.orders
    keep = [m for m in range(-3, 4)]
    effs, labels, angles = [], [], []
    for m in keep:
        i = res.order_index(m, 0)
        effs.append(res.T_orders[i])
        labels.append(f"{m:+d}")
        angles.append(res.theta_out_trn[i])
    fig, ax = plt.subplots(figsize=(7, 3.8))
    bars = ax.bar(labels, np.array(effs) * 100, color=AMBER, edgecolor=DEEP, width=0.6)
    for b, a in zip(bars, angles):
        if np.isfinite(a):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1,
                    f"{a:.0f}°", ha="center", va="bottom", fontsize=9, color=DEEP)
    ax.set_xlabel("transmitted diffraction order $m$")
    ax.set_ylabel("efficiency (%)")
    ax.set_title("TiO₂ grating @ 650 nm — power per exit lane (exit angle above)")
    ax.set_ylim(0, max(effs) * 100 * 1.2)
    ax.grid(axis="x")
    save(fig, "grating_orders.png")


# -- 3a. metasurface phase library (Lesson 3) -------------------------------
def metasurface_phase():
    period, N = 420e-9, 96
    radii = np.linspace(0.12, 0.46, 22)
    T, phase = [], []
    for r in radii:
        pillar = shapes.circle(radius=r, grid_shape=(N, N))
        rcwa = RCWA(period_x=period, period_y=period, resolution=(N, N),
                    n_orders=(9, 9))
        rcwa.add_uniform_layer(np.inf, "Air")
        rcwa.add_layer(600e-9, pillar, ["Air", "TiO2"])
        rcwa.add_uniform_layer(np.inf, "SiO2")
        rcwa.set_source(wavelength=532e-9, theta=0, polarization="linear")
        _, _, res = rcwa.simulate()
        T.append(res.T_total)
        phase.append(res.T_phase)
    phase = np.unwrap(phase)
    phase -= phase[0]
    fig, ax1 = plt.subplots(figsize=(7, 3.8))
    ax2 = ax1.twinx()
    ax2.grid(False)
    l1, = ax1.plot(radii, np.array(T) * 100, color=ORANGE, lw=2.2, label="transmittance")
    l2, = ax2.plot(radii, np.degrees(phase), color=BLUE, lw=2.2, label="phase")
    ax1.set_xlabel("pillar radius (fraction of period)")
    ax1.set_ylabel("transmittance (%)", color=ORANGE)
    ax2.set_ylabel("transmission phase (deg)", color=BLUE)
    ax1.tick_params(axis="y", colors=ORANGE)
    ax2.tick_params(axis="y", colors=BLUE)
    ax1.set_title("TiO₂ nanopillar @ 532 nm — the metalens phase library")
    ax1.legend([l1, l2], ["transmittance", "phase"], frameon=False, loc="center left")
    save(fig, "metasurface_phase.png")


# -- 3b. metasurface near field (Lesson 3 + hero) ---------------------------
def metasurface_field():
    period, N = 420e-9, 96
    pillar = shapes.circle(radius=0.32, grid_shape=(N, N))
    rcwa = RCWA(period_x=period, period_y=period, resolution=(N, N), n_orders=(10, 10))
    rcwa.add_uniform_layer(np.inf, "Air")
    rcwa.add_layer(600e-9, pillar, ["Air", "TiO2"])
    rcwa.add_uniform_layer(np.inf, "SiO2")
    rcwa.set_source(wavelength=532e-9, theta=0, polarization="linear")
    rcwa.simulate()
    xz = rcwa.get_fields(plane="xz", nx=200, y_position=period / 2)["xz"]
    ax = plot_field(xz, component="intensity")
    ax.set_title("|E|² inside a TiO₂ nanopillar (xz cross-section)")
    save(ax.figure, "metasurface_field.png")


# -- 4. parameter sweep map (Lesson 4) --------------------------------------
def sweep_map():
    period, N = 450e-9, 64
    disk = shapes.circle(radius=0.3, grid_shape=(N, N))
    wavelengths = np.linspace(400e-9, 800e-9, 44)
    heights = np.linspace(100e-9, 400e-9, 30)
    Rmap = np.empty((heights.size, wavelengths.size))
    for j, h in enumerate(heights):
        rcwa = RCWA(period_x=period, period_y=period, resolution=(N, N), n_orders=(7, 7))
        rcwa.add_uniform_layer(np.inf, "Air")
        rcwa.add_layer(h, disk, ["Air", "Si3N4"])
        rcwa.add_uniform_layer(np.inf, "SiO2")
        for i, w in enumerate(wavelengths):
            rcwa.set_source(wavelength=w, theta=0, polarization="linear")
            Rmap[j, i] = rcwa.simulate()[2].R_total
    fig, ax = plt.subplots(figsize=(7, 4.2))
    im = ax.pcolormesh(wavelengths * 1e9, heights * 1e9, Rmap * 100,
                       shading="auto", cmap="inferno")
    ax.set_xlabel("wavelength (nm)")
    ax.set_ylabel("pillar height (nm)")
    ax.set_title("Si₃N₄ disk — reflectance over (wavelength, height)")
    ax.grid(False)
    fig.colorbar(im, ax=ax, label="reflectance (%)")
    save(fig, "sweep_map.png")


# -- 5. polarization (Lesson 5) ---------------------------------------------
def polarization():
    period, N = 500e-9, 80
    bar = shapes.rectangle(center=(0.5, 0.5), size=(0.7, 0.25), grid_shape=(N, N))
    rcwa = RCWA(period_x=period, period_y=period, resolution=(N, N), n_orders=(8, 8))
    rcwa.add_uniform_layer(np.inf, "Air")
    rcwa.add_layer(220e-9, bar, ["Air", "Si"])
    rcwa.add_uniform_layer(np.inf, "SiO2")
    angles = np.linspace(0, 180, 37)
    T = []
    for psi in angles:
        rcwa.set_source(wavelength=700e-9, theta=0, polarization="linear",
                        linear_pol_angle=psi)
        T.append(rcwa.simulate()[2].T_total)
    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.plot(angles, np.array(T) * 100, color=ORANGE, lw=2.4)
    ax.axvline(0, color="0.7", lw=1, ls=":")
    ax.axvline(90, color="0.7", lw=1, ls=":")
    ax.set_xlabel("linear polarization angle (deg)   [0 = along bar, 90 = across]")
    ax.set_ylabel("transmittance (%)")
    ax.set_title("Si nanobar @ 700 nm — form birefringence")
    ax.set_xticks([0, 45, 90, 135, 180])
    save(fig, "polarization.png")


# -- 6. angular dispersion map (Lesson 6) -----------------------------------
def dispersion_map():
    period, N = 500e-9, 64
    disk = shapes.circle(radius=0.3, grid_shape=(N, N))
    rcwa = RCWA(period_x=period, period_y=period, resolution=(N, N), n_orders=(8, 8))
    rcwa.add_uniform_layer(np.inf, "Air")
    rcwa.add_layer(180e-9, disk, ["Air", "TiO2"])
    rcwa.add_uniform_layer(np.inf, "SiO2")
    wavelengths = np.linspace(450e-9, 750e-9, 50)
    thetas = np.linspace(0, 50, 34)
    Rmap = np.empty((thetas.size, wavelengths.size))
    for j, th in enumerate(thetas):
        for i, w in enumerate(wavelengths):
            rcwa.set_source(wavelength=w, theta=th, polarization="linear")
            Rmap[j, i] = rcwa.simulate()[2].R_total
    fig, ax = plt.subplots(figsize=(7, 4.2))
    im = ax.pcolormesh(wavelengths * 1e9, thetas, Rmap * 100,
                       shading="auto", cmap="magma")
    ax.set_xlabel("wavelength (nm)")
    ax.set_ylabel("incidence angle (deg)")
    ax.set_title("TiO₂ metasurface — angle–wavelength dispersion")
    ax.grid(False)
    fig.colorbar(im, ax=ax, label="reflectance (%)")
    save(fig, "dispersion_map.png")


# -- 7. AR coating before/after (gallery + inverse + home) ------------------
def ar_coating():
    lib = default_library
    period, N = 180e-9, 64
    # A representative subwavelength Si3N4 moth-eye (a small central pillar).
    topo = shapes.circle(center=(0.5, 0.5), radius=0.22, grid_shape=(N, N))
    rcwa = RCWA(period_x=period, period_y=period, resolution=(N, N), n_orders=(7, 7))
    rcwa.add_uniform_layer(np.inf, "Air")
    rcwa.add_layer(95e-9, topo, ["Air", "Si3N4"])
    rcwa.add_uniform_layer(np.inf, "SiO2")
    wl = np.linspace(300e-9, 600e-9, 61)
    R = []
    for w in wl:
        rcwa.set_source(wavelength=w, theta=0, polarization="linear")
        R.append(rcwa.simulate()[2].R_total)
    nsub = np.array([complex(lib.get("SiO2", w)).real for w in wl])
    R_bare = ((nsub - 1) / (nsub + 1)) ** 2
    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.plot(wl * 1e9, R_bare * 100, "--", color="0.55", lw=1.8, label="bare glass")
    ax.plot(wl * 1e9, np.array(R) * 100, color=ORANGE, lw=2.4,
            label="Si₃N₄ moth-eye AR")
    ax.fill_between(wl * 1e9, np.array(R) * 100, R_bare * 100,
                    color=AMBER, alpha=0.15)
    ax.set_xlabel("wavelength (nm)")
    ax.set_ylabel("reflectance (%)")
    ax.set_title("Inverse-designed anti-reflection coating on glass")
    ax.legend(frameon=False)
    ax.set_ylim(0, 5)
    save(fig, "ar_coating.png")


# -- 8. convergence (Performance) -------------------------------------------
def convergence():
    from ikarus.tools.convergence import convergence_curve
    period, N = 820e-9, 128
    sq = shapes.rectangle(center=(0.5, 0.5), size=(0.5, 0.5), grid_shape=(N, N))
    rcwa = RCWA(period_x=period, period_y=period, resolution=(N, N), n_orders=(6, 6))
    rcwa.add_uniform_layer(np.inf, "Air")
    rcwa.add_layer(300e-9, sq, ["Air", "aSi"])
    rcwa.add_uniform_layer(np.inf, "SiO2")
    rcwa.set_source(wavelength=700e-9, theta=0, polarization="linear", linear_pol_angle=90)
    orders = np.arange(3, 15)
    Ms, defect = convergence_curve(rcwa, orders, metric="energy")
    harmonics = (2 * Ms + 1) ** 2
    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.semilogy(harmonics, defect + 1e-12, "o-", color=ORANGE, lw=2)
    ax.set_xlabel("number of Fourier harmonics  $P=(2M+1)^2$")
    ax.set_ylabel("energy defect  |R + T − 1|")
    ax.set_title("Convergence — aSi square, TM (the slow case)")
    ax.grid(True, which="both", alpha=0.25)
    save(fig, "convergence.png")


# -- 9. shape gallery (Shapes / Core Concepts) ------------------------------
def shape_gallery():
    N = 200
    items = [
        ("circle", shapes.circle(radius=0.3, grid_shape=(N, N))),
        ("ellipse(45°)", shapes.ellipse(radii=(0.35, 0.16), angle=45, grid_shape=(N, N))),
        ("rectangle", shapes.rectangle(size=(0.55, 0.3), grid_shape=(N, N))),
        ("ring", shapes.ring(inner_radius=0.2, outer_radius=0.36, grid_shape=(N, N))),
        ("cross", shapes.cross(arm_length=0.7, arm_width=0.22, grid_shape=(N, N))),
        ("polygon (hex)", shapes.polygon(
            [(0.5, 0.85), (0.8, 0.67), (0.8, 0.33), (0.5, 0.15), (0.2, 0.33), (0.2, 0.67)],
            grid_shape=(N, N))),
    ]
    from matplotlib.colors import ListedColormap
    cmap = ListedColormap(["#fff3e0", DEEP])
    fig, axes = plt.subplots(2, 3, figsize=(7.5, 5))
    for ax, (name, arr) in zip(axes.ravel(), items):
        ax.imshow(arr.T, origin="lower", cmap=cmap, extent=[0, 1, 0, 1])
        ax.set_title(name, fontsize=11)
        ax.set_xticks([]); ax.set_yticks([])
        ax.grid(False)
        for s in ax.spines.values():
            s.set_visible(True); s.set_color("0.7")
    fig.suptitle("ikarus.shapes — topology primitives", y=0.99)
    save(fig, "shape_gallery.png")


# -- 10. parametric-shape inverse design (Lesson 7) -------------------------
def lesson7_parametric():
    from ikarus.inverse import MetaAtom, free, optimize, Target
    from ikarus.shapes import Cross
    from matplotlib.colors import ListedColormap

    atom = MetaAtom(period=700e-9, cover="Air", substrate="SiO2")
    atom.add_pattern(topology=Cross(arm_length=free(0.3, 0.95),
                                    arm_width=free(0.1, 0.45),
                                    angle=free(0, 90), grid_shape=(96, 96)),
                     materials=["Air", "Si"], height=free(0.3e-6, 0.9e-6))
    best = optimize(atom, Target.maximize("T", at=1300e-9),
                    n_orders=6, pop=16, n_gen=10, seed=0, verbose=False)
    p = best.params
    rcwa = best.metaatom

    wl = np.linspace(1.0e-6, 1.6e-6, 61)
    T, R = [], []
    for w in wl:
        rcwa.set_source(wavelength=w, theta=0, polarization="linear")
        _, _, res = rcwa.simulate()
        T.append(res.T_total)
        R.append(res.R_total)

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(9, 3.8),
                                   gridspec_kw={"width_ratios": [1, 1.5]})
    # optimized topology
    topo = rcwa.layers[1].topology
    axL.imshow(topo.T, origin="lower", extent=[0, 1, 0, 1],
               cmap=ListedColormap(["#fff3e0", DEEP]))
    axL.set_xticks([]); axL.set_yticks([]); axL.grid(False)
    axL.set_title("optimized Si cross", fontsize=11)
    axL.set_xlabel(f"arms {p['shape__arm_length']:.2f}×{p['shape__arm_width']:.2f}, "
                   f"{p['shape__angle']:.0f}°, h={p['height']*1e9:.0f} nm", fontsize=8.5)
    # spectrum
    axR.plot(wl * 1e9, np.array(T) * 100, color=ORANGE, lw=2.2, label="Transmittance")
    axR.plot(wl * 1e9, np.array(R) * 100, color=BLUE, lw=2.0, label="Reflectance")
    axR.axvline(1300, color="0.6", ls=":", lw=1.2)
    axR.set_xlabel("wavelength (nm)"); axR.set_ylabel("efficiency (%)")
    axR.set_title("response of the evolved meta-atom")
    axR.legend(frameon=False, loc="center left")
    axR.set_ylim(0, 105)
    fig.suptitle("Lesson 7 — inverse design over a shape class's own parameters", y=1.02)
    save(fig, "lesson7_inverse_shape.png")


# -- 11b. two-layer dielectric AR stack (Lesson 8 hero) ---------------------
def structure_arstack():
    from ikarus.inverse import Structure, free, optimize, Target
    from ikarus.shapes import Circle

    class ARStack(Structure):
        cover, substrate, resolution = "Air", "SiO2", 64
        period = free(0.20e-6, 0.40e-6)
        h1 = free(0.05e-6, 0.30e-6)
        h2 = free(0.05e-6, 0.30e-6)
        r1 = free(0.10, 0.48)
        r2 = free(0.10, 0.48)

        def define(self, p):
            self.add_layer(p.h1, Circle(radius=p.r1), ["Air", "Si3N4"])
            self.add_layer(p.h2, Circle(radius=p.r2), ["Air", "TiO2"])

    best = optimize(ARStack(), Target.minimize("R", at=600e-9),
                    n_orders=6, pop=10, n_gen=6, seed=0, verbose=False)
    rcwa = best.rcwa
    wl = np.linspace(450e-9, 750e-9, 31)
    R, T = [], []
    for w in wl:
        rcwa.set_source(wavelength=w, theta=0, polarization="linear")
        res = rcwa.simulate()[2]
        R.append(res.R_total); T.append(res.T_total)
    R, T = np.array(R), np.array(T)
    nsub = np.array([complex(default_library.get("SiO2", w)).real for w in wl])
    R_bare = ((nsub - 1) / (nsub + 1)) ** 2

    rcwa.set_source(wavelength=600e-9, theta=0, polarization="linear"); rcwa.simulate()
    eps = rcwa.get_fields(plane="xz", nx=150, y_position=best.params["period"] / 2)["xz"].eps

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(10, 3.8),
                                   gridspec_kw={"width_ratios": [1, 1.5]})
    axL.imshow(eps.real.T, origin="upper", aspect="auto", cmap="viridis")
    axL.set_xticks([]); axL.set_yticks([]); axL.grid(False)
    axL.set_title("optimized 2-layer stack (xz)", fontsize=11)
    axR.plot(wl * 1e9, R_bare * 100, ":", color="0.55", lw=1.6, label="bare glass R")
    axR.plot(wl * 1e9, R * 100, "-", color="#f4511e", lw=2.4, label="R (AR stack)")
    axR.plot(wl * 1e9, T * 100, "-", color="#1565c0", lw=2.0, label="T")
    axR.plot(wl * 1e9, (R + T) * 100, "--", color="0.4", lw=1.2, label="R + T")
    axR.set_xlabel("wavelength (nm)"); axR.set_ylabel("efficiency (%)")
    axR.set_ylim(0, 105); axR.legend(frameon=False, fontsize=8.5, loc="center right")
    axR.grid(alpha=0.3); axR.set_title("lossless dielectrics → R + T = 100%")
    fig.suptitle("Lesson 8 — two-layer AR stack optimized as one Structure", y=1.02)
    save(fig, "structure_arstack.png")


# -- 11. multi-layer Structure moth-eye (Lesson 8 advanced) -----------------
def structure_motheye():
    from ikarus.inverse import Structure, free, optimize, Target
    from ikarus.shapes import Circle
    from matplotlib.colors import ListedColormap

    class MothEye(Structure):
        cover, substrate, resolution = "Air", "Si", 72
        N = 8
        period = free(150e-9, 240e-9)
        height = free(200e-9, 1000e-9)
        r_base = free(0.15, 0.5)
        gamma = free(0.5, 3.0)

        def define(self, p):
            for i in range(self.N):
                r = p.r_base * ((i + 0.5) / self.N) ** p.gamma
                self.add_layer(p.height / self.N, Circle(radius=r), ["Air", "Si"])

    best = optimize(MothEye(), Target.minimize("R", band=(300e-9, 600e-9, 4), worst_case=True),
                    n_orders=6, pop=8, n_gen=5, seed=0, verbose=False)
    rcwa = best.rcwa
    rcwa.set_source(wavelength=400e-9, theta=0, polarization="linear"); rcwa.simulate()
    eps = rcwa.get_fields(plane="xz", nx=140, y_position=best.params["period"] / 2)["xz"].eps

    wl = np.linspace(300e-9, 600e-9, 25)
    R = []
    for w in wl:
        rcwa.set_source(wavelength=w, theta=0, polarization="linear")
        R.append(rcwa.simulate()[2].R_total)
    nsub = np.array([complex(default_library.get("Si", w)).real for w in wl])
    R_bare = ((nsub - 1) / (nsub + 1)) ** 2

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(9, 3.8),
                                   gridspec_kw={"width_ratios": [1, 1.6]})
    axL.imshow(eps.real.T, origin="upper", aspect="auto", cmap=ListedColormap(["#fff3e0", DEEP]))
    axL.set_xticks([]); axL.set_yticks([]); axL.grid(False)
    axL.set_title("optimized cone (xz)", fontsize=11)
    axR.plot(wl * 1e9, R_bare * 100, "--", color="0.5", lw=1.8, label="bare silicon")
    axR.plot(wl * 1e9, np.array(R) * 100, "-", color=ORANGE, lw=2.4, label="moth-eye Structure")
    axR.set_xlabel("wavelength (nm)"); axR.set_ylabel("reflectance (%)")
    axR.set_ylim(bottom=0); axR.legend(frameon=False); axR.grid(alpha=0.3)
    axR.set_title("optimized as one Structure via optimize()")
    fig.suptitle("Lesson 8 — a moth-eye optimized as one Structure (8 slices, 4 shared parameters)",
                 y=1.02)
    save(fig, "structure_motheye.png")


# -- 12. home-page hero: structure -> spectrum -> field ----------------------
def hero_showcase():
    """The landing-page hero: one structure, told three ways -- what you build
    (3-D pillar array), what Ikarus computes (spectrum), what light does inside
    (field cross-section). All three panels are the SAME simulation."""
    period, N, M = 750e-9, 96, 8
    radius, height = 0.28, 350e-9
    disk = shapes.circle(center=(0.5, 0.5), radius=radius, grid_shape=(N, N))

    rcwa = RCWA(period_x=period, period_y=period, resolution=(N, N), n_orders=(M, M))
    rcwa.add_uniform_layer(np.inf, "Air")
    rcwa.add_layer(height, disk, ["Air", "aSi"])
    rcwa.add_uniform_layer(np.inf, "SiO2")

    wl = np.linspace(1050e-9, 1650e-9, 61)
    R, T = [], []
    for w in wl:
        rcwa.set_source(wavelength=w, theta=0, polarization="linear")
        _, _, res = rcwa.simulate()
        R.append(res.R_total); T.append(res.T_total)
    R, T = np.array(R), np.array(T)
    wl_res = wl[int(np.argmax(R))]                      # strongest resonance

    fig = plt.figure(figsize=(13.4, 4.1))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.05, 1.25, 1.0], wspace=0.25)

    # (a) the structure -- 3-D pillar array on a substrate slab.  matplotlib's
    # 3-D z-sorting is per-artist and unreliable, so we disable it and paint the
    # pillars ourselves, farthest-from-camera first with increasing zorder.
    AZIM, ELEV = -62, 24
    ax3 = fig.add_subplot(gs[0], projection="3d")
    ax3.computed_zorder = False
    xx, yy = np.meshgrid(np.linspace(0, 3, 2), np.linspace(0, 3, 2))
    ax3.plot_surface(xx, yy, np.zeros_like(xx), color="#ffe0b2", alpha=1.0,
                     shade=False, zorder=0)
    theta = np.linspace(0, 2 * np.pi, 40)
    zc = np.linspace(0, 0.7, 2)
    th, zz = np.meshgrid(theta, zc)
    cam = np.deg2rad(AZIM)
    centers = sorted(
        ((cx, cy) for cx in (0.5, 1.5, 2.5) for cy in (0.5, 1.5, 2.5)),
        key=lambda c: c[0] * np.cos(cam) + c[1] * np.sin(cam))   # farthest first
    for i, (cx, cy) in enumerate(centers):
        z0 = 1 + 2 * i                                            # painter's order
        ax3.plot_surface(cx + radius * np.cos(th), cy + radius * np.sin(th),
                         zz, color=ORANGE, shade=True,
                         linewidth=0, antialiased=True, zorder=z0)
        disk_xy = np.c_[cx + radius * np.cos(theta), cy + radius * np.sin(theta)]
        top = mpl.collections.PolyCollection(
            [disk_xy], facecolors="#ffab91", edgecolors="#d84315",
            linewidths=0.7, zorder=z0 + 1)
        ax3.add_collection3d(top, zs=0.7, zdir="z")
    ax3.set_box_aspect((1, 1, 0.55))
    ax3.set_zlim(0, 1.15); ax3.set_xlim(0, 3); ax3.set_ylim(0, 3)
    ax3.view_init(elev=ELEV, azim=AZIM)
    ax3.set_axis_off(); ax3.grid(False)
    ax3.set_title("the structure you build\n(aSi pillars on glass)", fontsize=11)

    # (b) the spectrum Ikarus computes
    axs = fig.add_subplot(gs[1])
    axs.plot(wl * 1e9, R, color=ORANGE, lw=2.4, label="R")
    axs.plot(wl * 1e9, T, color=BLUE, lw=1.8, alpha=0.75, label="T")
    axs.axvline(wl_res * 1e9, color="0.6", lw=1.0, ls=":")
    axs.annotate("resonance", (wl_res * 1e9, 0.5), textcoords="offset points",
                 xytext=(-9, 0), fontsize=9, color="0.4", rotation=90,
                 ha="center", va="center")
    axs.set_xlabel("wavelength (nm)"); axs.set_ylabel("efficiency")
    axs.set_ylim(0, 1.04)
    axs.legend(frameon=False, ncol=2, loc="upper center",
               bbox_to_anchor=(0.5, -0.20), columnspacing=2.5)
    axs.set_title("the physics Ikarus computes", fontsize=11)

    # (c) the field inside, at the resonance (light enters from the top)
    axf = fig.add_subplot(gs[2])
    rcwa.set_source(wavelength=wl_res, theta=0, polarization="linear")
    rcwa.simulate()
    xz = rcwa.get_fields(plane="xz", nx=180, y_position=period / 2)["xz"]
    plot_field(xz, component="intensity", ax=axf)
    axf.set_title(f"…and what light does inside\n(|E|² at {wl_res*1e9:.0f} nm, xz)",
                  fontsize=11)
    save(fig, "hero_showcase.png")


# -- 13. theory: the RCWA pipeline schematic ---------------------------------
def theory_pipeline():
    """Schematic of the method: periodic layers, one incident wave, discrete
    reflected/transmitted orders, one eigensolve per layer, S-matrix cascade."""
    fig, ax = plt.subplots(figsize=(10.6, 4.6))
    ax.set_xlim(0, 10); ax.set_ylim(0, 6); ax.set_axis_off(); ax.grid(False)

    # layer bands
    ax.add_patch(plt.Rectangle((0.6, 3.9), 6.2, 1.9, color="#e3f2fd"))       # cover
    ax.add_patch(plt.Rectangle((0.6, 2.6), 6.2, 1.3, color="#fff3e0"))       # layer
    ax.add_patch(plt.Rectangle((0.6, 0.4), 6.2, 2.2, color="#ffe0b2"))       # substrate
    for cx in np.arange(1.0, 6.6, 0.9):                                       # pillars
        ax.add_patch(plt.Rectangle((cx, 2.6), 0.45, 1.3, color=ORANGE))
    ax.text(6.6, 5.45, "cover (air)", ha="right", fontsize=10, color="#1565c0")
    ax.text(0.8, 2.75, "patterned layer(s)", fontsize=10, color=DEEP,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.75, pad=1.5))
    ax.text(6.6, 0.6, "substrate", ha="right", fontsize=10, color="#8d6e63")

    # incident + diffracted orders
    ax.annotate("", xy=(2.5, 3.95), xytext=(1.6, 5.75),
                arrowprops=dict(arrowstyle="-|>", lw=2.6, color="0.25"))
    ax.text(1.25, 5.62, "one incident\nplane wave", fontsize=9.5, ha="center",
            va="top", color="0.25")
    for dx, m in ((-0.1, "-1"), (0.9, "0"), (1.9, "+1")):
        ax.annotate("", xy=(2.5 + dx + 0.9, 5.75), xytext=(2.5, 3.95),
                    arrowprops=dict(arrowstyle="-|>", lw=1.8, color=ORANGE))
        ax.text(2.5 + dx + 1.0, 5.55, m, fontsize=9, color=ORANGE)
    for dx, m in ((-0.9, "-1"), (0.0, "0"), (0.9, "+1")):
        ax.annotate("", xy=(2.9 + dx + 0.5 * (dx != 0) * np.sign(dx), 0.55),
                    xytext=(2.9, 2.55),
                    arrowprops=dict(arrowstyle="-|>", lw=1.8, color=BLUE))
        ax.text(2.95 + dx + 0.5 * (dx != 0) * np.sign(dx), 0.28, m,
                fontsize=9, color=BLUE)
    ax.text(4.9, 4.95, "reflected orders", fontsize=9.5, color=ORANGE)
    ax.text(4.5, 1.15, "transmitted orders", fontsize=9.5, color=BLUE)

    # the algorithm, as boxes on the right
    steps = [
        (4.85, "periodic ε(x, y) → Fourier series\n(a convolution matrix per layer)"),
        (3.45, "one eigensolve per layer\n(its natural modes + phases)"),
        (2.05, "layers joined by S-matrices\n(Redheffer ⋆ — never overflows)"),
        (0.65, "amplitudes → R, T, phase,\nfields, per-order efficiencies"),
    ]
    for y, txt in steps:
        ax.add_patch(mpl.patches.FancyBboxPatch(
            (7.35, y), 2.45, 1.0, boxstyle="round,pad=0.08",
            facecolor="white", edgecolor=ORANGE, linewidth=1.4))
        ax.text(8.575, y + 0.5, txt, ha="center", va="center", fontsize=8.6)
    for y in (4.85, 3.45, 2.05):
        ax.annotate("", xy=(8.575, y - 0.3), xytext=(8.575, y - 0.1),
                    arrowprops=dict(arrowstyle="-|>", lw=1.4, color="0.45"))
    save(fig, "theory_pipeline.svg")     # pure schematic -> true vector graphics


# -- 14. theory: why factorization matters (Gibbs + convergence race) --------
def theory_factorization():
    """(a) a truncated Fourier series rings at permittivity jumps; (b) the race:
    Laurent vs Li vs the normal-vector method on a high-contrast cylinder (the
    exact case validated against FMMax; the default converges by n_orders ~ 8)."""
    # (a) Gibbs ringing of the direct series
    x = np.linspace(0, 1, 2000, endpoint=False)
    eps = np.where((x > 0.25) & (x < 0.75), 12.25, 1.0)
    Mg = 10
    c = np.fft.fft(eps) / eps.size
    recon = np.real(sum(c[m] * np.exp(2j * np.pi * m * x)
                        for m in range(-Mg, Mg + 1)))

    # (b) convergence race on the FMMax-validated cylinder
    N = 128
    mask = shapes.circle(center=(0.5, 0.5), radius=0.30, grid_shape=(N, N))
    Ms = [4, 6, 8, 10, 12]
    curves = {}
    for fac in ("laurent", "li", "auto"):
        vals = []
        for M in Ms:
            rc = RCWA(period_x=400e-9, period_y=400e-9, resolution=(N, N),
                      n_orders=(M, M), factorization=fac)
            rc.add_uniform_layer(np.inf, "Air")
            rc.add_layer(200e-9, mask.astype(int), [1.0, 3.5])
            rc.add_uniform_layer(np.inf, "Air")
            rc.set_source(wavelength=700e-9, theta=0, polarization="linear",
                          linear_pol_angle=0)
            vals.append(rc.simulate()[2].R_total)
        curves[fac] = np.array(vals)

    fig, (axg, axr) = plt.subplots(1, 2, figsize=(11.2, 4.0))
    axg.plot(x, eps, color="0.3", lw=2.0, label="true ε(x)")
    axg.plot(x, recon, color=ORANGE, lw=1.6,
             label=f"direct Fourier series (M={Mg})")
    axg.set_xlabel("x / period"); axg.set_ylabel("permittivity ε")
    axg.legend(frameon=False, loc="upper left", fontsize=9)
    axg.set_title("the problem: truncated series ring at jumps", fontsize=11)

    styles = {"laurent": ("0.55", "--", "Laurent (direct rule)"),
              "li": (BLUE, "-.", "Li two-step (axis-aligned exact)"),
              "auto": (ORANGE, "-", "normal-vector — the default")}
    for fac, (color, ls, label) in styles.items():
        axr.plot(Ms, curves[fac], ls, color=color, lw=2.2, marker="o",
                 ms=5, label=label)
    axr.axhline(0.9397, color="0.75", lw=1.0, ls=":")
    axr.text(4.1, 0.9445, "converged value (cross-checked vs FMMax)",
             fontsize=8.5, color="0.45")
    axr.set_xlabel("n_orders  M"); axr.set_ylabel("reflectance R")
    axr.set_xticks(Ms)
    axr.legend(frameon=False, fontsize=9, loc="lower right")
    axr.set_title("the fix, measured: high-contrast cylinder", fontsize=11)
    save(fig, "theory_factorization.png")


# -- 15. theory: the normal-vector tangent field ------------------------------
def theory_tangent_field():
    """The boundary-following tangent field of the normal-vector method,
    computed by ikarus.core._normalvector.tangent_field on real masks."""
    from matplotlib.colors import ListedColormap
    from ikarus.core._normalvector import tangent_field

    N, step = 128, 9
    masks = {
        "cylinder": shapes.circle(center=(0.5, 0.5), radius=0.30, grid_shape=(N, N)),
        "ring": shapes.ring(inner_radius=0.16, outer_radius=0.34, grid_shape=(N, N)),
    }
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.6))
    for ax, (name, mask) in zip(axes, masks.items()):
        ge = np.where(np.asarray(mask) > 0, 12.25, 1.0).astype(complex)
        tx, ty = tangent_field(ge)
        ax.imshow(np.asarray(mask).T, origin="lower",
                  cmap=ListedColormap(["#fff8f0", "#ffcc80"]))
        ii = np.arange(step // 2, N, step)
        X, Y = np.meshgrid(ii, ii, indexing="ij")
        ax.quiver(X, Y, tx.real[X, Y], ty.real[X, Y], color=DEEP,
                  scale=30, width=0.004, pivot="mid")
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
        ax.set_title(name, fontsize=11)
    fig.suptitle("the normal-vector method's tangent field — smooth, unit-length, "
                 "boundary-following everywhere", fontsize=11.5, y=0.99)
    save(fig, "theory_tangent_field.png")


if __name__ == "__main__":
    import sys
    ALL = [thin_film_spectrum, grating_orders, metasurface_phase,
           metasurface_field, polarization, convergence, shape_gallery,
           ar_coating, lesson7_parametric, structure_arstack, structure_motheye,
           sweep_map, dispersion_map, hero_showcase, theory_pipeline,
           theory_factorization, theory_tangent_field]
    only = set(sys.argv[1:])
    print("Generating documentation figures ->", ASSETS)
    for fn in ALL:
        if not only or fn.__name__ in only:
            fn()
    print("Done.")
