"""Generate the figures for the Validation & Accuracy docs page.

Unlike ``gen_docs_figures.py`` this needs the optional cross-check solvers
(``fmmax``, ``grcwa``, ``torcwa``); it is run *manually*, and the resulting PNGs
are committed into ``docs/assets/`` so the docs build never needs those tools::

    pip install "ikarus-rcwa[all]" fmmax grcwa torcwa
    python scripts/gen_validation_figures.py

The reference harnesses live in ``ikarus/tests/validation`` and are each validated
against analytic Fresnel to < 1e-6 first (see ``test_crosscode.py``).
"""

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "4")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

from pathlib import Path

import numpy as np
import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt

from ikarus import RCWA, shapes
from ikarus.tests.validation import fmmax_reference as fm
from ikarus.tests.validation import grcwa_reference as gr
from ikarus.tests.validation import torcwa_reference as tw

ASSETS = Path(__file__).resolve().parent.parent / "docs" / "assets"
ASSETS.mkdir(parents=True, exist_ok=True)

# -- refined, brand-aligned palette -----------------------------------------
# Warm (orange/gold/bronze) = Ikarus & faithful methods, the heroes; muted slate
# = the other tools, which recede; charcoal for structure.  Data lines carry a
# little transparency (+ solid/dashed pairing) so overlapping curves stay legible.
ORANGE = "#e2570b"   # Ikarus normal-vector (default) — brand orange
AMBER = "#eaa00c"    # Ikarus Li inverse rule — warm gold
GREEN = "#a56a2b"    # FMMax reference — warm bronze
DEEP = "#33404a"     # charcoal — text, annotations, tags
BLUE = "#4f6b7a"     # Ikarus laurent / direct rule — muted slate
PURPLE = "#88a0ac"   # torcwa — lighter steel
RED = "#37505c"      # grcwa — dark slate
GREY = "#5b6b75"     # true-value / reference lines
plt.rcParams.update({
    "figure.facecolor": "white", "savefig.facecolor": "white",
    "axes.facecolor": "white", "font.size": 11, "axes.grid": True,
    "grid.alpha": 0.25, "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 130,
})


def save(fig, name):
    fig.savefig(ASSETS / name, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print("wrote", name)


# canonical structures (identical to the tests) -----------------------------
PERIOD, N_HI, H_TM, WL = 400e-9, 3.5, 300e-9, 700e-9         # high-contrast TM grating
CYL_H = 200e-9                                                # 2-D cylinder height


def ikarus_grating_R(fac, M, Nx=1024):
    topo = np.zeros((Nx, 2), dtype=int)
    topo[: Nx // 2, :] = 1
    rc = RCWA(period_x=PERIOD, period_y=PERIOD, resolution=(Nx, 2),
              n_orders=(M, 0), factorization=fac)
    rc.add_uniform_layer(np.inf, "Air")
    rc.add_layer(H_TM, topo, ["Air", N_HI])
    rc.add_uniform_layer(np.inf, "Air")
    rc.set_source(wavelength=WL, theta=0, polarization="linear", linear_pol_angle=90)
    return rc.simulate()[2].R_total


def ikarus_cylinder_R(fac, M, N=96):
    mask = shapes.circle(center=(0.5, 0.5), radius=0.30, grid_shape=(N, N))
    rc = RCWA(period_x=PERIOD, period_y=PERIOD, resolution=(N, N),
              n_orders=(M, M), factorization=fac)
    rc.add_uniform_layer(np.inf, "Air")
    rc.add_layer(CYL_H, mask.astype(int), [1.0, N_HI])
    rc.add_uniform_layer(np.inf, "Air")
    rc.set_source(wavelength=WL, theta=0, polarization="linear", linear_pol_angle=0)
    return rc.simulate()[2].R_total


# ===========================================================================
# Figure 1 -- exact where an exact answer exists (Fresnel baseline)
# ===========================================================================
def fig_fresnel():
    wl = np.linspace(400e-9, 900e-9, 240)
    n_slab, thick = 2.5, 500e-9

    def analytic(w):
        n1, n2 = 1.0, n_slab
        r12, r23 = (n1 - n2) / (n1 + n2), (n2 - n1) / (n2 + n1)
        beta = 2 * np.pi * n2 * thick / w
        r = (r12 + r23 * np.exp(2j * beta)) / (1 + r12 * r23 * np.exp(2j * beta))
        return abs(r) ** 2

    R_a = np.array([analytic(w) for w in wl])
    wl_pts = np.linspace(400e-9, 900e-9, 26)
    R_ik = []
    for w in wl_pts:
        rc = RCWA(period_x=300e-9, period_y=300e-9, n_orders=0)
        rc.add_uniform_layer(np.inf, 1.0)
        rc.add_uniform_layer(thick, n_slab)
        rc.add_uniform_layer(np.inf, 1.0)
        rc.set_source(wavelength=w, theta=0, polarization="linear")
        R_ik.append(rc.simulate()[2].R_total)
    R_ik = np.array(R_ik)
    err = np.abs(R_ik - np.array([analytic(w) for w in wl_pts]))

    fig, (ax, axe) = plt.subplots(2, 1, figsize=(7.2, 5.4), height_ratios=[3, 1.4],
                                  sharex=True)
    ax.plot(wl * 1e9, R_a * 100, color=GREY, lw=2, label="analytic Fresnel")
    ax.plot(wl_pts * 1e9, R_ik * 100, "o", color=ORANGE, ms=6, label="Ikarus")
    ax.set_ylabel("reflectance (%)")
    ax.set_title("Ikarus is exact where an exact answer exists\n"
                 "(uniform slab, n = 2.5, 500 nm — Fabry–Pérot fringes)", fontsize=11)
    ax.legend(frameon=False, loc="upper right")
    axe.semilogy(wl_pts * 1e9, np.maximum(err, 1e-16), "o-", color=BLUE, ms=4, lw=1)
    axe.axhline(1e-6, color=RED, ls="--", lw=1, label="1e-6")
    axe.set_ylabel("|error|")
    axe.set_xlabel("wavelength (nm)")
    axe.set_ylim(1e-16, 1e-4)
    axe.legend(frameon=False, loc="upper right", fontsize=9)
    save(fig, "validation_fresnel.png")


# ===========================================================================
# Figures 2 & 3 -- high-contrast TM grating convergence (shared data)
# ===========================================================================
def tm_sweep():
    Ms = np.array([4, 6, 8, 10, 12, 14, 16, 20, 24, 30])
    data = {
        "normal": np.array([ikarus_grating_R("normal", int(m)) for m in Ms]),
        "li": np.array([ikarus_grating_R("li", int(m)) for m in Ms]),
        "laurent": np.array([ikarus_grating_R("laurent", int(m)) for m in Ms]),
        "torcwa": np.array([tw.grating_R0(N_HI, PERIOD, H_TM, WL, int(m)) for m in Ms]),
    }
    # converged truth from FMMax NORMAL (independent code)
    eps = [fm.uniform_grid(1.0, (128, 128)),
           fm.binary_grating_grid(N_HI, 0.5, (128, 128)),
           fm.uniform_grid(1.0, (128, 128))]
    R_true, _ = fm.stack_RT(eps, [0.0, H_TM, 0.0], PERIOD, WL, 500, "normal", "TM")
    return Ms, data, R_true


def fig_tm_convergence(Ms, data, R_true):
    fig, ax = plt.subplots(figsize=(7.4, 5.0))
    ax.axhline(R_true * 100, color=GREY, ls="--", lw=1.6, alpha=0.9,
               label=f"true value = {R_true*100:.1f}%  (FMMax, converged)")
    ax.plot(Ms, data["normal"] * 100, "o-", color=ORANGE, lw=2.6, ms=7, alpha=0.9,
            label="Ikarus — normal-vector (default)")
    ax.plot(Ms, data["li"] * 100, "s--", color=AMBER, lw=2, ms=6, alpha=0.9,
            label="Ikarus — Li inverse rule")
    ax.plot(Ms, data["laurent"] * 100, "v-", color=BLUE, lw=2.6, ms=6, alpha=0.8,
            label="Ikarus — direct (Laurent) rule")
    ax.plot(Ms, data["torcwa"] * 100, "x--", color=PURPLE, lw=1.8, ms=8, alpha=0.95,
            label="torcwa (direct rule)")
    ax.set_xlabel("Fourier orders retained  (M)")
    ax.set_ylabel("zeroth-order reflectance (%)")
    ax.set_title("High-contrast TM grating (n = 3.5): faithful methods snap to the\n"
                 "answer; the direct rule crawls in from above", fontsize=11)
    ax.annotate("faithful: converged\nby M ≈ 10", xy=(12.5, R_true * 100 + 0.2),
                xytext=(18.5, 11.9), fontsize=9, color=DEEP, ha="center",
                arrowprops=dict(arrowstyle="->", color=DEEP, lw=1))
    ax.annotate("direct rule (grcwa/torcwa):\nstill wrong at M = 30",
                xy=(30, data["laurent"][-1] * 100), xytext=(15.5, data["laurent"][-1] * 100 + 4),
                fontsize=9, color=RED,
                arrowprops=dict(arrowstyle="->", color=RED, lw=1))
    ax.legend(frameon=False, fontsize=9.5, loc="upper right")
    save(fig, "validation_tm_convergence.png")


def fig_tm_error(Ms, data, R_true):
    fig, ax = plt.subplots(figsize=(7.4, 5.0))
    for key, c, mk, ls, lbl in [
        ("li", ORANGE, "o", "-", "Ikarus faithful (normal-vector ≡ Li for 1-D)"),
        ("laurent", BLUE, "v", "-", "Ikarus direct rule"),
        ("torcwa", PURPLE, "x", "--", "torcwa (direct rule)"),
    ]:
        err = np.abs(data[key] - R_true)
        ax.loglog(Ms, np.maximum(err, 1e-6), mk + ls, color=c, lw=2.2, ms=6.5,
                  alpha=0.9, label=lbl)
    guide = 0.9 * (Ms.astype(float) / Ms[0]) ** -1.0 * np.abs(data["laurent"][0] - R_true)
    ax.loglog(Ms, guide, ":", color=GREY, lw=1.6, label="slope ∝ 1/M  (guide)")
    ax.set_xlabel("Fourier orders retained  (M)")
    ax.set_ylabel("|reflectance − true value|")
    ax.set_title("Same data, error vs. orders: the direct rule converges only as\n"
                 "O(1/M); the faithful methods fall off a cliff", fontsize=11)
    ax.legend(frameon=False, fontsize=9.5)
    save(fig, "validation_tm_error.png")


# ===========================================================================
# Figure 4 -- 2-D curved cylinder (normal-vector vs Li on a curved boundary)
# ===========================================================================
def fig_cylinder():
    Ms = np.array([4, 6, 8, 10, 12, 14, 16])
    R_n = np.array([ikarus_cylinder_R("normal", int(m)) for m in Ms])
    R_li = np.array([ikarus_cylinder_R("li", int(m)) for m in Ms])
    N = 256
    air = np.ones((N, N), dtype=complex)
    mask = np.asarray(shapes.circle(center=(0.5, 0.5), radius=0.30, grid_shape=(N, N)))
    cyl = np.where(mask.astype(bool), complex(N_HI) ** 2, 1.0)
    R_true, _ = fm.stack_RT([air, cyl, air], [0.0, CYL_H, 0.0], PERIOD, WL, 500, "normal", "TE")

    fig, ax = plt.subplots(figsize=(7.4, 5.0))
    ax.axhline(R_true * 100, color=GREY, ls="--", lw=1.6, alpha=0.9,
               label=f"true value = {R_true*100:.1f}%  (FMMax NORMAL)")
    ax.plot(Ms, R_n * 100, "o-", color=ORANGE, lw=2.6, ms=7, alpha=0.9,
            label="Ikarus — normal-vector (default)")
    ax.plot(Ms, R_li * 100, "s--", color=AMBER, lw=2, ms=6, alpha=0.9,
            label="Ikarus — Li inverse rule")
    ax.set_xlabel("Fourier orders per axis  (M)")
    ax.set_ylabel("reflectance (%)")
    ax.set_title("2-D curved cylinder (n = 3.5): on curved boundaries the\n"
                 "normal-vector method converges faster than even the inverse rule",
                 fontsize=11)
    ax.legend(frameon=False, fontsize=9.5, loc="lower right")
    save(fig, "validation_cylinder.png")


# ===========================================================================
# Figure 5 -- the bottom line, at a practical truncation
# ===========================================================================
def fig_summary(R_true):
    # each solver at a *practical* truncation a user would actually pick
    fmmax_faithful = R_true
    ik_normal = ikarus_grating_R("normal", 12)
    ik_li = ikarus_grating_R("li", 12)
    ik_laurent = ikarus_grating_R("laurent", 12)
    torcwa_direct = tw.grating_R0(N_HI, PERIOD, H_TM, WL, 12)
    grcwa_direct, _ = gr.grating_RT(N_HI, PERIOD, H_TM, WL, nG=400)

    labels = ["FMMax\nNORMAL", "Ikarus\nnormal", "Ikarus\nLi",
              "Ikarus\nlaurent", "torcwa", "grcwa\n(nG=400)"]
    vals = np.array([fmmax_faithful, ik_normal, ik_li,
                     ik_laurent, torcwa_direct, grcwa_direct]) * 100
    colors = [GREEN, ORANGE, AMBER, BLUE, PURPLE, RED]

    fig, ax = plt.subplots(figsize=(7.6, 5.2))
    ax.axvspan(-0.5, 2.5, color=ORANGE, alpha=0.07)     # faithful group (warm)
    ax.axvspan(2.5, 5.5, color=BLUE, alpha=0.09)        # direct-rule group (cool)
    bars = ax.bar(labels, vals, color=colors, edgecolor="white", width=0.7)
    ax.axhline(R_true * 100, color=GREY, ls="--", lw=1.6)
    ax.text(2.5, 12.3, f"true value = {R_true*100:.1f}%", color=DEEP,
            fontsize=9.5, ha="center", style="italic")
    for b, v in zip(bars, vals):
        err = (v - R_true * 100) / (R_true * 100) * 100
        tag = f"{v:.1f}%" + (f"\n(+{err:.0f}%)" if err > 1 else "\n✓")
        ax.text(b.get_x() + b.get_width() / 2, v + 0.4, tag, ha="center",
                va="bottom", fontsize=8.5, color=DEEP)
    ax.set_ylabel("zeroth-order reflectance (%)")
    ax.set_ylim(0, max(vals) * 1.30)
    ax.text(1.0, max(vals) * 1.20, "faithful  ✓", color=ORANGE, ha="center",
            fontsize=11, fontweight="bold")
    ax.text(4.0, max(vals) * 1.20, "direct rule — wrong", color=RED, ha="center",
            fontsize=11, fontweight="bold")
    ax.set_title("The bottom line — high-contrast TM grating at a practical truncation",
                 fontsize=11)
    save(fig, "validation_summary.png")


if __name__ == "__main__":
    fig_fresnel()
    Ms, data, R_true = tm_sweep()
    fig_tm_convergence(Ms, data, R_true)
    fig_tm_error(Ms, data, R_true)
    fig_cylinder()
    fig_summary(R_true)
    print("done ->", ASSETS)
