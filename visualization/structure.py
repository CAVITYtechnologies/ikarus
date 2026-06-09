"""Plot the layer stack and per-layer topology maps."""

from __future__ import annotations

import numpy as np


def _lazy_plt():
    import matplotlib.pyplot as plt
    return plt


def _material_colors(names):
    plt = _lazy_plt()
    cmap = plt.get_cmap("tab10")
    uniq = list(dict.fromkeys(names))
    return {n: cmap(i % 10) for i, n in enumerate(uniq)}


def _layer_label(layer, idx) -> str:
    if layer.name:
        return layer.name
    if layer.is_uniform:
        return str(layer.material)
    return f"pattern[{idx}]"


def plot_stack(rcwa, ax=None, savefig: str | None = None, finite_frac: float = 0.25):
    """Draw the layer stack as an xz cross-section, color-coded by material.

    Semi-infinite cover/substrate are drawn with a finite visual height
    (``finite_frac`` of the total interior thickness).
    """
    plt = _lazy_plt()
    if ax is None:
        _, ax = plt.subplots(figsize=(4, 6))

    interior = rcwa.layers[1:-1]
    total = sum(l.height for l in interior) or 1.0
    cap = finite_frac * total

    labels = [_layer_label(l, i) for i, l in enumerate(rcwa.layers)]
    colors = _material_colors(labels)

    # Draw top (cover) downward: start at z = -cap (cover) .. through interior .. substrate.
    z = -cap
    spans = [(rcwa.layers[0], cap)]
    for l in interior:
        spans.append((l, l.height))
    spans.append((rcwa.layers[-1], cap))

    # spans has exactly one entry per layer, so index aligns with `labels`.
    for i, (layer, h) in enumerate(spans):
        label = labels[i]
        ax.add_patch(plt.Rectangle((0, z), 1.0, h, facecolor=colors[label],
                                   edgecolor="k", lw=0.8))
        ax.text(0.5, z + h / 2, label, ha="center", va="center", fontsize=9)
        z += h

    ax.set_xlim(0, 1)
    ax.set_ylim(z, -cap)  # invert so cover is on top
    ax.set_xticks([])
    ax.set_ylabel("z (m)")
    ax.set_title("Layer stack (xz)")
    if savefig:
        ax.figure.savefig(savefig, bbox_inches="tight", dpi=150)
    return ax


def plot_topology(rcwa, layer_index: int, wavelength: float | None = None,
                  ax=None, savefig: str | None = None):
    """Show a patterned layer's permittivity (real part) over the unit cell."""
    plt = _lazy_plt()
    layer = rcwa.layers[layer_index]
    if layer.is_uniform:
        raise ValueError(f"layer {layer_index} is uniform; nothing to map")
    wl = wavelength or (rcwa.source.wavelength if rcwa.source else 550e-9)
    eps = layer.permittivity_grid(wl, rcwa.materials, rcwa.resolution)

    if ax is None:
        _, ax = plt.subplots(figsize=(5, 5))
    extent = [0, rcwa.period_x, 0, rcwa.period_y]
    im = ax.imshow(eps.real.T, origin="lower", extent=extent, aspect="auto",
                   cmap="viridis")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(f"Layer {layer_index} Re(eps) @ {wl*1e9:.0f} nm")
    ax.figure.colorbar(im, ax=ax, label="Re(eps)")
    if savefig:
        ax.figure.savefig(savefig, bbox_inches="tight", dpi=150)
    return ax
