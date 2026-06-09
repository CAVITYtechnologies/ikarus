"""Plot reconstructed real-space fields (magnitude or phase)."""

from __future__ import annotations

import numpy as np

_COMPONENTS = {"Ex": ("E", 0), "Ey": ("E", 1), "Ez": ("E", 2),
               "Hx": ("H", 0), "Hy": ("H", 1), "Hz": ("H", 2)}


def _lazy_plt():
    import matplotlib.pyplot as plt
    return plt


def _extract(field_map, component: str):
    """Return a 2-D real array and a colorbar label for ``component``."""
    if component == "intensity":
        return field_map.intensity, "|E|^2"
    if component in ("E", "H"):
        arr = getattr(field_map, component)
        return np.sum(np.abs(arr) ** 2, axis=-1), f"|{component}|^2"
    if component.endswith("phase"):
        attr, idx = _COMPONENTS[component[:2]]
        return np.angle(getattr(field_map, attr)[..., idx]), f"arg({component[:2]})"
    attr, idx = _COMPONENTS[component]
    return np.abs(getattr(field_map, attr)[..., idx]), f"|{component}|"


def plot_field(field_map, component: str = "intensity", ax=None,
               savefig: str | None = None, cmap: str | None = None,
               overlay: bool = True, overlay_color: str = "white",
               overlay_alpha: float = 0.45):
    """Plot a 2-D :class:`~ikarus.core.fields.FieldMap` cross-section.

    ``component`` is ``'intensity'``, a field component (``'Ex'``..``'Hz'``) for
    magnitude, or ``'<comp>phase'`` (e.g. ``'Eyphase'``) for phase.

    If ``overlay`` is true and the map carries the structure permittivity
    (``field_map.eps``, attached by :meth:`ikarus.RCWA.get_fields`), the material
    boundaries are drawn as semi-transparent contours: the topology outline for
    ``xy`` slices, the layer interfaces for ``xz`` / ``yz`` cross-sections.
    """
    plt = _lazy_plt()
    data, label = _extract(field_map, component)
    coords = list(field_map.coords.items())
    (a_name, a), (b_name, b) = coords[0], coords[1]

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))
    if cmap is None:
        cmap = "twilight" if "phase" in component else "inferno"
    extent = [b.min(), b.max(), a.min(), a.max()]
    im = ax.imshow(data, origin="lower", extent=extent, aspect="auto", cmap=cmap)

    if overlay and getattr(field_map, "eps", None) is not None:
        _overlay_structure(ax, field_map.eps, extent, overlay_color, overlay_alpha)

    ax.set_xlabel(f"{b_name} (m)")
    ax.set_ylabel(f"{a_name} (m)")
    ax.set_title(label + (f"  (z={field_map.z:.2e} m)" if field_map.z is not None else ""))
    ax.figure.colorbar(im, ax=ax, label=label)
    if savefig:
        ax.figure.savefig(savefig, bbox_inches="tight", dpi=150)
    return ax


def _overlay_structure(ax, eps, extent, color, alpha):
    """Draw material boundaries (contours at permittivity jumps) over a field."""
    eps = np.real(eps)
    vals = np.unique(np.round(eps, 6))
    if vals.size < 2:
        return  # uniform region -> nothing to outline
    levels = (vals[:-1] + vals[1:]) / 2.0
    ax.contour(eps, levels=levels, extent=extent, origin="lower",
               colors=color, alpha=alpha, linewidths=1.0)


def plot_field_xy(field_dict, component: str = "intensity", savefig: str | None = None):
    """Plot one or more ``xy`` field maps (the dict returned by ``get_fields``)."""
    plt = _lazy_plt()
    maps = list(field_dict.values())
    n = len(maps)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), squeeze=False)
    for ax, fm in zip(axes[0], maps):
        plot_field(fm, component=component, ax=ax)
    if savefig:
        fig.savefig(savefig, bbox_inches="tight", dpi=150)
    return fig
