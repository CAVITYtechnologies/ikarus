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
               savefig: str | None = None, cmap: str | None = None):
    """Plot a 2-D :class:`~ikarus.core.fields.FieldMap` cross-section.

    ``component`` is ``'intensity'``, a field component (``'Ex'``..``'Hz'``) for
    magnitude, or ``'<comp>phase'`` (e.g. ``'Eyphase'``) for phase.
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
    ax.set_xlabel(f"{b_name} (m)")
    ax.set_ylabel(f"{a_name} (m)")
    ax.set_title(label + (f"  (z={field_map.z:.2e} m)" if field_map.z is not None else ""))
    ax.figure.colorbar(im, ax=ax, label=label)
    if savefig:
        ax.figure.savefig(savefig, bbox_inches="tight", dpi=150)
    return ax


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
