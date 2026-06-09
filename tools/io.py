"""HDF5 export/import of simulation results.

Results are stored in a self-describing HDF5 layout: scalar totals and complex
zero-order coefficients, per-order efficiencies and exit angles, the simulation
metadata (geometry, source, harmonic count) and -- optionally -- reconstructed
field maps.  Files are readable by any HDF5 viewer (``h5dump``, ``h5py``,
``HDFView``) and re-loadable into a plain dict via :func:`load_results`.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def _require_h5py():
    try:
        import h5py
    except ImportError as exc:  # pragma: no cover
        raise ImportError("HDF5 I/O requires the 'h5py' package") from exc
    return h5py


def _store_coeff(group, name, value):
    """Store a scalar/dict reflection or transmission coefficient."""
    if isinstance(value, dict):  # circular: {'co', 'cross'}
        sub = group.create_group(name)
        for key, val in value.items():
            sub.create_dataset(key, data=np.complex128(val))
        sub.attrs["type"] = "circular"
    else:
        group.create_dataset(name, data=np.complex128(value))


def _metadata(rcwa) -> dict:
    src = rcwa.source
    layers = []
    for lay in rcwa.layers:
        layers.append({
            "height": float(lay.height) if np.isfinite(lay.height) else "inf",
            "uniform": bool(lay.is_uniform),
            "material": str(lay.material) if lay.is_uniform else None,
            "materials": [str(m) for m in (lay.materials or [])],
            "name": lay.name,
        })
    meta = {
        "period_x": rcwa.period_x,
        "period_y": rcwa.period_y,
        "n_orders": list(rcwa.n_orders),
        "resolution": list(rcwa.resolution),
        "layers": layers,
    }
    if src is not None:
        meta["source"] = {
            "wavelength": src.wavelength, "theta": src.theta, "phi": src.phi,
            "polarization": src.polarization, "linear_pol_angle": src.linear_pol_angle,
        }
    return meta


def save_results(rcwa, path, include=("T", "R", "metadata"), result=None):
    """Write the most recent (or supplied) simulation result to ``path`` (HDF5)."""
    h5py = _require_h5py()
    if result is None:
        _, _, result = rcwa.simulate()

    include = set(include)
    with h5py.File(path, "w") as f:
        f.attrs["format"] = "ikarus-results"
        f.attrs["version"] = "1"

        if "R" in include:
            _store_coeff(f, "R", result.R)
            f.create_dataset("R_total", data=float(result.R_total))
            f.create_dataset("R_orders", data=np.asarray(result.R_orders))
        if "T" in include:
            _store_coeff(f, "T", result.T)
            f.create_dataset("T_total", data=float(result.T_total))
            f.create_dataset("T_orders", data=np.asarray(result.T_orders))

        p, q = result.orders
        f.create_dataset("order_p", data=np.asarray(p))
        f.create_dataset("order_q", data=np.asarray(q))
        f.create_dataset("theta_out_trn", data=np.asarray(result.theta_out_trn))
        f.create_dataset("phi_out_trn", data=np.asarray(result.phi_out_trn))
        f.create_dataset("theta_out_ref", data=np.asarray(result.theta_out_ref))
        f.create_dataset("phi_out_ref", data=np.asarray(result.phi_out_ref))
        f.create_dataset("energy_balance", data=float(result.energy_balance))

        if "metadata" in include:
            f.attrs["metadata"] = json.dumps(_metadata(rcwa))

        if "fields" in include:
            fmaps = rcwa.get_fields(plane="xz")
            grp = f.create_group("fields")
            for label, fm in fmaps.items():
                sub = grp.create_group(label)
                sub.create_dataset("E", data=fm.E)
                sub.create_dataset("H", data=fm.H)
                for axis, vals in fm.coords.items():
                    sub.create_dataset(f"coord_{axis}", data=np.asarray(vals))
    return Path(path)


def load_results(path) -> dict:
    """Load an Ikarus HDF5 result file into a nested dict."""
    h5py = _require_h5py()
    out = {}
    with h5py.File(path, "r") as f:
        def read(name, obj):
            if isinstance(obj, h5py.Dataset):
                out[name] = obj[()]
        f.visititems(read)
        if "metadata" in f.attrs:
            out["metadata"] = json.loads(f.attrs["metadata"])
        out["energy_balance"] = float(f["energy_balance"][()]) if "energy_balance" in f else None
    return out
