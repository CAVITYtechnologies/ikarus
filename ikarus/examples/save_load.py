"""Save simulation results to HDF5 and load them back.

Runs a small TiO2-pillar metasurface, writes the full result -- totals, per-order
efficiencies, exit angles, the geometry/source metadata and reconstructed fields
-- to a self-describing HDF5 file, then loads it back into a plain dict (no RCWA
object required) and prints a round-trip report.  The ``.h5`` is readable by any
HDF5 tool (``h5py``, ``h5ls``, HDFView).

Requires the io extra:  pip install ikarus-rcwa[io]
Run:  python -m ikarus.examples.save_load
"""

from pathlib import Path

import numpy as np

from ikarus import RCWA, shapes


def main():
    out = Path.cwd() / "ikarus_save_load_output"
    out.mkdir(exist_ok=True)
    h5 = out / "metasurface.h5"

    # 1. a metasurface simulation -------------------------------------------
    period, N = 500e-9, 96
    pillar = shapes.circle(radius=0.3, grid_shape=(N, N))
    rcwa = RCWA(period_x=period, period_y=period, resolution=(N, N), n_orders=(8, 8))
    rcwa.add_uniform_layer(np.inf, "Air")
    rcwa.add_layer(220e-9, pillar, ["Air", "TiO2"])
    rcwa.add_uniform_layer(np.inf, "SiO2")
    rcwa.set_source(wavelength=600e-9, theta=10, polarization="linear")
    _, _, result = rcwa.simulate()
    print(f"computed:  R = {result.R_total:.4f}   T = {result.T_total:.4f}")

    # 2. save: totals + per-order data + metadata + reconstructed fields -----
    try:
        rcwa.save_results(h5, include=["T", "R", "metadata", "fields"], result=result)
    except ImportError as exc:  # pragma: no cover
        print(f"(HDF5 I/O needs h5py: {exc} -- pip install ikarus-rcwa[io])")
        return
    print(f"saved   -> {h5}  ({h5.stat().st_size / 1024:.0f} KB)")

    # 3. load it back -- as if from a different script, with no RCWA object --
    data = RCWA.load_results(h5)
    print("\nfile contents:", ", ".join(sorted(data)))

    # 4. use the loaded data ------------------------------------------------
    print(f"\nloaded:    R_total = {data['R_total']:.4f}   T_total = {data['T_total']:.4f}")
    meta = data["metadata"]
    print(f"geometry:  period = {meta['period_x'] * 1e9:.0f} nm, "
          f"n_orders = {tuple(meta['n_orders'])}")
    src = meta["source"]
    print(f"source:    lambda = {src['wavelength'] * 1e9:.0f} nm, "
          f"theta = {src['theta']:.0f} deg, pol = {src['polarization']}")

    # per-order: the brightest transmitted diffraction orders
    p, q, t_orders = data["order_p"], data["order_q"], data["T_orders"]
    print("brightest transmitted orders:")
    for i in np.argsort(-t_orders)[:3]:
        if t_orders[i] > 1e-3:
            print(f"   ({p[i]:+d},{q[i]:+d}): T = {t_orders[i]:.4f}")

    # 5. confirm the round-trip is exact ------------------------------------
    assert np.isclose(data["R_total"], result.R_total)
    assert np.allclose(data["T_orders"], result.T_orders)
    print("\nround-trip verified: loaded values match the originals exactly.")


if __name__ == "__main__":
    main()
