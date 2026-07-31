"""Benchmark the CPU core against an accelerator on 2-D metasurface solves.

RCWA cost is dominated by a dense complex eigendecomposition of a
``2*(2M+1)**2`` matrix, so an accelerator only wins once that matrix is large.
This sweeps ``n_orders`` and prints the CPU vs device time and speedup, so you can
read off the crossover point and the win at the large end.

Usage (e.g. on the A6000 box)::

    pip install "git+https://github.com/CAVITYtechnologies/ikarus@feature/gpu-device"
    pip install -U "jax[cuda12]"          # CUDA-enabled jax/jaxlib
    python scripts/benchmark_device.py            # cpu vs cuda (default)
    python scripts/benchmark_device.py metal      # cpu vs Apple Metal

Notes:
* Each ``simulate()`` on the device path currently includes JAX *tracing* overhead
  (the forward solve is not yet jitted), so small sizes look worse than the raw
  compute would be; the large-M numbers are the honest indicator of the GPU win.
* The first device call compiles (XLA) -- excluded here via a warm-up solve.
"""

import sys
import time

import numpy as np

from ikarus import RCWA, shapes

DEVICE = sys.argv[1] if len(sys.argv) > 1 else "cuda"
ORDERS = [6, 8, 10, 12, 15, 18, 22, 26]


def build(device, M, N=128):
    mask = shapes.circle(center=(0.5, 0.5), radius=0.32, grid_shape=(N, N))
    rc = RCWA(period_x=500e-9, period_y=500e-9, resolution=(N, N),
              n_orders=(M, M), device=device)
    rc.add_uniform_layer(np.inf, "Air")
    rc.add_layer(300e-9, mask.astype(int), [1.0, 3.5])
    rc.add_uniform_layer(np.inf, "SiO2")
    rc.set_source(wavelength=633e-9, theta=0, polarization="linear")
    return rc


def timeit(rc, repeats=3):
    rc.simulate()                              # warm-up (JIT/XLA compile)
    t0 = time.perf_counter()
    for _ in range(repeats):
        r = rc.simulate()[2]
    return (time.perf_counter() - t0) / repeats, r.R_total


def main():
    try:
        import jax
        print("JAX devices:", jax.devices())
    except Exception as e:  # noqa: BLE001
        print("could not import jax:", e)

    print(f"\n2-D pillar metasurface (128x128 grid), CPU core vs device={DEVICE!r}\n")
    print(f"{'M':>4} {'eig dim':>9} {'cpu (s)':>10} {DEVICE+' (s)':>12} "
          f"{'speedup':>9} {'|dR|':>9}")
    print("-" * 60)
    for M in ORDERS:
        dim = 2 * (2 * M + 1) ** 2
        t_c, R_c = timeit(build("cpu", M))
        try:
            t_g, R_g = timeit(build(DEVICE, M))
            print(f"{M:>4} {dim:>9} {t_c:>10.3f} {t_g:>12.3f} "
                  f"{t_c / t_g:>8.1f}x {abs(R_c - R_g):>9.1e}")
        except Exception as e:  # noqa: BLE001
            print(f"{M:>4} {dim:>9} {t_c:>10.3f}   {DEVICE} ERROR: "
                  f"{type(e).__name__}: {str(e)[:50]}")


if __name__ == "__main__":
    main()
