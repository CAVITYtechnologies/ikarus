"""Independent cross-code reference via **torcwa** (torch; direct/Laurent rule).

torcwa (C. Kim, https://github.com/kch3782/torcwa) is a torch FMM solver that uses
the **direct (Laurent) Fourier rule**.  Because it is naturally 1-D-truncatable
(``order=[M, 0]``), it lines up exactly with Ikarus's ``laurent`` mode at
``n_orders=(M, 0)`` -- same rule, same truncation -- which lets us show two things
at once:

* Ikarus's ``laurent`` mode reproduces torcwa to ~1e-3 (Ikarus is a strict
  superset: it does what torcwa does, *and* offers the faithful mode torcwa lacks);
* both stay off the faithful ~0.100 on high-contrast TM, converging only O(1/M).

Validated against analytic Fresnel to < 1e-6.  ``torcwa`` (and torch) are optional
cross-check dependencies, **not** Ikarus requirements.
"""

from __future__ import annotations

import numpy as np


def _reflect_R0(eps_layer, thickness, period, wavelength, order, pol="TM"):
    """Zeroth-order reflectance of one patterned/uniform layer between air/air.

    ``eps_layer`` is a torch tensor (patterned grid) or a scalar permittivity.
    period < wavelength here, so only order 0 propagates and R0 == total R.
    """
    import torch
    import torcwa

    sim = torcwa.rcwa(freq=1.0 / wavelength, order=list(order), L=[period, period],
                      dtype=torch.complex128, device=torch.device("cpu"))
    sim.add_input_layer(eps=1.0)
    sim.add_output_layer(eps=1.0)
    sim.set_incident_angle(inc_ang=0.0, azi_ang=0.0)
    sim.add_layer(thickness=thickness, eps=eps_layer)
    sim.solve_global_smatrix()
    amp = [1.0, 0.0] if pol.upper() == "TM" else [0.0, 1.0]   # TM=Ex, TE=Ey
    sim.source_planewave(amplitude=amp, direction="forward", notation="xy")
    inp = "x" if pol.upper() == "TM" else "y"
    oth = "y" if inp == "x" else "x"
    # power_norm=True returns power-normalized *amplitudes*; |.|^2 is the efficiency.
    co = sim.S_parameters(orders=[0, 0], direction="forward", port="reflection",
                          polarization=inp + inp, ref_order=[0, 0], power_norm=True)
    cr = sim.S_parameters(orders=[0, 0], direction="forward", port="reflection",
                          polarization=oth + inp, ref_order=[0, 0], power_norm=True)
    return float(abs(complex(co)) ** 2 + abs(complex(cr)) ** 2)


def slab_R0(n_slab, thickness, wavelength, pol="TM"):
    """Uniform-slab R0 -- factorization-independent, validates the harness."""
    import torch

    eps = torch.tensor(complex(n_slab) ** 2, dtype=torch.complex128)
    return _reflect_R0(eps, thickness, 0.5 * wavelength, wavelength, [3, 0], pol)


def grating_R0(n_hi, period, height, wavelength, order_m, duty=0.5, Nx=512, pol="TM"):
    """Total R of a 1-D lamellar grating via torcwa at ``order=[order_m, 0]``.

    Same structure Ikarus/FMMax/grcwa see (high index in the first ``duty`` of x).
    """
    import torch

    line = np.ones((Nx, Nx))
    line[: int(Nx * duty), :] = float(n_hi) ** 2
    eps = torch.tensor(line, dtype=torch.complex128)
    return _reflect_R0(eps, height, period, wavelength, [int(order_m), 0], pol)
