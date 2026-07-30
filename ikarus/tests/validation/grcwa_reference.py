"""Independent cross-code reference via **grcwa** (direct/Laurent rule).

grcwa (Jin & Fan, https://github.com/weiliangjinca/grcwa) is a NumPy FMM solver
that uses the **direct (Laurent) Fourier rule**.  It is one of the tools users
reach for, so it is a useful, concrete stand-in for "what a direct-rule solver
does" on high-contrast TM: it converges only as ``O(1/M)`` to the true answer and
is badly off at practical truncations.  Ikarus's ``laurent`` mode reproduces the
same behaviour (same rule), while Ikarus's default and FMMax's NORMAL do not.

Validated against analytic Fresnel to < 1e-6.  Lengths are normalized to the
period (grcwa is scale-invariant; ``freq = 1 / wavelength`` in ``c = 1`` units).

grcwa is an optional cross-check dependency, **not** an Ikarus requirement.
"""

from __future__ import annotations

import numpy as np


def slab_RT(n_slab, thickness, wavelength, pol="TM", nG=11):
    """Uniform-slab R/T -- factorization-independent, validates the harness."""
    import grcwa

    grcwa.set_backend("numpy")
    # Work in wavelength units (freq=1) with a SUBWAVELENGTH cell (period 0.5 wl) so
    # every higher order is evanescent -- no grazing-order singularity, and uniform
    # => nG-free. (A cell of exactly 1 wl puts the +/-1 order at cutoff: singular.)
    obj = grcwa.obj(int(nG), [0.5, 0.0], [0.0, 0.5], 1.0, 0.0, 0.0, verbose=0)
    obj.Add_LayerUniform(1.0, 1.0)                     # semi-infinite cover
    obj.Add_LayerUniform(thickness / wavelength, complex(n_slab) ** 2)
    obj.Add_LayerUniform(1.0, 1.0)                     # semi-infinite substrate
    obj.Init_Setup()
    p, s = (1.0, 0.0) if pol.upper() == "TM" else (0.0, 1.0)   # grcwa p==TM, s==TE
    obj.MakeExcitationPlanewave(p, 0.0, s, 0.0, order=0)
    R, T = obj.RT_Solve(normalize=1)
    return float(R), float(T)


def grating_RT(n_hi, period, height, wavelength, nG, duty=0.5, Nx=256, pol="TM"):
    """Total R/T of a 1-D lamellar grating (high index in the first ``duty`` of x).

    Matches the structure Ikarus/FMMax see.  Uses the direct rule, so on a
    high-contrast TM grating expect a value well above the faithful answer,
    decreasing slowly (``O(1/M)``) as ``nG`` grows.
    """
    import grcwa

    grcwa.set_backend("numpy")
    freq = period / wavelength                         # normalize lengths to the period
    obj = grcwa.obj(int(nG), [1.0, 0.0], [0.0, 1.0], freq, 0.0, 0.0, verbose=0)
    obj.Add_LayerUniform(1.0, 1.0)                     # air cover
    obj.Add_LayerGrid(height / period, Nx, Nx)         # patterned layer
    obj.Add_LayerUniform(1.0, 1.0)                     # air substrate
    obj.Init_Setup()
    eps = np.ones((Nx, Nx))
    eps[: int(Nx * duty), :] = float(n_hi) ** 2        # first fraction of x = n_hi
    obj.GridLayer_geteps(eps.flatten())
    p, s = (1.0, 0.0) if pol.upper() == "TM" else (0.0, 1.0)
    obj.MakeExcitationPlanewave(p, 0.0, s, 0.0, order=0)
    R, T = obj.RT_Solve(normalize=1)
    return float(R), float(T)
