"""Reproducible cross-code validation of Ikarus's Fourier factorization.

Historically Ikarus's "validated against FMMax" numbers were derived once by hand
and only the *result* was hard-coded (e.g. the self-extrapolated ``R~0.100`` in
``test_factorization.py``).  These tests recompute the external reference **live**,
turning a one-off manual check into a reproducible one.

``fmmax`` is an optional cross-check dependency, not an Ikarus requirement, so the
whole module is skipped where it is not installed (e.g. the publish CI).
"""

import numpy as np
import pytest

pytest.importorskip("fmmax")

from ikarus import RCWA
from ikarus.tests.validation.fmmax_reference import (
    binary_grating_grid,
    stack_RT,
    uniform_grid,
)
from ikarus.tests.validation.fresnel_reference import fresnel_stack

# Canonical high-contrast TM grating (identical to test_factorization.py).
PERIOD, N_HI, H, WL = 400e-9, 3.5, 300e-9, 700e-9


def _ikarus_grating_R(factorization, M, Nx=1024):
    topo = np.zeros((Nx, 2), dtype=int)
    topo[: Nx // 2, :] = 1  # high index first -> same structure the FMMax grid uses
    rc = RCWA(period_x=PERIOD, period_y=PERIOD, resolution=(Nx, 2),
              n_orders=(M, 0), factorization=factorization)
    rc.add_uniform_layer(np.inf, "Air")
    rc.add_layer(H, topo, ["Air", N_HI])
    rc.add_uniform_layer(np.inf, "Air")
    rc.set_source(wavelength=WL, theta=0, polarization="linear", linear_pol_angle=90)
    return rc.simulate()[2].R_total


def test_fmmax_harness_reproduces_fresnel():
    """The harness itself is correct: a uniform slab matches analytic Fresnel.

    Uniform layers are factorization-independent, so this isolates the FMMax
    excitation / flux / S-matrix wiring from the physics under test.
    """
    eps = [uniform_grid(1.0), uniform_grid(2.5), uniform_grid(1.0)]
    R, T = stack_RT(eps, [0.0, 200e-9, 0.0], period=500e-9, wavelength=633e-9,
                    num_terms=1, pol="TE")
    R_analytic = fresnel_stack([1.0, 2.5, 1.0], [200e-9], 633e-9, 0.0, "s")[0]
    assert abs(R - R_analytic) < 1e-6
    assert abs((R + T) - 1.0) < 1e-9


def test_ikarus_faithful_matches_fmmax_high_contrast_tm():
    """THE cross-code result: on a high-contrast TM grating Ikarus's faithful
    factorization agrees with FMMax's (an independent code), while the direct
    (Laurent) rule -- what grcwa/torcwa use -- lands on a different, wrong value.
    """
    eps = [uniform_grid(1.0, (128, 128)),
           binary_grating_grid(N_HI, 0.5, (128, 128)),
           uniform_grid(1.0, (128, 128))]
    thick = [0.0, H, 0.0]

    R_fmmax_faithful, _ = stack_RT(eps, thick, PERIOD, WL, num_terms=400,
                                   formulation="normal", pol="TM")
    R_fmmax_direct, _ = stack_RT(eps, thick, PERIOD, WL, num_terms=400,
                                 formulation="fft", pol="TM")

    R_ik_normal = _ikarus_grating_R("normal", M=16)
    R_ik_li = _ikarus_grating_R("li", M=16)
    R_ik_laurent = _ikarus_grating_R("laurent", M=20)

    # 1. Ikarus faithful == FMMax faithful (two independent codes, same rule).
    assert abs(R_ik_normal - R_fmmax_faithful) < 3e-3
    assert abs(R_ik_li - R_fmmax_faithful) < 3e-3

    # 2. The faithful answer is ~0.10, i.e. the self-extrapolated target is real.
    assert 0.095 < R_fmmax_faithful < 0.105

    # 3. The direct rule is wrong in BOTH codes (Ikarus laurent ~0.137,
    #    FMMax FFT still ~0.16-0.20 at this truncation -- converging only O(1/M)).
    assert abs(R_ik_laurent - R_fmmax_faithful) > 0.03
    assert abs(R_fmmax_direct - R_fmmax_faithful) > 0.03
