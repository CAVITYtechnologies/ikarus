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

from ikarus import RCWA, shapes
from ikarus.tests.validation import grcwa_reference, torcwa_reference
from ikarus.tests.validation.fmmax_reference import (
    binary_grating_grid,
    stack_RT,
    uniform_grid,
)
from ikarus.tests.validation.fresnel_reference import fresnel_stack

# fmmax / grcwa are optional cross-check deps (not Ikarus requirements); each test
# importorskips the tool it needs, so they skip independently (and both skip in the
# publish CI, which installs neither). The *_reference modules import their heavy
# dep lazily, so importing them at module load is always safe.

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


# 2-D curved cylinder (identical to test_normalvector.py): the case where the
# normal-vector method is the real differentiator over even Li's separable rule.
CYL_PERIOD, CYL_N, CYL_H, CYL_WL = 400e-9, 3.5, 200e-9, 700e-9


def _ikarus_cylinder_R(factorization, M, N=96):
    mask = shapes.circle(center=(0.5, 0.5), radius=0.30, grid_shape=(N, N))
    rc = RCWA(period_x=CYL_PERIOD, period_y=CYL_PERIOD, resolution=(N, N),
              n_orders=(M, M), factorization=factorization)
    rc.add_uniform_layer(np.inf, "Air")
    rc.add_layer(CYL_H, mask.astype(int), [1.0, CYL_N])
    rc.add_uniform_layer(np.inf, "Air")
    rc.set_source(wavelength=CYL_WL, theta=0, polarization="linear", linear_pol_angle=0)  # TE
    return rc.simulate()[2].R_total


def _fmmax_cylinder_grid(N):
    mask = np.asarray(shapes.circle(center=(0.5, 0.5), radius=0.30, grid_shape=(N, N)))
    return np.where(mask.astype(bool), complex(CYL_N) ** 2, 1.0)


def test_fmmax_harness_reproduces_fresnel():
    """The harness itself is correct: a uniform slab matches analytic Fresnel.

    Uniform layers are factorization-independent, so this isolates the FMMax
    excitation / flux / S-matrix wiring from the physics under test.
    """
    pytest.importorskip("fmmax")
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
    pytest.importorskip("fmmax")
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


def test_grcwa_harness_reproduces_fresnel():
    """Sanity: the grcwa harness matches analytic Fresnel (validates conventions)."""
    pytest.importorskip("grcwa")
    R, T = grcwa_reference.slab_RT(2.5, 200e-9, 633e-9, pol="TM")
    R_analytic = fresnel_stack([1.0, 2.5, 1.0], [200e-9], 633e-9, 0.0, "p")[0]
    assert abs(R - R_analytic) < 1e-6
    assert abs((R + T) - 1.0) < 1e-9


def test_grcwa_direct_rule_is_wrong_on_high_contrast_tm():
    """grcwa uses the direct (Laurent) rule, so on the canonical high-contrast TM
    grating it lands well above the faithful ~0.10 and only crawls down as O(1/M) --
    still badly off even at 400 orders. This is the concrete "why switch to Ikarus"
    result: a user's direct-rule tool needs far more orders to reach the truth.
    Ikarus's own ``laurent`` mode sits in the same wrong regime (same rule).
    """
    pytest.importorskip("grcwa")
    R100, _ = grcwa_reference.grating_RT(N_HI, PERIOD, H, WL, nG=100)
    R200, _ = grcwa_reference.grating_RT(N_HI, PERIOD, H, WL, nG=200)
    R400, _ = grcwa_reference.grating_RT(N_HI, PERIOD, H, WL, nG=400)

    # far from the faithful answer, even at 400 orders...
    assert R400 > 0.15                       # true value is ~0.100
    # ...and monotonically decreasing toward it (the O(1/M) direct-rule signature).
    assert R100 > R200 > R400

    # Ikarus's laurent mode is in the same wrong regime (it is the same rule),
    # while Ikarus's default faithful mode is not.
    assert _ikarus_grating_R("laurent", M=20) > 0.12
    assert abs(_ikarus_grating_R("normal", M=16) - 0.100) < 5e-3


def test_torcwa_harness_reproduces_fresnel():
    """Sanity: the torcwa harness matches analytic Fresnel (validates conventions)."""
    pytest.importorskip("torcwa")
    R = torcwa_reference.slab_R0(2.5, 200e-9, 633e-9, pol="TM")
    R_analytic = fresnel_stack([1.0, 2.5, 1.0], [200e-9], 633e-9, 0.0, "p")[0]
    assert abs(R - R_analytic) < 1e-6


def test_torcwa_matches_ikarus_laurent_and_misses_faithful():
    """torcwa is the direct rule too, but 1-D-truncatable (order=[M,0]) so it lines
    up exactly with Ikarus's ``laurent`` (M,0). Two results in one:

    * Ikarus ``laurent`` reproduces torcwa to ~1e-3 at matched truncation -- Ikarus
      is a strict superset of what a direct-rule tool computes;
    * both stay off the faithful ~0.100 and only crawl down as O(1/M).
    """
    pytest.importorskip("torcwa")
    R10 = torcwa_reference.grating_R0(N_HI, PERIOD, H, WL, order_m=10)
    R20 = torcwa_reference.grating_R0(N_HI, PERIOD, H, WL, order_m=20)
    R30 = torcwa_reference.grating_R0(N_HI, PERIOD, H, WL, order_m=30)

    # direct-rule signature: far from faithful 0.100, decreasing toward it as O(1/M).
    assert R10 > R20 > R30 > 0.11

    # the superset proof: Ikarus laurent (M,0) == torcwa order=[M,0] (same rule).
    assert abs(R20 - _ikarus_grating_R("laurent", M=20)) < 5e-3
    assert abs(R30 - _ikarus_grating_R("laurent", M=30)) < 5e-3

    # while Ikarus's default faithful mode is already at the true answer.
    assert abs(_ikarus_grating_R("normal", M=16) - 0.100) < 5e-3


def test_ikarus_normal_matches_fmmax_on_2d_curved_cylinder():
    """2-D forward-efficiency cross-check on a CURVED boundary -- where the
    normal-vector method is the real differentiator (it beats even Li's separable
    rule).  Ikarus ``normal`` must match FMMax's ``NORMAL`` (an independent
    implementation of the same Fast Fourier Factorization), making the previously
    hard-coded ``0.92 < R < 0.96`` window in test_normalvector.py a live check.
    """
    pytest.importorskip("fmmax")
    N = 256
    air = np.ones((N, N), dtype=complex)
    eps = [air, _fmmax_cylinder_grid(N), air]
    R_fmmax_normal, _ = stack_RT(eps, [0.0, CYL_H, 0.0], CYL_PERIOD, CYL_WL,
                                 num_terms=500, formulation="normal", pol="TE")

    # Two independent normal-vector implementations agree on the curved cylinder.
    assert abs(_ikarus_cylinder_R("normal", M=14) - R_fmmax_normal) < 6e-3
    assert 0.92 < R_fmmax_normal < 0.96      # the formerly hard-coded window, now live

    # The separable li rule lags at low order on the curved boundary (the reason the
    # normal-vector method exists): normal is already ahead of li at M=8.
    assert _ikarus_cylinder_R("normal", M=8) - _ikarus_cylinder_R("li", M=8) > 0.02
