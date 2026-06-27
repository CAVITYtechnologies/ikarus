"""Tests for the normal-vector (Fast Fourier Factorization) method.

The normal-vector method (``factorization="normal"``, used by the default
``"auto"``) applies the inverse rule along the true local boundary normal, giving
FFF-accelerated convergence on curved/oblique high-contrast structures while
reducing exactly to the separable ``"li"`` rule on axis-aligned geometry.  The
absolute reflectances pinned below were validated against FMMax
(``Formulation.NORMAL``) to <=2e-3; see ``ikarus/core/_normalvector.py``.
"""

import numpy as np

from ikarus import RCWA, shapes
from ikarus.core.fourier import HarmonicGrid
from ikarus.core._normalvector import inplane_tensor, tangent_terms
from ikarus.core.solver import _mixed_convolution, _ANOMALY_LOSS


def _grating_grid(nx=256, ny=4, n_hi=3.5):
    ge = np.ones((nx, ny), dtype=complex)
    ge[nx // 2:, :] = n_hi ** 2
    return np.conj(ge) - _ANOMALY_LOSS


def test_tangent_terms_reduce_for_axis_aligned():
    # 1-D x-grating: tangent field is along y (tx=0, ty=1) -> Pxx=1, others 0.
    Pxx, Pxy, Pyx, Pyy = tangent_terms(np.zeros((4, 4)), np.ones((4, 4)))
    assert np.allclose(Pxx, 1) and np.allclose(Pyy, 0)
    assert np.allclose(Pxy, 0) and np.allclose(Pyx, 0)


def test_inplane_tensor_reduces_to_separable_li_1d():
    """With a 1-D tangent field, the full tensor must equal the separable `li`
    operators exactly (no off-diagonal coupling)."""
    ge = _grating_grid()
    grid = HarmonicGrid(10, 0)
    nx, ny = ge.shape
    E00, E01, E10, E11 = inplane_tensor(ge, np.zeros((nx, ny)), np.ones((nx, ny)), grid)

    Exx = _mixed_convolution(ge, grid, "x")   # inverse rule across x (normal)
    Eyy = _mixed_convolution(ge, grid, "y")   # Laurent along y (tangential)

    assert np.abs(E01).max() < 1e-9 and np.abs(E10).max() < 1e-9   # no coupling
    assert np.abs(E00 - Eyy).max() < 1e-9                          # tangential -> Laurent
    assert np.abs(E11 - Exx).max() < 1e-9                          # normal -> inverse rule


# --- end-to-end tests -------------------------------------------------------

def _cylinder_rcwa(factorization, M, N=96):
    mask = shapes.circle(center=(0.5, 0.5), radius=0.30, grid_shape=(N, N))
    rc = RCWA(period_x=400e-9, period_y=400e-9, resolution=(N, N),
              n_orders=(M, M), factorization=factorization)
    rc.add_uniform_layer(np.inf, "Air")
    rc.add_layer(200e-9, mask.astype(int), [1.0, 3.5])
    rc.add_uniform_layer(np.inf, "Air")
    rc.set_source(wavelength=700e-9, theta=0, polarization="linear",
                  linear_pol_angle=0)
    return rc.simulate()[2]


def test_auto_is_the_default():
    """The default factorization is the user-friendly automatic one."""
    rc = RCWA(period_x=400e-9, period_y=400e-9)
    assert rc.factorization == "auto"


def test_normal_reduces_to_li_for_1d_grating():
    """End-to-end: on an axis-aligned 1-D grating the normal-vector method must
    return *exactly* the separable inverse rule (the tangent field is constant)."""
    topo = np.zeros((512, 2), dtype=int)
    topo[256:, :] = 1
    def R(fac):
        rc = RCWA(period_x=400e-9, period_y=400e-9, resolution=(512, 2),
                  n_orders=(14, 0), factorization=fac)
        rc.add_uniform_layer(np.inf, "Air")
        rc.add_layer(300e-9, topo, [1.0, 3.5])
        rc.add_uniform_layer(np.inf, "Air")
        rc.set_source(wavelength=700e-9, theta=0, polarization="linear",
                      linear_pol_angle=90)   # TM: the rule that matters
        return rc.simulate()[2].R_total
    assert abs(R("normal") - R("li")) < 1e-9


def test_normal_conserves_energy_on_curved():
    res = _cylinder_rcwa("normal", M=10)
    assert abs(res.energy_balance - 1.0) < 1e-7


def test_normal_accelerates_and_corrects_li_on_curved():
    """The core FFF property + a guard on the off-diagonal *sign*: on a curved
    high-contrast cylinder the normal-vector method is already near its converged
    value at low order, pulling well ahead of the still-climbing ``li`` rule.
    A wrong off-diagonal sign collapses ``normal`` back onto ``li`` (gap ~0.003)."""
    r_li = _cylinder_rcwa("li", M=8).R_total
    r_n8 = _cylinder_rcwa("normal", M=8).R_total
    r_n14 = _cylinder_rcwa("normal", M=14).R_total
    # normal jumps clearly ahead of li at the same low order (FFF acceleration).
    assert r_n8 - r_li > 0.02
    # and it is already converged: M=8 and M=14 agree far better than li does.
    assert abs(r_n8 - r_n14) < 0.5 * abs(r_li - r_n14)
    # absolute regression guard (FMMax-NORMAL validated this cylinder at ~0.94).
    assert 0.92 < r_n14 < 0.96


def test_auto_matches_explicit_normal():
    assert abs(_cylinder_rcwa("auto", M=8).R_total
               - _cylinder_rcwa("normal", M=8).R_total) < 1e-12
