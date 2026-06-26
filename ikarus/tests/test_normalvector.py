"""Tests for the normal-vector (FFF) in-plane permittivity tensor (WIP feature).

The full normal-vector factorization is being built on `feature/normal-vector` and
validated against FMMax. These tests pin the pieces that are already complete.
"""

import numpy as np

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
