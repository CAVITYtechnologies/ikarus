"""Tests for harmonic indexing and convolution matrices."""

import numpy as np

from ikarus.core.fourier import HarmonicGrid, convolution_matrix


def test_harmonic_counts_and_zero_index():
    g = HarmonicGrid(2, 3)
    assert g.num_x == 5 and g.num_y == 7
    assert g.size == 35
    p, q = g.index_arrays()
    i0 = g.zero_order_index()
    assert p[i0] == 0 and q[i0] == 0


def test_uniform_cell_gives_scalar_times_identity():
    g = HarmonicGrid(3, 3)
    cell = np.full((32, 32), 4.0 + 0j)
    C = convolution_matrix(cell, g)
    assert np.allclose(C, 4.0 * np.eye(g.size))


def test_convolution_is_hermitian_for_real_cell():
    g = HarmonicGrid(4, 4)
    rng = np.random.default_rng(0)
    cell = rng.random((64, 64))  # real permittivity
    C = convolution_matrix(cell, g)
    assert np.allclose(C, C.conj().T, atol=1e-12)


def test_dc_coefficient_is_mean():
    g = HarmonicGrid(2, 2)
    cell = np.zeros((32, 32))
    cell[:16] = 1.0
    cell[16:] = 5.0  # mean = 3
    C = convolution_matrix(cell, g)
    i0 = g.zero_order_index()
    assert abs(C[i0, i0] - 3.0) < 1e-12


def test_too_coarse_resolution_raises():
    g = HarmonicGrid(20, 0)
    cell = np.ones((8, 4))  # far too coarse for +/-40 difference orders
    try:
        convolution_matrix(cell, g)
        assert False, "expected ValueError"
    except ValueError:
        pass
