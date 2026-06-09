"""Fourier-space machinery for RCWA.

This module builds the *harmonic basis* used by the modal method and the
*convolution matrices* (a.k.a. Toeplitz / "Toeplitz-of-Toeplitz" matrices) that
represent multiplication by a periodic function in Fourier space.

Conventions
-----------
A periodic function ``f(x, y)`` with periods ``(Lx, Ly)`` is expanded as

    f(x, y) = sum_{m,n} f_{mn} exp( +i (2*pi*m/Lx) x + i (2*pi*n/Ly) y ).

Multiplication of two periodic functions ``h = f * g`` becomes, in the truncated
Fourier basis, a matrix-vector product ``h_vec = F @ g_vec`` where ``F`` is the
*convolution matrix* with entries ``F[(m,n),(m',n')] = f_{m-m', n-n'}``.

The harmonic orders are flattened to a single index in row-major order over
``(p, q)`` with ``p`` (the x-order) varying slowest.  Helper :func:`harmonic_map`
returns the explicit ordering so every other module agrees on it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class HarmonicGrid:
    """Bookkeeping for the truncated set of Fourier harmonics.

    Parameters
    ----------
    n_orders_x, n_orders_y:
        Maximum positive order kept in each direction.  Orders run from
        ``-n_orders`` to ``+n_orders`` inclusive, so the count in x is
        ``2*n_orders_x + 1``.
    """

    n_orders_x: int
    n_orders_y: int

    @property
    def num_x(self) -> int:
        return 2 * self.n_orders_x + 1

    @property
    def num_y(self) -> int:
        return 2 * self.n_orders_y + 1

    @property
    def size(self) -> int:
        """Total number of harmonics ``P = (2*Mx+1)*(2*My+1)``."""
        return self.num_x * self.num_y

    @property
    def orders_x(self) -> np.ndarray:
        return np.arange(-self.n_orders_x, self.n_orders_x + 1)

    @property
    def orders_y(self) -> np.ndarray:
        return np.arange(-self.n_orders_y, self.n_orders_y + 1)

    def index_arrays(self) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(p, q)`` integer order arrays of length :attr:`size`.

        Row-major flattening: ``p`` (x-order) is the slow index, ``q`` the fast
        index.  This matches a ``np.meshgrid(..., indexing='ij')`` followed by a
        C-order ``ravel``.
        """
        p, q = np.meshgrid(self.orders_x, self.orders_y, indexing="ij")
        return p.ravel(), q.ravel()

    def zero_order_index(self) -> int:
        """Flat index of the specular ``(0, 0)`` harmonic."""
        return self.n_orders_x * self.num_y + self.n_orders_y


def harmonic_map(n_orders_x: int, n_orders_y: int) -> HarmonicGrid:
    """Convenience constructor for :class:`HarmonicGrid`."""
    return HarmonicGrid(int(n_orders_x), int(n_orders_y))


def convolution_matrix(cell: np.ndarray, grid: HarmonicGrid) -> np.ndarray:
    """Build the convolution (Toeplitz) matrix of a periodic cell function.

    Parameters
    ----------
    cell:
        Real-space samples of the periodic function on a ``(Nx, Ny)`` grid that
        tiles the unit cell uniformly.  May be complex (e.g. a lossy
        permittivity distribution).
    grid:
        Harmonic truncation describing how many orders to keep.

    Returns
    -------
    np.ndarray
        Complex matrix of shape ``(P, P)`` with ``P = grid.size``.

    Notes
    -----
    We obtain the Fourier coefficients with a 2-D FFT (normalized by the number
    of samples) and then gather the coefficient ``f_{Δp, Δq}`` for every pair of
    harmonics.  The required spread of difference orders is
    ``±2*n_orders`` in each direction, so the sampling grid must be at least
    ``4*n_orders + 1`` to avoid aliasing of the differences; callers are
    responsible for providing a sufficiently fine ``cell``.
    """
    cell = np.asarray(cell)
    nx, ny = cell.shape

    # Fourier coefficients f_{mn}.  np.fft.fft2 uses the exp(-i...) convention;
    # to match the +i expansion above we conjugate the exponent by using
    # fftshift on a forward transform and indexing with the natural sign.
    coeffs = np.fft.fftshift(np.fft.fft2(cell)) / (nx * ny)

    # Center of the (shifted) coefficient array corresponds to order 0.
    cx, cy = nx // 2, ny // 2

    p, q = grid.index_arrays()  # length P
    # Difference orders between every pair of harmonics.
    dp = p[:, None] - p[None, :]
    dq = q[:, None] - q[None, :]

    # Guard against requesting orders that the FFT grid cannot resolve.
    max_dp, max_dq = np.abs(dp).max(), np.abs(dq).max()
    if max_dp > cx or max_dq > cy:
        raise ValueError(
            "Cell resolution too coarse for the requested harmonic orders: "
            f"need at least {2 * max_dp + 1}x{2 * max_dq + 1} samples, "
            f"got {nx}x{ny}."
        )

    return coeffs[cx + dp, cy + dq]


def reciprocal_vectors(period_x: float, period_y: float):
    """Return the reciprocal-lattice spacings ``(2*pi/Lx, 2*pi/Ly)``."""
    return 2.0 * np.pi / period_x, 2.0 * np.pi / period_y
