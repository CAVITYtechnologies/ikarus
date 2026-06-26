"""Normal-vector (Fast Fourier Factorization) in-plane permittivity tensor.

The two-step ("li") rule applies the inverse rule along x and y separately -- exact
for axis-aligned boundaries, but only approximate for curved/oblique ones.  The
**normal-vector method** applies the inverse rule along the *true local boundary
normal* everywhere, restoring fast convergence for arbitrary curved high-contrast
structures.

This module builds the full in-plane permittivity operator (the transverse
permittivity tensor ``E2`` of Liu & Fan, *JOSA A* 2012, eq. 51), mirroring the
construction in FMMax's ``fmm_matrices.transverse_permittivity_vector`` but using
Ikarus's own convolution-matrix conventions.

Status: WIP on ``feature/normal-vector``.  The tensor construction
(:func:`inplane_tensor`) and the tangent field (:func:`tangent_field`,
double-angle orientation diffusion) are **validated**: the field is unit-magnitude
and correctly azimuthal on a circle, the tensor reduces to ``li`` exactly for
axis-aligned geometry, and ``normal`` reduces to ``li`` exactly for 1-D.

REMAINING GAP (the eigenproblem integration): feeding the full tensor (incl.
off-diagonal) into Ikarus's Rumpf ``Q`` (``solver.layer_modes``) only *partially*
captures its effect, so curved high-contrast layers are improved but not fully
accelerated to match FMMax ``Formulation.NORMAL`` (cylinder: ``normal`` ~0.92 at
``M=16`` vs FMMax ~0.939 fast).  FMMax left-multiplies the transverse-permittivity
tensor onto the script-k matrix (Liu 2012 eq. 28: ``matrix = E2 @ omega_script_k
- k_matrix``), a different structure than Rumpf ``P@Q``.  The robust fix is to
mirror that eigenproblem for ``factorization="normal"`` patterned layers.
See ``~/Desktop/ikarus_metamirror/fmmax_validation/``.
"""

from __future__ import annotations

import numpy as np

from .fourier import HarmonicGrid, convolution_matrix


def tangent_field(eps_grid_eng: np.ndarray, smoothing: float | None = None):
    r"""Unit tangent vector field ``(tx, ty)`` for the normal-vector method.

    The inverse rule must be applied along the *true local boundary normal*
    everywhere, which needs a **smooth, unit-magnitude** field over the whole cell
    -- not just at edges.  A raw gradient fails: it vanishes in the bulk and
    *cancels* where opposite normals meet (e.g. a circle's centre).

    We use **double-angle orientation diffusion** (the standard robust orientation
    field).  The boundary normal direction ``theta`` is encoded as the *double
    angle* ``z = (g_x + i g_y)^2 = |grad|^2 e^{2 i theta}`` so that opposite normals
    (``theta`` and ``theta + pi``) add rather than cancel; ``z`` is diffused into
    the bulk by a periodic Gaussian blur, and the orientation is recovered as
    ``theta = angle(z) / 2``.  The tangent is that normal rotated 90 deg, unit
    magnitude everywhere.

    For an axis-aligned (1-D) boundary this yields a constant field and the tensor
    reduces *exactly* to the separable ``li`` rule.  ``smoothing`` is the diffusion
    length in pixels (default ``~max(grid)/12``); the *converged* result is
    insensitive to it -- the field sets convergence rate, not the limit.
    """
    s = np.abs(np.asarray(eps_grid_eng)).astype(float)
    nx, ny = s.shape
    if smoothing is None:
        smoothing = max(nx, ny) / 12.0
    # periodic central-difference gradient = the (un-normalized) boundary normal.
    gx = 0.5 * (np.roll(s, -1, axis=0) - np.roll(s, 1, axis=0))
    gy = 0.5 * (np.roll(s, -1, axis=1) - np.roll(s, 1, axis=1))
    z = (gx + 1j * gy) ** 2                      # |grad|^2 * exp(2 i theta_normal)
    try:
        from scipy.ndimage import gaussian_filter
        sig = (smoothing if nx > 1 else 0, smoothing if ny > 1 else 0)
        z = gaussian_filter(z.real, sig, mode="wrap") + \
            1j * gaussian_filter(z.imag, sig, mode="wrap")
    except Exception:                            # scipy optional
        pass
    theta = np.angle(z) / 2.0                     # recovered normal orientation
    nrm_x, nrm_y = np.cos(theta), np.sin(theta)   # unit normal everywhere
    tx, ty = nrm_y, -nrm_x                         # tangent = normal rotated 90 deg
    return tx.astype(complex), ty.astype(complex)


def tangent_terms(tx: np.ndarray, ty: np.ndarray):
    r"""Real-space projection terms ``(Pxx, Pxy, Pyx, Pyy)`` of Liu (2012) eq. 50.

    Built from the (complex) tangent vector field ``(tx, ty)``.  Note the Liu-Fan
    diagonal-swap correction (``Pxx`` uses ``|ty|^2``, ``Pyy`` uses ``|tx|^2``),
    which is what makes a 1-D x-grating (``tx=0``) reduce to the inverse rule along
    x -- i.e. exactly the current ``li`` result.
    """
    denom = np.abs(tx) ** 2 + np.abs(ty) ** 2
    denom = np.where(np.isclose(denom, 0.0), 1.0, denom)
    Pyy = np.abs(tx) ** 2 / denom
    Pyx = np.conj(tx) * ty / denom
    Pxy = tx * np.conj(ty) / denom
    Pxx = np.abs(ty) ** 2 / denom
    return Pxx, Pxy, Pyx, Pyy


def inplane_tensor(eps_grid_eng: np.ndarray, tx: np.ndarray, ty: np.ndarray,
                   grid: HarmonicGrid):
    r"""Full in-plane permittivity tensor blocks ``(E00, E01, E10, E11)``.

    Mirrors ``E2 = blkdiag(<<eps>>) - blkdiag(<<eps>> - <<1/eps>>^{-1}) @ P`` with
    ``P = [[<<Pyy>>, <<Pyx>>], [<<Pxy>>, <<Pxx>>]]`` (Liu 2012 eq. 51).  Each block
    is a ``(P, P)`` convolution operator in Ikarus's conventions.

    ``eps_grid_eng`` is the engineering-convention permittivity grid (already
    conjugated + anomaly-regularized), consistent with the rest of the solver.
    For a uniform cell or an axis-aligned boundary this reduces to the ``li``
    operators; the off-diagonal ``E01``/``E10`` are what curved boundaries need.
    """
    ge = np.asarray(eps_grid_eng)
    eps_hat = convolution_matrix(ge, grid)               # <<eps>>
    eta_inv = np.linalg.inv(convolution_matrix(1.0 / ge, grid))  # <<1/eps>>^{-1}
    delta = eps_hat - eta_inv                            # 0 where eps is continuous

    Pxx, Pxy, Pyx, Pyy = tangent_terms(tx, ty)
    cm = lambda f: convolution_matrix(np.asarray(f, dtype=complex), grid)
    E00 = eps_hat - delta @ cm(Pyy)
    E01 = -delta @ cm(Pyx)
    E10 = -delta @ cm(Pxy)
    E11 = eps_hat - delta @ cm(Pxx)
    return E00, E01, E10, E11
