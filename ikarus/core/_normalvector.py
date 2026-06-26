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

Status: WIP on ``feature/normal-vector``.  The tensor construction below is
complete; the tangent-field generator (:func:`tangent_field`, a Newton-step solve
after Schuster/Liu) and the integration into ``layer_modes``' ``Q`` matrix are
being validated empirically against FMMax (``Formulation.NORMAL``) -- see
``~/Desktop/ikarus_metamirror/fmmax_validation/``.
"""

from __future__ import annotations

import numpy as np

from .fourier import HarmonicGrid, convolution_matrix


def tangent_field(eps_grid_eng: np.ndarray, smoothing: float = 1.0):
    r"""Tangent vector field ``(tx, ty)`` for the normal-vector method.

    The local boundary **normal** is the gradient of a (periodically) smoothed copy
    of the permittivity; the **tangent** is that normal rotated 90 deg.  Boundaries
    of any orientation get a faithful normal, so the inverse rule is applied along
    the true normal everywhere (not just along x/y as the separable rule does).

    ``smoothing`` is the Gaussian blur width in pixels -- the length over which the
    rule is blended across an edge.

    .. note::
        **WIP placeholder.** This smoothed-gradient field is *zero in the bulk*
        (away from boundaries) and cancels at symmetry centres, so it does not yet
        give a faithful unit-magnitude field everywhere -- ``factorization="normal"``
        is therefore not yet validated.  The shipping version will use the
        variational construction (a single Newton step on the field's Fourier
        coefficients, after FMMax/Schuster) which yields a smooth unit field over
        the whole cell.  The tensor assembly and Q-integration that consume this
        field are already verified correct (they reduce to ``li`` exactly given a
        proper field).
    """
    s = np.abs(np.asarray(eps_grid_eng)).astype(float)
    if smoothing and smoothing > 0:
        try:
            from scipy.ndimage import gaussian_filter
            s = gaussian_filter(s, smoothing, mode="wrap")
        except Exception:                       # scipy optional -> fall back to raw gradient
            pass
    # periodic central-difference gradient = the (un-normalized) normal direction
    gx = 0.5 * (np.roll(s, -1, axis=0) - np.roll(s, 1, axis=0))
    gy = 0.5 * (np.roll(s, -1, axis=1) - np.roll(s, 1, axis=1))
    tx, ty = gy, -gx                            # tangent = normal rotated 90 deg
    norm = np.hypot(tx, ty)
    safe = norm > 1e-12
    tx = np.where(safe, tx / np.where(safe, norm, 1.0), 0.0)
    ty = np.where(safe, ty / np.where(safe, norm, 1.0), 0.0)
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
