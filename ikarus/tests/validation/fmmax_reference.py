"""Independent cross-code reference via **FMMax**.

FMMax (https://github.com/facebookresearch/fmmax) is a separate, peer-reviewed
JAX implementation of the Fourier Modal Method.  We use it as an *external* check
on Ikarus's Fourier factorization -- the property that actually distinguishes a
faithful high-contrast solver:

* faithful factorization  -- Ikarus ``normal``/``li``  <->  FMMax ``NORMAL``/``JONES_DIRECT``
* direct (Laurent) rule    -- Ikarus ``laurent``        <->  FMMax ``FFT``

On a high-contrast TM structure the faithful pair must agree with each other and
converge fast to the true answer, while the direct rule converges only as
``O(1/M)`` (Lalanne & Morris 1996; Li 1996).

The harness below is validated end-to-end against the analytic stratified-medium
(Fresnel) solution to < 1e-6 (see ``test_crosscode.py``), so agreement on a
grating is a genuine cross-code result, not a coincidence of conventions.

``fmmax`` is an optional cross-check dependency, **not** an Ikarus requirement;
importing this module without it raises, and the tests ``importorskip`` it.
"""

from __future__ import annotations

import numpy as np

# Map our factorization vocabulary onto FMMax's formulation enum.
_FORMULATIONS = {
    "normal": "NORMAL",          # faithful: normal-vector / Fast Fourier Factorization
    "jones": "JONES_DIRECT",     # faithful: vector Jones method
    "fft": "FFT",                # direct rule (the grcwa/torcwa-equivalent)
    "laurent": "FFT",            # alias: FFT *is* the direct/Laurent rule in FMMax
}


def stack_RT(
    eps_layers,
    thicknesses,
    period,
    wavelength,
    num_terms,
    formulation: str = "normal",
    pol: str = "TM",
):
    """Total reflectance/transmittance of a layered periodic stack, via FMMax.

    Parameters
    ----------
    eps_layers:
        List of 2-D permittivity grids ``eps(x, y)`` -- cover, interior..., substrate.
        The first and last are the semi-infinite ports.  All grids share one shape.
    thicknesses:
        Matching list of layer thicknesses (meters).  The two port thicknesses are
        ignored (semi-infinite), so any value -- e.g. ``0.0`` -- is fine there.
    period:
        Square-cell pitch (scalar) or ``(period_x, period_y)`` (meters).
    wavelength:
        Free-space wavelength (meters).
    num_terms:
        Approximate number of Fourier terms (FMMax picks a circular truncation).
    formulation:
        ``'normal'``/``'jones'`` (faithful) or ``'fft'``/``'laurent'`` (direct rule).
    pol:
        ``'TE'`` or ``'TM'`` at normal incidence.

    Returns
    -------
    (R, T): floats, summed over all propagating orders (lossless => R + T == 1).
    """
    import jax

    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    import fmmax

    form = getattr(fmmax.Formulation, _FORMULATIONS[formulation.lower()])
    px, py = (period, period) if np.isscalar(period) else period
    plv = fmmax.LatticeVectors(u=jnp.array([px, 0.0]), v=jnp.array([0.0, py]))
    expansion = fmmax.generate_expansion(plv, approximate_num_terms=int(num_terms))
    in_plane_wavevector = jnp.zeros((2,))  # normal incidence
    wl = jnp.asarray(float(wavelength))

    solves = [
        fmmax.eigensolve_isotropic_media(
            wl, in_plane_wavevector, plv, jnp.asarray(np.asarray(eps, dtype=complex)),
            expansion, formulation=form,
        )
        for eps in eps_layers
    ]
    smat = fmmax.stack_s_matrix(solves, [jnp.asarray(float(t)) for t in thicknesses])

    # Incident: unit forward amplitude at the zeroth order, chosen polarization.
    # Eigenmode ordering is [family-0 (TE), family-1 (TM)] each of length num_terms.
    n_terms = expansion.num_terms
    bc = np.asarray(expansion.basis_coefficients)
    idx0 = int(np.argmin(np.abs(bc).sum(axis=1)))
    off = 0 if pol.upper() == "TE" else n_terms
    fwd = jnp.zeros((2 * n_terms, 1), dtype=complex).at[idx0 + off, 0].set(1.0)

    fwd_end = smat.s11 @ fwd        # transmitted (forward) in the substrate
    bwd_start = smat.s21 @ fwd      # reflected  (backward) in the cover

    inc_f, _ = fmmax.amplitude_poynting_flux(fwd, jnp.zeros_like(fwd), solves[0])
    _, ref_b = fmmax.amplitude_poynting_flux(jnp.zeros_like(bwd_start), bwd_start, solves[0])
    trn_f, _ = fmmax.amplitude_poynting_flux(fwd_end, jnp.zeros_like(fwd_end), solves[-1])

    s_in = float(jnp.sum(inc_f))
    R = float(-jnp.sum(ref_b) / s_in)   # backward flux is negative -> flip sign
    T = float(jnp.sum(trn_f) / s_in)
    return R, T


def uniform_grid(n_index, shape=(16, 16)):
    """A uniform permittivity grid for a material of refractive index ``n_index``."""
    return np.full(shape, complex(n_index) ** 2)


def binary_grating_grid(n_hi, duty=0.5, shape=(128, 128)):
    """A 1-D lamellar grating grid: first ``duty`` fraction along x is ``n_hi``.

    Matches Ikarus's ``topo[: Nx*duty] = 1`` convention (high index occupies the
    first part of the period), so the two solvers see the *same* structure.
    """
    grid = np.ones(shape, dtype=complex)
    grid[: int(shape[0] * duty), :] = complex(n_hi) ** 2
    return grid
