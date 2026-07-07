"""Differentiable non-Hermitian eigendecomposition (custom VJP).

``jax``'s stock ``jnp.linalg.eig`` does not support reverse-mode gradients with
respect to the eigen*vectors*, which the RCWA scattering matrix needs.  This
module wraps ``jax.lax.linalg.eig`` with the eigenvector gradient of Boeddeker
et al. (2019), eq. 4.77 (https://arxiv.org/abs/1701.00392).

The gradient expression contains ``1 / (lambda_i - lambda_j)``, which blows up
for (near-)degenerate eigenvalues -- the classic differentiable-RCWA pitfall.
We regularize with a **scale-aware Lorentzian broadening**,

    1/delta  ->  conj(delta) / (|delta|^2 + eps),
    eps = max(eps_relative * max|delta|^2, eps_minimum),

following the analysis in FMMax's ``eig.py`` (which surveys the torcwa / grcwa /
rcwa_tf variants of this trick and their failure modes).  Validated in the
Phase-0 spike: gradients through the full Ikarus pipeline match central finite
differences to ~1e-7 typical / 1.8e-6 worst-case.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

_EPS_RELATIVE = 1e-12
_EPS_MINIMUM = 1e-24


@jax.custom_vjp
def eig(matrix: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Eigenvalues and right eigenvectors of a general complex matrix.

    Differentiable in reverse mode (see module docstring).  CPU-backed: JAX
    lowers general ``eig`` to LAPACK on the host, exactly like the NumPy core.
    """
    return jax.lax.linalg.eig(matrix, compute_left_eigenvectors=False)


def _eig_fwd(matrix):
    w, v = jax.lax.linalg.eig(matrix, compute_left_eigenvectors=False)
    return (w, v), (w, v)


def _eig_bwd(res, grads):
    w, v = res
    gw, gv = grads
    # Lorentzian-broadened 1/(lambda_i - lambda_j), zero on the diagonal.
    delta = w[None, :] - w[:, None]
    rng = jnp.max(jnp.abs(delta) ** 2)
    eps = jnp.maximum(_EPS_RELATIVE * rng, _EPS_MINIMUM)
    F = delta.conj() / (jnp.abs(delta) ** 2 + eps)
    n = w.shape[0]
    di = jnp.arange(n)
    F = F.at[di, di].set(0.0)

    # Boeddeker eq. 4.77 (JAX convention: cotangents w.r.t. the parameters, so
    # conjugate in and out).
    gw_c, gv_c = jnp.conj(gw), jnp.conj(gv)
    vH = jnp.conj(v.T)
    eye = jnp.eye(n, dtype=bool)
    rhs = (jnp.diag(gw_c)
           + jnp.conj(F) * (vH @ gv_c)
           - jnp.conj(F) * (vH @ v) @ jnp.where(eye, jnp.real(vH @ gv_c), 0.0)
           ) @ vH
    grad_matrix = jnp.linalg.solve(vH, rhs)
    return (jnp.conj(grad_matrix),)


eig.defvjp(_eig_fwd, _eig_bwd)
