"""Differentiable (JAX) forward solver -- the engine for adjoint inverse design.

Experimental. Requires the optional ``[grad]`` extra::

    pip install "ikarus-rcwa[grad]"

The NumPy core (:mod:`ikarus.core`) remains the canonical solver; this module
mirrors it (pinned to ~1e-13 by the test suite) so that ``jax.grad`` of any
figure of merit is the adjoint method: the gradient with respect to *every*
input pixel costs about one extra forward solve.

Quick taste::

    import jax, jax.numpy as jnp
    from ikarus import HarmonicGrid
    from ikarus.grad import solve

    def loss(rho):                      # rho: (Nx, Ny) density in [0, 1]
        eps = 1.0 + rho * (12.25 - 1.0)
        sol = solve([eps], [300e-9], 1.0, 1.0, HarmonicGrid(8, 8),
                    0.0, 0.0, 500e-9, 500e-9, 700e-9, (1.0, 0.0),
                    factorization="li")
        return -sol.R_total             # maximize reflection

    g = jax.grad(loss)(rho0)            # d(loss)/d(every pixel), one adjoint

Higher-level topology-optimization drivers (density filtering, binarization
projection, optax loops through the ``MetaAtom``/``Structure`` API) build on
this and land in a later phase.
"""

try:
    import jax as _jax  # noqa: F401
    _HAS_JAX = True
except ModuleNotFoundError:
    _HAS_JAX = False

if _HAS_JAX:
    import jax as _jax_mod

    if not _jax_mod.config.jax_enable_x64:
        # RCWA needs double precision; JAX defaults to float32.
        _jax_mod.config.update("jax_enable_x64", True)

    from ._eig import eig
    from ._solver import GradSolution, solve, tangent_fields_for

    __all__ = ["solve", "GradSolution", "tangent_fields_for", "eig"]
else:  # pragma: no cover - exercised only without the extra installed
    __all__ = []

    def __getattr__(name):
        raise ModuleNotFoundError(
            "ikarus.grad requires JAX, which is an optional dependency. "
            "Install it with:  pip install \"ikarus-rcwa[grad]\""
        )
