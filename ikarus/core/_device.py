"""Optional accelerator (GPU / Apple Metal) forward path.

The default engine is the NumPy core.  Passing ``device != "cpu"`` to
:class:`~ikarus.RCWA` routes the *forward* solve through :func:`ikarus.grad.solve`
-- the parity-tested JAX mirror of the core -- which runs wherever JAX runs:
NVIDIA CUDA, Apple Metal, or JAX-on-CPU.  Same physics, same ``SimulationResult``;
only the device changes.  Nothing else about the API moves.

**Scope of the accelerated path.**  Forward per-order efficiencies, the complex
zeroth-order coefficients (hence phase) and exit angles, for **isotropic** stacks
and every factorization rule.  *Not yet* on the accelerator: anisotropic (tensor)
layers, and real-space field reconstruction (:meth:`RCWA.get_fields`) -- both raise
a clear error asking for ``device="cpu"``.

**Where it pays off.**  RCWA cost is dominated by a dense complex eigendecomposition
of a ``2*(2M+1)**2`` matrix, so the accelerator wins only when that matrix is large
(big ``n_orders`` in 2-D).  For small or 1-D problems the CPU core is faster --
kernel-launch and host<->device transfer dominate.
"""

from __future__ import annotations

import warnings
from types import SimpleNamespace

import numpy as np

from .solver import uniform_modes, wavevector_matrices

_CPU = {"cpu"}
# 'cpu-jax' forces the JAX path onto the CPU -- used to prove parity with the
# NumPy core without any accelerator present (the whole test story on a Mac).
_ACCEL = {"gpu", "cuda", "metal", "tpu", "auto", "cpu-jax"}


def normalize_device(device) -> str:
    """Canonicalize a user device string; raise on anything unexpected."""
    d = str(device or "cpu").lower()
    if d in _CPU:
        return "cpu"
    if d in _ACCEL:
        return d
    raise ValueError(
        f"unknown device {device!r}; expected 'cpu', 'gpu'/'cuda', 'metal', "
        f"'auto', or 'cpu-jax'")


def _pick_jax_device(requested: str):
    """Resolve a concrete JAX device, falling back to CPU with a clear warning.

    Platform names differ across plugins (CUDA registers ``'gpu'``, Apple's plugin
    ``'METAL'``), so rather than hard-code them we simply take the first
    non-CPU device JAX can see -- which is whatever accelerator plugin is installed.
    """
    import jax

    if requested == "cpu-jax":
        return jax.devices("cpu")[0]
    accel = [d for d in jax.devices() if d.platform != "cpu"]
    # Apple Metal (jax-metal) can't do float64 or complex eigendecomposition yet --
    # the two operations RCWA depends on -- so it cannot accelerate this solver.
    # Drop it here so we fall back cleanly instead of crashing mid-solve with an
    # XLA "UNIMPLEMENTED" error. (Revisit if a future jax-metal gains f64/complex.)
    metal = [d for d in accel if d.platform.lower() == "metal"]
    usable = [d for d in accel if d.platform.lower() != "metal"]
    if usable:
        return usable[0]
    if metal:
        warnings.warn(
            "an Apple Metal GPU is present, but jax-metal does not yet support "
            "float64 or complex eigendecomposition -- the operations RCWA needs -- "
            "so it cannot accelerate this solver. Running on JAX-CPU instead; use an "
            "NVIDIA CUDA GPU for acceleration.", RuntimeWarning, stacklevel=3)
    elif requested != "auto":
        warnings.warn(
            f"device={requested!r} requested but JAX sees no usable accelerator. "
            f"Install the matching plugin (jax[cuda12] for NVIDIA); running on "
            f"JAX-CPU for now.", RuntimeWarning, stacklevel=3)
    return jax.devices("cpu")[0]


def jax_forward(*, eps_grids, heights, eps_ref, eps_trn, grid, kx0, ky0,
                period_x, period_y, wavelength, polarization_xy, factorization,
                device):
    """Run the forward solve on a JAX device; return a core-compatible solution.

    The returned object duck-types every attribute ``RCWA._package`` reads, so the
    accelerated path produces the *same* ``SimulationResult`` as the NumPy core.
    """
    try:
        import jax
        from ..grad import solve as jax_solve
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "device != 'cpu' needs the differentiable JAX solver: install it with "
            "pip install \"ikarus-rcwa[grad]\" (plus an accelerator plugin such as "
            "jax[cuda12] or jax-metal to actually use a GPU).") from exc

    for g in eps_grids:
        if isinstance(g, tuple):
            raise NotImplementedError(
                "anisotropic (tensor) layers are not supported on device != 'cpu' "
                "yet -- use device='cpu' for anisotropic materials.")

    jdev = _pick_jax_device(device)
    with jax.default_device(jdev):
        sol = jax_solve(
            eps_grids=[np.asarray(g, dtype=complex) for g in eps_grids],
            heights=list(heights), eps_ref=eps_ref, eps_trn=eps_trn, grid=grid,
            kx0=kx0, ky0=ky0, period_x=period_x, period_y=period_y,
            wavelength=wavelength, polarization_xy=polarization_xy,
            factorization=factorization)
        R_orders = np.asarray(sol.R_orders)
        T_orders = np.asarray(sol.T_orders)
        rx, ry, rz = np.asarray(sol.rx), np.asarray(sol.ry), np.asarray(sol.rz)
        tx, ty, tz = np.asarray(sol.tx), np.asarray(sol.ty), np.asarray(sol.tz)

    # Geometry for the exit angles -- identical to what the core stores.
    Kx, Ky = wavevector_matrices(grid, kx0, ky0, period_x, period_y, wavelength)
    Kz_ref = uniform_modes(eps_ref, Kx, Ky)[2]
    Kz_trn = uniform_modes(eps_trn, Kx, Ky)[2]

    return SimpleNamespace(
        grid=grid, R_orders=R_orders, T_orders=T_orders,
        R_total=float(R_orders.sum()), T_total=float(T_orders.sum()),
        rx=rx, ry=ry, rz=rz, tx=tx, ty=ty, tz=tz,
        Kx=Kx, Ky=Ky, Kz_ref=Kz_ref, Kz_trn=Kz_trn,
        eps_ref=eps_ref, eps_trn=eps_trn,
        device=str(jdev), _accelerated=True,
    )
