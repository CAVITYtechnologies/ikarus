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
    """Resolve a concrete JAX device.

    ``"auto"`` gracefully uses the best available accelerator (CPU if none, no
    error).  An **explicit** accelerator request (``"cuda"``/``"gpu"``/``"tpu"``/
    ``"metal"``) that cannot be honored **raises** -- so you never silently
    benchmark the CPU while believing you are on the GPU (the native-Windows trap,
    where JAX has no CUDA build).  Want a quiet fallback? use ``"auto"`` or ``"cpu"``.

    Platform names differ across plugins (CUDA registers ``'gpu'``, Apple's plugin
    ``'METAL'``), so we select by "first non-CPU, non-Metal device JAX can see".
    """
    import jax

    if requested == "cpu-jax":
        return jax.devices("cpu")[0]
    devs = jax.devices()
    # Apple Metal (jax-metal) can't do float64 or complex eig -- the two operations
    # RCWA depends on -- so it is never "usable" for this solver.
    usable = [d for d in devs
              if d.platform != "cpu" and d.platform.lower() != "metal"]
    if requested == "auto":
        return usable[0] if usable else jax.devices("cpu")[0]
    if requested == "metal":
        raise RuntimeError(
            "device='metal': Apple Metal (jax-metal) does not support float64 or "
            "complex eigendecomposition -- the operations RCWA needs -- so it cannot "
            "run this solver. Use device='cpu', or a CUDA GPU on Linux/WSL2.")
    # explicit 'cuda' / 'gpu' / 'tpu'
    if usable:
        return usable[0]
    raise RuntimeError(
        f"device={requested!r} requested but JAX sees no usable GPU "
        f"(jax.devices() = {devs}). JAX GPU builds are Linux/WSL2-only -- native "
        f"Windows and macOS jaxlib are CPU-only. Install \"jax[cuda12]\" on "
        f"Linux/WSL2 for the NVIDIA GPU, or use device='cpu' (or device='auto' to "
        f"fall back to CPU without error).")


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
