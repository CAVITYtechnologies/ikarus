"""Tests for the differentiable (JAX) solver mirror, ikarus.grad.

Contract: ikarus.grad.solve must reproduce the NumPy core (the canonical,
FMMax-validated implementation) to near machine precision across the geometry
zoo, and its jax.grad must match finite differences. Skipped wholesale if the
optional [grad] extra (JAX) is not installed.
"""

import numpy as np
import pytest

jax = pytest.importorskip("jax")

import jax.numpy as jnp  # noqa: E402

from ikarus import shapes  # noqa: E402
from ikarus.core.fourier import HarmonicGrid  # noqa: E402
from ikarus.core import solver as npsolver  # noqa: E402
from ikarus.grad import solve, tangent_fields_for  # noqa: E402

WL = 700e-9
PITCH = 400e-9


def _np_reference(eps_grids, heights, grid, factorization, pol=(1.0, 0.0),
                  kx0=0.0, ky0=0.0, eps_ref=1.0, eps_trn=1.0):
    sol = npsolver.solve_stack(
        eps_grids=list(eps_grids), heights=list(heights),
        eps_ref=eps_ref, eps_trn=eps_trn, grid=grid, kx0=kx0, ky0=ky0,
        period_x=PITCH, period_y=PITCH, wavelength=WL,
        polarization_xy=pol, factorization=factorization)
    return sol


def _grating_grid(nx=256, ny=2, n_hi=3.5):
    g = np.ones((nx, ny), dtype=complex)
    g[nx // 2:, :] = n_hi ** 2
    return g


def _cylinder_grid(N=64, n_hi=3.5):
    mask = shapes.circle(center=(0.5, 0.5), radius=0.30, grid_shape=(N, N))
    return np.where(np.asarray(mask) > 0, n_hi ** 2, 1.0).astype(complex)


def test_pins_1d_grating_li():
    grid = HarmonicGrid(10, 0)
    g = _grating_grid()
    ref = _np_reference([g], [300e-9], grid, "li")
    sol = solve([g], [300e-9], 1.0, 1.0, grid, 0.0, 0.0, PITCH, PITCH, WL,
                (1.0, 0.0), factorization="li")
    assert abs(float(sol.R_total) - ref.R_total) < 1e-12
    assert abs(float(sol.T_total) - ref.T_total) < 1e-12
    # complex amplitudes (physics convention) must match too -- phase objectives
    # depend on them.
    np.testing.assert_allclose(np.array(sol.rx), ref.rx, atol=1e-12)


def test_pins_2d_cylinder_laurent_and_li():
    grid = HarmonicGrid(5, 5)
    g = _cylinder_grid()
    for fac in ("laurent", "li"):
        ref = _np_reference([g], [200e-9], grid, fac, pol=(0.0, 1.0))
        sol = solve([g], [200e-9], 1.0, 1.0, grid, 0.0, 0.0, PITCH, PITCH, WL,
                    (0.0, 1.0), factorization=fac)
        assert abs(float(sol.R_total) - ref.R_total) < 1e-12, fac


def test_pins_2d_cylinder_normal_vector():
    """The flagship factorization: identical tangent field -> identical result."""
    grid = HarmonicGrid(5, 5)
    g = _cylinder_grid()
    ref = _np_reference([g], [200e-9], grid, "auto")
    sol = solve([g], [200e-9], 1.0, 1.0, grid, 0.0, 0.0, PITCH, PITCH, WL,
                (1.0, 0.0), factorization="auto")
    assert abs(float(sol.R_total) - ref.R_total) < 1e-12


def test_pins_multilayer_oblique_mixed_pol():
    """Uniform film + patterned layer + uniform film, oblique, mixed pol,
    lossy substrate -- the everything test."""
    grid = HarmonicGrid(4, 4)
    g = _cylinder_grid(N=48)
    film = np.full((48, 48), 2.25 + 0j)          # uniform SiO2-ish layer
    layers = [film, g, np.full((48, 48), 4.0 + 0.1j)]
    heights = [120e-9, 200e-9, 80e-9]
    kx0 = np.sin(np.deg2rad(25.0))
    pol = (0.6, 0.8)
    ref = _np_reference(layers, heights, grid, "li", pol=pol, kx0=kx0,
                        eps_trn=2.1)
    sol = solve(layers, heights, 1.0, 2.1, grid, kx0, 0.0, PITCH, PITCH, WL,
                pol, factorization="li")
    assert abs(float(sol.R_total) - ref.R_total) < 1e-12
    assert abs(float(sol.T_total) - ref.T_total) < 1e-12


def test_gradient_matches_finite_differences():
    """d(R)/d(rho_i) via jax.grad vs central finite differences on the core."""
    grid = HarmonicGrid(6, 0)
    NX = 128
    rng = np.random.default_rng(3)
    rho = np.clip(0.5 + 0.3 * rng.standard_normal(NX), 0.05, 0.95)

    def eps_of(r):
        line = 1.0 + jnp.asarray(r) * (12.25 - 1.0)
        return jnp.tile(line[:, None], (1, 2)).astype(complex)

    def R(r):
        return solve([eps_of(r)], [300e-9], 1.0, 1.0, grid, 0.0, 0.0,
                     PITCH, PITCH, WL, (1.0, 0.0), factorization="li").R_total

    g = np.array(jax.grad(R)(jnp.asarray(rho)))
    step = 1e-5
    for i in (17, 64, 100):
        rp, rm = rho.copy(), rho.copy()
        rp[i] += step
        rm[i] -= step
        fd = (float(R(jnp.asarray(rp))) - float(R(jnp.asarray(rm)))) / (2 * step)
        assert abs(g[i] - fd) <= 1e-5 * max(abs(fd), 1e-6), f"pixel {i}"


def test_gradient_wrt_height():
    grid = HarmonicGrid(6, 0)
    g0 = _grating_grid(nx=128)

    def R(h):
        return solve([jnp.asarray(g0)], [h], 1.0, 1.0, grid, 0.0, 0.0,
                     PITCH, PITCH, WL, (1.0, 0.0), factorization="li").R_total

    gh = float(jax.grad(R)(300e-9))
    step = 1e-12
    fd = (float(R(300e-9 + step)) - float(R(300e-9 - step))) / (2 * step)
    assert abs(gh - fd) <= 1e-5 * abs(fd)


def test_jit_with_precomputed_tangent_fields():
    """The optimizer-loop pattern: tangent fields precomputed outside jit."""
    grid = HarmonicGrid(5, 5)
    g = _cylinder_grid()
    tf = tangent_fields_for([g])

    @jax.jit
    def R(eps):
        return solve([eps], [200e-9], 1.0, 1.0, grid, 0.0, 0.0, PITCH, PITCH,
                     WL, (1.0, 0.0), factorization="auto",
                     tangent_fields=tf).R_total

    ref = _np_reference([g], [200e-9], grid, "auto")
    assert abs(float(R(jnp.asarray(g))) - ref.R_total) < 1e-12
    # and it is differentiable under jit
    grad = jax.jit(jax.grad(lambda e: solve(
        [e], [200e-9], 1.0, 1.0, grid, 0.0, 0.0, PITCH, PITCH, WL,
        (1.0, 0.0), factorization="auto", tangent_fields=tf).R_total))
    gval = np.array(grad(jnp.asarray(g)))
    assert np.all(np.isfinite(gval.real))


def test_tracing_without_tangent_fields_raises():
    grid = HarmonicGrid(3, 3)
    g = _cylinder_grid(N=32)
    with pytest.raises(ValueError, match="tangent_fields"):
        jax.grad(lambda e: solve(
            [e], [200e-9], 1.0, 1.0, grid, 0.0, 0.0, PITCH, PITCH, WL,
            (1.0, 0.0), factorization="auto").R_total)(jnp.asarray(g))


def test_anisotropic_layers_rejected():
    grid = HarmonicGrid(3, 3)
    comps = tuple(np.ones((32, 32), dtype=complex) for _ in range(5))
    with pytest.raises(NotImplementedError, match="anisotropic"):
        solve([comps], [200e-9], 1.0, 1.0, grid, 0.0, 0.0, PITCH, PITCH, WL,
              (1.0, 0.0))
