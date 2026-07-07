"""Density-based topology-optimization machinery (filter + projection).

The standard relax-and-project scheme for freeform pixel maps:

1. each pixel is a continuous density ``rho in [0, 1]``;
2. a **conic filter** (radius = half the minimum feature size) smooths the
   density, enforcing a fabricable length scale;
3. a **tanh projection** with sharpness ``beta`` pushes the filtered density
   toward binary; ``beta`` is ramped over the optimization so the design ends
   effectively two-level;
4. the final design is the hard threshold of the filtered density at
   ``eta = 0.5``.

All operations are differentiable JAX ops (the filter is a periodic FFT
convolution, matching the periodic unit cell).
"""

from __future__ import annotations

import numpy as np
import jax.numpy as jnp


def conic_kernel_fft(shape: tuple[int, int], radius_px: float) -> np.ndarray:
    """FFT of the normalized conic (linear hat) kernel on a periodic grid.

    Static (NumPy) -- compute once and close over it; only the density is
    traced.  ``radius_px <= 0.5`` returns an identity kernel (no filtering).
    """
    nx, ny = shape
    if radius_px <= 0.5:
        k = np.zeros(shape)
        k[0, 0] = 1.0
        return np.fft.fft2(k)
    x = np.arange(nx)
    y = np.arange(ny)
    dx = np.minimum(x, nx - x)[:, None]          # periodic distances
    dy = np.minimum(y, ny - y)[None, :]
    r = np.hypot(dx, dy)
    k = np.maximum(0.0, 1.0 - r / radius_px)
    k /= k.sum()
    return np.fft.fft2(k)


def conic_filter(rho: jnp.ndarray, kernel_fft) -> jnp.ndarray:
    """Periodic convolution of the density with the conic kernel."""
    return jnp.real(jnp.fft.ifft2(jnp.fft.fft2(rho) * kernel_fft))


def tanh_projection(rho: jnp.ndarray, beta, eta: float = 0.5) -> jnp.ndarray:
    """Smoothed Heaviside pushing ``rho`` toward {0, 1}; sharpness ``beta``."""
    num = jnp.tanh(beta * eta) + jnp.tanh(beta * (rho - eta))
    den = jnp.tanh(beta * eta) + jnp.tanh(beta * (1.0 - eta))
    return num / den


def beta_schedule(steps: int, beta_min: float = 8.0, beta_max: float = 256.0):
    """Geometric beta ramp: doubles in equal segments from min to max."""
    n_stages = max(1, int(np.round(np.log2(beta_max / beta_min))) + 1)
    betas = beta_min * 2.0 ** np.minimum(
        np.floor(np.arange(steps) * n_stages / max(steps, 1)), n_stages - 1)
    return np.minimum(betas, beta_max)
