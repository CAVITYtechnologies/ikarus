"""Ikarus: high-precision 2-D RCWA photonics simulation.

Rigorous coupled-wave analysis (Fourier modal method) for 2-D periodic photonic
structures -- metasurfaces, gratings and photonic crystals -- with full vectorial
polarization, anisotropic materials, field reconstruction and HDF5 I/O.

Quick start::

    import numpy as np
    from ikarus import RCWA

    rcwa = RCWA(period_x=1e-6, period_y=1e-6, resolution=64, n_orders=15)
    rcwa.add_uniform_layer(height=np.inf, material='Air')
    rcwa.add_uniform_layer(height=200e-9, material='Si')
    rcwa.add_uniform_layer(height=np.inf, material='SiO2')
    rcwa.set_source(wavelength=1550e-9, theta=0, polarization='linear')
    T, R, result = rcwa.simulate()
"""

from __future__ import annotations

from .core import (
    RCWA,
    SimulationResult,
    Source,
    Layer,
    Material,
    MaterialLibrary,
    default_library,
    HarmonicGrid,
)
from . import shapes

__version__ = "0.1.0"

__all__ = [
    "RCWA",
    "SimulationResult",
    "Source",
    "Layer",
    "Material",
    "MaterialLibrary",
    "default_library",
    "HarmonicGrid",
    "shapes",
    "__version__",
]
