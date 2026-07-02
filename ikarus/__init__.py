"""Ikarus: high-precision 2-D RCWA photonics simulation.

Rigorous coupled-wave analysis (Fourier modal method) for 2-D periodic photonic
structures -- metasurfaces, gratings and photonic crystals -- with full vectorial
polarization, field reconstruction, inverse design and HDF5 I/O.

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
    AnisotropicMaterial,
    Material,
    MaterialLibrary,
    default_library,
    uniaxial,
    HarmonicGrid,
)
from . import shapes
from .sweep import Sweep, SweepResult
from ._progress import progress

__version__ = "0.9.0"


def ai_guide() -> str:
    """Return the condensed Ikarus reference written for AI assistants / LLMs.

    The text ships inside the installed package, so any session can load full
    expert context regardless of where it is running::

        python -c "import ikarus; print(ikarus.ai_guide())"

    It is the same content as the docs' *Ikarus for AI Assistants* page.
    """
    from pathlib import Path

    return (Path(__file__).parent / "AI_GUIDE.md").read_text(encoding="utf-8")

__all__ = [
    "RCWA",
    "SimulationResult",
    "Source",
    "Layer",
    "AnisotropicMaterial",
    "Material",
    "MaterialLibrary",
    "default_library",
    "uniaxial",
    "HarmonicGrid",
    "shapes",
    "Sweep",
    "SweepResult",
    "progress",
    "ai_guide",
    "__version__",
]
