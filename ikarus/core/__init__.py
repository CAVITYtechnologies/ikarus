"""Core RCWA engine: solver, layers, materials, source and Fourier machinery."""

from .rcwa import RCWA, SimulationResult
from .source import Source
from .layer import Layer
from .materials import (
    AnisotropicMaterial,
    Material,
    MaterialLibrary,
    default_library,
    uniaxial,
)
from .fourier import HarmonicGrid

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
]
