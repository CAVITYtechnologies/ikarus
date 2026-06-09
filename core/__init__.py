"""Core RCWA engine: solver, layers, materials, source and Fourier machinery."""

from .rcwa import RCWA, SimulationResult
from .source import Source
from .layer import Layer
from .materials import Material, MaterialLibrary, default_library
from .fourier import HarmonicGrid

__all__ = [
    "RCWA",
    "SimulationResult",
    "Source",
    "Layer",
    "Material",
    "MaterialLibrary",
    "default_library",
    "HarmonicGrid",
]
