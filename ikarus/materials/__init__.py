"""Bundled optical materials and the material API.

The ``*.json`` files in this directory are Ikarus's shipped material database.
The public materials API is re-exported here for discoverability, so both

    from ikarus import default_library
    from ikarus.materials import default_library

work (the objects themselves live in :mod:`ikarus.core.materials`).
"""

from ikarus.core.materials import (
    AnisotropicMaterial,
    Material,
    MaterialLibrary,
    default_library,
    uniaxial,
)

__all__ = [
    "default_library",
    "MaterialLibrary",
    "Material",
    "AnisotropicMaterial",
    "uniaxial",
]
