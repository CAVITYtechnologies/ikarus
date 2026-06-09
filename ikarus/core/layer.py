"""Layer definitions for the RCWA stack.

A :class:`Layer` knows its physical thickness and how to produce the real-space
permittivity distribution ``eps(x, y)`` sampled on the unit-cell grid at a given
wavelength.  Two flavours exist:

* **Uniform** layers -- a single material filling the cell.  The two outermost
  layers of a stack must be uniform and semi-infinite (``height = inf``); they
  are the cover (incidence) and substrate (transmission) regions.
* **Patterned** layers -- an integer ``topology`` pixel map selecting among a
  list of ``materials``.

Anisotropic materials are supported by allowing a material to resolve to a 3x3
permittivity tensor; uniform-isotropic is the common fast path.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .materials import MaterialLibrary


@dataclass
class Layer:
    height: float
    # Uniform layer: a single material specifier.
    material: object | None = None
    # Patterned layer: integer pixel map + material list.
    topology: np.ndarray | None = None
    materials: list | None = None
    # Per-layer sampling resolution (falls back to the solver's global value).
    resolution: tuple[int, int] | None = None
    name: str = ""

    def __post_init__(self) -> None:
        if self.height < 0:
            raise ValueError("layer height must be non-negative (or inf)")
        if self.material is None and self.topology is None:
            raise ValueError("layer needs either `material` or `topology`")
        if self.topology is not None:
            self.topology = np.asarray(self.topology)
            if self.materials is None:
                raise ValueError("patterned layer requires a `materials` list")
            n_mat = int(self.topology.max()) + 1
            if len(self.materials) < n_mat:
                raise ValueError(
                    f"topology references {n_mat} materials but only "
                    f"{len(self.materials)} provided"
                )

    # -- queries -----------------------------------------------------------
    @property
    def is_uniform(self) -> bool:
        return self.topology is None

    @property
    def is_semi_infinite(self) -> bool:
        return np.isinf(self.height)

    # -- permittivity ------------------------------------------------------
    def permittivity_grid(
        self,
        wavelength: float,
        library: MaterialLibrary,
        resolution: tuple[int, int],
    ) -> np.ndarray:
        """Sample ``eps(x, y)`` on an ``(Nx, Ny)`` grid for this layer.

        For a uniform layer the grid is filled with a constant.  For a patterned
        layer the integer topology is resampled (nearest-neighbour) to the
        requested resolution and each material index replaced by its complex
        permittivity at ``wavelength``.
        """
        nx, ny = self.resolution or resolution
        if self.is_uniform:
            eps = library.permittivity(self.material, wavelength)
            return np.full((nx, ny), eps, dtype=complex)

        topo = self._resample_topology(nx, ny)
        eps_values = np.array(
            [library.permittivity(m, wavelength) for m in self.materials],
            dtype=complex,
        )
        return eps_values[topo]

    def uniform_permittivity(self, wavelength: float, library: MaterialLibrary) -> complex:
        """Scalar permittivity of a uniform layer (raises if patterned)."""
        if not self.is_uniform:
            raise ValueError("layer is patterned, no single permittivity")
        return library.permittivity(self.material, wavelength)

    def _resample_topology(self, nx: int, ny: int) -> np.ndarray:
        """Nearest-neighbour resample of the integer topology to ``(nx, ny)``."""
        tnx, tny = self.topology.shape
        if (tnx, tny) == (nx, ny):
            return self.topology
        ix = (np.arange(nx) * tnx / nx).astype(int).clip(0, tnx - 1)
        iy = (np.arange(ny) * tny / ny).astype(int).clip(0, tny - 1)
        return self.topology[np.ix_(ix, iy)]
