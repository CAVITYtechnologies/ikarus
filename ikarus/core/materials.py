"""Material database and dispersion handling.

Materials are stored as small JSON files (see ``ikarus/materials/*.json``) holding
tabulated complex refractive index ``n + i k`` versus wavelength, or the
parameters of a Lorentz-oscillator dispersion model.  The :class:`MaterialLibrary`
loads them lazily, interpolates to any requested wavelength with a cubic spline,
and returns the complex permittivity ``eps = (n + i k)**2``.

A material may also be specified inline as a plain number (constant index) which
is convenient for quick tests, e.g. ``add_uniform_layer(material=1.5)``.

Sign convention
---------------
We use the physics ``exp(-i omega t)`` time convention, so absorbing media have
``k > 0`` and ``Im(eps) > 0``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy.interpolate import interp1d

# Directory holding the shipped JSON material files.
_DB_DIR = Path(__file__).resolve().parent.parent / "materials"


@dataclass
class Material:
    """A single optical material with wavelength-dependent index.

    Use :meth:`from_dict` / :meth:`from_file` to construct from the JSON schema,
    or :meth:`constant` for a non-dispersive index.
    """

    name: str
    # Either tabulated data ...
    wavelength_nm: np.ndarray | None = None
    n: np.ndarray | None = None
    k: np.ndarray | None = None
    # ... or a Lorentz model: eps(w) = eps_inf + sum_j f_j w0_j^2 /
    #     (w0_j^2 - w^2 - i gamma_j w), with w = angular frequency in rad/s.
    lorentz: dict | None = None
    comment: str = ""

    _interp_n = None
    _interp_k = None

    # -- constructors ------------------------------------------------------
    @classmethod
    def constant(cls, value: complex, name: str = "const") -> "Material":
        nk = complex(value)
        return cls(name=name, wavelength_nm=np.array([1.0, 1e7]),
                   n=np.array([nk.real, nk.real]),
                   k=np.array([nk.imag, nk.imag]),
                   comment="constant index")

    @classmethod
    def from_dict(cls, data: dict) -> "Material":
        if "lorentz" in data:
            return cls(name=data["name"], lorentz=data["lorentz"],
                       comment=data.get("comment", ""))
        return cls(
            name=data["name"],
            wavelength_nm=np.asarray(data["wavelength_nm"], dtype=float),
            n=np.asarray(data["n"], dtype=float),
            k=np.asarray(data.get("k", np.zeros_like(data["n"])), dtype=float),
            comment=data.get("comment", ""),
        )

    @classmethod
    def from_file(cls, path: str | Path) -> "Material":
        with open(path, "r") as fh:
            return cls.from_dict(json.load(fh))

    # -- evaluation --------------------------------------------------------
    def _ensure_interpolators(self) -> None:
        if self.lorentz is not None or self._interp_n is not None:
            return
        wl = self.wavelength_nm
        # Cubic where we have enough points, else linear; always extrapolate by
        # clamping to the nearest tabulated value (bounds_error=False, fill).
        kind = "cubic" if wl.size >= 4 else "linear"
        self._interp_n = interp1d(
            wl, self.n, kind=kind, bounds_error=False,
            fill_value=(self.n[0], self.n[-1]),
        )
        self._interp_k = interp1d(
            wl, self.k, kind=kind, bounds_error=False,
            fill_value=(self.k[0], self.k[-1]),
        )

    def index(self, wavelength: float | np.ndarray) -> np.ndarray | complex:
        """Complex refractive index ``n + i k`` at ``wavelength`` (meters)."""
        wl_nm = np.asarray(wavelength, dtype=float) * 1e9
        if self.lorentz is not None:
            eps = self._lorentz_eps(wl_nm)
            nk = np.sqrt(eps)
            # Ensure k >= 0 (physical, absorbing) under exp(-i w t).
            nk = np.where(nk.imag < 0, np.conj(nk), nk)
            return nk if nk.ndim else complex(nk)
        self._ensure_interpolators()
        n = self._interp_n(wl_nm)
        k = self._interp_k(wl_nm)
        out = n + 1j * k
        return out if out.ndim else complex(out)

    def permittivity(self, wavelength: float | np.ndarray) -> np.ndarray | complex:
        """Relative permittivity ``eps = (n + i k)**2`` at ``wavelength``."""
        nk = self.index(wavelength)
        return nk * nk

    def _lorentz_eps(self, wl_nm: np.ndarray) -> np.ndarray:
        c = 299792458.0
        w = 2.0 * np.pi * c / (np.asarray(wl_nm) * 1e-9)
        p = self.lorentz
        eps = np.full_like(w, p.get("eps_inf", 1.0), dtype=complex)
        for osc in p["oscillators"]:
            f, w0, gamma = osc["f"], osc["w0"], osc["gamma"]
            eps = eps + f * w0**2 / (w0**2 - w**2 - 1j * gamma * w)
        return eps


@dataclass
class AnisotropicMaterial:
    """An anisotropic (birefringent) material: a permittivity *tensor*.

    The three principal refractive indices are ordinary material specifiers
    (a name like ``'SiO2'``, a number, or a :class:`Material`), so **dispersive
    anisotropy** works out of the box -- e.g. tabulated ordinary/extraordinary
    indices of calcite.  ``angle`` (degrees) rotates the principal axes in the
    x-y plane, which produces the off-diagonal ``eps_xy = eps_yx`` tensor
    components; this is how wave plates at arbitrary orientation are built.

    Scope: the permittivity tensor is ``[[eps_xx, eps_xy, 0], [eps_yx, eps_yy, 0],
    [0, 0, eps_zz]]`` -- any in-plane orientation plus a distinct z response.
    Tilted-optic-axis media (``eps_xz``/``eps_yz``) and magneto-optic gyrotropy
    (``eps_xy != eps_yx``) are out of scope.

    Anywhere Ikarus accepts a material you may equivalently pass a plain
    ``(n_x, n_y, n_z)`` tuple, or use :func:`uniaxial` for the common
    ordinary/extraordinary parametrization.
    """

    n_x: object
    n_y: object
    n_z: object
    angle: float = 0.0          # in-plane rotation of the principal axes (deg)
    name: str = "anisotropic"

    def permittivity_tensor(
        self, wavelength: float, library: "MaterialLibrary",
    ) -> tuple[complex, complex, complex, complex, complex]:
        """Tensor components ``(eps_xx, eps_xy, eps_yx, eps_yy, eps_zz)``."""
        e1 = library.permittivity(self.n_x, wavelength)   # principal, pre-rotation
        e2 = library.permittivity(self.n_y, wavelength)
        ezz = library.permittivity(self.n_z, wavelength)
        phi = np.deg2rad(self.angle)
        c, s = np.cos(phi), np.sin(phi)
        exx = e1 * c * c + e2 * s * s
        eyy = e1 * s * s + e2 * c * c
        exy = (e1 - e2) * s * c                            # reciprocal: eps_xy == eps_yx
        return exx, exy, exy, eyy, ezz


def uniaxial(n_o, n_e, axis: str | float = "z", name: str = "uniaxial") -> AnisotropicMaterial:
    """Convenience constructor for a uniaxial (birefringent) material.

    ``n_o``/``n_e`` are the ordinary/extraordinary indices (any material spec --
    a name, a number, or a :class:`Material`, so dispersion is supported).
    ``axis`` sets the optic (extraordinary) axis:

    * ``"z"`` (default) -- a *c-plate*: isotropic in-plane, distinct z response.
    * ``"x"`` / ``"y"`` -- an *a-plate*: wave-plate behaviour at normal incidence.
    * a number (degrees) -- optic axis in the x-y plane at that angle from x.
    """
    if axis == "z":
        return AnisotropicMaterial(n_o, n_o, n_e, name=name)
    if axis == "x":
        return AnisotropicMaterial(n_e, n_o, n_o, name=name)
    if axis == "y":
        return AnisotropicMaterial(n_o, n_e, n_o, name=name)
    if isinstance(axis, (int, float)) and not isinstance(axis, bool):
        return AnisotropicMaterial(n_e, n_o, n_o, angle=float(axis), name=name)
    raise ValueError(f"axis must be 'x', 'y', 'z' or an in-plane angle in degrees, got {axis!r}")


class MaterialLibrary:
    """Lazy-loading registry of named materials.

    Resolves material specifiers used throughout the API:

    * a ``str`` naming a database entry (``'Si'``) or a JSON file path,
    * a ``Material`` instance,
    * a number -> constant index,
    * an :class:`AnisotropicMaterial` or a plain ``(n_x, n_y, n_z)`` tuple
      -> anisotropic (tensor) material.
    """

    def __init__(self, db_dir: str | Path = _DB_DIR):
        self._db_dir = Path(db_dir)
        self._cache: dict[str, Material] = {}

    # -- registry ----------------------------------------------------------
    def available(self) -> list[str]:
        """Names of all materials discoverable in the database directory."""
        if not self._db_dir.exists():
            return sorted(self._cache)
        files = {p.stem for p in self._db_dir.glob("*.json")}
        return sorted(files | set(self._cache))

    def register(self, material: Material) -> None:
        self._cache[material.name] = material

    # -- anisotropy ---------------------------------------------------------
    @staticmethod
    def _coerce_anisotropic(spec) -> "AnisotropicMaterial | None":
        """Return an :class:`AnisotropicMaterial` if ``spec`` denotes one, else None.

        A plain 3-element tuple/list of scalar specs is shorthand for a diagonal
        tensor ``(n_x, n_y, n_z)``.
        """
        if isinstance(spec, AnisotropicMaterial):
            return spec
        if isinstance(spec, (tuple, list)) and len(spec) == 3:
            return AnisotropicMaterial(*spec)
        return None

    def is_anisotropic(self, spec) -> bool:
        return self._coerce_anisotropic(spec) is not None

    def permittivity_tensor(
        self, spec, wavelength: float,
    ) -> tuple[complex, complex, complex, complex, complex]:
        """Tensor components ``(eps_xx, eps_xy, eps_yx, eps_yy, eps_zz)`` of any
        material spec.  Isotropic specs return the scalar on the diagonal."""
        aniso = self._coerce_anisotropic(spec)
        if aniso is not None:
            return aniso.permittivity_tensor(wavelength, self)
        eps = self.permittivity(spec, wavelength)
        return eps, 0.0 + 0.0j, 0.0 + 0.0j, eps, eps

    def resolve(self, spec) -> Material:
        """Turn any accepted material specifier into a :class:`Material`."""
        if isinstance(spec, AnisotropicMaterial) or (
            isinstance(spec, (tuple, list)) and len(spec) == 3
        ):
            raise TypeError(
                "anisotropic material used where a scalar (isotropic) material is "
                "required -- use permittivity_tensor() for anisotropic specs"
            )
        if isinstance(spec, Material):
            return spec
        if isinstance(spec, (int, float, complex)):
            return Material.constant(spec)
        if isinstance(spec, str):
            if spec in self._cache:
                return self._cache[spec]
            path = self._db_dir / f"{spec}.json"
            if path.exists():
                mat = Material.from_file(path)
                self._cache[spec] = mat
                return mat
            # Maybe it is a direct path to a JSON file.
            if Path(spec).exists():
                mat = Material.from_file(spec)
                self._cache[mat.name] = mat
                return mat
            raise KeyError(
                f"Unknown material {spec!r}. Available: {self.available()}"
            )
        raise TypeError(f"Cannot interpret material specifier {spec!r}")

    # -- convenience used by the spec's API examples -----------------------
    def get(self, spec, wavelength: float) -> complex:
        """Return complex index ``n + i k`` of ``spec`` at ``wavelength``."""
        return self.resolve(spec).index(wavelength)

    def permittivity(self, spec, wavelength: float) -> complex:
        return self.resolve(spec).permittivity(wavelength)

    def add_from_file(self, path: str | Path, name: str | None = None,
                      persist: bool = False) -> Material:
        """Load a material from a CSV (``lambda_nm, n, k``) or JSON file.

        If ``persist`` is true the material is also written into the database
        directory as JSON so it is available to future sessions.
        """
        path = Path(path)
        if path.suffix.lower() == ".json":
            mat = Material.from_file(path)
        else:
            mat = _material_from_csv(path, name or path.stem)
        if name:
            mat.name = name
        self.register(mat)
        if persist:
            self.save(mat)
        return mat

    def save(self, material: Material) -> Path:
        """Serialize a tabulated material to the database directory as JSON."""
        self._db_dir.mkdir(parents=True, exist_ok=True)
        out = self._db_dir / f"{material.name}.json"
        payload = {
            "name": material.name,
            "comment": material.comment,
            "wavelength_nm": np.asarray(material.wavelength_nm).tolist(),
            "n": np.asarray(material.n).tolist(),
            "k": np.asarray(material.k).tolist(),
        }
        with open(out, "w") as fh:
            json.dump(payload, fh, indent=2)
        return out


def _material_from_csv(path: Path, name: str) -> Material:
    """Parse a whitespace/comma-delimited file with columns ``lambda_nm n k``."""
    data = np.genfromtxt(path, delimiter=None, comments="#")
    if data.ndim == 1:
        data = data[None, :]
    if data.shape[1] < 2:
        raise ValueError(
            f"{path}: expected at least 2 columns (wavelength_nm, n[, k])"
        )
    wl = data[:, 0]
    n = data[:, 1]
    k = data[:, 2] if data.shape[1] >= 3 else np.zeros_like(n)
    order = np.argsort(wl)
    return Material(name=name, wavelength_nm=wl[order], n=n[order], k=k[order],
                    comment=f"imported from {path.name}")


# A module-level default library so ``from ikarus import materials`` style access
# (and the RCWA façade) share one cache.
default_library = MaterialLibrary()
