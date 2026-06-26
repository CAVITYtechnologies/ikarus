"""The :class:`RCWA` façade -- the user-facing entry point.

This class collects the layer stack, the illumination :class:`~ikarus.core.source.Source`
and the numerical settings, then orchestrates the stateless solver to produce
diffraction efficiencies, complex reflection/transmission coefficients and (via
:mod:`ikarus.core.fields`) real-space field maps.

Typical use::

    rcwa = RCWA(period_x=1e-6, period_y=1e-6, resolution=64, n_orders=15)
    rcwa.add_uniform_layer(height=np.inf, material='Air')   # cover
    rcwa.add_uniform_layer(height=200e-9, material='Si')
    rcwa.add_uniform_layer(height=np.inf, material='SiO2')  # substrate
    rcwa.set_source(wavelength=1550e-9, theta=0, polarization='linear')
    T, R, result = rcwa.simulate()
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .fourier import HarmonicGrid
from .layer import Layer
from .materials import MaterialLibrary, default_library
from .solver import FieldSolution, solve_stack
from .source import Source


@dataclass
class SimulationResult:
    """Rich result returned as the third element of :meth:`RCWA.simulate`.

    Holds the full per-order data and a back-reference to the solver solution so
    that field reconstruction and order-resolved analysis remain available.
    """

    T: object  # scalar complex (linear) or {'co','cross'} (circular)
    R: object
    T_total: float
    R_total: float
    T_orders: np.ndarray
    R_orders: np.ndarray
    orders: tuple[np.ndarray, np.ndarray]  # (p, q) integer order arrays
    theta_out_trn: np.ndarray  # exit polar angle per order (deg), NaN if evanescent
    phi_out_trn: np.ndarray
    theta_out_ref: np.ndarray
    phi_out_ref: np.ndarray
    energy_balance: float  # R_total + T_total (==1 for lossless)
    solution: FieldSolution

    def order_index(self, p: int, q: int) -> int:
        ps, qs = self.orders
        match = np.where((ps == p) & (qs == q))[0]
        if match.size == 0:
            raise KeyError(f"order ({p}, {q}) not in truncated set")
        return int(match[0])

    @staticmethod
    def _phase(coeff):
        if isinstance(coeff, dict):
            return {k: float(np.angle(v)) for k, v in coeff.items()}
        return float(np.angle(coeff))

    @property
    def T_phase(self):
        """Phase (radians) of the zero-order transmission coefficient.

        A float for linear polarization, or ``{'co', 'cross'}`` for circular.
        """
        return self._phase(self.T)

    @property
    def R_phase(self):
        """Phase (radians) of the zero-order reflection coefficient."""
        return self._phase(self.R)


class RCWA:
    def __init__(
        self,
        period_x: float,
        period_y: float,
        resolution: int | tuple[int, int] = 32,
        n_orders: int | tuple[int, int] = 25,
        dtype=np.complex128,
        materials: Optional[MaterialLibrary] = None,
        convergence_tol: float = 1e-4,
        factorization: str = "li",
    ):
        if period_x <= 0 or period_y <= 0:
            raise ValueError("periods must be positive (meters)")
        if factorization not in ("laurent", "li"):
            raise ValueError(
                f"factorization must be 'laurent' or 'li', got {factorization!r}")
        self.period_x = float(period_x)
        self.period_y = float(period_y)
        self.resolution = self._as_pair(resolution)
        self._n_orders = self._as_pair(n_orders)
        self.dtype = dtype
        self.convergence_tol = convergence_tol
        self.factorization = factorization

        self.materials = materials or default_library
        self.layers: list[Layer] = []
        self.source: Optional[Source] = None
        self._last_solution: Optional[FieldSolution] = None
        # Convergence cache (for simulate(auto_converge='once')).
        self._converged: bool = False

    # -- configuration -----------------------------------------------------
    @staticmethod
    def _as_pair(value) -> tuple[int, int]:
        if isinstance(value, (tuple, list)):
            return int(value[0]), int(value[1])
        return int(value), int(value)

    @property
    def n_orders(self) -> tuple[int, int]:
        return self._n_orders

    @n_orders.setter
    def n_orders(self, value) -> None:
        self._n_orders = self._as_pair(value)
        self._converged = False

    @property
    def shapes(self):
        """Access the shape-primitive library: ``rcwa.shapes.circle(...)``."""
        from .. import shapes as _shapes
        return _shapes

    def add_uniform_layer(self, height: float, material, name: str = "") -> Layer:
        """Append a uniform (single-material) layer.  Use ``height=np.inf`` for
        the semi-infinite cover/substrate (first and last layers)."""
        layer = Layer(height=height, material=material, name=name)
        self.layers.append(layer)
        self._converged = False
        return layer

    def add_layer(self, height: float, topology, materials, resolution=None,
                  name: str = "") -> Layer:
        """Append a patterned layer.

        ``topology`` may be an integer array, a parametric
        :class:`~ikarus.shapes.parametric.Shape`, or any object exposing an
        ``img`` array attribute (e.g. an external topology class).
        """
        res = self._as_pair(resolution) if resolution is not None else None
        topology = self._coerce_topology(topology, res)
        layer = Layer(height=height, topology=topology, materials=list(materials),
                      resolution=res, name=name)
        self.layers.append(layer)
        self._converged = False
        return layer

    def _coerce_topology(self, topology, res):
        """Render a parametric/external topology object to an integer array."""
        if hasattr(topology, "to_grid"):            # ikarus parametric Shape
            return np.asarray(topology.to_grid(res or self.resolution)).astype(int)
        if hasattr(topology, "img"):                # external class (e.g. Topology-Species)
            return np.asarray(topology.img).astype(int)
        return np.asarray(topology)

    def set_source(self, **kwargs) -> Source:
        """Create or update the illumination.  Unspecified fields are retained
        from the previous call so sweeps can change one parameter at a time."""
        if self.source is None:
            if "wavelength" not in kwargs:
                raise ValueError("first set_source call requires `wavelength`")
            self.source = Source(**kwargs)
        else:
            self.source = self.source.copy_with(**kwargs)
        return self.source

    # -- core solve --------------------------------------------------------
    def _validate_stack(self) -> None:
        if len(self.layers) < 2:
            raise ValueError("need at least a cover and a substrate layer")
        cover, substrate = self.layers[0], self.layers[-1]
        for end, lay in (("cover", cover), ("substrate", substrate)):
            if not lay.is_semi_infinite or not lay.is_uniform:
                raise ValueError(
                    f"the {end} layer must be uniform and semi-infinite "
                    f"(height=inf)"
                )
        for lay in self.layers[1:-1]:
            if lay.is_semi_infinite:
                raise ValueError("interior layers must have finite thickness")

    def _fft_sampling(self) -> tuple[int, int]:
        """Real-space sampling for the convolution matrices (avoids aliasing of
        the required difference orders ±2*M)."""
        mx, my = self._n_orders
        nx = max(self.resolution[0], 4 * mx + 1)
        ny = max(self.resolution[1], 4 * my + 1)
        return nx, ny

    def _solve(self) -> FieldSolution:
        if self.source is None:
            raise ValueError("call set_source(...) before simulating")
        self._validate_stack()

        cover, substrate = self.layers[0], self.layers[-1]
        wl = self.source.wavelength
        eps_ref = cover.uniform_permittivity(wl, self.materials)
        eps_trn = substrate.uniform_permittivity(wl, self.materials)

        # The cover index sets the incident wavevector.
        self.source.n_incident = np.sqrt(eps_ref)
        kx0, ky0 = self.source.kx0_ky0()

        grid = HarmonicGrid(*self._n_orders)
        sampling = self._fft_sampling()

        eps_grids, heights = [], []
        for lay in self.layers[1:-1]:
            eps_grids.append(lay.permittivity_grid(wl, self.materials, sampling))
            heights.append(lay.height)

        pol = self.source.polarization_vector()
        solution = solve_stack(
            eps_grids=eps_grids, heights=heights,
            eps_ref=eps_ref, eps_trn=eps_trn, grid=grid,
            kx0=kx0, ky0=ky0,
            period_x=self.period_x, period_y=self.period_y, wavelength=wl,
            polarization_xy=(pol[0], pol[1]),
            factorization=self.factorization,
        )
        self._last_solution = solution
        return solution

    def simulate(self, auto_converge: str = "never", converge_tol: Optional[float] = None,
                 max_orders: int = 200, verbose: bool = False,
                 check_convergence: bool = False) -> tuple:
        """Run a simulation and return ``(T, R, result)``.

        ``auto_converge`` selects harmonic-order convergence behaviour:
        ``'never'`` uses the current ``n_orders``; ``'once'`` finds and caches an
        optimal value; ``'always'`` re-converges every call.  Convergence is judged
        on the **complex zeroth-order R/T coefficients** (magnitude *and* phase),
        not the energy balance.  See :mod:`ikarus.tools.convergence`.

        ``check_convergence=True`` re-solves once at a higher ``n_orders`` and warns
        if the zeroth-order R/T (incl. phase) are still moving -- a cheap safety net
        for a single solve (skip it inside tight sweep/optimization loops).
        """
        if auto_converge != "never":
            from ..tools.convergence import auto_converge_orders
            auto_converge_orders(
                self, mode=auto_converge,
                tol=converge_tol or self.convergence_tol,
                max_orders=max_orders, verbose=verbose,
            )

        solution = self._solve()
        if check_convergence and auto_converge == "never":
            from ..tools.convergence import check_convergence as _check
            _check(self, baseline=solution, tol=converge_tol or 1e-3)
        return self._package(solution)

    # -- result packaging --------------------------------------------------
    def _package(self, sol: FieldSolution) -> tuple:
        grid = sol.grid
        p, q = grid.index_arrays()
        i0 = grid.zero_order_index()

        theta_r, phi_r = self._exit_angles(sol, region="ref")
        theta_t, phi_t = self._exit_angles(sol, region="trn")

        if self.source.polarization in ("RCP", "LCP"):
            T = self._circular_coeffs(sol, region="trn")
            R = self._circular_coeffs(sol, region="ref")
        else:
            T = self._linear_coeff(sol, region="trn", i0=i0)
            R = self._linear_coeff(sol, region="ref", i0=i0)

        result = SimulationResult(
            T=T, R=R,
            T_total=sol.T_total, R_total=sol.R_total,
            T_orders=sol.T_orders, R_orders=sol.R_orders,
            orders=(p, q),
            theta_out_trn=theta_t, phi_out_trn=phi_t,
            theta_out_ref=theta_r, phi_out_ref=phi_r,
            energy_balance=sol.R_total + sol.T_total,
            solution=sol,
        )
        self._check_energy_balance(result)
        return T, R, result

    def _check_energy_balance(self, result: "SimulationResult") -> None:
        """Emit warnings for unphysical energy balance (spec section 7.3).

        ``R + T`` should not exceed 1 for a passive structure.  Mild excess
        usually means an unconverged or wrong-sign loss; a large excess signals
        numerical breakdown of the eigenmode/scattering algebra at very high
        ``n_orders`` (try reducing the order count or increasing resolution)."""
        bal = result.energy_balance
        if bal > 1.5 or not np.isfinite(bal):
            warnings.warn(
                f"Energy balance R+T={bal:.3e} is far above 1: the solution is "
                f"numerically unstable at n_orders={self._n_orders}. Reduce "
                f"n_orders (high-contrast structures lose conditioning at very "
                f"high order) or report this case.",
                RuntimeWarning, stacklevel=3,
            )
        elif bal > 1.01:
            warnings.warn(
                f"Energy balance R+T={bal:.4f} exceeds 1.01. If the structure is "
                f"lossless this indicates incomplete convergence (increase "
                f"n_orders); if a material has gain check the sign of its k.",
                RuntimeWarning, stacklevel=3,
            )

    def _linear_coeff(self, sol: FieldSolution, region: str, i0: int) -> complex:
        """0th-order complex amplitude with ``|coeff|^2 == efficiency``."""
        if region == "trn":
            fx, fy, fz, eff = sol.tx[i0], sol.ty[i0], sol.tz[i0], sol.T_orders[i0]
        else:
            fx, fy, fz, eff = sol.rx[i0], sol.ry[i0], sol.rz[i0], sol.R_orders[i0]
        pol = self.source.polarization_vector()
        proj = fx * np.conj(pol[0]) + fy * np.conj(pol[1]) + fz * np.conj(pol[2])
        phase = np.exp(1j * np.angle(proj)) if abs(proj) > 0 else 1.0
        return np.sqrt(max(eff, 0.0)) * phase

    def _circular_coeffs(self, sol: FieldSolution, region: str) -> dict:
        """Co/cross circular amplitudes of the 0th order (see Phase 2)."""
        from .polarization import circular_decomposition
        return circular_decomposition(self.source, sol, region)

    def _exit_angles(self, sol: FieldSolution, region: str):
        """Polar/azimuth exit angles (deg) per order; NaN for evanescent."""
        grid = sol.grid
        kx = np.diag(sol.Kx).real
        ky = np.diag(sol.Ky).real
        Kz = np.diag(sol.Kz_ref if region == "ref" else sol.Kz_trn)
        n = np.sqrt(sol.eps_ref if region == "ref" else sol.eps_trn).real
        kt = np.sqrt(kx**2 + ky**2)
        with np.errstate(invalid="ignore"):
            theta = np.degrees(np.arctan2(kt, np.abs(Kz.real)))
            phi = np.degrees(np.arctan2(ky, kx))
        theta[np.abs(Kz.real) < 1e-9] = np.nan
        return theta, phi

    # -- field extraction --------------------------------------------------
    def get_fields(self, z_positions=None, plane: str = "xy", nx: int = 64,
                   ny: int = 64, x_position: float = 0.0, y_position: float = 0.0):
        """Reconstruct real-space ``E``/``H`` fields from the last simulation.

        ``plane='xy'`` returns one field map per ``z`` in ``z_positions`` (depths
        in meters, ``z=0`` at the cover/first-layer interface, increasing into the
        stack).  ``plane='xz'``/``'yz'`` return a single cross-section sweeping the
        full stack at fixed ``y_position``/``x_position``.

        Returns a dict of :class:`~ikarus.core.fields.FieldMap`.
        """
        if self._last_solution is None:
            self._solve()
        from .fields import reconstruct
        if z_positions is None:
            z_positions = [0.0]
        maps = reconstruct(self._last_solution, z_positions, nx=nx, ny=ny,
                           plane=plane, x_position=x_position, y_position=y_position)
        # Attach the real-space permittivity on each map's grid so field plots
        # can overlay the structure outline.
        for fmap in maps.values():
            try:
                fmap.eps = self._structure_eps(fmap, plane, x_position, y_position)
            except Exception:
                fmap.eps = None
        return maps

    def _structure_eps(self, fmap, plane: str, x_position: float, y_position: float):
        """Real part of eps sampled on a FieldMap's grid (for plot overlays)."""
        wl = self.source.wavelength
        lib = self.materials
        interior = self.layers[1:-1]
        heights = [lay.height for lay in interior]
        edges = np.concatenate([[0.0], np.cumsum(heights)]) if heights else np.array([0.0])
        total = float(edges[-1])

        def region_at(z):
            if z < 0:
                return self.layers[0]
            if z > total:
                return self.layers[-1]
            k = int(np.searchsorted(edges, z, side="right") - 1)
            return interior[min(max(k, 0), len(interior) - 1)]

        def profile(layer, axis, frac, n):
            if layer.is_uniform:
                return np.full(n, lib.permittivity(layer.material, wl).real)
            g = layer.permittivity_grid(wl, lib, (n, n)).real
            idx = min(int(frac * n), n - 1)
            return g[:, idx] if axis == "x" else g[idx, :]

        if plane == "xy":
            nx, ny = len(fmap.coords["x"]), len(fmap.coords["y"])
            lay = region_at(fmap.z)
            if lay.is_uniform:
                return np.full((nx, ny), lib.permittivity(lay.material, wl).real)
            return lay.permittivity_grid(wl, lib, (nx, ny)).real

        if plane == "xz":
            axis, frac = "x", (y_position % self.period_y) / self.period_y
            ax_arr = fmap.coords["x"]
        else:  # yz
            axis, frac = "y", (x_position % self.period_x) / self.period_x
            ax_arr = fmap.coords["y"]
        zc = fmap.coords["z"]
        eps = np.empty((len(ax_arr), len(zc)))
        cache: dict = {}
        for jz, zz in enumerate(zc):
            lay = region_at(float(zz))
            if id(lay) not in cache:
                cache[id(lay)] = profile(lay, axis, frac, len(ax_arr))
            eps[:, jz] = cache[id(lay)]
        return eps

    def visualize_structure(self, plane: str = "xz", layer_index: Optional[int] = None,
                            **kwargs):
        """Plot the layer stack (``plane='xz'``) or a layer's topology
        (``plane='xy'``).  See :mod:`ikarus.visualization`."""
        from ..visualization import structure as _struct
        if plane == "xy":
            return _struct.plot_topology(self, layer_index or 0, **kwargs)
        return _struct.plot_stack(self, **kwargs)

    def visualize_fields(self, field_map=None, component: str = "intensity", **kwargs):
        """Plot a reconstructed field map.  See :mod:`ikarus.visualization`."""
        from ..visualization import fields as _vf
        if field_map is None:
            field_map = self.get_fields(plane="xz")
        return _vf.plot_field(field_map, component=component, **kwargs)

    def save_results(self, path, include=("T", "R", "metadata"), result=None):
        """Save simulation results to HDF5.  See :mod:`ikarus.tools.io`."""
        from ..tools import io
        return io.save_results(self, path, include=include, result=result)

    @staticmethod
    def load_results(path):
        """Load results previously saved with :meth:`save_results`."""
        from ..tools import io
        return io.load_results(path)

    # -- accessors ---------------------------------------------------------
    @property
    def last_solution(self) -> Optional[FieldSolution]:
        return self._last_solution
