"""Multi-layer parametric structures for inverse design.

Where a :class:`~ikarus.inverse.dof.MetaAtom` optimizes a *single* patterned
layer, a :class:`Structure` optimizes an **entire layer stack** -- several
patterned layers, free heights, a free period, and (crucially) *shared / derived*
geometry, where many layers are computed from a few parameters.

Subclass it, declare each parameter as ``free(...)`` (a degree of freedom) or a
plain value (fixed), and implement :meth:`define` to lay out the stack.  The
instance satisfies the same ``variables()`` / ``build()`` protocol that
:func:`~ikarus.inverse.optimize` consumes, so you optimize it exactly like a
``MetaAtom``::

    from ikarus.inverse import Structure, free, optimize, Target
    from ikarus.shapes import Circle, Cross

    class TwoLayer(Structure):
        cover, substrate, resolution = "Air", "SiO2", 96
        period  = free(0.3e-6, 0.9e-6)     # free
        h1      = free(0.1e-6, 0.4e-6)     # free
        h2      = 0.20e-6                  # fixed
        radius  = free(0.10, 0.45)         # free
        arm_len = free(0.30, 0.90)         # free

        def define(self, p):               # p = resolved parameters
            self.add_layer(p.h1, Circle(radius=p.radius), ["Si", "Air"])   # air hole in Si
            self.add_layer(p.h2, Cross(arm_length=p.arm_len, arm_width=0.2), ["Air", "Si"])

    best = optimize(TwoLayer(), Target.minimize("R", at=1550e-9))
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from ..core.rcwa import RCWA
from .dof import Free, Pixels

# Attributes that configure the stack rather than act as optimizable parameters.
_RESERVED = {"cover", "substrate", "resolution", "polarization", "pol_angle"}


class Structure:
    """Base class for a multi-layer parametric structure (a "buildable" design).

    Declare parameters as class attributes -- ``free(lo, hi)`` for a degree of
    freedom, a plain number for a fixed parameter -- and implement :meth:`define`.
    A ``period`` parameter is required (free or fixed); ``cover``, ``substrate``,
    ``resolution``, ``polarization`` and ``pol_angle`` are configuration, not DOF.
    """

    cover = "Air"
    substrate = "SiO2"
    resolution = 96
    polarization = "linear"
    pol_angle = 0.0

    def __init__(self, **overrides):
        # per-instance tweaks of (usually fixed) parameters, e.g. MothEye(N=12)
        for key, value in overrides.items():
            setattr(self, key, value)
        self._layers: list = []

    # -- the method the user writes -----------------------------------------
    def define(self, p) -> None:
        """Lay out the interior layers using the resolved parameters ``p``.

        Call :meth:`add_layer` for each interior layer.  ``p`` is a namespace whose
        attributes are the *concrete* values of every declared parameter (free
        parameters resolved to the optimizer's pick, fixed ones as declared).  The
        cover, substrate and period are added by the base class.
        """
        raise NotImplementedError("subclass Structure and implement define(self, p)")

    def add_layer(self, height, topology, materials) -> None:
        """Add one interior layer (called from :meth:`define`).

        ``topology`` may be an integer array, a parametric
        :class:`~ikarus.shapes.parametric.Shape`, or any object exposing an ``img``
        array; for a single-material layer pass a plain material as ``topology`` is
        not needed -- use a uniform layer instead via a one-material list.
        """
        self._layers.append((float(height), topology, list(materials)))

    # -- parameter discovery ------------------------------------------------
    def _declared(self) -> dict:
        """Every declared parameter ``{name: value}`` (free or fixed)."""
        params: dict = {}
        for klass in reversed(type(self).__mro__):
            for name, value in vars(klass).items():
                if name.startswith("_") or name in _RESERVED:
                    continue
                if callable(value) or isinstance(value, (classmethod, staticmethod, property)):
                    continue
                params[name] = value
        for name, value in vars(self).items():
            if not name.startswith("_") and name not in _RESERVED:
                params[name] = value
        return params

    # -- the protocol optimize() consumes -----------------------------------
    def variables(self) -> dict:
        """Return ``{name: ('real', (lo, hi)) | ('binary',)}`` for every free DOF."""
        declared = self._declared()
        if "period" not in declared:
            raise ValueError("a Structure must declare a `period` parameter (free or fixed)")
        out: dict = {}
        for name, value in declared.items():
            if isinstance(value, Free):
                out[name] = ("real", (value.low, value.high))
            elif isinstance(value, Pixels):
                for k in range(value.n_free):
                    out[f"{name}__px{k}"] = ("binary",)
        return out

    @property
    def n_dof(self) -> int:
        return len(self.variables())

    def build(self, params: dict, n_orders) -> RCWA:
        """Resolve ``params`` and assemble the whole stack as an :class:`~ikarus.RCWA`."""
        resolved: dict = {}
        for name, value in self._declared().items():
            if isinstance(value, Free):
                resolved[name] = params[name]
            elif isinstance(value, Pixels):
                bits = np.array([params[f"{name}__px{k}"] for k in range(value.n_free)])
                resolved[name] = value.expand(bits)
            else:
                resolved[name] = value
        p = SimpleNamespace(**resolved)
        if not hasattr(p, "period"):
            raise ValueError("a Structure must declare a `period` parameter (free or fixed)")

        self._layers = []
        self.define(p)
        if not self._layers:
            raise ValueError("define(self, p) added no layers -- call self.add_layer(...)")

        res = (self.resolution if isinstance(self.resolution, (tuple, list))
               else (self.resolution, self.resolution))
        no = n_orders if isinstance(n_orders, (tuple, list)) else (n_orders, n_orders)
        rcwa = RCWA(period_x=float(p.period), period_y=float(p.period),
                    resolution=res, n_orders=no, materials=None)
        rcwa.add_uniform_layer(np.inf, self.cover)
        for height, topology, materials in self._layers:
            if len(materials) == 1:
                rcwa.add_uniform_layer(height, materials[0])
            else:
                rcwa.add_layer(height, topology, materials)
        rcwa.add_uniform_layer(np.inf, self.substrate)
        return rcwa
