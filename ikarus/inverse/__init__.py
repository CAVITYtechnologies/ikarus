"""Gradient-free inverse design for metaatoms.

Define a metaatom with free degrees of freedom, declare what you want, optimize::

    from ikarus.inverse import MetaAtom, free, pixels, Target, optimize

    atom = MetaAtom(period=free(0.4e-6, 0.9e-6), cover="Air", substrate="SiO2")
    atom.add_pattern(topology=pixels(12, 12, symmetry="c4v"),
                     materials=["Air", "Si"], height=free(0.3e-6, 0.9e-6))

    best = optimize(atom, Target.maximize("R", order=(0, 0), at=1550e-9))
    print(best.report());  best.metaatom   # ready to simulate / visualize
"""

from .dof import MetaAtom, free, pixels, Free, Pixels
from .targets import Target
from .optimize import optimize, OptimizeResult

__all__ = ["MetaAtom", "free", "pixels", "Free", "Pixels",
           "Target", "optimize", "OptimizeResult"]
