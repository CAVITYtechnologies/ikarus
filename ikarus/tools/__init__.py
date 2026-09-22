"""Command-line and utility tools: material import, convergence, HDF5 I/O.

The public helpers are re-exported here for discoverability, so both

    from ikarus.tools import check_convergence
    from ikarus.tools.convergence import check_convergence

work (the functions themselves live in the submodules).

``add_material`` is a command-line entry point, not a function; run it as::

    python -m ikarus.tools.add_material my_material.csv --name MyMaterial
"""

from .convergence import auto_converge_orders, check_convergence, convergence_curve
from .io import load_results, save_results

__all__ = [
    "auto_converge_orders",
    "check_convergence",
    "convergence_curve",
    "save_results",
    "load_results",
]
