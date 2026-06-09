"""Circular-polarization co/cross decomposition of the specular order.

For circularly polarized illumination the reflection/transmission are reported as
a ``{'co', 'cross'}`` dictionary of complex amplitudes.  ``co`` is the component
with the *same* handedness as the incident wave, ``cross`` the opposite.  The
amplitudes are normalized so that ``|co|**2`` and ``|cross|**2`` are the power
efficiencies diffracted into each handedness for the zero order, hence
``|co|**2 + |cross|**2`` equals that order's total efficiency.
"""

from __future__ import annotations

import numpy as np

from .solver import FieldSolution
from .source import Source


def _transverse_basis(direction: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return real TE/TM unit vectors transverse to ``direction``."""
    d = direction / np.linalg.norm(direction)
    if abs(d[0]) < 1e-12 and abs(d[1]) < 1e-12:
        # Normal propagation: match the Source's normal-incidence convention.
        te = np.array([0.0, 1.0, 0.0])
        tm = np.array([1.0, 0.0, 0.0]) * np.sign(d[2] if d[2] != 0 else 1.0)
        return te, tm
    te = np.cross([0.0, 0.0, 1.0], d)
    te /= np.linalg.norm(te)
    tm = np.cross(d, te)
    tm /= np.linalg.norm(tm)
    return te, tm


def circular_decomposition(source: Source, sol: FieldSolution, region: str) -> dict:
    """Decompose the zero-order outgoing field into co/cross circular amplitudes.

    Parameters
    ----------
    source:
        The (circularly polarized) illumination.
    sol:
        Solved :class:`~ikarus.core.solver.FieldSolution`.
    region:
        ``'trn'`` for transmission, ``'ref'`` for reflection.
    """
    i0 = sol.grid.zero_order_index()
    kx0 = sol.Kx[i0, i0].real
    ky0 = sol.Ky[i0, i0].real

    if region == "trn":
        F = np.array([sol.tx[i0], sol.ty[i0], sol.tz[i0]])
        kz = np.diag(sol.Kz_trn)[i0]
        eff = sol.T_orders[i0]
        zsign = +1.0
    else:
        F = np.array([sol.rx[i0], sol.ry[i0], sol.rz[i0]])
        kz = np.diag(sol.Kz_ref)[i0]
        eff = sol.R_orders[i0]
        zsign = -1.0

    direction = np.array([kx0, ky0, zsign * kz.real])
    te, tm = _transverse_basis(direction)

    # Project the (transverse) field onto the right/left circular basis.
    f_te = F @ te
    f_tm = F @ tm
    c_rcp = (f_te - 1j * f_tm) / np.sqrt(2.0)  # conj(e_R) . F, e_R=(te+i tm)/sqrt2
    c_lcp = (f_te + 1j * f_tm) / np.sqrt(2.0)

    # Rescale projections so |c|^2 sums to the order efficiency.
    norm = np.abs(c_rcp) ** 2 + np.abs(c_lcp) ** 2
    scale = np.sqrt(eff / norm) if norm > 1e-300 else 0.0
    c_rcp *= scale
    c_lcp *= scale

    if source.polarization == "RCP":
        return {"co": c_rcp, "cross": c_lcp}
    return {"co": c_lcp, "cross": c_rcp}
