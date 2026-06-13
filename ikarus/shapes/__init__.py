"""Shape primitives for building topology pixel maps.

Two flavours:

* **Functional primitives** (:func:`circle`, :func:`rectangle`, ...) return a
  fixed integer ``np.ndarray`` of shape ``grid_shape`` -- pass one directly as a
  ``topology`` to :meth:`ikarus.RCWA.add_layer`.  Filled pixels take ``value``
  (default 1), the rest ``background`` (default 0).  :func:`rotate` spins any such
  map by an angle.
* **Parametric shape classes** (:class:`Circle`, :class:`Cross`, ...) carry named
  parameters that may be left ``free(...)`` for inverse design.  See
  :mod:`ikarus.shapes.parametric`.
"""

from .primitives import (circle, rectangle, ring, polygon, ellipse, cross,
                         combine, rotate)
from .parametric import (Shape, Circle, Ellipse, Rectangle, Ring, Cross,
                         SplitRing)

__all__ = [
    # functional primitives
    "circle", "rectangle", "ring", "polygon", "ellipse", "cross", "combine",
    "rotate",
    # parametric shape classes
    "Shape", "Circle", "Ellipse", "Rectangle", "Ring", "Cross", "SplitRing",
]
