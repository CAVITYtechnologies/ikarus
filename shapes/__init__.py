"""Shape primitives for building topology pixel maps.

Each primitive returns an integer ``np.ndarray`` of shape ``grid_shape`` that can
be passed directly to :meth:`ikarus.RCWA.add_layer` as a ``topology``.  Filled
pixels take ``value`` (default 1), the rest take ``background`` (default 0).
"""

from .primitives import circle, rectangle, ring, polygon, ellipse, cross, combine

__all__ = ["circle", "rectangle", "ring", "polygon", "ellipse", "cross", "combine"]
