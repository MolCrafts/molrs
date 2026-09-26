"""Pure numeric primitives — ``molrs::op``.

Weighted rigid superposition (Horn quaternion with a scale-free eigen-gap
test) and weighted centroids. Points cross as ``(k, 3)`` float64 arrays and
rotations as ``(3, 3)`` row-major matrices. All computation is in Rust; this
module is a thin re-export of ``_lib.op``.
"""

from ._lib import op as _op

DEFAULT_GAP_TOL = _op.DEFAULT_GAP_TOL
Fit = _op.Fit
superpose = _op.superpose
centroid = _op.centroid

__all__ = [
    "DEFAULT_GAP_TOL",
    "Fit",
    "superpose",
    "centroid",
]
