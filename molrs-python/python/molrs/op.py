"""Pure numeric primitives — ``molrs::op``.

Weighted rigid superposition (Horn quaternion with a scale-free eigen-gap
test), weighted centroids and seeded uniform SO(3) sampling. Points cross as
``(k, 3)`` float64 arrays and rotations as ``(3, 3)`` row-major matrices. All
computation is in Rust; this module is a thin re-export of ``_lib.op``.
"""

from ._lib import op as _op

DEFAULT_GAP_TOL = _op.DEFAULT_GAP_TOL
Fit = _op.Fit
superpose = _op.superpose
superpose_many = _op.superpose_many
centroid = _op.centroid
random_rotations = _op.random_rotations
random_angles = _op.random_angles
rotation_from_uniform = _op.rotation_from_uniform

__all__ = [
    "DEFAULT_GAP_TOL",
    "Fit",
    "superpose",
    "superpose_many",
    "centroid",
    "random_rotations",
    "random_angles",
    "rotation_from_uniform",
]
