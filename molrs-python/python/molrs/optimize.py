"""Geometry optimization — ``molrs::optimize``.

An optimizer minimizes a :class:`~molrs.ff.potential.Potentials` aggregate over a
coordinate vector; it does not build the potentials and does not own a force
field. Construct the potentials with :mod:`molrs.ff.potential`, hand them here.
"""

from ._native import (
    Lbfgs,
    OptimizationReport,
)

__all__ = [
    "Lbfgs",
    "OptimizationReport",
]
