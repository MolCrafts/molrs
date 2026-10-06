"""Geometry optimization — ``molrs::optimize``.

An optimizer minimizes a :class:`~molrs.ff.potential.Potentials` aggregate over a
coordinate vector; it does not build the potentials and does not own a force
field. Construct the potentials with :mod:`molrs.ff.potential`, hand them here.
"""

from __future__ import annotations

from ._lib import (
    LBFGS,
    OptReport,
)

__all__ = [
    "LBFGS",
    "OptReport",
]
