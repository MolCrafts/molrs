"""Geometry optimization — ``molrs::optimize``.

An optimizer minimizes a :class:`~molrs.ff.Potentials` aggregate over a
coordinate vector; it does not build the potentials and does not own a force
field. Construct the potentials from :mod:`molrs.ff`, hand them here.
"""

from __future__ import annotations

from ._lib import (
    LBFGS as LBFGS,
    OptReport as OptReport,
)

__all__ = [
    "LBFGS",
    "OptReport",
]
