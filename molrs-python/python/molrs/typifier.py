"""Typifier interfaces and built-in force-field typifiers.

A typifier labels a molecular graph and returns the same concrete graph type:
``typify(mol: T) -> T``. Built-in all-atom typifiers therefore accept and return
``Atomistic``. Coarse-grained typifiers should accept and return ``CoarseGrain``.
"""

from __future__ import annotations

from typing import TypeVar

from ._lib import Graph, MMFF94STypifier, MMFF94Typifier, OPLSAATypifier, Typifier
from .views import Atomistic

TGraph = TypeVar("TGraph", bound=Graph)


__all__ = [
    "Typifier",
    "TGraph",
    "OPLSAATypifier",
    "MMFF94Typifier",
    "MMFF94STypifier",
    "Atomistic",
    "Graph",
]
