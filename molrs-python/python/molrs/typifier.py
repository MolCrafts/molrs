"""Typifier interfaces and built-in force-field typifiers.

A typifier labels a molecular graph and returns the same concrete graph type:
``typify(mol: T) -> T``. Built-in all-atom typifiers therefore accept and return
``Atomistic``. Coarse-grained typifiers should accept and return ``CoarseGrain``.
"""

from __future__ import annotations

from typing import Protocol, TypeVar

from .molrs import Atomistic, Graph, MMFFTypifier, OPLSAATypifier

TGraph = TypeVar("TGraph", bound=Graph)


class Typifier(Protocol[TGraph]):
    """Structural protocol for graph typifiers."""

    def typify(self, mol: TGraph) -> TGraph: ...


__all__ = [
    "Typifier",
    "TGraph",
    "OPLSAATypifier",
    "MMFFTypifier",
    "Atomistic",
    "Graph",
]
