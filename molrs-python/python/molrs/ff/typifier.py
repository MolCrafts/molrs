"""Typifier base, its ``Match``, and the built-in force-field typifiers.

A typifier implements one hook, ``match(graph) -> Match``: it reads a graph and
returns the annotations to write, positional against the graph's nodes and
against each relation kind's own rows (``graph.links.exact_bucket(cls)``). The
base :class:`Typifier` owns the rest. ``typify(mol)`` copies ``mol``, calls
``match`` on the copy, stamps the match onto the copy and defines its types in
the output force field — ``forcefield()``, of which ``typify`` is the only
writer. A subclass must not define ``typify``.

The built-in typifiers accept and return ``Atomistic``. :class:`ElementTypifier`
labels by element symbol alone and defines no force field; it is exported from
this module only.
"""

from __future__ import annotations

from .._lib import (
    AtdTypifier,
    ElementTypifier,
    Match,
    MMFF94STypifier,
    MMFF94Typifier,
    OPLSAATypifier,
    Typifier,
)

__all__ = [
    "Typifier",
    "Match",
    "OPLSAATypifier",
    "MMFF94Typifier",
    "MMFF94STypifier",
    "AtdTypifier",
    "ElementTypifier",
]
