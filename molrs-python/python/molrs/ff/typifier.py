"""Typifier base, its ``Match``, and the built-in force-field typifiers.

A typifier implements one hook, ``match(graph) -> Match``: it reads a graph and
returns the annotations to write, positional against the graph's nodes and
against each relation kind's own rows (``graph.links.exact_bucket(cls)``). The
base :class:`Typifier` owns the rest. ``typify(mol)`` copies ``mol``, calls
``match`` on the copy, stamps the match onto the copy and defines its types in
the output force field — ``forcefield()``, of which ``typify`` is the only
writer. A subclass must not define ``typify``.

The built-in typifiers accept and return ``Atomistic``. :class:`ElementTypifier`
labels by element symbol alone and defines no force field.

:func:`assign_cmaps` builds a typed frame's ``cmaps`` block from its
dihedrals, against a force field's CMAP types.
"""

from .._lib import (
    AtdTypifier,
    ElementTypifier,
    GaffTypifier,
    Match,
    MMFF94STypifier,
    MMFF94Typifier,
    OPLSAATypifier,
    Typifier,
    assign_cmaps,
)

__all__ = [
    "AtdTypifier",
    "ElementTypifier",
    "GaffTypifier",
    "MMFF94STypifier",
    "MMFF94Typifier",
    "Match",
    "OPLSAATypifier",
    "Typifier",
    "assign_cmaps",
]
