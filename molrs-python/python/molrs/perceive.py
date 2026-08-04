"""Chemical perception — ``molrs::perceive``.

One layer above ``core`` and below ``ff`` / ``io`` / ``conformer``: rings (SSSR),
aromaticity, hydrogens, stereochemistry, rotatable bonds, Gasteiger charges, and
SMARTS/SMIRKS matching.

:class:`Perceive` is a builder — every ``find_*`` method is graph-in / graph-out
and non-mutating, so a pipeline reads as a chain of graphs. :class:`RingInfo`
answers the other question: it *reports* ring facts and never touches the
molecule.

SMARTS lives here because a pattern is a query over a *perceived* graph —
matching needs ring membership and aromaticity, not a text format. The SMILES
front-end is a format, and lives in :mod:`molrs.io`.
"""

from __future__ import annotations

from ._lib import (
    Perceive as Perceive,
    RingInfo as RingInfo,
    SmartsMatch as SmartsMatch,
    SmartsPattern as SmartsPattern,
)

__all__ = [
    "Perceive",
    "RingInfo",
    "SmartsMatch",
    "SmartsPattern",
]
