"""Chemical perception — ``molrs::perceive``.

*Perception* derives chemical facts a connectivity graph implies but does not
state. One layer above ``core`` and below ``ff`` / ``io`` / ``conformer``:
rings (SSSR, the smallest set of smallest rings), aromaticity, hydrogens,
stereochemistry, rotatable bonds, SMARTS (SMILES Arbitrary Target
Specification) substructure matching, and coarse-grained bead-pattern
matching. Gasteiger charges are a charge model and live in :mod:`molrs.ff`.

:class:`Perceive` is a builder — every ``find_*`` method is graph-in / graph-out
and non-mutating, so a pipeline reads as a chain of graphs. :class:`RingInfo`
answers the other question: it *reports* ring facts and never touches the
molecule.

SMARTS lives here because a pattern is a query over a *perceived* graph —
matching needs ring membership and aromaticity, not a text format. The SMILES
front-end is a format, and lives in :mod:`molrs.io`.

:class:`SubgraphMatcher` is the coarse-grained counterpart: it snapshots a bead
pattern (a :class:`~molrs.CoarseGrain`, e.g. from
``CGSmilesIR(...).to_coarsegrain()``) and lists every occurrence of it in a
target ``CoarseGrain`` as bead-handle groups. It does not partition
overlapping groups.
"""

from __future__ import annotations

from ._lib import (
    Perceive as Perceive,
    RingInfo as RingInfo,
    SmartsMatch as SmartsMatch,
    SmartsPattern as SmartsPattern,
    SubgraphMatcher as SubgraphMatcher,
)

__all__ = [
    "Perceive",
    "RingInfo",
    "SmartsMatch",
    "SmartsPattern",
    "SubgraphMatcher",
]
