"""SMILES, SMARTS and CGsmiles notation — ``molrs::io::smiles``.

:class:`SmilesIR` is :mod:`molrs.io`'s because SMILES is a *format*: text in,
molecule out, exactly like PDB or XYZ; :func:`molrs.io.read_smiles` is its
one-molecule door. SMARTS is not — a pattern is a query over a perceived
graph — so it lives in :mod:`molrs.perceive`.

:class:`CGSmilesIR` is the same kind of door onto the CGsmiles notation, which
writes a molecule at one or more *coarse-grained* resolutions — a resolution
at which one particle, a *bead*, stands in for a whole group of atoms: text
in, one :class:`CGGraph` per resolution level plus the fragment tables that
resolve them out, and ``to_atomistic()`` expands the lowest level into atoms.
That expansion is topology only — atoms, bonds and the per-atom ``frag_id``
saying which bead each atom came from. A line notation states no geometry, so
coordinates, hydrogens and perception remain separate steps. The records it
hands out — :class:`CGGraph`, :class:`CGNode`, :class:`CGEdge`,
:class:`CGFragmentDef`, :class:`ResolvedPair`, :class:`PairEnd` and
:class:`BondingDescriptor` — are read-only views over the parsed value, so no
fact of the notation has to be re-parsed, decoded or unpacked from a bare
tuple position on the Python side.

A :class:`BondingDescriptor` reports its ``kind`` as the grammar glyph
(``"$"``, ``"<"``, ``">"``, ``"!"``), which is both what a user writes and
what a stored port's ``port_kind`` prop holds — one spelling for the notation,
the column and this boundary, so a descriptor kind reaches
:meth:`Atomistic.def_port <molrs.core.Atomistic.def_port>` untranslated. The enums
the notation does not spell out keep lowercase variant names:
``BondingDescriptor.order`` and ``ResolvedPair.kind`` are bond kinds
(``"single"``, ``"aromatic"``, …) and ``PairEnd.end`` is ``"sub"`` or
``"body"``.

Every refusal raised by this family of notations — by the parser, by the
expansion of a CGsmiles string, or by an emit — is a :class:`SmilesError`, one
class carrying the four facts the Rust error owns: ``kind``, the variant name
of the rule that was broken (``"UnclosedBranch"``, ``"UnexpectedEnd"``,
``"CgNotExpandable"``, …); ``span``, the byte range of the offending text as a
``(start, end)`` pair whose end is clamped to ``len(input)``; ``input``, the
offending string, empty for the errors raised past the parser, which are handed
an IR and never see the text it came from; and ``notation``, lowercase
``"smiles"``, ``"smarts"`` or ``"cgsmiles"``. It subclasses
:class:`ValueError`, so ``except ValueError`` keeps catching it, and
``str(e)`` is the message Rust renders, caret line included.

There is no ``CGSmilesReader``, deliberately: a parser of one string into
an IR is not a path-backed cursor, and the one-shot door onto a notation is
a ``molrs.io.read_<fmt>`` function, not a class.
"""

from .._lib import (
    BondingDescriptor,
    CGEdge,
    CGFragmentDef,
    CGGraph,
    CGNode,
    CGSmilesIR,
    PairEnd,
    ResolvedPair,
    SmilesError,
    SmilesIR,
)

__all__ = [
    "BondingDescriptor",
    "CGEdge",
    "CGFragmentDef",
    "CGGraph",
    "CGNode",
    "CGSmilesIR",
    "PairEnd",
    "ResolvedPair",
    "SmilesError",
    "SmilesIR",
]
