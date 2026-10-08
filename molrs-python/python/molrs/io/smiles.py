"""SMILES notation — ``molrs::io::smiles``.

:class:`SmilesIr` is :mod:`molrs.io`'s because SMILES is a *format*: text in,
molecule out, exactly like PDB or XYZ. Its doors are functions of
:mod:`molrs.io`: :func:`~molrs.io.read_smiles_str` reads one molecule, and
:func:`~molrs.io.write_smiles_str` writes one. SMARTS is not a format — a
pattern is a query over a perceived graph — so it lives, wholly, in
:mod:`molrs.perceive` (:class:`~molrs.perceive.SmartsPattern`, including a
pattern generated from an atom's environment).

:class:`SmilesIr` is the parsed text: ``SmilesIr(s)`` parses SMILES,
:meth:`SmilesIr.from_fragment` a fragment body with bonding descriptors
(``[<]OCC[>]``), :meth:`SmilesIr.from_atomistic` builds one from a molecule;
``to_atomistic()``, ``components()`` and ``to_template()`` turn it into
graphs. A :class:`BondingDescriptor` is one such joining-site marker, as the
CGsmiles records (:mod:`molrs.io.cgsmiles`) hand it out.

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
"""

from .._native import BondingDescriptor, SmilesError, SmilesIr

__all__ = [
    "BondingDescriptor",
    "SmilesError",
    "SmilesIr",
]
