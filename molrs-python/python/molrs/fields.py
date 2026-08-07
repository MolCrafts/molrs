"""I/O-boundary column-name translation.

Only the per-format rename maps live here. The canonical field *names* are
:mod:`molrs.keys` and their dtypes are :mod:`molrs.schema` — this module used
to carry a third copy, a hand-written ``FieldSpec`` table that declared the
relation endpoints ``int64`` while Rust stored them ``uint32``, so a frame
built through it lost its whole topology on the way back into Rust, silently.

Ask the schema for a column's dtype::

    molrs.schema.column("charge").dtype       # "float"
    molrs.schema.column("charge").numpy_dtype # "float64"
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import numpy as np

# Import from the compiled leaf, not the package: `molrs/__init__` imports
# `io`, which imports this module, so a package-level import would cycle.
from ._lib import schema as _schema

__all__ = [
    "FieldFormatter",
    "GroFieldFormatter",
    "PdbFieldFormatter",
    "LammpsFieldFormatter",
    "XyzFieldFormatter",
    "Mol2FieldFormatter",
]


class FieldFormatter:
    """Translates between format-native and canonical column names.

    Subclasses map ``{format_key: canonical_key}``. Registrations are isolated
    per subclass via ``__init_subclass__``.
    """

    _field_formatters: ClassVar[dict[str, str]] = {}

    def __init_subclass__(cls, **kwargs: object) -> None:
        super().__init_subclass__(**kwargs)
        cls._field_formatters = dict(cls._field_formatters)

    @classmethod
    def register_field(cls, format_key: str, canonical_key: str) -> None:
        """Register a mapping at runtime."""
        cls._field_formatters[format_key] = canonical_key

    def canonicalize(self, block):
        """Reader exit: format-native names → canonical (in place)."""
        for fmt_key, canonical in self._field_formatters.items():
            if fmt_key in block and canonical not in block:
                block.rename(fmt_key, canonical)
        return block

    def localize(self, block):
        """Writer entry: canonical names → format-native (in place)."""
        for fmt_key, canonical in self._field_formatters.items():
            if canonical in block and fmt_key not in block:
                block.rename(canonical, fmt_key)
        return block

    def canonicalize_frame(self, frame):
        """Canonicalize every block of a Frame (in place)."""
        for key in list(frame.keys()):
            self.canonicalize(frame[key])
        return frame

    def localize_frame(self, frame):
        """Localize every block of a Frame (in place)."""
        for key in list(frame.keys()):
            self.localize(frame[key])
        return frame


class GroFieldFormatter(FieldFormatter):
    """GRO ↔ canonical names."""

    _field_formatters: ClassVar[dict[str, str]] = {
        "resid": "res_id",
        "resname": "res_name",
        "atom_name": "name",
        "atom_id": "id",
    }


class PdbFieldFormatter(FieldFormatter):
    """PDB ↔ canonical names."""

    _field_formatters: ClassVar[dict[str, str]] = {
        "res_seq": "res_id",
        "resname": "res_name",
        "symbol": "element",
    }


class LammpsFieldFormatter(FieldFormatter):
    """LAMMPS ↔ canonical names.

    Only charge/mol renames. **Never** map ``type`` ↔ ``type_id``: the frame
    schema declares ``type`` as *string* (force-field label) and ``type_id`` as
    *uint* (numeric ordinal). Renaming the uint column onto ``type`` either
    fails the schema check or hides ``type_id`` from the Rust data writer.
    """

    _field_formatters: ClassVar[dict[str, str]] = {
        "q": "charge",
        "mol": "mol_id",
    }


class XyzFieldFormatter(FieldFormatter):
    """XYZ ↔ canonical names."""

    _field_formatters: ClassVar[dict[str, str]] = {
        "symbol": "element",
        "species": "element",
    }


class Mol2FieldFormatter(FieldFormatter):
    """Tripos MOL2 ↔ canonical names.

    ``atom_type`` is the SYBYL type label, mapped onto the shared string
    ``type`` column. Substructure fields are residues.
    """

    _field_formatters: ClassVar[dict[str, str]] = {
        "atom_type": "type",
        "subst_id": "res_id",
        "subst_name": "res_name",
        "sybyl_bond_type": "type",
    }
