"""Canonical field registry and I/O boundary formatters — single source of truth.

Every canonical field name is taken from the Rust ``molrs.keys`` constants (which
mirror ``molrs_core::keys``), so the field-name convention is defined in exactly
one place and the Rust and Python layers cannot drift. ``FieldSpec`` pairs each
canonical key with its storage dtype; :class:`FieldFormatter` translates between
format-native column names and canonical names at the I/O boundary.

Architecture::

    molrs.keys (Rust)                — canonical field-name strings
        ↓
    FieldSpec                        — canonical key + numpy dtype
        ↓
    FieldFormatter                   — per-format mapping registry
        ↓                              canonicalize() / localize() on Block / Frame
    GroFieldFormatter(FieldFormatter) — GRO-specific mappings
    PdbFieldFormatter(FieldFormatter) — PDB-specific mappings
    ...

molpy re-exports this module wholesale (``from molrs.fields import *``) and adds
only force-field-specific param formatting on top.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import numpy as np

# Relative import of the compiled `keys` submodule — leaf extension, so this is
# cycle-free even when fields.py is imported during package initialization.
from ._lib import keys

# ===================================================================
#                       FieldSpec
# ===================================================================


@dataclass(frozen=True)
class FieldSpec:
    """Specification of a canonical field in the internal data model.

    Attributes:
        key: Canonical field name (always sourced from ``molrs.keys``).
        dtype: NumPy dtype for this field's column.
        doc: Human-readable description with units.
    """

    key: str
    dtype: np.dtype
    doc: str


# ===================================================================
#               Canonical Atom Fields
# ===================================================================

ATOM_ID = FieldSpec(keys.ID, np.dtype(np.int64), "atom ID (1-indexed)")
ATOM_TYPE = FieldSpec(keys.TYPE, np.dtype("U64"), "force field type label")
CHARGE = FieldSpec(keys.CHARGE, np.dtype(np.float64), "partial charge (e)")
MASS = FieldSpec(keys.MASS, np.dtype(np.float64), "atomic mass (amu)")
MOL_ID = FieldSpec(keys.MOL_ID, np.dtype(np.int64), "molecule ID (1-indexed)")
ELEMENT = FieldSpec(keys.ELEMENT, np.dtype("U4"), "element symbol")
SYMBOL = FieldSpec(keys.SYMBOL, np.dtype("U4"), "atom/site symbol label")
# Atom name — PDB caps the column at 4 chars on write, so U8 is ample. ``NAME``
# is an alias kept for callers that referenced the generic name field.
ATOM_NAME = FieldSpec(keys.NAME, np.dtype("U8"), "atom name")
NAME = ATOM_NAME
POS_X = FieldSpec(keys.X, np.dtype(np.float64), "x coordinate (Angstrom)")
POS_Y = FieldSpec(keys.Y, np.dtype(np.float64), "y coordinate (Angstrom)")
POS_Z = FieldSpec(keys.Z, np.dtype(np.float64), "z coordinate (Angstrom)")
XYZ = FieldSpec(keys.XYZ, np.dtype(np.float64), "position vector (Angstrom)")
VEL_X = FieldSpec(keys.VX, np.dtype(np.float64), "x velocity (Angstrom/fs)")
VEL_Y = FieldSpec(keys.VY, np.dtype(np.float64), "y velocity (Angstrom/fs)")
VEL_Z = FieldSpec(keys.VZ, np.dtype(np.float64), "z velocity (Angstrom/fs)")
RES_ID = FieldSpec(keys.RES_ID, np.dtype(np.int64), "residue ID")
RES_NAME = FieldSpec(keys.RES_NAME, np.dtype("U8"), "residue name")
ORDER = FieldSpec(keys.ORDER, np.dtype(np.float64), "bond order")
BEAD_TYPE = FieldSpec(keys.BEAD_TYPE, np.dtype("U64"), "coarse-grained bead type")

# Short aliases used at some call sites.
X = POS_X
Y = POS_Y
Z = POS_Z
TYPE = ATOM_TYPE
ID = ATOM_ID

# ===================================================================
#               Canonical Topology (relation) Fields
# ===================================================================

BOND_TYPE = FieldSpec(keys.TYPE, np.dtype("U64"), "bond type label")
BOND_ATOMI = FieldSpec(keys.ATOMI, np.dtype(np.int64), "first atom index (0-indexed)")
BOND_ATOMJ = FieldSpec(keys.ATOMJ, np.dtype(np.int64), "second atom index (0-indexed)")

ANGLE_TYPE = FieldSpec(keys.TYPE, np.dtype("U64"), "angle type label")
ANGLE_ATOMI = FieldSpec(keys.ATOMI, np.dtype(np.int64), "first atom index (0-indexed)")
ANGLE_ATOMJ = FieldSpec(keys.ATOMJ, np.dtype(np.int64), "vertex atom index (0-indexed)")
ANGLE_ATOMK = FieldSpec(keys.ATOMK, np.dtype(np.int64), "third atom index (0-indexed)")

DIHEDRAL_TYPE = FieldSpec(keys.TYPE, np.dtype("U64"), "dihedral type label")
DIHEDRAL_ATOMI = FieldSpec(keys.ATOMI, np.dtype(np.int64), "first atom index (0-indexed)")
DIHEDRAL_ATOMJ = FieldSpec(keys.ATOMJ, np.dtype(np.int64), "second atom index (0-indexed)")
DIHEDRAL_ATOMK = FieldSpec(keys.ATOMK, np.dtype(np.int64), "third atom index (0-indexed)")
DIHEDRAL_ATOML = FieldSpec(keys.ATOML, np.dtype(np.int64), "fourth atom index (0-indexed)")


# ===================================================================
#                       FieldFormatter
# ===================================================================


class FieldFormatter:
    """Translates between format-specific and canonical field names.

    Subclasses define ``_field_formatters`` as a class-level registry mapping
    format-native column names to :class:`FieldSpec` objects. Registrations are
    isolated per subclass via ``__init_subclass__``.

    Example::

        class LammpsFieldFormatter(FieldFormatter):
            _field_formatters = {
                "q":   CHARGE,   # LAMMPS "q" → canonical "charge"
                "mol": MOL_ID,   # LAMMPS "mol" → canonical "mol_id"
            }
    """

    _field_formatters: ClassVar[dict[str, FieldSpec]] = {}

    def __init_subclass__(cls, **kwargs: object) -> None:
        super().__init_subclass__(**kwargs)
        cls._field_formatters = dict(cls._field_formatters)

    @classmethod
    def register_field(cls, format_key: str, spec: FieldSpec) -> None:
        """Register a field mapping at runtime."""
        cls._field_formatters[format_key] = spec

    # ── Block-level translation ──────────────────────────────────

    def canonicalize(self, block):
        """Reader exit: rename format-specific keys to canonical (in-place)."""
        for fmt_key, spec in self._field_formatters.items():
            if fmt_key in block and spec.key not in block:
                block.rename(fmt_key, spec.key)
        return block

    def localize(self, block):
        """Writer entry: rename canonical keys to format-specific (in-place)."""
        for fmt_key, spec in self._field_formatters.items():
            if spec.key in block and fmt_key not in block:
                block.rename(spec.key, fmt_key)
        return block

    # ── Frame-level convenience ──────────────────────────────────

    def canonicalize_frame(self, frame):
        """Canonicalize all blocks in a Frame (in-place)."""
        for key in list(frame.keys()):
            self.canonicalize(frame[key])
        return frame

    def localize_frame(self, frame):
        """Localize all blocks in a Frame (in-place)."""
        for key in list(frame.keys()):
            self.localize(frame[key])
        return frame


# ===================================================================
#               Per-format Formatters
# ===================================================================


class GroFieldFormatter(FieldFormatter):
    """GRO format ↔ canonical field name translation."""

    _field_formatters: ClassVar[dict[str, FieldSpec]] = {
        "resid": RES_ID,
        "resname": RES_NAME,
        "atom_name": ATOM_NAME,
        "atom_id": ATOM_ID,
    }


class PdbFieldFormatter(FieldFormatter):
    """PDB format ↔ canonical field name translation."""

    _field_formatters: ClassVar[dict[str, FieldSpec]] = {
        "resid": RES_ID,
        "resname": RES_NAME,
        "symbol": ELEMENT,
    }


class LammpsFieldFormatter(FieldFormatter):
    """LAMMPS data format ↔ canonical field name translation."""

    _field_formatters: ClassVar[dict[str, FieldSpec]] = {
        "q": CHARGE,
        "mol": MOL_ID,
    }


class XyzFieldFormatter(FieldFormatter):
    """XYZ format ↔ canonical field name translation."""

    _field_formatters: ClassVar[dict[str, FieldSpec]] = {
        "symbol": ELEMENT,
    }


__all__ = [
    "FieldSpec",
    "FieldFormatter",
    "GroFieldFormatter",
    "PdbFieldFormatter",
    "LammpsFieldFormatter",
    "XyzFieldFormatter",
    # atom fields
    "ATOM_ID", "ATOM_TYPE", "CHARGE", "MASS", "MOL_ID", "ELEMENT", "SYMBOL",
    "ATOM_NAME", "NAME", "POS_X", "POS_Y", "POS_Z", "XYZ",
    "VEL_X", "VEL_Y", "VEL_Z", "RES_ID", "RES_NAME", "ORDER", "BEAD_TYPE",
    "X", "Y", "Z", "TYPE", "ID",
    # relation fields
    "BOND_TYPE", "BOND_ATOMI", "BOND_ATOMJ",
    "ANGLE_TYPE", "ANGLE_ATOMI", "ANGLE_ATOMJ", "ANGLE_ATOMK",
    "DIHEDRAL_TYPE", "DIHEDRAL_ATOMI", "DIHEDRAL_ATOMJ",
    "DIHEDRAL_ATOMK", "DIHEDRAL_ATOML",
]
