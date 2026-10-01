"""Frame vocabulary, projected from the compiled Rust tables.

``ColumnSpec`` / ``BlockSpec`` live here as a real package path so pickle and
``from molrs.schema import ColumnSpec`` resolve the same class the binder
declares (``module = "molrs.schema"``).

The canonical block names (``ATOMS``, ``BONDS``, ``ANGLES``, ``DIHEDRALS``,
``IMPROPERS``, ``PAIRS``, ``EXCLUSIONS``; ``TOPOLOGY`` is the bonded relation
blocks in increasing arity) are Rust's ``store::schema::block_names``, and
:func:`relation_endpoints` says which block and columns a relation block's
rows point into.
"""

from ._lib import schema as _schema

ColumnSpec = _schema.ColumnSpec
BlockSpec = _schema.BlockSpec
columns = _schema.columns
blocks = _schema.blocks
VOCAB_VERSION = _schema.VOCAB_VERSION
column = _schema.column
block = _schema.block
relation_endpoints = _schema.relation_endpoints
to_json = _schema.to_json
to_markdown = _schema.to_markdown
ATOMS = _schema.ATOMS
BONDS = _schema.BONDS
ANGLES = _schema.ANGLES
DIHEDRALS = _schema.DIHEDRALS
IMPROPERS = _schema.IMPROPERS
PAIRS = _schema.PAIRS
EXCLUSIONS = _schema.EXCLUSIONS
TOPOLOGY = _schema.TOPOLOGY

__all__ = [
    "ANGLES",
    "ATOMS",
    "BONDS",
    "DIHEDRALS",
    "EXCLUSIONS",
    "IMPROPERS",
    "PAIRS",
    "TOPOLOGY",
    "VOCAB_VERSION",
    "BlockSpec",
    "ColumnSpec",
    "block",
    "blocks",
    "column",
    "columns",
    "relation_endpoints",
    "to_json",
    "to_markdown",
]
