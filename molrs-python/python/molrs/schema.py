"""Frame vocabulary, projected from the compiled Rust tables.

``ColumnSpec`` / ``BlockSpec`` live here as a real package path so pickle and
``from molrs.schema import ColumnSpec`` resolve the same class the binder
declares (``module = "molrs.schema"``).
"""

from ._lib import schema as _schema

ColumnSpec = _schema.ColumnSpec
BlockSpec = _schema.BlockSpec
columns = _schema.columns
blocks = _schema.blocks
VOCAB_VERSION = _schema.VOCAB_VERSION
column = _schema.column
block = _schema.block
to_json = _schema.to_json
to_markdown = _schema.to_markdown

__all__ = [
    "BlockSpec",
    "ColumnSpec",
    "VOCAB_VERSION",
    "block",
    "blocks",
    "column",
    "columns",
    "to_json",
    "to_markdown",
]
