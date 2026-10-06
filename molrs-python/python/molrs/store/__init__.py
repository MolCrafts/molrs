"""The column store and the frame — ``molrs::store``.

* :class:`Block` — a heterogeneous column store (numpy arrays).
* :class:`Frame` — named blocks plus a :class:`~molrs.spatial.Box` and
  :class:`FrameMeta` metadata (:class:`MetaValue` typed scalars,
  :class:`MetaDocument` nested documents).
* :class:`Trajectory` — an in-memory frame sequence with its
  :class:`ScalarObservable` / :class:`VectorObservable` records.
* :exc:`BlockDtypeError` — a column value the store cannot hold (object,
  None-bearing or ragged); a ``TypeError``.
* :mod:`molrs.store.keys` / :mod:`molrs.store.schema` — the column
  vocabulary: canonical names and their dtypes.
"""

# `collections.abc.Mapping` is aliased so that it is not exported here.
from collections.abc import Mapping as _AbcMapping
from collections.abc import MutableMapping as _AbcMutableMapping

from .._lib import (
    Block,
    BlockDtypeError,
    Frame,
    FrameMeta,
    MetaDocument,
    MetaValue,
    ScalarObservable,
    Trajectory,
    VectorObservable,
)
from . import keys, schema

# `frame.meta` implements the full mapping protocol in Rust; this makes
# `isinstance(frame.meta, MutableMapping)` say so too.
_AbcMutableMapping.register(FrameMeta)
# Registration supplies isinstance only — MetaDocument implements its own
# surface. Callers that branch on Mapping rather than dict:
# molvis/python/src/molvis/wire.py:389,530
# molrec/tests/molrs_adapter.py:110-113
_AbcMapping.register(MetaDocument)

__all__ = [
    "Block",
    "BlockDtypeError",
    "Frame",
    "FrameMeta",
    "MetaDocument",
    "MetaValue",
    "ScalarObservable",
    "Trajectory",
    "VectorObservable",
    "keys",
    "schema",
]
