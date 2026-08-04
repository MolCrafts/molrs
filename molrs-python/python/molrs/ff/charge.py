"""Partial-charge models — ``molrs::ff::charge``.

Three models behind one ``assign(mol)`` call. :class:`BccModel` (AM1-BCC /
ABCG2) and :class:`MullikenModel` need a QM population as input;
:class:`GasteigerModel` (PEOE) is topological and takes no QM at all — it
accepts the ``qm`` argument and ignores it, which is what keeps the three
interchangeable at a call site.
"""

from __future__ import annotations

from .._lib import (
    BccModel as BccModel,
    GasteigerModel as GasteigerModel,
    MullikenModel as MullikenModel,
)

__all__ = [
    "BccModel",
    "GasteigerModel",
    "MullikenModel",
]
