"""3D conformer generation — ``molrs::conformer``.

ETKDGv3 constraints, 4D distance-geometry embedding, experimental-torsion
refinement, MMFF94 cleanup and stereo guards, behind one entry point::

    mol, report = molrs.conformer.Conformer(...).generate(mol)

The report types carry per-stage diagnostics, so a failed embedding says which
stage gave up rather than returning coordinates nobody should trust.
"""

from __future__ import annotations

from ._lib import (
    Conformer as Conformer,
    ConformerReport as ConformerReport,
    ConformerStageReport as ConformerStageReport,
)

__all__ = [
    "Conformer",
    "ConformerReport",
    "ConformerStageReport",
]
