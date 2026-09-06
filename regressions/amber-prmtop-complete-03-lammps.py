r"""LAMMPS include from a prmtop-derived force field declares special_bonds.

Goldens: 1/SCNB = 0.5 and 1/SCEE ≈ 0.833333 at writer precision 6.
No AmberTools. Runner:

    uv --directory molrs-python run python ../regressions/amber-prmtop-complete-03-lammps.py
"""
from __future__ import annotations

from pathlib import Path

import molrs

# Reuse the 02 fixture text.
_src = Path(__file__).with_name("amber-prmtop-complete-02-forcefield.py").read_text()
ns: dict = {}
exec(compile(_src.split("def _rel")[0] + "\n", "gaff_mini.py", "exec"), ns)
GAFF_MINI = ns["GAFF_MINI"]

ff = molrs.ff.read_amber_prmtop_ff_str(GAFF_MINI)
text = molrs.ff.write_lammps_forcefield_str(ff)
assert "special_bonds lj" in text
assert "0.500000" in text
assert "0.833333" in text
print("amber-prmtop-complete-03-lammps ok: special_bonds 0.5 / 0.833333")
