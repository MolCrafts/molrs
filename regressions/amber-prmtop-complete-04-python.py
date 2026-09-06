r"""Python ForceField exposes AMBER 1-4 weights from a prmtop.

Goldens: lj_14 = 0.5, coul_14 = 1/1.2. No AmberTools. Runner:

    uv --directory molrs-python run python ../regressions/amber-prmtop-complete-04-python.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

import molrs

_src = Path(__file__).with_name("amber-prmtop-complete-02-forcefield.py").read_text()
ns: dict = {}
exec(compile(_src.split("def _rel")[0] + "\n", "gaff_mini.py", "exec"), ns)
GAFF_MINI = ns["GAFF_MINI"]

ff = molrs.ff.read_amber_prmtop_ff_str(GAFF_MINI)
np.testing.assert_allclose(ff.special_bonds_lj, [0.0, 0.0, 0.5])
np.testing.assert_allclose(ff.special_bonds_coul, [0.0, 0.0, 1.0 / 1.2])
assert not hasattr(molrs.io, "read_prmtop")
print("amber-prmtop-complete-04-python ok: special_bonds 0.5 / 1/1.2, no read_prmtop")
