r"""A hand-written Amber prmtop, read through molrs.

Covers the whole read: the structure `molrs.io.read_amber_prmtop` produces, the
force field `molrs.ff.read_amber_prmtop_ff_str` produces from the same file, and
the 1-4 scaling that travels from the prmtop into `ForceField` and out through
the LAMMPS include.

Goldens are literals from the Amber PARM spec
<https://ambermd.org/FileFormats.php> (accessed 2026-09-04), from the fixture
text itself, and from the A/B closed form with published GAFF R*/ε
(c3: R* = 1.9080 Å, ε = 0.1094 kcal/mol). No AmberTools, ParmEd, RDKit or any
other third-party scientific package is imported or subprocessed.

Runner:

    uv --directory molrs-python run python ../regressions/read_amber_prmtop.py
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

import molrs

LITFSI = """%VERSION  VERSION_STAMP = V0001.000
%FLAG TITLE
%FORMAT(20a4)
TFSI
%FLAG POINTERS
%FORMAT(10I8)
      16       6       0      14       0      25       0      27       0       0
      65       2      14      25      27       7      12       4       7       0
       0       0       0       0       0       0       0       1      15       0
       0
%FLAG ATOM_NAME
%FORMAT(20a4)
F   C   F1  F2  S   O   O3  N   S1  O1  O2  C1  F4  F5  F3  LI
%FLAG CHARGE
%FORMAT(5E16.8)
 -4.94977802E+00  1.01079098E+01 -4.94977802E+00 -4.94977802E+00  2.70364265E+01
 -1.11739144E+01 -1.19684066E+01 -1.92992379E+01  3.06571975E+01 -1.19684066E+01
 -1.19684066E+01  1.01079098E+01 -4.94977802E+00 -4.94977802E+00 -4.94977802E+00
  1.82223000E+01
%FLAG ATOMIC_NUMBER
%FORMAT(10I8)
       9       6       9       9      16       8       8       7      16       8
       8       6       9       9       9       3
%FLAG MASS
%FORMAT(5E16.8)
  1.90000000E+01  1.20100000E+01  1.90000000E+01  1.90000000E+01  3.20600000E+01
  1.60000000E+01  1.60000000E+01  1.40100000E+01  3.20600000E+01  1.60000000E+01
  1.60000000E+01  1.20100000E+01  1.90000000E+01  1.90000000E+01  1.90000000E+01
  6.94000000E+00
%FLAG AMBER_ATOM_TYPE
%FORMAT(20a4)
f   c3  f   f   s6  o   o   ne  sy  o   o   c3  f   f   f   Li+
%FLAG NUMBER_EXCLUDED_ATOMS
%FORMAT(10I8)
       7       7       5       4       7       3       2       7       6       5
       4       3       2       1       1       1
%FLAG EXCLUDED_ATOMS_LIST
%FORMAT(10I8)
       2       3       4       5       6       7       8       3       4       5
       6       7       8       9       4       5       6       7       8       5
       6       7       8       6       7       8       9      10      11      12
       7       8       9       8       9       9      10      11      12      13
      14      15      10      11      12      13      14      15      11      12
      13      14      15      12      13      14      15      13      14      15
      14      15      15       0       0
%FLAG RESIDUE_LABEL
%FORMAT(20a4)
TF  LI
%FLAG RESIDUE_POINTER
%FORMAT(10I8)
       1      16
%FLAG BONDS_INC_HYDROGEN
%FORMAT(10I8)
%FLAG BONDS_WITHOUT_HYDROGEN
%FORMAT(10I8)
      33      36       1      33      39       1      33      42       1      24
      27       2      24      30       2      24      33       3      21      24
       4      12      15       5      12      18       5      12      21       6
       3       6       1       3       9       1       3      12       7       0
       3       1
%FLAG ANGLES_INC_HYDROGEN
%FORMAT(10I8)
%FLAG ANGLES_WITHOUT_HYDROGEN
%FORMAT(10I8)
      39      33      42       1      36      33      39       1      36      33
      42       1      30      24      33       2      27      24      30       3
      27      24      33       2      24      33      36       4      24      33
      39       4      24      33      42       4      21      24      27       5
      21      24      30       5      21      24      33       6      18      12
      21       7      15      12      18       8      15      12      21       7
      12      21      24       9       9       3      12      10       6       3
       9       1       6       3      12      10       3      12      15      11
       3      12      18      11       3      12      21      12       0       3
       6       1       0       3       9       1       0       3      12      10
%FLAG DIHEDRALS_INC_HYDROGEN
%FORMAT(10I8)
%FLAG DIHEDRALS_WITHOUT_HYDROGEN
%FORMAT(10I8)
      30      24      33      36       1      30      24      33      39       1
      30      24      33      42       1      27      24      33      36       1
      27      24      33      39       1      27      24      33      42       1
      21      24      33      36       1      21      24      33      39       1
      21      24      33      42       1      18      12      21      24       2
      15      12      21      24       2      12      21      24      27       3
      12      21     -24      27       4      12      21      24      30       3
      12      21     -24      30       4      12      21      24      33       3
      12      21     -24      33       4       9       3      12      15       1
       9       3      12      18       1       9       3      12      21       1
       6       3      12      15       1       6       3      12      18       1
       6       3      12      21       1       3      12      21      24       2
       0       3      12      15       1       0       3      12      18       1
       0       3      12      21       1
%FLAG TREE_CHAIN_CLASSIFICATION
%FORMAT(20a4)
E   M   E   E   M   E   E   M   M   E   E   M   E   E   E   BLA
%FLAG JOIN_ARRAY
%FORMAT(10I8)
       0       0       0       0       0       0       0       0       0       0
       0       0       0       0       0       0
%FLAG IROTAT
%FORMAT(10I8)
       0       0       0       0       0       0       0       0       0       0
       0       0       0       0       0       0
%FLAG SOLVENT_POINTERS
%FORMAT(3I8)
       2       2       3
%FLAG ATOMS_PER_MOLECULE
%FORMAT(10I8)
      15       1
%FLAG RADIUS_SET
%FORMAT(1a80)
modified Bondi radii (mbondi2)
%FLAG RADII
%FORMAT(5E16.8)
  1.50000000E+00  1.70000000E+00  1.50000000E+00  1.50000000E+00  1.80000000E+00
  1.50000000E+00  1.50000000E+00  1.55000000E+00  1.80000000E+00  1.50000000E+00
  1.50000000E+00  1.70000000E+00  1.50000000E+00  1.50000000E+00  1.50000000E+00
  1.50000000E+00
%FLAG SCREEN
%FORMAT(5E16.8)
  8.80000000E-01  7.20000000E-01  8.80000000E-01  8.80000000E-01  9.60000000E-01
  8.50000000E-01  8.50000000E-01  7.90000000E-01  9.60000000E-01  8.50000000E-01
  8.50000000E-01  7.20000000E-01  8.80000000E-01  8.80000000E-01  8.80000000E-01
  8.00000000E-01
%FLAG BOX_DIMENSIONS
%FORMAT(5E16.8)
  9.00000000E+01  3.00000000E+01  3.00000000E+01  3.00000000E+01
"""

STAR4 = """%VERSION  VERSION_STAMP = V0001.000
%FLAG TITLE
%FORMAT(20a4)
STAR
%FLAG POINTERS
%FORMAT(10I8)
       4       2       0       3       0       3       0       3       0       0
       7       1       3       3       3       1       1       2       2       0
       0       0       0       0       0       0       0       0       4       0
       0
%FLAG ATOM_NAME
%FORMAT(20a4)
C1  C2  C3  C4
%FLAG CHARGE
%FORMAT(5E16.8)
  0.00000000E+00  0.00000000E+00  0.00000000E+00  0.00000000E+00
%FLAG ATOMIC_NUMBER
%FORMAT(10I8)
       6       6       6       6
%FLAG MASS
%FORMAT(5E16.8)
  1.20100000E+01  1.20100000E+01  1.20100000E+01  1.20100000E+01
%FLAG NUMBER_EXCLUDED_ATOMS
%FORMAT(10I8)
       3       2       1       1
%FLAG EXCLUDED_ATOMS_LIST
%FORMAT(10I8)
       2       3       4       3       4       4       0
%FLAG RESIDUE_LABEL
%FORMAT(20a4)
MOL
%FLAG RESIDUE_POINTER
%FORMAT(10I8)
       1
%FLAG AMBER_ATOM_TYPE
%FORMAT(20a4)
c3  c3  c3  c3
%FLAG BONDS_INC_HYDROGEN
%FORMAT(10I8)
%FLAG BONDS_WITHOUT_HYDROGEN
%FORMAT(10I8)
       0       3       1       3       6       1       3       9       1
%FLAG ANGLES_INC_HYDROGEN
%FORMAT(10I8)
%FLAG ANGLES_WITHOUT_HYDROGEN
%FORMAT(10I8)
       0       3       6       1       0       3       9       1       6       3
       9       1
%FLAG DIHEDRALS_INC_HYDROGEN
%FORMAT(10I8)
%FLAG DIHEDRALS_WITHOUT_HYDROGEN
%FORMAT(10I8)
       0       3       6       9       1       0       3      -6      -9       2
       3       0       6      -9       2
"""

GAFF_MINI = """%VERSION VERSION_STAMP = V0001.000
%FLAG TITLE
%FORMAT(20a4)
GAFF_MINI
%FLAG POINTERS
%FORMAT(10I8)
       4       2       1       2       1       1       0       1       0       0
       0       1       2       1       1       2       2       2       2       0
       0       0       0       0       0       0       0       0       4       0
       0
%FLAG ATOM_NAME
%FORMAT(20a4)
C1  C2  C3  H1
%FLAG CHARGE
%FORMAT(5E16.8)
  0.00000000E+00  0.00000000E+00  0.00000000E+00  0.00000000E+00
%FLAG MASS
%FORMAT(5E16.8)
  1.20100000E+01  1.20100000E+01  1.20100000E+01  1.00800000E+00
%FLAG AMBER_ATOM_TYPE
%FORMAT(20a4)
c3  c3  c3  hc
%FLAG ATOM_TYPE_INDEX
%FORMAT(10I8)
       1       1       1       2
%FLAG NONBONDED_PARM_INDEX
%FORMAT(10I8)
       1       2       2       3
%FLAG BOND_FORCE_CONSTANT
%FORMAT(5E16.8)
  3.00000000E+02  3.40000000E+02
%FLAG BOND_EQUIL_VALUE
%FORMAT(5E16.8)
  1.53500000E+00  1.09000000E+00
%FLAG ANGLE_FORCE_CONSTANT
%FORMAT(5E16.8)
  5.00000000E+01  4.00000000E+01
%FLAG ANGLE_EQUIL_VALUE
%FORMAT(5E16.8)
  1.91113553E+00  1.91113553E+00
%FLAG DIHEDRAL_FORCE_CONSTANT
%FORMAT(5E16.8)
  1.50000000E-01  2.00000000E-01
%FLAG DIHEDRAL_PERIODICITY
%FORMAT(5E16.8)
  3.00000000E+00  2.00000000E+00
%FLAG DIHEDRAL_PHASE
%FORMAT(5E16.8)
  0.00000000E+00  0.00000000E+00
%FLAG SCEE_SCALE_FACTOR
%FORMAT(5E16.8)
  1.20000000E+00  1.20000000E+00
%FLAG SCNB_SCALE_FACTOR
%FORMAT(5E16.8)
  2.00000000E+00  2.00000000E+00
%FLAG LENNARD_JONES_ACOEF
%FORMAT(5E16.8)
  1.04308023E+06  9.717081166135172E+04  7.516077034091E+03
%FLAG LENNARD_JONES_BCOEF
%FORMAT(5E16.8)
  6.75612248E+02  1.2691914994192737E+02  2.17257827878E+01
%FLAG BONDS_INC_HYDROGEN
%FORMAT(10I8)
       6       9       2
%FLAG BONDS_WITHOUT_HYDROGEN
%FORMAT(10I8)
       0       3       1       3       6       1
%FLAG ANGLES_INC_HYDROGEN
%FORMAT(10I8)
       3       6       9       2
%FLAG ANGLES_WITHOUT_HYDROGEN
%FORMAT(10I8)
       0       3       6       1
%FLAG DIHEDRALS_INC_HYDROGEN
%FORMAT(10I8)
%FLAG DIHEDRALS_WITHOUT_HYDROGEN
%FORMAT(10I8)
       0       3       6       9       1
"""

NBFIX = """%VERSION VERSION_STAMP = V0001.000
%FLAG TITLE
%FORMAT(20a4)
GAFF_MINI
%FLAG POINTERS
%FORMAT(10I8)
       4       2       1       2       1       1       0       1       0       0
       0       1       2       1       1       2       2       2       2       0
       0       0       0       0       0       0       0       0       4       0
       0
%FLAG ATOM_NAME
%FORMAT(20a4)
C1  C2  C3  H1
%FLAG CHARGE
%FORMAT(5E16.8)
  0.00000000E+00  0.00000000E+00  0.00000000E+00  0.00000000E+00
%FLAG MASS
%FORMAT(5E16.8)
  1.20100000E+01  1.20100000E+01  1.20100000E+01  1.00800000E+00
%FLAG AMBER_ATOM_TYPE
%FORMAT(20a4)
c3  c3  c3  hc
%FLAG ATOM_TYPE_INDEX
%FORMAT(10I8)
       1       1       1       2
%FLAG NONBONDED_PARM_INDEX
%FORMAT(10I8)
       1       2       2       3
%FLAG BOND_FORCE_CONSTANT
%FORMAT(5E16.8)
  3.00000000E+02  3.40000000E+02
%FLAG BOND_EQUIL_VALUE
%FORMAT(5E16.8)
  1.53500000E+00  1.09000000E+00
%FLAG ANGLE_FORCE_CONSTANT
%FORMAT(5E16.8)
  5.00000000E+01  4.00000000E+01
%FLAG ANGLE_EQUIL_VALUE
%FORMAT(5E16.8)
  1.91113553E+00  1.91113553E+00
%FLAG DIHEDRAL_FORCE_CONSTANT
%FORMAT(5E16.8)
  1.50000000E-01  2.00000000E-01
%FLAG DIHEDRAL_PERIODICITY
%FORMAT(5E16.8)
  3.00000000E+00  2.00000000E+00
%FLAG DIHEDRAL_PHASE
%FORMAT(5E16.8)
  0.00000000E+00  0.00000000E+00
%FLAG SCEE_SCALE_FACTOR
%FORMAT(5E16.8)
  1.20000000E+00  1.20000000E+00
%FLAG SCNB_SCALE_FACTOR
%FORMAT(5E16.8)
  2.00000000E+00  2.00000000E+00
%FLAG LENNARD_JONES_ACOEF
%FORMAT(5E16.8)
  1.04308023E+06  9.814251977796524E+04  7.516077034091E+03
%FLAG LENNARD_JONES_BCOEF
%FORMAT(5E16.8)
  6.75612248E+02  1.281883414413926E+02  2.17257827878E+01
%FLAG BONDS_INC_HYDROGEN
%FORMAT(10I8)
       6       9       2
%FLAG BONDS_WITHOUT_HYDROGEN
%FORMAT(10I8)
       0       3       1       3       6       1
%FLAG ANGLES_INC_HYDROGEN
%FORMAT(10I8)
       3       6       9       2
%FLAG ANGLES_WITHOUT_HYDROGEN
%FORMAT(10I8)
       0       3       6       1
%FLAG DIHEDRALS_INC_HYDROGEN
%FORMAT(10I8)
%FLAG DIHEDRALS_WITHOUT_HYDROGEN
%FORMAT(10I8)
       0       3       6       9       1
"""



# ── structure ───────────────────────────────────────────────────────────────

def _read(text: str):
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "fixture.prmtop"
        path.write_text(text)
        return molrs.io.read_amber_prmtop(path)


frame = _read(LITFSI)
atoms = frame["atoms"]
assert atoms.nrows == 16
res_name = list(atoms.view("res_name"))
assert res_name[:15] == ["TF"] * 15
assert res_name[15] == "LI"
mol_id = [int(x) for x in atoms.view("mol_id")]
assert mol_id[:15] == [1] * 15
assert mol_id[15] == 2
charge = atoms.view("charge")
assert abs(float(charge[15]) - 1.0) < 1e-12

dihedrals = frame["dihedrals"]
assert dihedrals.nrows == 27
flag = list(dihedrals.view("exclude_14"))
assert [i for i, v in enumerate(flag) if v] == [12, 14, 16]

excl = frame["exclusions"]
assert excl.nrows == 63
assert int(excl.view("atomi")[0]) == 0
assert int(excl.view("atomj")[0]) == 1

box = frame.box
assert box is not None
lengths = [float(x) for x in box.lengths]
assert all(abs(L - 30.0) < 1e-9 for L in lengths)
assert float(frame.meta["oldbeta"]) == 90.0

star = _read(STAR4)
d = star["dihedrals"]
imp = star["impropers"]
assert d.nrows == 3
assert imp.nrows == 2

def _key(block, row):
    return tuple(int(block.view(c)[row]) for c in ("atomi", "atomj", "atomk", "atoml", "type_id"))

d_flag = list(d.view("exclude_14"))
i_flag = list(imp.view("exclude_14"))
d_map = {_key(d, i): d_flag[i] for i in range(d.nrows)}
for i in range(imp.nrows):
    assert i_flag[i] == d_map[_key(imp, i)]
assert i_flag[0] is True or i_flag[0] == True
assert i_flag[1] is False or i_flag[1] == False


# ── force field ─────────────────────────────────────────────────────────────

ff = molrs.ff.read_amber_prmtop_ff_str(GAFF_MINI)

def _rel(got: float, expected: float, tol: float = 1e-6) -> None:
    assert abs(got - expected) <= tol * abs(expected), (got, expected)


lj = ff.get_style("pair", "lj/cut")
coul = ff.get_style("pair", "coul/cut")
assert lj is not None and coul is not None
c3 = next(t for t in lj.types if t.name in ("c3", "c3-c3") or t.name.startswith("c3"))
eps = float(c3["epsilon"])
sigma = float(c3["sigma"])
_rel(eps, 0.1094)
_rel(sigma, 3.3996695)

coul_p = ff.style_params("pair", "coul/cut")
lj_p = ff.style_params("pair", "lj/cut")
assert abs(float(coul_p["coulomb"]) - 332.05221729) < 1e-8
assert abs(float(coul_p["dielectric"]) - 1.0) < 1e-12
assert abs(float(coul_p["cutoff"]) - 10.0) < 1e-12
assert abs(float(lj_p["cutoff"]) - 9.0) < 1e-12

text = molrs.ff.write_lammps_forcefield_str(ff, skip_pair_style=True)
assert "pair_style" not in text
assert "pair_coeff c3 c3" in text

try:
    molrs.ff.read_amber_prmtop_ff_str(NBFIX)
except Exception as e:
    msg = str(e)
    assert "c3" in msg and "hc" in msg
else:
    raise SystemExit("NBFIX variant should have raised")


# ── 1-4 scaling, prmtop → ForceField → LAMMPS include ───────────────────────

np.testing.assert_allclose(ff.special_bonds_lj, [0.0, 0.0, 0.5])
np.testing.assert_allclose(ff.special_bonds_coul, [0.0, 0.0, 1.0 / 1.2])

include = molrs.ff.write_lammps_forcefield_str(ff)
assert "special_bonds lj" in include
assert "0.500000" in include          # 1 / SCNB
assert "0.833333" in include          # 1 / SCEE, at writer precision 6

print(
    "read_amber_prmtop ok: 16 atoms, exclude_14={12,14,16}, exclusions=63, "
    "box=30, oldbeta=90; c3 sigma/eps, coulomb=332.05221729; "
    "special_bonds 0.5 / 0.833333"
)
