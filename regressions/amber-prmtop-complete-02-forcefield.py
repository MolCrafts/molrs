r"""Read a hand-written GAFF-shaped prmtop force field through molrs.ff.

Goldens are hand-derived from the A/B closed form and published GAFF
R*/ε (c3: R*=1.9080 Å, ε=0.1094 kcal/mol). No AmberTools/ParmEd.

Runner:

    uv --directory molrs-python run python ../regressions/amber-prmtop-complete-02-forcefield.py
"""
from __future__ import annotations

import molrs

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


def _rel(got: float, expected: float, tol: float = 1e-6) -> None:
    assert abs(got - expected) <= tol * abs(expected), (got, expected)


ff = molrs.ff.read_amber_prmtop_ff_str(GAFF_MINI)
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

print(
    "amber-prmtop-complete-02-forcefield ok: c3 sigma/eps coulomb=332.05221729 "
    "lj/cut+coul/cut skip_pair_style"
)
