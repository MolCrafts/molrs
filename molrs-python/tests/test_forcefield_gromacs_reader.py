"""Seam smoke test for the GROMACS topology force-field reader.

The directive model (what is read, what is refused, what a skip does) is proved
in Rust (``ff::forcefield::readers::gromacs``). This test only asserts that the
public ``molrs.ff.read_gromacs_top_ff_str`` maps a refused directive to
``ValueError`` and that its ``skip_directives`` keyword reaches the Rust reader.
"""

import pytest

import molrs

# One OPLS-AA atom type (GROMACS v2026.3 ``ffnonbonded.itp`` row) followed by a
# ``[ constrainttypes ]`` section, which the reader refuses unless skipped.
_TOP = """\
[ defaults ]
; nbfunc  comb-rule  gen-pairs  fudgeLJ  fudgeQQ
1  3  yes  0.5  0.5

[ atomtypes ]
; name  bond_type  at.num  mass  charge  ptype  sigma  epsilon
opls_135  CT  6  12.01100  -0.180  A  3.50000e-01  2.76144e-01

[ constrainttypes ]
; i  j  funct  b0
CT  HC  1  0.10900
"""


def test_constrainttypes_is_refused_without_skip_and_read_past_with_it():
    with pytest.raises(ValueError, match="constrainttypes"):
        molrs.ff.read_gromacs_top_ff_str(_TOP)

    ff = molrs.ff.read_gromacs_top_ff_str(_TOP, skip_directives=["constrainttypes"])
    assert isinstance(ff, molrs.ff.ForceField)
