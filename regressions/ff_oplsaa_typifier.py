r"""OPLS-AA from a SMILES string, composed step by step.

The library exposes primitives; assembling them is the caller's job, and this is
what that assembly looks like end to end: parse, perceive hydrogens, embed,
enumerate topology, typify, build the pair list, compile potentials, evaluate.
There is no ``typifier.build()`` shortcut, by design — the Frame where a missing
term would be visible must stay in the caller's hands.

No third-party scientific package. Runner:

    uv --directory molrs-python run python ../regressions/ff_oplsaa_typifier.py
"""

from __future__ import annotations

import molrs
from molrs.conformer import Conformer
from molrs.perceive import Perceive

mol = molrs.io.SmilesIR("CCO").to_atomistic()
mol = Perceive().find_hydrogens(mol)
mol, _report = Conformer(seed=0).generate(mol)
mol.generate_topology(gen_angle=True, gen_dihedral=True)

typifier = molrs.ff.OPLSAATypifier()
frame = typifier.typify(mol).to_frame()
ff = typifier.forcefield()
frame["pairs"] = molrs.ff.intramolecular_pairs(frame, ff)

potentials = ff.to_potentials(frame)
coords = molrs.ff.extract_coords(frame)
energy, forces = potentials.calc_energy_forces(coords)

assert energy == energy, "energy is NaN"
assert forces.shape == (frame["atoms"].nrows, 3), forces.shape

print(f"ff_oplsaa_typifier ok: ethanol E = {energy:.6f} kcal/mol")
