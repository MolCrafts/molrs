"""Compose-only OPLS path — no typifier.build() (0.12)."""
from __future__ import annotations

import molrs
from molrs.conformer import Conformer
from molrs.perceive import Perceive

assert not hasattr(molrs.ff.OPLSAATypifier(), "build")
assert not hasattr(molrs, "Entity")
assert not hasattr(molrs.ff, "PairLJ126Style")

mol = molrs.io.SmilesIR("CCO").to_atomistic()
mol = Perceive().find_hydrogens(mol)
mol, _report = Conformer(seed=0).generate(mol)
mol.generate_topology(gen_angle=True, gen_dihedral=True)

typifier = molrs.ff.OPLSAATypifier()
typed = typifier.typify(mol)
frame = typed.to_frame()
frame["pairs"] = molrs.ff.intramolecular_pairs(frame)
pots = typifier.forcefield().to_potentials(frame)
coords = molrs.ff.extract_coords(frame)
energy, forces = pots.calc_energy_forces(coords)
assert energy == energy  # finite
print("legacy-delete regression ok", energy)
