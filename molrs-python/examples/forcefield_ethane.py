"""Generate 3D coordinates for ethane, then evaluate MMFF94 energy and forces.

Pipeline (0.12):
  Atomistic → Conformer.generate → MMFF94Typifier.typify
    → intramolecular_pairs → ForceField.to_potentials → calc_energy_forces
"""

from __future__ import annotations

import numpy as np
from molrs import Atomistic
from molrs.conformer import Conformer
from molrs.ff import MMFF94Typifier, extract_coords, intramolecular_pairs

# --- 1. Build ethane skeleton: C-C (no coordinates, no hydrogens) ---
mol = Atomistic()
c1 = mol.add_atom("C")
c2 = mol.add_atom("C")
mol.add_bond(c1, c2)

print(f"Input: {mol}")

# --- 2. Generate 3D coordinates (adds hydrogens automatically) ---
mol3d, report = Conformer(speed="medium", seed=42).generate(mol)

print(f"\nAfter embed: {mol3d}")
print(f"  final_energy (internal UFF) = {report.final_energy:.4f}")

# --- 3. Typify and evaluate MMFF94 ---
typifier = MMFF94Typifier()
frame = typifier.typify(mol3d).to_frame()
frame["pairs"] = intramolecular_pairs(frame)
pots = typifier.forcefield().to_potentials(frame)
coords = extract_coords(frame)
energy, forces = pots.calc_energy_forces(coords)

print(f"\nMMFF94 energy = {energy:.6f} kcal/mol")
print(f"forces shape = {np.asarray(forces).shape}")
