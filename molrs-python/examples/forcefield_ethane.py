"""Generate 3D coordinates for ethane, then evaluate MMFF94 energy and forces.

Demonstrates the full pipeline:
  Atomistic (no coords) → generate_3d → MMFFTypifier.build → Potentials.eval
"""

import numpy as np
from molrs import (
    Atomistic,
    EmbedOptions,
    MMFFTypifier,
    generate_3d,
    extract_coords,
)

# --- 1. Build ethane skeleton: C-C (no coordinates, no hydrogens) ---
mol = Atomistic()
c1 = mol.add_atom("C")
c2 = mol.add_atom("C")
mol.add_bond(c1, c2)

print(f"Input: {mol}")

# --- 2. Generate 3D coordinates (adds hydrogens automatically) ---
opts = EmbedOptions(speed="medium", seed=42)
result = generate_3d(mol, opts)
mol3d = result.mol
report = result.report

print(f"\nAfter embed: {mol3d}")
print(f"  final_energy (internal UFF) = {report.final_energy:.4f}")

# --- 3. Typify with MMFF94 and build potentials ---
typifier = MMFFTypifier()
try:
    potentials = typifier.build(mol3d)
    print(f"\n{potentials}")

    # --- 4. Extract coordinates and evaluate ---
    frame = mol3d.to_frame()
    coords = extract_coords(frame)
    energy, forces = potentials.eval(coords)

    print(f"\nMMFF94 energy: {energy:.4f} kcal/mol")

    n_atoms = len(coords) // 3
    forces_3n = forces.reshape(n_atoms, 3)
    print(f"\nPer-atom forces (kcal/mol/A):")
    for i in range(n_atoms):
        print(f"  [{i}] f = [{forces_3n[i,0]:+8.4f}, "
              f"{forces_3n[i,1]:+8.4f}, {forces_3n[i,2]:+8.4f}]")

    # Newton's 3rd law check
    fsum = forces_3n.sum(axis=0)
    print(f"\nForce sum: [{fsum[0]:+.2e}, {fsum[1]:+.2e}, {fsum[2]:+.2e}] "
          f"(should be ~0)")

except Exception as e:
    print(f"\nBuild skipped (incomplete parameter coverage): {e}")
    print("(embed completed successfully)")
