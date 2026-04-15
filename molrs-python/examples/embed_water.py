"""Generate 3D coordinates for a water molecule.

Demonstrates: Atomistic construction → generate_3d → inspect report.
Input is just atoms + bonds, no coordinates.
"""

from molrs import Atomistic, EmbedOptions, generate_3d

# Build water: O-H-H (no coordinates)
mol = Atomistic()
o = mol.add_atom("O")
h1 = mol.add_atom("H")
h2 = mol.add_atom("H")
mol.add_bond(o, h1)
mol.add_bond(o, h2)

print(f"Input: {mol}")
print(f"  atoms={mol.n_atoms}, bonds={mol.n_bonds}")

# Generate 3D coordinates
opts = EmbedOptions(speed="medium", seed=42)
result = generate_3d(mol, opts)

out = result.mol
report = result.report

print(f"\nOutput: {out}")
print(f"  atoms={out.n_atoms} (hydrogens may be added)")
print(f"  final_energy={report.final_energy:.4f}")

print("\nStages:")
for stage in report.stages:
    print(f"  {stage.stage}: steps={stage.steps}, converged={stage.converged}, "
          f"elapsed={stage.elapsed_ms}ms")

if report.warnings:
    print("\nWarnings:")
    for w in report.warnings:
        print(f"  - {w}")

# Export to Frame and inspect coordinates
frame = out.to_frame()
atoms = frame["atoms"]
print("\nAtom coordinates:")
x = atoms.view("x")
y = atoms.view("y")
z = atoms.view("z")
for i in range(len(x)):
    print(f"  [{i}] ({x[i]:+.4f}, {y[i]:+.4f}, {z[i]:+.4f})")
