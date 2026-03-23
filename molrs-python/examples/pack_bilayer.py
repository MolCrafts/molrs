#!/usr/bin/env python3
"""
Lipid bilayer packing example.

Reads lipid and water PDB structures from the molrs-pack examples,
then packs a bilayer system: water below, lipid lower leaflet,
lipid upper leaflet, water above.

Based on the Rust example at:
  molrs-pack/examples/pack_bilayer/main.rs

Usage:
  cd molrs-python
  python examples/pack_bilayer.py --input-dir ../../molrs-pack/examples/pack_bilayer
"""

import argparse
import os
import molrs

DEFAULT_BILAYER_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "molrs-pack", "examples", "pack_bilayer"
)


def main():
    parser = argparse.ArgumentParser(description="Pack a lipid bilayer example.")
    parser.add_argument(
        "--input-dir",
        default=DEFAULT_BILAYER_DIR,
        help="Directory containing water.pdb and palmitoil.pdb",
    )
    args = parser.parse_args()

    water = molrs.read_pdb(os.path.join(args.input_dir, "water.pdb"))
    lipid = molrs.read_pdb(os.path.join(args.input_dir, "palmitoil.pdb"))
    print(f"water: {water['atoms'].nrows} atoms, lipid: {lipid['atoms'].nrows} atoms")

    # Water below the bilayer: z in [-10, 0]
    water_low = (
        molrs.Target(water, count=500)
        .with_constraint(molrs.InsideBox([0, 0, -10], [40, 40, 0]))
        .with_name("water_low")
    )

    # Water above the bilayer: z in [28, 38]
    water_high = (
        molrs.Target(water, count=500)
        .with_constraint(molrs.InsideBox([0, 0, 28], [40, 40, 38]))
        .with_name("water_high")
    )

    # Lower lipid leaflet: z in [0, 14]
    lipid_low = (
        molrs.Target(lipid, count=50)
        .with_constraint(molrs.InsideBox([0, 0, 0], [40, 40, 14]))
        .with_name("lipid_low")
    )

    # Upper lipid leaflet: z in [14, 28]
    lipid_high = (
        molrs.Target(lipid, count=50)
        .with_constraint(molrs.InsideBox([0, 0, 14], [40, 40, 28]))
        .with_name("lipid_high")
    )

    targets = [water_low, water_high, lipid_low, lipid_high]
    total_atoms = sum(t.count * t.natoms for t in targets)
    print(f"Packing {sum(t.count for t in targets)} molecules ({total_atoms} atoms)...")

    packer = molrs.Packer(tolerance=2.0, precision=0.01)
    result = packer.pack(targets, max_loops=300, seed=2026)

    print(f"converged={result.converged}, fdist={result.fdist:.4f}, frest={result.frest:.4f}")
    print(f"positions: {result.positions.shape}, elements: {len(result.elements)}")

    # Write XYZ with correct element symbols
    out_path = os.path.join(os.path.dirname(__file__), "bilayer_output.xyz")
    natoms = result.positions.shape[0]
    with open(out_path, "w") as f:
        f.write(f"{natoms}\n")
        f.write(f"bilayer converged={result.converged}\n")
        for elem, pos in zip(result.elements, result.positions):
            f.write(f"{elem:2s}  {pos[0]:10.4f}  {pos[1]:10.4f}  {pos[2]:10.4f}\n")

    print(f"Wrote {out_path}")

    # Show element distribution
    from collections import Counter
    counts = Counter(result.elements)
    print(f"Element counts: {dict(sorted(counts.items()))}")


if __name__ == "__main__":
    main()
