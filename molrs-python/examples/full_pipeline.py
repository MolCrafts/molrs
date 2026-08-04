"""Full pipeline: build molecules from scratch, generate 3D, evaluate forces.

0.12 surface: molrs.conformer / molrs.ff only (no top-level typifier re-exports).
"""

from __future__ import annotations

from molrs import Atomistic
from molrs.conformer import Conformer
from molrs.ff import MMFF94Typifier, extract_coords, intramolecular_pairs


def build_methane() -> Atomistic:
    """CH4 — just carbon, hydrogens added by embed."""
    mol = Atomistic()
    mol.add_atom("C")
    return mol


def build_ethanol() -> Atomistic:
    """C-C-O skeleton."""
    mol = Atomistic()
    c1 = mol.add_atom("C")
    c2 = mol.add_atom("C")
    o = mol.add_atom("O")
    mol.add_bond(c1, c2)
    mol.add_bond(c2, o)
    return mol


def build_benzene() -> Atomistic:
    """6-membered aromatic ring."""
    mol = Atomistic()
    carbons = [mol.add_atom("C") for _ in range(6)]
    for i in range(6):
        rh = mol.add_bond(carbons[i], carbons[(i + 1) % 6])
        mol.set_relation_prop("bonds", rh, "order", 1.5)
    return mol


def build_acetic_acid() -> Atomistic:
    """CH3-C(=O)-OH skeleton."""
    mol = Atomistic()
    c_me = mol.add_atom("C")
    c_co = mol.add_atom("C")
    o_dbl = mol.add_atom("O")
    o_oh = mol.add_atom("O")
    mol.add_bond(c_me, c_co)
    rh = mol.add_bond(c_co, o_dbl)
    mol.set_relation_prop("bonds", rh, "order", 2.0)
    mol.add_bond(c_co, o_oh)
    return mol


molecules = {
    "methane": build_methane(),
    "ethanol": build_ethanol(),
    "benzene": build_benzene(),
    "acetic_acid": build_acetic_acid(),
}

typifier = MMFF94Typifier()

for name, mol in molecules.items():
    print(f"=== {name} ===")
    print(f"  input: atoms={mol.n_atoms}, bonds={mol.n_relations('bonds')}")

    mol3d, report = Conformer(speed="medium", seed=123).generate(mol)
    print(f"  conformer: atoms={mol3d.n_atoms}, energy={report.final_energy:.2f}")

    try:
        frame = typifier.typify(mol3d).to_frame()
        frame["pairs"] = intramolecular_pairs(frame)
        pots = typifier.forcefield().to_potentials(frame)
        coords = extract_coords(frame)
        energy, forces = pots.calc_energy_forces(coords)
        print(f"  MMFF94 energy={energy:.4f}, forces shape={forces.shape}")
    except Exception as exc:  # noqa: BLE001 — demo script
        print(f"  MMFF94 skip: {exc}")
