"""Write the molrec_version 1 fixtures with the published molrs 0.15.0.

    uv venv --python 3.12 v015 && VIRTUAL_ENV=v015 uv pip install molcrafts-molrs==0.15.0
    v015/bin/python generate.py <this directory>

Each record is packed (`<name>.mrec.zip`, molrs 0.15.0's `molrs.io.mrec.pack`).

Every record is N-methylacetamide (one seeded conformer) with a force field in
molrs 0.15's definitions: harmonic `k` of E = ½k(x − x0)², angle values in
radians, `bond morse` `D`, `pair morse` `D0`, `dihedral fourier`, MMFF
out-of-plane rows with the centre second and the per-instance `theta0` column
in radians. `energies.json` holds what molrs 0.15.0 computes for each record;
molrs 0.16 must compute the same from the converted record.
"""

import json
import math
import shutil
import sys
from itertools import combinations_with_replacement
from pathlib import Path

import numpy as np

import molrs

OUT = Path(sys.argv[1])


def molecule():
    mol = molrs.io.smiles.SmilesIr("CC(=O)NC").to_atomistic()
    mol, _ = molrs.conformer.Conformer(seed=7).generate(mol)
    return mol


def block(columns):
    b = molrs.Block()
    for key, values in columns.items():
        b.insert(key, values)
    return b


def packed(path):
    """Pack the record at `path` into `<path>.zip`; no directory is left."""
    molrs.io.mrec.pack(str(path))
    shutil.rmtree(path, ignore_errors=True)


def write(name, frame, ff, energies):
    path = OUT / f"{name}.mrec"
    molrs.io.write_mrec_system(str(path), frame, forcefield=ff)
    packed(path)
    energy, forces = molrs.ff.PotentialCompiler(ff).compile(frame).calc_energy_forces(frame)
    energies[name] = {"energy": float(energy), "forces": np.asarray(forces).ravel().tolist()}


def mmff(energies):
    typifier = molrs.ff.typifier.Mmff94Typifier()
    frame = typifier.typify(molecule()).to_frame()
    ff = typifier.forcefield()
    write("mmff", frame, ff, energies)
    # The same system without its force field: the frame's own columns are
    # all a reader has to convert it by.
    molrs.io.write_mrec_system(str(OUT / "mmff-frame-only.mrec"), frame)
    packed(OUT / "mmff-frame-only.mrec")


def topology():
    """The MMFF-typed molecule's geometry and topology, typed by element."""
    src = molrs.ff.typifier.Mmff94Typifier().typify(molecule()).to_frame()
    atoms = src["atoms"]
    element = [str(e) for e in atoms["element"]]
    frame = molrs.Frame()
    frame["atoms"] = block(
        {
            "x": np.asarray(atoms["x"]),
            "y": np.asarray(atoms["y"]),
            "z": np.asarray(atoms["z"]),
            "type": element,
            "charge": np.asarray(atoms["charge"]),
        }
    )
    ends = {"bonds": 2, "angles": 3, "dihedrals": 4, "impropers": 4}
    columns = ["atomi", "atomj", "atomk", "atoml"]
    rows = {}
    for kind, n in ends.items():
        idx = [np.asarray(src[kind][c], dtype=np.uint32) for c in columns[:n]]
        labels = []
        for row in zip(*idx):
            names = [element[int(a)] for a in row]
            if kind != "impropers" and names[::-1] < names:
                names = names[::-1]
            labels.append("-".join(names))
        frame[kind] = block({**dict(zip(columns[:n], idx)), "type": labels})
        rows[kind] = sorted(set(labels))
    return frame, sorted(set(element)), rows


def base_ff(name, units, elements):
    ff = molrs.ff.ForceField(name, units=units)
    masses = {"C": 12.011, "H": 1.008, "N": 14.007, "O": 15.999}
    atom = ff.def_style("atom", "full")
    types = {e: atom.def_type(e, mass=masses[e]) for e in elements}
    return ff, types


def ends(types, label):
    return [types[e] for e in label.split("-")]


def classic(energies):
    """harmonic bond / angle (½k, radians), multi-term periodic dihedral,
    harmonic improper (chi0 in radians), lj/cut + coul/cut, special bonds."""
    frame, elements, rows = topology()
    ff, types = base_ff("classic", "real", elements)
    ff.set_special_bonds([0.0, 0.0, 0.5], [0.0, 0.0, 0.8333])
    bond = ff.def_style("bond", "harmonic")
    for i, label in enumerate(rows["bonds"]):
        bond.def_type(label, *ends(types, label), k=600.0 + 37.0 * i, r0=1.05 + 0.07 * i)
    angle = ff.def_style("angle", "harmonic")
    for i, label in enumerate(rows["angles"]):
        angle.def_type(label, *ends(types, label), k=80.0 + 9.0 * i, theta0=1.85 + 0.031 * i)
    dihedral = ff.def_style("dihedral", "periodic")
    for i, label in enumerate(rows["dihedrals"]):
        dihedral.def_type(
            label,
            *ends(types, label),
            k1=0.7 + 0.1 * i,
            periodicity1=1.0,
            phase1=0.3 + 0.05 * i,
            k2=0.25,
            periodicity2=2.0,
            phase2=math.pi,
            k3=0.1,
            periodicity3=3.0,
            phase3=0.0,
        )
    improper = ff.def_style("improper", "harmonic")
    for i, label in enumerate(rows["impropers"]):
        improper.def_type(label, *ends(types, label), k=12.0 + i, chi0=0.11 + 0.02 * i)
    lj = ff.def_style("pair", "lj/cut", {"cutoff": 10.0, "mixing": "arithmetic"})
    for i, e in enumerate(elements):
        lj.def_type(e, types[e], types[e], epsilon=0.05 + 0.03 * i, sigma=2.5 + 0.3 * i)
    ff.def_style("pair", "coul/cut", {"coulomb": 332.0716, "cutoff": 10.0})
    write("classic", frame, ff, energies)


def variants(energies):
    """bond morse `D`, angle class2 (theta0 in radians), dihedral charmm
    (phase in radians, w = 0), improper periodic (phase in radians), pair
    morse `D0`."""
    frame, elements, rows = topology()
    ff, types = base_ff("variants", "real", elements)
    bond = ff.def_style("bond", "morse")
    for i, label in enumerate(rows["bonds"]):
        bond.def_type(label, *ends(types, label), D=90.0 + 5.0 * i, alpha=2.0, r0=1.1 + 0.05 * i)
    angle = ff.def_style("angle", "class2")
    for i, label in enumerate(rows["angles"]):
        angle.def_type(
            label, *ends(types, label), theta0=1.9 + 0.02 * i, k2=50.0, k3=-12.0, k4=4.0
        )
    dihedral = ff.def_style("dihedral", "charmm")
    for i, label in enumerate(rows["dihedrals"]):
        dihedral.def_type(
            label, *ends(types, label), k=0.4 + 0.1 * i, periodicity=3.0, phase=0.2 * i, w=0.0
        )
    improper = ff.def_style("improper", "periodic")
    for i, label in enumerate(rows["impropers"]):
        improper.def_type(
            label, *ends(types, label), k=1.1 + 0.1 * i, periodicity=2.0, phase=math.pi
        )
    morse = ff.def_style("pair", "morse", {"cutoff": 8.0})
    for i, (a, b) in enumerate(combinations_with_replacement(elements, 2)):
        morse.def_type(f"{a}-{b}", types[a], types[b], D0=0.1 + 0.01 * i, alpha=1.5, r0=3.5)
    write("variants", frame, ff, energies)


def fourier_lj(energies):
    """`dihedral fourier` (the alias of periodic), harmonic terms, in `lj`
    units: molrs 0.15 stated no angle unit beside the `lj` preset."""
    frame, elements, rows = topology()
    ff, types = base_ff("fourier-lj", "lj", elements)
    bond = ff.def_style("bond", "harmonic")
    for i, label in enumerate(rows["bonds"]):
        bond.def_type(label, *ends(types, label), k=300.0 + 11.0 * i, r0=1.2)
    angle = ff.def_style("angle", "harmonic")
    for i, label in enumerate(rows["angles"]):
        angle.def_type(label, *ends(types, label), k=40.0 + 3.0 * i, theta0=2.0 - 0.01 * i)
    dihedral = ff.def_style("dihedral", "fourier")
    for i, label in enumerate(rows["dihedrals"]):
        dihedral.def_type(
            label,
            *ends(types, label),
            k1=0.5,
            periodicity1=1.0,
            phase1=0.7 + 0.03 * i,
            k2=0.2 + 0.01 * i,
            periodicity2=3.0,
            phase2=math.pi / 3.0,
        )
    lj = ff.def_style("pair", "lj/cut", {"cutoff": 6.0, "mixing": "geometric"})
    for i, e in enumerate(elements):
        lj.def_type(e, types[e], types[e], epsilon=0.8 + 0.1 * i, sigma=1.0 + 0.05 * i)
    write("fourier-lj", frame, ff, energies)


def class2_metal(energies):
    """`dihedral class2` (phi in radians) and `angle class2` in `metal`."""
    frame, elements, rows = topology()
    ff, types = base_ff("class2-metal", "metal", elements)
    bond = ff.def_style("bond", "class2")
    for label in rows["bonds"]:
        bond.def_type(label, *ends(types, label), r0=1.3, k2=20.0, k3=-30.0, k4=40.0)
    angle = ff.def_style("angle", "class2")
    for i, label in enumerate(rows["angles"]):
        angle.def_type(label, *ends(types, label), theta0=1.95 + 0.01 * i, k2=3.0, k3=-1.0, k4=0.5)
    dihedral = ff.def_style("dihedral", "class2")
    for i, label in enumerate(rows["dihedrals"]):
        dihedral.def_type(
            label,
            *ends(types, label),
            k1=0.02,
            phi1=0.1 * i,
            k2=0.01,
            phi2=math.pi / 2.0,
            k3=0.005,
            phi3=-0.4,
        )
    write("class2-metal", frame, ff, energies)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    energies = {}
    mmff(energies)
    classic(energies)
    variants(energies)
    fourier_lj(energies)
    class2_metal(energies)
    (OUT / "energies.json").write_text(json.dumps(energies, indent=1) + "\n")
    for name, value in energies.items():
        print(name, value["energy"])


if __name__ == "__main__":
    main()
