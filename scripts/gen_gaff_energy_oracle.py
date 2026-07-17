"""Regenerate the GAFF electrostatic/energy oracle with AmberTools25.

The committed numbers are not produced by molrs.  For each required ion this
script performs the external chain

    RDKit SDF -> antechamber AM1-BCC/GAFF -> parmchk2 -> tleap -> sander

at the exact deterministic geometry used by ``gen_am1bcc_oracle.py``.  It also
backs the Coulomb conversion factor out of sander's printed electrostatic energy:

    k = (EELEC + 1-4 EEL) / Σ scale(i,j) q_i q_j / r_ij

where graph-distance 1/2 pairs are excluded and graph-distance 3 pairs use
SCEE=1.2.  Thus the constant and the energy oracle are measurements of the
external implementation, not values copied from documentation.

Usage:
    python scripts/gen_gaff_energy_oracle.py
"""

from __future__ import annotations

import json
import math
import os
import re
import subprocess
import tempfile
from collections import deque
from pathlib import Path

from rdkit import rdBase

from gen_am1bcc_oracle import MOLECULES, embed


ROOT = Path(__file__).resolve().parents[1]
AMBERHOME = Path(
    os.environ.get(
        "AMBERHOME", "/opt/homebrew/Caskroom/miniconda/base/envs/AmberTools25"
    )
)
ANTECHAMBER = AMBERHOME / "bin" / "antechamber"
PARMCHK2 = AMBERHOME / "bin" / "parmchk2"
TLEAP = AMBERHOME / "bin" / "tleap"
SANDER = AMBERHOME / "bin" / "sander"
SCEE = 1.2
AMBER_COULOMB = 332.052_217_29

# These ions are mandatory witnesses, not a convenient neutral subset.  Select
# them from the AM1-BCC oracle generator so both tools have one geometry recipe,
# seed and SMILES list rather than two copies that can drift.
REQUIRED_IONS = ("acetate", "methylammonium", "imidazolium")
CASES = [case for case in MOLECULES if case[0] in REQUIRED_IONS]
if tuple(case[0] for case in CASES) != REQUIRED_IONS:
    raise RuntimeError(f"AM1-BCC generator is missing required ions: {REQUIRED_IONS}")


def run(cmd: list[str], cwd: Path) -> str:
    result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(
            f"command failed ({' '.join(cmd)}):\n{result.stdout}\n{result.stderr}"
        )
    return result.stdout + result.stderr


def parse_mol2(path: Path) -> tuple[list[list[float]], list[float], list[tuple[int, int]]]:
    coords: list[list[float]] = []
    charges: list[float] = []
    bonds: list[tuple[int, int]] = []
    section = ""
    for line in path.read_text().splitlines():
        if line.startswith("@<TRIPOS>"):
            section = line.strip()
            continue
        fields = line.split()
        if section == "@<TRIPOS>ATOM" and fields:
            coords.append([float(fields[2]), float(fields[3]), float(fields[4])])
            charges.append(float(fields[8]))
        elif section == "@<TRIPOS>BOND" and fields:
            bonds.append((int(fields[1]) - 1, int(fields[2]) - 1))
    return coords, charges, bonds


def graph_distance(adjacency: list[list[int]], start: int, target: int) -> int:
    todo: deque[tuple[int, int]] = deque([(start, 0)])
    seen = {start}
    while todo:
        node, distance = todo.popleft()
        if node == target:
            return distance
        for other in adjacency[node]:
            if other not in seen:
                seen.add(other)
                todo.append((other, distance + 1))
    raise RuntimeError("disconnected molecule")


def coulomb_pair_sum(
    coords: list[list[float]], charges: list[float], bonds: list[tuple[int, int]]
) -> tuple[float, float, int]:
    adjacency = [[] for _ in coords]
    for i, j in bonds:
        adjacency[i].append(j)
        adjacency[j].append(i)
    regular = 0.0
    scaled_14 = 0.0
    n14 = 0
    for i in range(len(coords)):
        for j in range(i + 1, len(coords)):
            distance = graph_distance(adjacency, i, j)
            if distance < 3:
                continue
            r = math.dist(coords[i], coords[j])
            term = charges[i] * charges[j] / r
            if distance == 3:
                scaled_14 += term / SCEE
                n14 += 1
            else:
                regular += term
    return regular, scaled_14, n14


def energy_field(text: str, label: str) -> float:
    prefix = r"(?<!1-4 )" if label in {"EEL", "VDWAALS"} else ""
    matches = re.findall(
        rf"{prefix}\b{re.escape(label)}\s*=\s*([-+0-9.Ee]+)", text
    )
    if not matches:
        raise RuntimeError(f"sander output has no {label!r} field:\n{text[-4000:]}")
    return float(matches[-1])


def generate_case(root: Path, name: str, smiles: str, charge: int) -> dict[str, object]:
    work = root / name
    work.mkdir()
    sdf = work / f"{name}.sdf"
    mol2 = work / f"{name}.mol2"
    frcmod = work / f"{name}.frcmod"
    embed(smiles, sdf)

    run(
        [
            str(ANTECHAMBER), "-i", sdf.name, "-fi", "sdf",
            "-o", mol2.name, "-fo", "mol2", "-c", "bcc",
            "-nc", str(charge), "-at", "gaff", "-pf", "y",
        ],
        work,
    )
    run(
        [str(PARMCHK2), "-i", mol2.name, "-f", "mol2", "-o", frcmod.name, "-s", "gaff"],
        work,
    )

    (work / "tleap.in").write_text(
        "source leaprc.gaff\n"
        f"loadamberparams {frcmod.name}\n"
        f"MOL = loadmol2 {mol2.name}\n"
        f"saveamberparm MOL {name}.prmtop {name}.inpcrd\n"
        "quit\n"
    )
    run([str(TLEAP), "-f", "tleap.in"], work)

    (work / "mdin").write_text(
        "GAFF external single point\n"
        "&cntrl\n"
        "  imin=1, maxcyc=0, ntb=0, igb=0, cut=999.0, ntpr=1,\n"
        "/\n"
    )
    run(
        [
            str(SANDER), "-O", "-i", "mdin", "-o", "mdout",
            "-p", f"{name}.prmtop", "-c", f"{name}.inpcrd",
            "-r", "restrt", "-inf", "mdinfo",
        ],
        work,
    )

    output = (work / "mdout").read_text()
    coords, charges, bonds = parse_mol2(mol2)
    regular_sum, scaled_14_sum, n14 = coulomb_pair_sum(coords, charges, bonds)
    eelec = energy_field(output, "EEL")
    eel14 = energy_field(output, "1-4 EEL")
    electrostatic = round(eelec + eel14, 4)
    pair_sum = regular_sum + scaled_14_sum
    recovered = electrostatic / pair_sum
    return {
        "name": name,
        "net_charge": charge,
        # mol2 charges carry six decimal places; mdout energies carry four.
        # Normalize the aggregate fields to those external precisions so a
        # regeneration is JSON-stable instead of exposing binary-sum noise.
        "sum_abs_charge": round(sum(abs(q) for q in charges), 6),
        "n_14_pairs": n14,
        "sander_total_energy": round(
            sum(
                energy_field(output, field)
                for field in [
                    "BOND",
                    "ANGLE",
                    "DIHED",
                    "VDWAALS",
                    "EEL",
                    "HBOND",
                    "1-4 VDW",
                    "1-4 EEL",
                    "RESTRAINT",
                ]
            ),
            4,
        ),
        "sander_electrostatic_energy": electrostatic,
        "sander_eelec": eelec,
        "sander_14_eel": eel14,
        "unscaled_coulomb_pair_sum": regular_sum + scaled_14_sum * SCEE,
        "scaled_coulomb_pair_sum": pair_sum,
        "recovered_coulomb": recovered,
    }


def main() -> None:
    missing = [str(p) for p in [ANTECHAMBER, PARMCHK2, TLEAP, SANDER] if not p.is_file()]
    if missing:
        raise SystemExit(f"AmberTools executables not found: {missing}")
    with tempfile.TemporaryDirectory(prefix="gaff-energy-oracle-") as tmp:
        records = [generate_case(Path(tmp), *case) for case in CASES]
    case_fields = (
        "name",
        "net_charge",
        "sum_abs_charge",
        "n_14_pairs",
        "sander_total_energy",
        "sander_electrostatic_energy",
        "recovered_coulomb",
    )
    cases = [{key: record[key] for key in case_fields} for record in records]
    oracle = {
        "generator": "scripts/gen_gaff_energy_oracle.py",
        "ambertools": "25",
        "rdkit": rdBase.rdkitVersion,
        "coulomb": AMBER_COULOMB,
        "cases": cases,
    }
    print(json.dumps(oracle, indent=2))


if __name__ == "__main__":
    main()
