"""Generate AM1-BCC regression oracle from AmberTools antechamber.

For each molecule we capture the full (input -> intermediate -> output) chain so
that the BCC stage can be regression-tested in ISOLATION from AM1:

  geometry + bonds/orders      <- the input molrs must be able to build
  am1_charges (equivalenced)   <- ANTECHAMBER_AM1BCC_PRE.AC  (BCC stage INPUT)
  bcc_atom_types               <- `-at bcc` run                (typifier oracle)
  bcc_charges (final)          <- output mol2                  (BCC stage OUTPUT)

molrs owns: bcc_atom_types and (am1_charges -> bcc_charges).
Atomiverse owns: producing am1_charges.  antechamber stands in for it here.
"""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import AllChem

AMBERHOME = Path("/opt/homebrew/Caskroom/miniconda/base/envs/AmberTools25")
ANTECHAMBER = AMBERHOME / "bin" / "antechamber"
SEED = 20260712

# (name, SMILES, net charge) — broad BCC chemistry + molcrafts electrolyte cases
MOLECULES = [
    ("methane",            "C",                   0),
    ("ethane",             "CC",                  0),
    ("ethene",             "C=C",                 0),
    ("acetylene",          "C#C",                 0),
    ("water",              "O",                   0),
    ("ammonia",            "N",                   0),
    ("methanol",           "CO",                  0),
    ("dimethyl_ether",     "COC",                 0),
    ("acetaldehyde",       "CC=O",                0),
    ("acetone",            "CC(C)=O",             0),
    ("acetic_acid",        "CC(=O)O",             0),
    ("acetate",            "CC(=O)[O-]",         -1),
    ("methylamine",        "CN",                  0),
    ("methylammonium",     "C[NH3+]",             1),
    ("acetonitrile",       "CC#N",                0),
    ("nitromethane",       "C[N+](=O)[O-]",       0),
    ("n_methylacetamide",  "CC(=O)NC",            0),
    ("dimethylformamide",  "CN(C)C=O",            0),
    ("benzene",            "c1ccccc1",            0),
    ("toluene",            "Cc1ccccc1",           0),
    ("phenol",             "Oc1ccccc1",           0),
    ("aniline",            "Nc1ccccc1",           0),
    ("pyridine",           "c1ccncc1",            0),
    ("imidazole",          "c1cnc[nH]1",          0),
    ("imidazolium",        "Cn1cc[nH+]c1",        1),
    ("thiophene",          "c1ccsc1",             0),
    ("methanethiol",       "CS",                  0),
    ("dimethyl_sulfoxide", "CS(C)=O",             0),
    ("chloromethane",      "CCl",                 0),
    ("fluorobenzene",      "Fc1ccccc1",           0),
    ("bromoethane",        "CCBr",                0),
    ("ethylene_carbonate", "C1COC(=O)O1",         0),
    ("dimethyl_carbonate", "COC(=O)OC",           0),
    ("dimethoxyethane",    "COCCOC",              0),
    ("methyl_methacrylate","CC(=C)C(=O)OC",       0),
    ("ethyl_acetate",      "CCOC(=O)C",           0),
    ("trimethyl_phosphate","COP(=O)(OC)OC",       0),
]


def embed(smiles: str, path: Path):
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    ps = AllChem.ETKDGv3()
    ps.randomSeed = SEED  # deterministic geometry -> reproducible oracle
    if AllChem.EmbedMolecule(mol, ps) != 0:
        raise RuntimeError(f"embed failed: {smiles}")
    AllChem.MMFFOptimizeMolecule(mol, maxIters=2000)
    Chem.MolToMolFile(mol, str(path))

    # The HONEST input a molrs user actually has: element + bond order +
    # aromatic flag + formal charge. NOT antechamber's BCC bond types --
    # molrs must PERCEIVE those (7=arom-single, 8=arom-double, 9=delocalized)
    # from this, exactly as antechamber's bondtype.c does.
    ORDER = {
        Chem.BondType.SINGLE: 1.0,
        Chem.BondType.DOUBLE: 2.0,
        Chem.BondType.TRIPLE: 3.0,
        Chem.BondType.AROMATIC: 1.5,
    }
    in_bonds = [
        [b.GetBeginAtomIdx(), b.GetEndAtomIdx(),
         ORDER[b.GetBondType()], bool(b.GetIsAromatic())]
        for b in mol.GetBonds()
    ]
    in_elements = [a.GetSymbol() for a in mol.GetAtoms()]
    formal = [a.GetFormalCharge() for a in mol.GetAtoms()]
    return in_elements, in_bonds, formal


def run_antechamber(work: Path, sdf: Path, out: Path, nc: int, at: str) -> None:
    cmd = [
        str(ANTECHAMBER),
        "-i", sdf.name, "-fi", "sdf",
        "-o", out.name, "-fo", "mol2",
        "-c", "bcc", "-nc", str(nc),
        "-at", at, "-pf", "n",
    ]
    r = subprocess.run(cmd, cwd=work, capture_output=True, text=True)
    if r.returncode != 0 or not (work / out.name).exists():
        raise RuntimeError(f"antechamber failed:\n{r.stdout}\n{r.stderr}")


def parse_ac(path: Path):
    """Parse an antechamber .AC file -> (charges, types, bonds)."""
    charges, types, bonds = [], [], []
    for line in path.read_text().splitlines():
        if line.startswith("ATOM"):
            f = line.split()
            charges.append(float(f[-2]))
            types.append(f[-1])
        elif line.startswith("BOND"):
            f = line.split()
            # BOND  idx  i  j  order  namei  namej   (1-based atom indices)
            bonds.append((int(f[2]) - 1, int(f[3]) - 1, int(f[4])))
    return charges, types, bonds


def parse_mol2(path: Path):
    """Parse a mol2 -> (elements, xyz, charges, types)."""
    els, xyz, charges, types = [], [], [], []
    section = None
    for line in path.read_text().splitlines():
        if line.startswith("@<TRIPOS>"):
            section = line.strip()
            continue
        if section == "@<TRIPOS>ATOM" and line.strip():
            f = line.split()
            xyz.append([float(f[2]), float(f[3]), float(f[4])])
            types.append(f[5])
            charges.append(float(f[8]))
            # element from the atom name (C1 -> C, Cl1 -> Cl)
            name = f[1]
            el = "".join(c for c in name if c.isalpha())
            els.append(el)
    return els, xyz, charges, types


def main() -> int:
    tmp = tempfile.TemporaryDirectory(prefix="am1bcc-oracle-")
    root = Path(tmp.name)   # antechamber scratch; never written into the repo

    records, failures = [], []
    for name, smiles, nc in MOLECULES:
        work = root / name
        work.mkdir()
        try:
            sdf = work / f"{name}.sdf"
            in_elements, in_bonds, formal = embed(smiles, sdf)

            # run 1: default (gaff) typing -> final BCC charges + PRE-BCC AM1 charges
            run_antechamber(work, sdf, work / f"{name}.mol2", nc, "gaff2")
            am1_charges, _gaff_types, bonds = parse_ac(work / "ANTECHAMBER_AM1BCC_PRE.AC")
            els, xyz, bcc_charges, _ = parse_mol2(work / f"{name}.mol2")

            # antechamber must have preserved SDF atom order, else the oracle
            # rows would not line up with the molrs-side input.
            assert els == in_elements, f"{name}: atom order drift {els} vs {in_elements}"

            # run 2: -at bcc -> the BCC atom types antechamber assigned
            run_antechamber(work, sdf, work / f"{name}_bcc.mol2", nc, "bcc")
            _, _, _, bcc_types = parse_mol2(work / f"{name}_bcc.mol2")

            # raw (non-equivalenced) sqm Mulliken, to document the -eq averaging
            raw = []
            sqm = (work / "sqm.out").read_text().splitlines()
            for i, line in enumerate(sqm):
                if "Mulliken Charge" in line:
                    for l2 in sqm[i + 1:]:
                        if "Total Mulliken" in l2:
                            break
                        p = l2.split()
                        if len(p) == 3:
                            raw.append(float(p[2]))
                    break

            n = len(els)
            assert len(am1_charges) == n and len(bcc_charges) == n and len(bcc_types) == n, name

            records.append({
                "name": name,
                "smiles": smiles,
                "net_charge": nc,
                "elements": els,
                "xyz": xyz,
                "formal_charges": formal,
                # --- INPUT molrs is given (what a real user has) ---
                "in_bonds": in_bonds,              # (i, j, order, is_aromatic)
                "am1_charges_raw": raw,            # sqm Mulliken, pre-equivalencing
                # --- ORACLE molrs must reproduce ---
                "am1_charges": am1_charges,        # after antechamber -eq equivalencing
                "bcc_bond_types": bonds,           # (i, j, bcc_bond_type) 1/2/3/7/8/9
                "bcc_atom_types": bcc_types,       # ATOMTYPE_BCC.DEF codes
                "bcc_charges": bcc_charges,        # final AM1-BCC charges
            })
            eq = "EQ" if raw and any(
                abs(a - b) > 1e-9 for a, b in zip(am1_charges, raw)
            ) else "  "
            print(f"  ok {eq} {name:22s} n={n:2d} q={sum(bcc_charges):+.4f}")
        except Exception as e:  # noqa: BLE001
            failures.append((name, str(e).splitlines()[0][:80]))
            print(f"  FAIL  {name:22s} {str(e).splitlines()[0][:60]}")

    emit_rust(records)
    if failures:
        print(f"{len(failures)} failures: {[f[0] for f in failures]}")
    return 0


def emit_rust(recs):

    L = []
    w = L.append
    w("//! AM1-BCC reference oracle — generated from AmberTools25 `antechamber -c bcc`.")
    w("//!")
    w("//! DO NOT HAND-EDIT. Regenerate with `scripts/gen_am1bcc_oracle.py`.")
    w("//!")
    w("//! Each case carries the full BCC pipeline in both directions:")
    w("//!   INPUT  — what a molrs user actually has: element, 3D coords, bond order,")
    w("//!            aromatic flag, formal charge, plus the AM1 base charges that")
    w("//!            Atomiverse (here: antechamber's sqm) supplies.")
    w("//!   ORACLE — what antechamber produced and molrs must reproduce: BCC bond")
    w("//!            types, BCC atom types, and the final AM1-BCC charges.")
    w("//!")
    w("//! `am1_charges` are antechamber's PRE-BCC charges (ANTECHAMBER_AM1BCC_PRE.AC),")
    w("//! i.e. sqm Mulliken AFTER topological-equivalence averaging (`-eq 1`, default).")
    w("//! `am1_charges_raw` is the un-averaged sqm Mulliken, kept so the equivalencing")
    w("//! stage can be tested on its own.")
    w("")
    w("#![allow(dead_code)]")
    # generated geometry: literal coordinates occasionally approximate a math
    # constant (e.g. z = 0.3180 A vs 1/pi = 0.31831). Not constants — data.
w("#![allow(clippy::approx_constant)]")
    w("")
    w("/// One antechamber reference molecule.")
    w("pub struct AntechamberCase {")
    w("    pub name: &'static str,")
    w("    pub smiles: &'static str,")
    w("    pub net_charge: i32,")
    w("    // --- input ---")
    w("    pub elements: &'static [&'static str],")
    w("    pub xyz: &'static [[f64; 3]],")
    w("    pub formal_charges: &'static [i32],")
    w("    /// (i, j, bond_order, is_aromatic)")
    w("    pub bonds: &'static [(usize, usize, f64, bool)],")
    w("    /// sqm Mulliken, before equivalence averaging")
    w("    pub am1_charges_raw: &'static [f64],")
    w("    // --- oracle ---")
    w("    /// AM1 charges after `-eq` equivalencing; the input to the BCC stage")
    w("    pub am1_charges: &'static [f64],")
    w("    /// (i, j, bcc_bond_type) — 1/2/3 = single/double/triple,")
    w("    /// 7 = aromatic-single, 8 = aromatic-double, 9 = delocalized")
    w("    pub bcc_bond_types: &'static [(usize, usize, i32)],")
    w("    /// ATOMTYPE_BCC.DEF codes")
    w("    pub bcc_atom_types: &'static [&'static str],")
    w("    /// final AM1-BCC charges")
    w("    pub bcc_charges: &'static [f64],")
    w("}")
    w("")


    def fl(xs, prec=6):
        return ", ".join(f"{x:.{prec}f}" for x in xs)


    w(f"/// {len(recs)} molecules spanning the BCC chemistry molcrafts actually uses.")
    w("pub const CASES: &[AntechamberCase] = &[")
    for r in recs:
        w("    AntechamberCase {")
        w(f'        name: "{r["name"]}",')
        w(f'        smiles: "{r["smiles"]}",')
        w(f'        net_charge: {r["net_charge"]},')
        els = ", ".join(f'"{e}"' for e in r["elements"])
        w(f"        elements: &[{els}],")
        w("        xyz: &[")
        for x, y, z in r["xyz"]:
            w(f"            [{x:.4f}, {y:.4f}, {z:.4f}],")
        w("        ],")
        w(f'        formal_charges: &[{", ".join(str(c) for c in r["formal_charges"])}],')
        w("        bonds: &[")
        for i, j, o, a in r["in_bonds"]:
            w(f"            ({i}, {j}, {o:.1f}, {str(a).lower()}),")
        w("        ],")
        w(f"        am1_charges_raw: &[{fl(r['am1_charges_raw'])}],")
        w(f"        am1_charges: &[{fl(r['am1_charges'])}],")
        w("        bcc_bond_types: &[")
        for i, j, t in r["bcc_bond_types"]:
            w(f"            ({i}, {j}, {t}),")
        w("        ],")
        ats = ", ".join(f'"{t}"' for t in r["bcc_atom_types"])
        w(f"        bcc_atom_types: &[{ats}],")
        w(f"        bcc_charges: &[{fl(r['bcc_charges'])}],")
        w("    },")
    w("];")
    w("")

    out = Path(__file__).resolve().parent.parent / "molrs/tests/ff/typifier/antechamber_oracle.rs"
    out.write_text("\n".join(L))
    print(f"wrote {out}  ({len(L)} lines, {len(recs)} cases)")

if __name__ == "__main__":
    sys.exit(main())
