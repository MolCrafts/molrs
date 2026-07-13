"""Generate AM1-BCC regression oracle from AmberTools antechamber.

For each molecule we capture the full (input -> intermediate -> output) chain so
that the BCC stage can be regression-tested in ISOLATION from AM1:

  geometry + bonds/orders      <- the input molrs must be able to build
  am1_charges (equivalenced)   <- ANTECHAMBER_AM1BCC_PRE.AC  (BCC stage INPUT)
  <table>_atom_types           <- one `-at <mode>` run each    (typifier oracle)
  bcc_charges (final)          <- `-c bcc` mol2                (BCC stage OUTPUT)
  abcg2_charges (final)        <- `-c abcg2` mol2              (ABCG2 stage OUTPUT)
  gas_charges (final)          <- `-c gas` mol2                (Gasteiger, NO QM)

molrs owns: the atom types and (am1_charges -> bcc_charges / abcg2_charges), and
the whole of gas_charges.  Atomiverse owns: producing am1_charges.  antechamber
stands in for it here.

The three CHARGE columns are the second axis, orthogonal to the seven TYPE
columns.  `bcc` and `abcg2` are the only two BCC correction families that exist
(BCCPARM.DAT / BCCPARM_ABCG2.DAT), they consume the SAME AM1 base charges (the
script asserts it per molecule), and they must come out of ONE `BccModel` with
nothing but the parameter set changed.  A model that special-cased either family
regresses the other column.

`gas_charges` is the corner of that axis with NO QM input at all: Gasteiger/PEOE
(GASPARM.DAT + ATOMTYPE_GAS.DEF) iterates on the topology alone, so antechamber
never calls sqm for it.  It is what proves the `ChargeModel` trait has not
quietly assumed "QM base charges plus a correction" -- the same trait, the same
`assign`, and this column comes out of an argument-free molecule.  Being purely
topological it is also inherently symmetric (methanol's three methyl H come out
IDENTICAL: 0.052691 x3), which is why it runs with `needs_equivalencing = false`
and yet never shows the conformer artefacts `-eq 1` exists to remove.

The seven type columns (see AT_MODES) are the SAME ATD/WILDATOM rule engine
driven by seven different `ATOMTYPE_*.DEF` tables, so they are what pins
"one engine, N tables": a table-specific hack shows up as a regression in the
other six columns.  Note GAS has an atom-type table but no BCC *correction*
table, which is why it can only be reached through the table-generic typifier,
never through the BCC-correction-family selector.
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

# (antechamber `-at` flag, oracle column, the ATOMTYPE_*.DEF the flag walks).
#
# The flag and the table are NOT spelled the same: `-at gaff` walks
# ATOMTYPE_GFF.DEF and `-at gaff2` walks ATOMTYPE_GFF2.DEF (antechamber's own
# help lists the flags as "gaff, gaff2, amber, bcc, abcg2, and sybyl"; `gas` is
# undocumented there but real).  The column is named after the TABLE, because
# that is the axis molrs's `AtdParameterSet` names -- calling the column `gaff`
# would make the Rust side look like it selects a *flag*.
AT_MODES = [
    ("bcc",   "bcc_atom_types",   "ATOMTYPE_BCC.DEF"),
    ("abcg2", "abcg2_atom_types", "ATOMTYPE_ABCG2.DEF"),
    ("gas",   "gas_atom_types",   "ATOMTYPE_GAS.DEF"),
    ("gaff",  "gff_atom_types",   "ATOMTYPE_GFF.DEF"),
    ("gaff2", "gff2_atom_types",  "ATOMTYPE_GFF2.DEF"),
    ("amber", "amber_atom_types", "ATOMTYPE_AMBER.DEF"),
    ("sybyl", "sybyl_atom_types", "ATOMTYPE_SYBYL.DEF"),
]

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


def run_antechamber(work: Path, sdf: Path, out: Path, nc: int, at: str,
                    charge: str = "bcc") -> None:
    """One antechamber run.

    `charge` selects the CHARGE model (`-c bcc` / `-c abcg2`), `at` the ATOM-TYPE
    table (`-at ...`).  They are independent: `-c abcg2` applies the ABCG2
    corrections through ABCG2's own internal typing whatever `-at` says, so the
    output mol2's type column is the `-at` table and its charge column is the `-c`
    model.  `-pf n` is mandatory throughout: `-pf y` deletes the intermediate
    ANTECHAMBER_AM1BCC_PRE.AC this oracle reads its AM1 base charges from.
    """
    cmd = [
        str(ANTECHAMBER),
        "-i", sdf.name, "-fi", "sdf",
        "-o", out.name, "-fo", "mol2",
        "-c", charge, "-nc", str(nc),
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

            # runs 2-8: one run per atom-type table. Same molecule, same engine
            # upstream -- only `-at` changes -- so the seven columns are directly
            # comparable and a table-specific special case cannot hide.
            # `-pf n` is mandatory: `-pf y` deletes ANTECHAMBER_AM1BCC_PRE.AC.
            atom_types = {}
            for at, col, _def in AT_MODES:
                out_mol2 = work / f"{name}_{at}.mol2"
                run_antechamber(work, sdf, out_mol2, nc, at)
                _, _, _, atom_types[col] = parse_mol2(out_mol2)

            # run 9: the OTHER correction family. Same molecule, same AM1 base
            # charges -- only BCCPARM.DAT -> BCCPARM_ABCG2.DAT changes. It gets its
            # own directory because it rewrites ANTECHAMBER_AM1BCC_PRE.AC, which we
            # read back to prove the two families really do share one AM1 column.
            abcg2_work = work / "abcg2"
            abcg2_work.mkdir()
            abcg2_sdf = abcg2_work / sdf.name
            abcg2_sdf.write_text(sdf.read_text())
            run_antechamber(abcg2_work, abcg2_sdf, abcg2_work / f"{name}.mol2",
                            nc, "gaff2", charge="abcg2")
            abcg2_am1, _, _ = parse_ac(abcg2_work / "ANTECHAMBER_AM1BCC_PRE.AC")
            _, _, abcg2_charges, _ = parse_mol2(abcg2_work / f"{name}.mol2")

            # run 10: the ZERO-QM corner. `-c gas` runs no sqm at all -- it
            # iterates GASPARM.DAT electronegativities over the bond graph -- so
            # it shares no column with the two above. Own directory for the same
            # reason abcg2 has one: antechamber's intermediates are named after
            # the run, not the molecule, and a second run in the same cwd would
            # overwrite the first one's.
            gas_work = work / "gas"
            gas_work.mkdir()
            gas_sdf = gas_work / sdf.name
            gas_sdf.write_text(sdf.read_text())
            run_antechamber(gas_work, gas_sdf, gas_work / f"{name}.mol2",
                            nc, "gaff2", charge="gas")
            _, _, gas_charges, _ = parse_mol2(gas_work / f"{name}.mol2")

            # Gasteiger is a TOPOLOGY-only model, so it cannot produce the
            # conformer artefact `-eq 1` exists to remove: equivalent atoms come
            # out bit-identical without any averaging. If this ever fired, the
            # `needs_equivalencing = false` half of the 2x2 would be wrong.
            assert not (gas_work / "sqm.out").exists(), (
                f"{name}: `-c gas` called sqm -- it is supposed to need no QM input"
            )

            # ONE am1_charges column, honestly shared. If `-c abcg2` ever fed its
            # correction stage a different base (a different sqm run, a different
            # `-eq` level), the fixture would be quietly lying to the ABCG2 test and
            # the Rust side would be chasing a mismatch that is not its own.
            assert abcg2_am1 == am1_charges, (
                f"{name}: `-c abcg2` consumed different AM1 base charges than "
                f"`-c bcc`; the shared `am1_charges` column is no longer valid"
            )

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
            assert len(am1_charges) == n and len(bcc_charges) == n, name
            assert len(abcg2_charges) == n, name
            assert len(gas_charges) == n, name
            for col, types in atom_types.items():
                assert len(types) == n, f"{name}: {col} typed {len(types)}/{n} atoms"

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
                # one column per ATOMTYPE_*.DEF table (see AT_MODES)
                **atom_types,
                "bcc_charges": bcc_charges,        # final AM1-BCC charges (-c bcc)
                "abcg2_charges": abcg2_charges,    # final ABCG2 charges  (-c abcg2)
                "gas_charges": gas_charges,        # Gasteiger/PEOE       (-c gas)
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
    w("//!            types, the seven atom-type columns, and the final charges of")
    w("//!            all three charge models (`-c bcc`, `-c abcg2` and `-c gas`).")
    w("//!")
    w("//! The three charge columns are one axis (which charge model), the seven type")
    w("//! columns another (which ATOMTYPE table). `bcc_charges` and `abcg2_charges`")
    w("//! are corrections of the SAME `am1_charges` — the generator asserts that the")
    w("//! two antechamber runs consumed identical AM1 base charges — so a charge model")
    w("//! that special-cased one family regresses the other column. `gas_charges` is")
    w("//! the zero-QM corner: `-c gas` runs no sqm at all (the generator asserts no")
    w("//! `sqm.out` is written), so it shares no input with the other two and a trait")
    w("//! that had assumed QM base charges could not reach it.")
    w("//!")
    w("//! The seven atom-type columns come from `-at {bcc,abcg2,gas,gaff,gaff2,amber,")
    w("//! sybyl}` on the SAME molecule: one rule engine, seven `ATOMTYPE_*.DEF` tables.")
    w("//! They are what pins the engine as table-generic — a per-table special case")
    w("//! regresses the others. The `-at` flag and the table are spelled differently:")
    w("//! `-at gaff` walks `ATOMTYPE_GFF.DEF`, `-at gaff2` walks `ATOMTYPE_GFF2.DEF`,")
    w("//! and the columns are named after the TABLE, as `AtdParameterSet` is.")
    w("//!")
    w("//! `am1_charges` are antechamber's PRE-BCC charges (ANTECHAMBER_AM1BCC_PRE.AC),")
    w("//! i.e. sqm Mulliken AFTER topological-equivalence averaging (`-eq 1`, default).")
    w("//! `am1_charges_raw` is the un-averaged sqm Mulliken, kept so the equivalencing")
    w("//! stage can be tested on its own.")
    w("")
    w("#![allow(dead_code)]")
    w("// Generated geometry: a literal coordinate can approximate a math constant")
    w("// (e.g. z = 0.3180 A vs 1/pi = 0.31831). These are data, not constants.")
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
    w("    /// ATOMTYPE_BCC.DEF codes (`antechamber -at bcc`)")
    w("    pub bcc_atom_types: &'static [&'static str],")
    w("    /// ATOMTYPE_ABCG2.DEF codes (`antechamber -at abcg2`)")
    w("    pub abcg2_atom_types: &'static [&'static str],")
    w("    /// ATOMTYPE_GAS.DEF codes (`antechamber -at gas`) — the Gasteiger table.")
    w("    /// GAS has no BCC correction table, so it is reachable only through the")
    w("    /// table-generic typifier.")
    w("    pub gas_atom_types: &'static [&'static str],")
    w("    /// ATOMTYPE_GFF.DEF codes (`antechamber -at gaff`) — GAFF atom types")
    w("    pub gff_atom_types: &'static [&'static str],")
    w("    /// ATOMTYPE_GFF2.DEF codes (`antechamber -at gaff2`) — GAFF2 atom types")
    w("    pub gff2_atom_types: &'static [&'static str],")
    w("    /// ATOMTYPE_AMBER.DEF codes (`antechamber -at amber`) — AMBER atom types")
    w("    pub amber_atom_types: &'static [&'static str],")
    w("    /// ATOMTYPE_SYBYL.DEF codes (`antechamber -at sybyl`) — SYBYL atom types")
    w("    pub sybyl_atom_types: &'static [&'static str],")
    w("    /// final AM1-BCC charges (`antechamber -c bcc`): `am1_charges` + BCCPARM.DAT")
    w("    pub bcc_charges: &'static [f64],")
    w("    /// final ABCG2 charges (`antechamber -c abcg2`): the SAME `am1_charges`,")
    w("    /// corrected with BCCPARM_ABCG2.DAT against ATOMTYPE_ABCG2.DEF types.")
    w("    ///")
    w("    /// The second correction family is the generality proof of the charge model:")
    w("    /// one engine, two parameter sets, no special case. That both families")
    w("    /// consume the same `am1_charges` is asserted by the generator, not assumed.")
    w("    pub abcg2_charges: &'static [f64],")
    w("    /// Gasteiger/PEOE charges (`antechamber -c gas`): GASPARM.DAT iterated over")
    w("    /// the bond graph, with `gas_atom_types` as the key.")
    w("    ///")
    w("    /// The ZERO-QM corner of the charge model: antechamber runs no sqm for this")
    w("    /// column (the generator asserts no `sqm.out` was written), so it shares")
    w("    /// NOTHING with `am1_charges` -- not a base charge, not a correction. A")
    w("    /// `ChargeModel` that could only be reached with QM charges in hand cannot")
    w("    /// produce it at all.")
    w("    ///")
    w("    /// Being purely topological it is also inherently symmetric: methanol's three")
    w("    /// methyl H are IDENTICAL here (0.052691 x3) where sqm splits them")
    w("    /// 0.053/0.098/0.053. That is why the model runs `needs_equivalencing = false`")
    w("    /// and still never carries a conformer artefact.")
    w("    pub gas_charges: &'static [f64],")
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
        for _at, col, _def in AT_MODES:
            ats = ", ".join(f'"{t}"' for t in r[col])
            w(f"        {col}: &[{ats}],")
        w(f"        bcc_charges: &[{fl(r['bcc_charges'])}],")
        w(f"        abcg2_charges: &[{fl(r['abcg2_charges'])}],")
        w(f"        gas_charges: &[{fl(r['gas_charges'])}],")
        w("    },")
    w("];")
    w("")

    out = Path(__file__).resolve().parent.parent / "molrs/tests/ff/typifier/antechamber_oracle.rs"
    out.write_text("\n".join(L))
    # rustfmt in place: without this the emitted file drifts from the committed one
    # the moment anyone runs `cargo fmt --all`, and the fixture stops round-tripping.
    subprocess.run(["rustfmt", "--edition", "2024", str(out)], check=True)
    print(f"wrote {out}  ({len(L)} lines, {len(recs)} cases)")

if __name__ == "__main__":
    sys.exit(main())
