//! ac-001 (chem-perceive-06) — antechamber's SECOND pass: the conjugated
//! 2-colouring, tested on molecules the 37-molecule oracle does not contain.
//!
//! `cd`, `cf`, `ch`, `nd`, `nf`, `pd`, `pf` are **not rows in `ATOMTYPE_GFF.DEF`**
//! (nor in `ATOMTYPE_GFF2.DEF`): the `.DEF` only ever emits the phase-1 name —
//! `cc`, `ce`, `cg`, `nc`, `ne`, `pc`, `pe`. antechamber renames half of every
//! conjugated system in a pass the `.DEF` does not describe: 2-colour the
//! conjugated subgraph, a **single** bond keeps the letter and a **double** bond
//! flips it, seeding each component at its phase-1 name in atom order. An atom in
//! no conjugated system keeps the phase-1 name.
//!
//! The partner names are upstream data, not an engine invention: `PARMCHK.DAT`
//! declares the pairing in its header and in a per-type `equivalent_flag` column
//! (`1 for cc/ce/cg/nc/ne/pc/pe`, `2 for cd/cf/ch/nd/nf/pd/pf`, `0 for others`),
//! which is why it becomes `AtdRule::alternate` rather than a `match` in the
//! engine. `ff/params.rs` pins that column; this file pins what the engine does
//! with it.
//!
//! **Why probes and not just the oracle.** GFF/GFF2 parity in `atd_antechamber.rs`
//! fails on 3 of 37 molecules (imidazole, imidazolium, thiophene) for want of this
//! pass, so a fix could be scored against 3 molecules that are all 5-membered
//! aromatic heterocycles. That is far too narrow a target: it says nothing about
//! the pass NOT firing (naphthalene, cyclohexene), nothing about non-aromatic
//! conjugation (benzoquinone), and nothing about a component whose two halves are
//! not symmetric (pyridone). These seven molecules pin the pass in both
//! directions, on its own, on chemistry outside the oracle — an implementation
//! that special-cases the three failing oracle molecules passes there and fails
//! here.
//!
//! ## Provenance of the expected types (DO NOT hand-edit)
//!
//! Built exactly as `scripts/gen_am1bcc_oracle.py` builds the oracle: SMILES →
//! RDKit `AddHs` → ETKDGv3 (`randomSeed = 20260712`) → MMFF → SDF; the INPUT
//! columns (element, xyz, formal charge, `(i, j, order, is_aromatic)`) are read
//! off that molecule, and the expected types are read off AmberTools25:
//!
//! ```text
//! $AMBERHOME/bin/antechamber -i <name>.sdf -fi sdf -o o.ac -fo ac \
//!                            -nc 0 -at {gaff,gaff2} -pf n     # ATOM lines
//! ```
//!
//! Note the input carries RDKit's *aromatic* bonds (order 1.5, `is_aromatic`), not
//! a Kekulé structure — that is what a molrs user has — so molrs must perceive the
//! alternation for itself, exactly as antechamber does from the SDF.

// ETKDG-embedded coordinates: a literal can land on a math constant
// (a biphenyl H at -0.7071 vs 1/sqrt(2)). These are data, not constants.
#![allow(clippy::approx_constant)]

use molrs::ff::typifier::Typifier;
use molrs::ff::typifier::atd::{AtdParameterSet, AtdTypifier};

use super::antechamber_oracle::CASES;
use super::oracle_mol::{atom_types, build_molecule};

/// One out-of-oracle probe molecule: the INPUT a molrs user has, plus what
/// `antechamber -at gaff` / `-at gaff2` answered for it.
struct ProbeCase {
    name: &'static str,
    smiles: &'static str,
    elements: &'static [&'static str],
    xyz: &'static [[f64; 3]],
    formal_charges: &'static [i32],
    /// (i, j, bond_order, is_aromatic)
    bonds: &'static [(usize, usize, f64, bool)],
    gff_atom_types: &'static [&'static str],
    gff2_atom_types: &'static [&'static str],
}

const PROBES: [ProbeCase; 7] = [
    ProbeCase {
        name: "pyrrole",
        smiles: "c1cc[nH]c1",
        elements: &["C", "C", "C", "N", "C", "H", "H", "H", "H", "H"],
        xyz: &[
            [0.3360, 1.1365, -0.1537],
            [1.1947, 0.0244, 0.0177],
            [0.3989, -1.0904, 0.1602],
            [-0.9094, -0.6897, 0.0808],
            [-0.9577, 0.6666, -0.1105],
            [0.6299, 2.1687, -0.2934],
            [2.2768, 0.0357, 0.0352],
            [0.6505, -2.1316, 0.3108],
            [-1.7137, -1.2997, 0.1523],
            [-1.9059, 1.1795, -0.1993],
        ],
        formal_charges: &[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        bonds: &[
            (0, 1, 1.5, true),
            (1, 2, 1.5, true),
            (2, 3, 1.5, true),
            (3, 4, 1.5, true),
            (4, 0, 1.5, true),
            (0, 5, 1.0, false),
            (1, 6, 1.0, false),
            (2, 7, 1.0, false),
            (3, 8, 1.0, false),
            (4, 9, 1.0, false),
        ],
        gff_atom_types: &["cc", "cc", "cd", "na", "cd", "ha", "ha", "h4", "hn", "h4"],
        gff2_atom_types: &["cc", "cc", "cd", "na", "cd", "ha", "ha", "h4", "hn", "h4"],
    },
    ProbeCase {
        name: "furan",
        smiles: "c1ccoc1",
        elements: &["C", "C", "C", "O", "C", "H", "H", "H", "H"],
        xyz: &[
            [0.6534, -0.7906, 0.0765],
            [-0.7503, -0.6984, -0.0846],
            [-1.0437, 0.6450, -0.1210],
            [0.0900, 1.3818, 0.0075],
            [1.1184, 0.5029, 0.1273],
            [1.2532, -1.6869, 0.1472],
            [-1.4612, -1.5085, -0.1645],
            [-1.9588, 1.2107, -0.2271],
            [2.0991, 0.9440, 0.2388],
        ],
        formal_charges: &[0, 0, 0, 0, 0, 0, 0, 0, 0],
        bonds: &[
            (0, 1, 1.5, true),
            (1, 2, 1.5, true),
            (2, 3, 1.5, true),
            (3, 4, 1.5, true),
            (4, 0, 1.5, true),
            (0, 5, 1.0, false),
            (1, 6, 1.0, false),
            (2, 7, 1.0, false),
            (4, 8, 1.0, false),
        ],
        gff_atom_types: &["cc", "cc", "cd", "os", "cd", "ha", "ha", "h4", "h4"],
        gff2_atom_types: &["cc", "cc", "cd", "os", "cd", "ha", "ha", "h4", "h4"],
    },
    ProbeCase {
        name: "benzoquinone",
        smiles: "O=C1C=CC(=O)C=C1",
        elements: &["O", "C", "C", "C", "C", "O", "C", "C", "H", "H", "H", "H"],
        xyz: &[
            [-1.8360, 1.9220, -0.0557],
            [-0.9912, 1.0376, -0.0301],
            [0.4505, 1.3546, -0.0004],
            [1.3734, 0.3884, 0.0276],
            [0.9912, -1.0376, 0.0301],
            [1.8360, -1.9220, 0.0557],
            [-0.4505, -1.3546, 0.0004],
            [-1.3734, -0.3884, -0.0276],
            [0.7139, 2.4054, -0.0026],
            [2.4349, 0.6039, 0.0497],
            [-0.7139, -2.4054, 0.0026],
            [-2.4349, -0.6038, -0.0496],
        ],
        formal_charges: &[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        bonds: &[
            (0, 1, 2.0, false),
            (1, 2, 1.0, false),
            (2, 3, 2.0, false),
            (3, 4, 1.0, false),
            (4, 5, 2.0, false),
            (4, 6, 1.0, false),
            (6, 7, 2.0, false),
            (7, 1, 1.0, false),
            (2, 8, 1.0, false),
            (3, 9, 1.0, false),
            (6, 10, 1.0, false),
            (7, 11, 1.0, false),
        ],
        gff_atom_types: &[
            "o", "c", "cc", "cd", "c", "o", "cc", "cd", "ha", "ha", "ha", "ha",
        ],
        gff2_atom_types: &[
            "o", "c", "cc", "cd", "c", "o", "cc", "cd", "ha", "ha", "ha", "ha",
        ],
    },
    ProbeCase {
        name: "pyridone",
        smiles: "O=c1cccc[nH]1",
        elements: &["O", "C", "C", "C", "C", "C", "N", "H", "H", "H", "H", "H"],
        xyz: &[
            [2.2869, -1.3155, 0.0321],
            [1.2199, -0.7136, 0.0170],
            [1.1708, 0.7722, 0.0298],
            [-0.0163, 1.3960, 0.0126],
            [-1.2469, 0.6395, -0.0182],
            [-1.2047, -0.6987, -0.0298],
            [0.0017, -1.3515, -0.0125],
            [2.1169, 1.2952, 0.0529],
            [-0.0774, 2.4791, 0.0214],
            [-2.1899, 1.1756, -0.0315],
            [-2.0972, -1.3149, -0.0527],
            [0.0362, -2.3633, -0.0211],
        ],
        formal_charges: &[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        bonds: &[
            (0, 1, 2.0, false),
            (1, 2, 1.0, false),
            (2, 3, 2.0, false),
            (3, 4, 1.0, false),
            (4, 5, 2.0, false),
            (5, 6, 1.0, false),
            (6, 1, 1.0, false),
            (2, 7, 1.0, false),
            (3, 8, 1.0, false),
            (4, 9, 1.0, false),
            (5, 10, 1.0, false),
            (6, 11, 1.0, false),
        ],
        gff_atom_types: &[
            "o", "c", "cc", "cd", "cd", "cc", "n", "ha", "ha", "ha", "h4", "hn",
        ],
        gff2_atom_types: &[
            "o", "c", "cc", "cd", "cd", "cc", "ns", "ha", "ha", "ha", "h4", "hn",
        ],
    },
    ProbeCase {
        name: "naphthalene",
        smiles: "c1ccc2ccccc2c1",
        elements: &[
            "C", "C", "C", "C", "C", "C", "C", "C", "C", "C", "H", "H", "H", "H", "H", "H", "H",
            "H",
        ],
        xyz: &[
            [-2.5154, 0.2715, -0.0915],
            [-2.2867, -1.0256, 0.3585],
            [-0.9795, -1.4886, 0.5177],
            [0.1162, -0.6589, 0.2286],
            [1.4375, -1.1090, 0.3833],
            [2.5154, -0.2715, 0.0914],
            [2.2867, 1.0256, -0.3585],
            [0.9795, 1.4886, -0.5177],
            [-0.1162, 0.6589, -0.2286],
            [-1.4375, 1.1090, -0.3833],
            [-3.5327, 0.6331, -0.2158],
            [-3.1251, -1.6785, 0.5860],
            [-0.8208, -2.5051, 0.8703],
            [1.6361, -2.1193, 0.7337],
            [3.5327, -0.6331, 0.2158],
            [3.1251, 1.6785, -0.5860],
            [0.8208, 2.5051, -0.8703],
            [-1.6361, 2.1193, -0.7337],
        ],
        formal_charges: &[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        bonds: &[
            (0, 1, 1.5, true),
            (1, 2, 1.5, true),
            (2, 3, 1.5, true),
            (3, 4, 1.5, true),
            (4, 5, 1.5, true),
            (5, 6, 1.5, true),
            (6, 7, 1.5, true),
            (7, 8, 1.5, true),
            (8, 9, 1.5, true),
            (9, 0, 1.5, true),
            (8, 3, 1.5, true),
            (0, 10, 1.0, false),
            (1, 11, 1.0, false),
            (2, 12, 1.0, false),
            (4, 13, 1.0, false),
            (5, 14, 1.0, false),
            (6, 15, 1.0, false),
            (7, 16, 1.0, false),
            (9, 17, 1.0, false),
        ],
        gff_atom_types: &[
            "ca", "ca", "ca", "ca", "ca", "ca", "ca", "ca", "ca", "ca", "ha", "ha", "ha", "ha",
            "ha", "ha", "ha", "ha",
        ],
        gff2_atom_types: &[
            "ca", "ca", "ca", "ca", "ca", "ca", "ca", "ca", "ca", "ca", "ha", "ha", "ha", "ha",
            "ha", "ha", "ha", "ha",
        ],
    },
    ProbeCase {
        name: "biphenyl",
        smiles: "c1ccc(-c2ccccc2)cc1",
        elements: &[
            "C", "C", "C", "C", "C", "C", "C", "C", "C", "C", "C", "C", "H", "H", "H", "H", "H",
            "H", "H", "H", "H", "H",
        ],
        xyz: &[
            [3.5483, 0.2077, -0.3112],
            [2.9143, -0.3428, 0.7996],
            [1.5196, -0.4246, 0.8401],
            [0.7346, 0.0430, -0.2271],
            [-0.7346, -0.0430, -0.1832],
            [-1.3813, -1.2650, 0.0668],
            [-2.7759, -1.3477, 0.1081],
            [-3.5483, -0.2077, -0.0991],
            [-2.9278, 1.0138, -0.3478],
            [-1.5331, 1.0944, -0.3899],
            [1.3948, 0.5952, -1.3376],
            [2.7894, 0.6767, -1.3804],
            [4.6327, 0.2712, -0.3437],
            [3.5029, -0.7071, 1.6374],
            [1.0436, -0.8478, 1.7225],
            [-0.7966, -2.1698, 0.2199],
            [-3.2565, -2.3035, 0.2994],
            [-4.6327, -0.2712, -0.0666],
            [-3.5270, 1.9064, -0.5068],
            [-1.0680, 2.0607, -0.5744],
            [0.8210, 0.9569, -2.1885],
            [3.2806, 1.1041, -2.2505],
        ],
        formal_charges: &[
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        ],
        bonds: &[
            (0, 1, 1.5, true),
            (1, 2, 1.5, true),
            (2, 3, 1.5, true),
            (3, 4, 1.0, false),
            (4, 5, 1.5, true),
            (5, 6, 1.5, true),
            (6, 7, 1.5, true),
            (7, 8, 1.5, true),
            (8, 9, 1.5, true),
            (3, 10, 1.5, true),
            (10, 11, 1.5, true),
            (11, 0, 1.5, true),
            (9, 4, 1.5, true),
            (0, 12, 1.0, false),
            (1, 13, 1.0, false),
            (2, 14, 1.0, false),
            (5, 15, 1.0, false),
            (6, 16, 1.0, false),
            (7, 17, 1.0, false),
            (8, 18, 1.0, false),
            (9, 19, 1.0, false),
            (10, 20, 1.0, false),
            (11, 21, 1.0, false),
        ],
        gff_atom_types: &[
            "ca", "ca", "ca", "cp", "cp", "ca", "ca", "ca", "ca", "ca", "ca", "ca", "ha", "ha",
            "ha", "ha", "ha", "ha", "ha", "ha", "ha", "ha",
        ],
        gff2_atom_types: &[
            "ca", "ca", "ca", "cp", "cp", "ca", "ca", "ca", "ca", "ca", "ca", "ca", "ha", "ha",
            "ha", "ha", "ha", "ha", "ha", "ha", "ha", "ha",
        ],
    },
    ProbeCase {
        name: "cyclohexene",
        smiles: "C1CCC=CC1",
        elements: &[
            "C", "C", "C", "C", "C", "C", "H", "H", "H", "H", "H", "H", "H", "H", "H", "H",
        ],
        xyz: &[
            [0.3027, -1.2907, 0.1332],
            [-0.9876, -0.7542, -0.4808],
            [-1.3674, 0.6029, 0.1172],
            [-0.1942, 1.5260, 0.2740],
            [1.0792, 1.1164, 0.1751],
            [1.4642, -0.3140, -0.0681],
            [0.5559, -2.2609, -0.3091],
            [0.1489, -1.4611, 1.2066],
            [-1.8047, -1.4676, -0.3246],
            [-0.8570, -0.6530, -1.5659],
            [-1.8271, 0.4581, 1.1022],
            [-2.1207, 1.0764, -0.5227],
            [-0.4082, 2.5732, 0.4727],
            [1.8857, 1.8382, 0.2771],
            [2.2834, -0.5904, 0.6053],
            [1.8468, -0.3993, -1.0922],
        ],
        formal_charges: &[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        bonds: &[
            (0, 1, 1.0, false),
            (1, 2, 1.0, false),
            (2, 3, 1.0, false),
            (3, 4, 2.0, false),
            (4, 5, 1.0, false),
            (5, 0, 1.0, false),
            (0, 6, 1.0, false),
            (0, 7, 1.0, false),
            (1, 8, 1.0, false),
            (1, 9, 1.0, false),
            (2, 10, 1.0, false),
            (2, 11, 1.0, false),
            (3, 12, 1.0, false),
            (4, 13, 1.0, false),
            (5, 14, 1.0, false),
            (5, 15, 1.0, false),
        ],
        gff_atom_types: &[
            "c3", "c3", "c3", "c2", "c2", "c3", "hc", "hc", "hc", "hc", "hc", "hc", "ha", "ha",
            "hc", "hc",
        ],
        gff2_atom_types: &[
            "c6", "c6", "c6", "c2", "c2", "c6", "hc", "hc", "hc", "hc", "hc", "hc", "ha", "ha",
            "hc", "hc",
        ],
    },
];

/// The names the 2-colouring pass may introduce: `AtdRule::alternate`'s range, and
/// exactly the types no `ATOMTYPE_GFF*.DEF` row declares.
const ALTERNATE_NAMES: [&str; 7] = ["cd", "cf", "ch", "nd", "nf", "pd", "pf"];

fn probe(name: &str) -> &'static ProbeCase {
    PROBES
        .iter()
        .find(|p| p.name == name)
        .unwrap_or_else(|| panic!("no probe named `{name}`"))
}

/// Type `case` with `set` and return the per-atom disagreements with antechamber.
fn disagreements(case: &ProbeCase, set: AtdParameterSet, want: &[&str]) -> Vec<String> {
    let (mol, ids) = build_molecule(case.elements, case.xyz, case.formal_charges, case.bonds);
    let typed = AtdTypifier::new(set)
        .typify(&mol)
        .unwrap_or_else(|e| panic!("{set:?} {}: {e}", case.name));
    let got = atom_types(&typed, &ids);
    want.iter()
        .enumerate()
        .filter(|(k, want_k)| got[*k].as_deref() != Some(**want_k))
        .map(|(k, want_k)| {
            format!(
                "atom {k}{}: got {:?}, antechamber says `{want_k}`",
                case.elements[k], got[k]
            )
        })
        .collect()
}

/// Both GAFF columns of one probe, atom for atom.
///
/// GFF and GFF2 are asserted together because the probe is one molecule and one
/// claim — "molrs types it as antechamber does" — and because the two tables are
/// not redundant here: cyclohexene's ring carbons are `c3` under GFF and `c6`
/// under GFF2, and pyridone's amide N is `n` under GFF and `ns` under GFF2.
fn assert_matches_antechamber(name: &str) {
    let case = probe(name);
    let mut failures: Vec<String> = Vec::new();
    for (set, want, flag) in [
        (AtdParameterSet::Gff, case.gff_atom_types, "gaff"),
        (AtdParameterSet::Gff2, case.gff2_atom_types, "gaff2"),
    ] {
        assert_eq!(
            want.len(),
            case.elements.len(),
            "{name}: the `-at {flag}` column has {} entries for {} atoms",
            want.len(),
            case.elements.len()
        );
        for line in disagreements(case, set, want) {
            failures.push(format!("  -at {flag}: {line}"));
        }
    }
    assert!(
        failures.is_empty(),
        "{name} ({}) disagrees with antechamber:\n{}",
        case.smiles,
        failures.join("\n")
    );
}

// ── the pass fires: conjugated systems are 2-coloured ───────────────────────

/// Pyrrole: `cc cc cd na cd`.
///
/// The 5-ring's four carbons are one conjugated component; walking it from atom 0,
/// the two bonds that are double in the Kekulé structure flip the letter. Nothing
/// in `ATOMTYPE_GFF.DEF` can produce the `cd`s — the file has no `cd` row — so an
/// engine that only walks the `.DEF` answers `cc cc cc na cc`, which is not what
/// antechamber says.
#[test]
fn pyrrole_alternates_cc_and_cd_around_the_conjugated_ring() {
    assert_matches_antechamber("pyrrole");
}

/// Furan: `cc cc cd os cd` — the same 2-colouring with an ether O in the ring
/// instead of an NH, so the pass cannot be keyed on the heteroatom.
#[test]
fn furan_alternates_cc_and_cd_around_the_conjugated_ring() {
    assert_matches_antechamber("furan");
}

/// Benzoquinone: `o c cc cd c o cc cd`.
///
/// Conjugation without aromaticity: the ring is a Kekulé cyclohexadienedione, RDKit
/// marks not one bond aromatic, and the two C=C pairs still come back `cc`/`cd`. A
/// pass gated on `is_aromatic` — the obvious way to make the three failing oracle
/// molecules pass — types all four carbons `cc` here.
#[test]
fn benzoquinone_is_two_coloured_although_no_bond_is_aromatic() {
    assert_matches_antechamber("benzoquinone");
}

/// 2-pyridone: `o c cc cd cd cc n` (GFF) / `... ns` (GFF2).
///
/// The asymmetric case. The conjugated component is the four ring CH carbons, and
/// its colouring is `cc cd cd cc` — NOT the strict alternation of pyrrole: the
/// 3–4 bond inside the component is single, so the letter is kept across it. A
/// pass that alternates by ring position rather than by BOND ORDER produces
/// `cc cd cc cd` and passes pyrrole, furan and all three failing oracle molecules
/// while being wrong.
#[test]
fn pyridone_keeps_the_letter_across_the_single_bond_inside_a_conjugated_component() {
    assert_matches_antechamber("pyridone");
}

// ── the pass does not fire: everything else is untouched ────────────────────

/// Naphthalene: all ten carbons are `ca`.
///
/// A pure aromatic ring system is typed by the `.DEF` (`ca`) and never reaches the
/// 2-colouring, so a fused, fully-conjugated 10-carbon π system must come back
/// uniform. This is the test an over-eager pass fails: 2-colouring *every* atom
/// with a perceived double bond would repaint half of naphthalene.
#[test]
fn naphthalene_is_all_ca_and_is_never_two_coloured() {
    assert_matches_antechamber("naphthalene");
}

/// Biphenyl: the two bridging carbons are `cp`, the other ten are `ca`.
///
/// Biphenyl is the shape most likely to be mistaken for a conjugated component
/// spanning two rings — the inter-ring bond is exactly the `cp` "biaryl" case — and
/// antechamber answers `cp cp`, not `cp cq`. So the pass must leave `cp` alone even
/// though `PARMCHK.DAT` gives `cp` a nonzero `equivalent_flag` too.
#[test]
fn biphenyl_bridges_with_two_cp_and_leaves_both_rings_ca() {
    assert_matches_antechamber("biphenyl");
}

/// Cyclohexene: `c3 c3 c3 c2 c2 c3` (GFF) / `c6 c6 c6 c2 c2 c6` (GFF2).
///
/// An isolated double bond is not a conjugated system: `c2` has no alternate at
/// all, and the sp3 ring carbons must stay `c3`/`c6`. A pass that fires on any
/// double bond would have something to say here; it must not.
#[test]
fn cyclohexene_has_an_isolated_double_bond_and_is_never_two_coloured() {
    assert_matches_antechamber("cyclohexene");
}

// ── the probes are what they claim to be ────────────────────────────────────

/// None of these molecules is in the 37-molecule oracle.
///
/// The point of the file. If a probe were also an oracle case, it would be scored
/// twice by `gff_atom_types_match_antechamber` and would add no new evidence — and
/// the 2-colouring pass would once again be tested only on molecules a targeted fix
/// could have been shaped around.
#[test]
fn every_probe_molecule_is_outside_the_thirty_seven_molecule_oracle() {
    for probe in &PROBES {
        let clash = CASES
            .iter()
            .find(|case| case.name == probe.name || case.smiles == probe.smiles);
        assert!(
            clash.is_none(),
            "probe `{}` ({}) is already oracle case `{}` — the probes exist to test the \
             2-colouring on chemistry the oracle does not reach",
            probe.name,
            probe.smiles,
            clash.map(|c| c.name).unwrap_or_default()
        );
    }
}

/// Non-vacuity: the probes pin BOTH answers the pass can give.
///
/// Four of them expect a name that no `ATOMTYPE_GFF*.DEF` row declares (`cd`), and
/// three expect none at all. Drop the first group and the pass is never exercised;
/// drop the second and nothing stops it from firing everywhere. A future edit that
/// quietly empties either half leaves the tests above green and meaningless.
#[test]
fn the_probes_pin_the_pass_both_firing_and_not_firing() {
    let (fires, quiet): (Vec<&ProbeCase>, Vec<&ProbeCase>) = PROBES.iter().partition(|probe| {
        probe
            .gff_atom_types
            .iter()
            .chain(probe.gff2_atom_types)
            .any(|t| ALTERNATE_NAMES.contains(t))
    });
    assert!(
        fires.len() >= 4 && quiet.len() >= 3,
        "the probe set must exercise the 2-colouring in both directions: {} molecules \
         expect an alternate name and {} expect none",
        fires.len(),
        quiet.len()
    );
}
