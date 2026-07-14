//! chem-perceive-15 — **the chain, end to end, against oracles molrs did not produce.**
//!
//! ```text
//! SMILES / SDF -> Perceive -> AtdTypifier -> ChargeModel -> ForceField -> Potentials -> E + F
//! ```
//!
//! Sixteen specs verified the stages. Each stage is green. **Nothing had ever run the
//! chain**, and the two worst defects on it lived precisely in that gap:
//!
//! * The generic MMFF path had **no electrostatic term at all** — 150 kcal/mol on
//!   caffeine — and every stage test passed, because the one fixture the energy test
//!   asserted (`e_ethane`) was one of exactly two whose MMFF charges are all zero.
//! * BCC bond-type perception and charge equivalencing were **missing entirely** —
//!   two whole algorithmic stages — and the suite was green, because the tests
//!   asserted what the code computed.
//!
//! So this target does two things a stage test structurally cannot:
//!
//! 1. **It carries ONE molecule through.** The graph that reaches `to_potentials` is
//!    the graph the typifier produced from the user's input, not a fresh one built
//!    from the oracle's answers. A stage that silently drops a column is invisible to
//!    the stages on either side of it; it is not invisible to the thing that walks the
//!    whole way.
//! 2. **It asserts against EXTERNAL truth** — antechamber (AmberTools25, 37
//!    molecules) and RDKit MMFF (11). Numbers molrs did not produce, and so cannot
//!    quietly agree with itself about.
//!
//! # The fixture lists are SCANNED, and the partitions are COMPUTED
//!
//! Not one molecule in this file is selected by hand. The MMFF fixtures come from a
//! directory scan; the antechamber molecules are the whole oracle; and every
//! *partition* below (zero-charge, delocalized-N) is a **predicate evaluated on the
//! molecule**, never a list of names. That is not stylistic. A hand-written list is a
//! list that can be shortened by hand — and the one that mattered had been shortened
//! to one.

use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::path::{Path, PathBuf};

use molrs::Atomistic;
use molrs::ff::charge::{BccModel, BccParameterSet, ChargeError, ChargeModel, GasteigerModel};
use molrs::ff::forcefield::ForceField;
use molrs::ff::forcefield::gaff::{GaffParameterSet, gaff_forcefield};
use molrs::ff::potential::intramolecular_pairs;
use molrs::ff::typifier::Typifier;
use molrs::ff::typifier::atd::{AtdParameterSet, AtdTypifier};
use molrs::ff::typifier::mmff::{MMFF94STypifier, MMFF94Typifier};
use molrs::store::frame::Frame;
use molrs::store::keys;
use molrs::system::molgraph::{Atom, PropValue};

#[path = "ff/typifier/antechamber_oracle.rs"]
mod antechamber_oracle;

/// The ONE oracle -> `Atomistic` builder, shared with the `ff` target rather than
/// copied. That sharing is a guard, not DRY: a second builder is a second chance to
/// hand the typifier antechamber's own *answers* (its perceived bond types) instead of
/// a user's *input*, and the copy would be the one nobody re-reads.
///
/// `allow(dead_code)` because this target uses two of its four helpers; the `ff` target
/// uses the others. The alternative — annotating the shared file itself — would blind
/// BOTH targets to a genuinely dead helper.
#[path = "ff/typifier/oracle_mol.rs"]
#[allow(dead_code)]
mod oracle_mol;

use antechamber_oracle::{AntechamberCase, CASES};
use oracle_mol::{build_case, build_molecule};

/// Charge parity with antechamber — the tolerance the Rust suite already uses
/// (`am1bcc_antechamber.rs`), restated rather than loosened. Loosening a tolerance to
/// make an acceptance green is a failure, not a pass.
const CHARGE_TOL: f64 = 1.0e-4;

/// Total-energy parity with RDKit MMFF, kcal/mol (`ff/mmff/energy.rs`).
const ENERGY_TOL: f64 = 1.0e-3;

// ===========================================================================
// The chain
// ===========================================================================

/// One molecule, carried from the user's input all the way to forces.
///
/// Every field is produced from the one above it. Nothing is rebuilt from the oracle:
/// `build_case` takes only the INPUT half (element, xyz, bond order, aromatic flag,
/// formal charge), which is what a molrs user actually has after reading an SDF.
struct Chain {
    name: &'static str,
    /// Stages 2-3: perception + atom typing, plus the charges, on ONE graph.
    typed: Atomistic,
    types: Vec<String>,
    /// Stage 4: the charge model, fed the raw (un-equivalenced) AM1 charges.
    charges: Vec<f64>,
    /// Stage 5: the force field, or why it could not be built.
    ff: Result<ForceField, String>,
    coords: Vec<f64>,
}

impl Chain {
    /// Run the documented chain on one oracle molecule.
    fn run(case: &AntechamberCase) -> Self {
        let (mol, ids) = build_case(case);

        // --- Perceive + AtdTypifier (the GAFF table) --------------------------
        // `AtdTypifier::typify` runs `Perceive::find_bond_types` itself; the graph it
        // returns carries the perceived facts AND the types. That single graph is what
        // every stage below consumes — which is the whole point of an end-to-end test.
        let typed = AtdTypifier::new(AtdParameterSet::Gff)
            .typify(&mol)
            .unwrap_or_else(|e| panic!("{}: ATD typing failed: {e}", case.name));
        let types: Vec<String> = ids
            .iter()
            .map(|aid| {
                typed
                    .get_atom(*aid)
                    .expect("the typed graph keeps the input atom ids")
                    .get_str(keys::TYPE)
                    .unwrap_or("<none>")
                    .to_owned()
            })
            .collect();

        // --- ChargeModel -----------------------------------------------------
        // Raw sqm Mulliken in; equivalencing and the BCC correction are the model's job.
        let charges = BccModel::new(BccParameterSet::Bcc)
            .expect("the compiled BCC table builds")
            .assign(&mol, Some(case.am1_charges_raw))
            .unwrap_or_else(|e| panic!("{}: BCC charge assignment failed: {e}", case.name));

        // --- ForceField ------------------------------------------------------
        // The charges ride on the molecule the force field is built from. A chain that
        // dropped them here would still produce an energy — just the wrong one.
        let mut charged = typed.clone();
        for (aid, q) in ids.iter().zip(&charges) {
            charged
                .set_atom(*aid, keys::CHARGE, *q)
                .expect("set charge");
        }

        // `gaff_forcefield` returns BOTH halves, and both are load-bearing: the force
        // field, and the molecule RE-LABELLED with the force field's own bond / angle /
        // dihedral type NAMES. Keeping only the force field and framing the input graph
        // gives a bonds block whose `type` column still holds the perceived integers —
        // and `to_potentials` refuses it, loudly, which is the whole point of the
        // `keys::TYPE` rule this chain fought over.
        let (labelled, ff) = match gaff_forcefield(GaffParameterSet::Gaff, &charged) {
            Ok((labelled, ff)) => (labelled, Ok(ff)),
            Err(e) => (charged, Err(e.to_string())),
        };

        let coords = case.xyz.iter().flat_map(|p| p.iter().copied()).collect();

        Self {
            name: case.name,
            typed: labelled,
            types,
            charges,
            ff,
            coords,
        }
    }

    /// Stages 6-7: `to_frame` -> neighbour list -> `to_potentials` -> E + F.
    fn energy_forces(&self) -> Option<(f64, Vec<f64>)> {
        let ff = self.ff.as_ref().ok()?;
        let mut frame = self.typed.to_frame();
        frame.insert("pairs", intramolecular_pairs(&frame));
        let pots = ff
            .to_potentials(&frame)
            .unwrap_or_else(|e| panic!("{}: to_potentials failed: {e}", self.name));
        Some(pots.calc_energy_forces(&self.coords))
    }
}

// ===========================================================================
// The chain runs, on EVERY molecule in the oracle
// ===========================================================================

/// The whole chain, on all 37 antechamber molecules, asserted at every stage.
///
/// One `Chain` per molecule, built once and interrogated at each stage, so a stage
/// that quietly drops what the one before it produced has nowhere to hide. Every
/// assertion is against antechamber — never against what molrs computed.
#[test]
fn the_chain_runs_on_every_antechamber_molecule() {
    assert_eq!(
        CASES.len(),
        37,
        "the oracle is 37 molecules; an oracle that shrank is an oracle that stopped testing"
    );

    let mut type_fails = Vec::new();
    let mut charge_fails = Vec::new();
    let mut energy_fails = Vec::new();
    let mut no_forcefield = Vec::new();

    for case in CASES {
        let chain = Chain::run(case);

        // Stage 3 — atom types vs `antechamber -at gaff`.
        for (i, (got, want)) in chain.types.iter().zip(case.gff_atom_types).enumerate() {
            if got != want {
                type_fails.push(format!(
                    "  {}: atom {i}: molrs `{got}`, gaff `{want}`",
                    case.name
                ));
            }
        }

        // Stage 4 — charges vs `antechamber -c bcc`.
        for (i, (got, want)) in chain.charges.iter().zip(case.bcc_charges).enumerate() {
            if (got - want).abs() > CHARGE_TOL {
                charge_fails.push(format!(
                    "  {}: atom {i}: molrs {got:.6}, antechamber {want:.6}",
                    case.name
                ));
            }
        }

        // Stages 5-7 — force field, potentials, energy, forces.
        match chain.energy_forces() {
            None => no_forcefield.push(format!(
                "  {}: {}",
                case.name,
                chain.ff.as_ref().unwrap_err()
            )),
            Some((e, f)) => {
                if !e.is_finite() {
                    energy_fails.push(format!("  {}: energy is {e}", case.name));
                }
                if f.iter().any(|x| !x.is_finite()) {
                    energy_fails.push(format!("  {}: a force component is not finite", case.name));
                }
                // An isolated molecule cannot push on itself. A term with a sign error
                // passes every energy check ever written and fails this one.
                for axis in 0..3 {
                    let net: f64 = f.iter().skip(axis).step_by(3).sum();
                    if net.abs() > 1e-8 {
                        energy_fails.push(format!(
                            "  {}: net force on axis {axis} is {net:.3e}, not 0",
                            case.name
                        ));
                    }
                }
            }
        }
    }

    let mut report = String::new();
    if !type_fails.is_empty() {
        let _ = writeln!(
            report,
            "ATOM TYPES vs antechamber:\n{}",
            type_fails.join("\n")
        );
    }
    if !charge_fails.is_empty() {
        let _ = writeln!(
            report,
            "CHARGES vs antechamber:\n{}",
            charge_fails.join("\n")
        );
    }
    if !energy_fails.is_empty() {
        let _ = writeln!(report, "ENERGY / FORCES:\n{}", energy_fails.join("\n"));
    }
    assert!(report.is_empty(), "the chain broke:\n{report}");

    // A molecule whose GAFF terms are not all in the table cannot reach an energy
    // through the public door: `gaff_forcefield` is exact-match-only by design, and the
    // estimator that would fill the gaps is not composed into it. That is a FACT about
    // how far the chain closes, and it is printed rather than silently skipped.
    println!(
        "chain closed on {}/{}; {} could not build a GAFF force field:\n{}",
        CASES.len() - no_forcefield.len(),
        CASES.len(),
        no_forcefield.len(),
        no_forcefield.join("\n")
    );
}

/// **A force field that does not declare its electrostatics silently omits them.**
///
/// This is the caffeine defect, asked of the chain rather than of MMFF. The generic
/// MMFF path produced an energy wrong by 150 kcal/mol because no `ForceField` ever
/// defined an electrostatic style — and every test that could have noticed was looking
/// at a molecule whose charges were all zero.
///
/// So: run the chain on molecules with real, non-zero, antechamber-blessed charges and
/// ask whether the electrostatic term EXISTS. Not whether it is right — whether it is
/// there. `CLAUDE.md` puts the rule in the imperative: *"A force field must **declare**
/// the constants it means: a style that omits `coulomb` / `dielectric` /
/// `coulomb14scale` is an `Err`, never a silent default."*
#[test]
fn the_force_field_the_chain_builds_declares_its_electrostatics() {
    let mut mute = Vec::new();

    for case in CASES {
        let chain = Chain::run(case);
        let Ok(ff) = chain.ff.as_ref() else { continue };

        // Charges present — the input class that can expose the hole, and the one
        // `e_ethane` structurally could not.
        let q: f64 = chain.charges.iter().map(|q| q.abs()).sum();
        if q < 1e-12 {
            continue;
        }

        // Does ANY style consume charge? (`coul/cut`, `coul/long`, `pme`, `ewald`, …)
        let declares = ff.styles().iter().any(|s| {
            let n = s.name.as_str();
            (s.category() == "pair" || s.category() == "kspace")
                && (n.contains("coul") || n.contains("pme") || n.contains("ewald"))
        });
        if !declares {
            let styles: Vec<String> = ff
                .styles()
                .iter()
                .map(|s| format!("{}/{}", s.category(), s.name))
                .collect();
            mute.push(format!(
                "  {} (sum|q| = {q:.4} e, net {}): {styles:?}",
                case.name, case.net_charge
            ));
        }
    }

    assert!(
        mute.is_empty(),
        "the chain builds force fields that carry CHARGES but declare NO ELECTROSTATIC \
         STYLE, so `to_potentials` returns an energy with no Coulomb term in it — \
         silently:\n{}\n\n\
         This is the caffeine hole, in the other force field. And the tell is already in \
         the tree: `special_bonds` carries `coul: [0, 0, 1/1.2]` — AMBER's SCEE, a 1-4 \
         Coulomb scale factor declared for a Coulomb term that does not exist. A constant \
         nothing consumes is the same smell as 4,065 XML rows nothing read.",
        mute.join("\n")
    );
}

// ===========================================================================
// The MMFF half of the chain — the RDKit oracle, scanned
// ===========================================================================

fn mmff_fixtures_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/ff/mmff/fixtures")
}

/// Every MMFF energy fixture on disk. **Scanned**, never listed.
fn mmff_fixtures() -> Vec<String> {
    let mut names: Vec<String> = std::fs::read_dir(mmff_fixtures_dir())
        .expect("read mmff fixtures dir")
        .map(|e| e.expect("dir entry").file_name())
        .filter_map(|f| {
            f.to_string_lossy()
                .strip_suffix(".energy.json")
                .map(str::to_owned)
        })
        .collect();
    names.sort();
    assert_eq!(
        names.len(),
        11,
        "the RDKit MMFF oracle is 11 molecules; the scan found {}. A scan that finds \
         nothing makes every assertion below vacuously green.",
        names.len()
    );
    names
}

/// V2000 SDF -> `Atomistic` + flat coords, preserving atom order, bond order, `M  CHG`.
fn load_sdf(name: &str) -> (Atomistic, Vec<f64>) {
    let path = mmff_fixtures_dir().join(format!("{name}.sdf"));
    let text = std::fs::read_to_string(&path).expect("read sdf");
    let lines: Vec<&str> = text.lines().collect();
    let counts = lines[3];
    let n_atoms: usize = counts[0..3].trim().parse().expect("n_atoms");
    let n_bonds: usize = counts[3..6].trim().parse().expect("n_bonds");

    let mut g = Atomistic::new();
    let mut ids = Vec::with_capacity(n_atoms);
    let mut coords = Vec::with_capacity(n_atoms * 3);
    for k in 0..n_atoms {
        let line = lines[4 + k];
        let x: f64 = line[0..10].trim().parse().expect("x");
        let y: f64 = line[10..20].trim().parse().expect("y");
        let z: f64 = line[20..30].trim().parse().expect("z");
        coords.extend_from_slice(&[x, y, z]);
        let element = line[31..34].trim().to_string();
        let mut atom = Atom::xyz(&element, x, y, z);
        atom.set("formal_charge", PropValue::Int(0));
        ids.push(g.add_atom(atom));
    }
    for k in 0..n_bonds {
        let line = lines[4 + n_atoms + k];
        let i: usize = line[0..3].trim().parse::<usize>().expect("bond i") - 1;
        let j: usize = line[3..6].trim().parse::<usize>().expect("bond j") - 1;
        let order: f64 = line[6..9].trim().parse().expect("bond order");
        let bid = g.add_bond(ids[i], ids[j]).expect("add bond");
        g.set_bond_prop(bid, keys::ORDER, PropValue::F64(order))
            .expect("bond");
    }
    for line in &lines[4 + n_atoms + n_bonds..] {
        let t = line.trim_end();
        if t == "M  END" || t == "$$$$" {
            break;
        }
        if let Some(rest) = t.strip_prefix("M  CHG") {
            let toks: Vec<&str> = rest.split_whitespace().collect();
            for pair in toks[1..].chunks(2) {
                if let [idx, chg] = pair {
                    let i: usize = idx.parse::<usize>().expect("chg idx") - 1;
                    let c: i32 = chg.parse().expect("chg");
                    g.set_atom(ids[i], "formal_charge", PropValue::Int(c))
                        .expect("set formal_charge");
                }
            }
        }
    }
    (g, coords)
}

/// The MMFF chain through the documented public door: typify -> Frame -> pairs.
fn mmff_frame<T: Typifier<Mol = Atomistic>>(typifier: &T, name: &str) -> (Frame, Vec<f64>) {
    let (mol, coords) = load_sdf(name);
    let typed = typifier
        .typify(&mol)
        .unwrap_or_else(|e| panic!("{name}: MMFF typify failed: {e}"));
    let mut frame = typed.to_frame();
    frame.insert("pairs", intramolecular_pairs(&frame));
    (frame, coords)
}

/// MMFF94 energy + forces + the typed frame, through the public chain.
fn mmff94_energy_forces(name: &str) -> (f64, Vec<f64>, Frame) {
    let t = MMFF94Typifier::new();
    let (frame, coords) = mmff_frame(&t, name);
    let pots = t
        .ff()
        .to_potentials(&frame)
        .unwrap_or_else(|e| panic!("{name}: to_potentials failed: {e}"));
    let (e, f) = pots.calc_energy_forces(&coords);
    (e, f, frame)
}

/// MMFF94s energy + forces, through the public chain.
fn mmff94s_energy_forces(name: &str) -> (f64, Vec<f64>) {
    let t = MMFF94STypifier::new();
    let (frame, coords) = mmff_frame(&t, name);
    let pots = t
        .ff()
        .to_potentials(&frame)
        .unwrap_or_else(|e| panic!("{name}: to_potentials failed: {e}"));
    pots.calc_energy_forces(&coords)
}

fn rdkit_energy(name: &str, field: &str) -> Option<f64> {
    let path = mmff_fixtures_dir().join(format!("{name}.energy.json"));
    let text = std::fs::read_to_string(path).expect("read energy json");
    let v: serde_json::Value = serde_json::from_str(&text).expect("parse energy json");
    v[field].as_f64()
}

/// The MMFF chain, on EVERY fixture, against RDKit's total energy.
#[test]
fn the_mmff_chain_matches_rdkit_on_every_fixture() {
    let mut fails = Vec::new();
    for name in mmff_fixtures() {
        let (got, _, _) = mmff94_energy_forces(&name);
        let want = rdkit_energy(&name, "mmff94_total_energy")
            .unwrap_or_else(|| panic!("{name}: the oracle carries no mmff94_total_energy"));
        let d = (got - want).abs();
        println!("{name:22} molrs={got:14.6}  rdkit={want:14.6}  d={d:.3e}");
        if d > ENERGY_TOL {
            fails.push(format!(
                "  {name}: molrs={got:.6} rdkit={want:.6} d={d:.3e}"
            ));
        }
    }
    assert!(
        fails.is_empty(),
        "the MMFF chain disagrees with RDKit:\n{}",
        fails.join("\n")
    );
}

// ===========================================================================
// THE REVERSE GATES — every one of them caught a real defect
// ===========================================================================

/// The per-atom MMFF partial charges the typifier baked onto a frame.
fn baked_charges(frame: &Frame) -> Vec<f64> {
    frame
        .get("atoms")
        .and_then(|b| b.get_float("charge"))
        .map(|c| c.iter().copied().collect::<Vec<f64>>())
        .expect("the MMFF typifier bakes a charge column")
}

/// The MMFF numeric atom types of a frame.
fn mmff_types(frame: &Frame) -> Vec<u32> {
    frame
        .get("atoms")
        .and_then(|b| b.get_string(keys::TYPE))
        .map(|t| t.iter().map(|s| s.parse().unwrap_or(0)).collect())
        .expect("the MMFF typifier writes a type column")
}

/// The electrostatic energy of a fixture — the charge-consuming pair style, alone.
fn mmff_electrostatic_energy(name: &str) -> f64 {
    let t = MMFF94Typifier::new();
    let (frame, coords) = mmff_frame(&t, name);
    let ff = t.ff();
    let style = ff
        .styles()
        .iter()
        .find(|s| s.category() == "pair" && s.name.contains("coul"))
        .unwrap_or_else(|| {
            panic!(
                "{name}: the MMFF force field declares NO electrostatic pair style — that \
                 is the 150 kcal/mol caffeine hole, exactly as it was"
            )
        });
    style
        .to_potential(&frame, ff.special_bonds())
        .unwrap_or_else(|e| panic!("{name}: the coul style failed to compile: {e}"))
        .map_or(0.0, |p| p.calc_energy_forces(&coords).0)
}

/// **Zero-charge molecules get EXACTLY 0.0 electrostatic energy — not "small".**
///
/// The forward assertion ("the electrostatic term exists") is satisfied by a term added
/// in the wrong place. This is the reverse one: a term that invents energy on a molecule
/// with no charges is not a fix, it is the next defect. And `0.0`, not `< 1e-6`: every
/// term of `Σ k·qᵢqⱼ / (D·(r+δ))` has a factor of zero in it, so the sum is zero *by
/// construction* — any tolerance at all would be hiding something.
///
/// The molecules are chosen by a **computed predicate** (`Σ|q| == 0`), never a list. The
/// version of this test that exists today hardcodes `["e_ethane", "e_butane"]`, and a
/// new zero-charge fixture would sit behind it, untested.
#[test]
fn zero_charge_molecules_get_exactly_zero_electrostatic_energy() {
    let mut neutral = Vec::new();
    for name in mmff_fixtures() {
        let (_, _, frame) = mmff94_energy_forces(&name);
        let q: f64 = baked_charges(&frame).iter().map(|q| q.abs()).sum();
        if q == 0.0 {
            neutral.push(name);
        }
    }
    assert!(
        !neutral.is_empty(),
        "no fixture has all-zero MMFF charges — the predicate selects nothing, so this \
         gate would pass without asserting anything at all"
    );
    println!("all-zero-charge fixtures (computed, not listed): {neutral:?}");

    for name in &neutral {
        let e = mmff_electrostatic_energy(name);
        assert_eq!(
            e, 0.0,
            "{name} carries NO charges, so its electrostatic energy must be exactly 0.0; \
             got {e:e}. A term that makes energy out of zero charges is not a fixed term — \
             it is a new bug wearing the fix's name."
        );
    }
}

/// **Without a delocalized nitrogen, MMFF94 and MMFF94s are BIT-IDENTICAL.**
///
/// MMFF94s re-parameterises exactly the out-of-plane rows of the delocalized trivalent
/// nitrogen types (10 and 40). On a molecule with none, the two variants ARE the same
/// force field, so their energies must be the same **bits** — not "close". Anything
/// else means the variant is leaking into rows it has no business touching.
///
/// And the converse, which is what makes the first half mean anything: on the molecules
/// that DO carry such a nitrogen, the two must **differ**. Otherwise a "they differ"
/// test can pass on a difference that does not exist — and `MMFF94STypifier` could once
/// silently emit MMFF94's potentials, because the test watching it compared only atom
/// types, which the two variants share by construction.
///
/// Both halves are computed from the typed molecule. Neither is a list.
#[test]
fn mmff94_and_mmff94s_are_bit_identical_without_a_delocalized_nitrogen() {
    let mut same = Vec::new();
    let mut differ = Vec::new();

    for name in mmff_fixtures() {
        let (e94, f94, frame) = mmff94_energy_forces(&name);
        let (e94s, f94s) = mmff94s_energy_forces(&name);
        // The predicate: does this molecule carry a type-10 / type-40 nitrogen?
        if mmff_types(&frame).iter().any(|&t| t == 10 || t == 40) {
            differ.push((name.clone(), e94, e94s));
        } else {
            same.push((name.clone(), e94, e94s, f94, f94s));
        }
    }

    assert!(
        !same.is_empty() && !differ.is_empty(),
        "the partition is vacuous: {} without a delocalized N, {} with. Both halves must \
         be non-empty, or one of the two assertions below asserts nothing.",
        same.len(),
        differ.len()
    );
    println!(
        "no delocalized N (computed): {:?}\nwith delocalized N (computed): {:?}",
        same.iter().map(|s| &s.0).collect::<Vec<_>>(),
        differ.iter().map(|d| &d.0).collect::<Vec<_>>()
    );

    for (name, e94, e94s, f94, f94s) in &same {
        assert_eq!(
            e94.to_bits(),
            e94s.to_bits(),
            "{name} has no delocalized nitrogen, so MMFF94 and MMFF94s are the SAME force \
             field and their energies must be the same bits. Got {e94:.12} vs {e94s:.12}."
        );
        assert!(
            f94.iter()
                .zip(f94s)
                .all(|(a, b)| a.to_bits() == b.to_bits()),
            "{name}: the two variants agree on the energy but not, bit for bit, on the forces"
        );
    }

    for (name, e94, e94s) in &differ {
        assert_ne!(
            e94, e94s,
            "{name} HAS a delocalized nitrogen — the one place MMFF94s changes a parameter \
             — and the two variants return the identical energy. MMFF94s is not reaching \
             the potentials: this is `MMFF94STypifier` silently emitting MMFF94."
        );
    }
}

/// **Benzene HAS impropers.** It had zero, silently, before this chain.
///
/// Six aromatic carbons are six trigonal centres, each contributing three Wilson
/// out-of-plane rows: 18. No forward "the oop energy is right" test can see this — with
/// no improper rows at all the oop term is 0.0, which for planar benzene is also very
/// nearly the right answer. The absence is visible only if you assert on the COUNT.
#[test]
fn benzene_has_impropers() {
    let (_, _, frame) = mmff94_energy_forces("e_benzene");
    let n = frame.get("impropers").and_then(|b| b.nrows()).unwrap_or(0);
    assert_eq!(
        n, 18,
        "benzene's six aromatic carbons are six trigonal centres x 3 Wilson rows = 18 \
         impropers; the frame carries {n}. Zero gives an out-of-plane energy of 0.0 — \
         which for planar benzene is very nearly right, and is how this went unseen."
    );
}

/// **Symmetry-equivalent oxygens carry EQUAL charges.** Acetate's two differed by 0.2014 e.
///
/// The whole point of the equivalencing stage. Symmetry was never asserted, so a
/// carboxylate could ship with one oxygen 0.2 e heavier than its own mirror image while
/// every charge test passed — each charge was within tolerance of an oracle value, just
/// the wrong one of the two.
///
/// Acetate comes from the oracle. Nitrate does not (the 37 carry no NO₃⁻), so it is
/// built through the SAME builder every oracle test uses — the guard that stops a test
/// handing the typifier a molecule no user has.
#[test]
fn symmetry_equivalent_oxygens_carry_equal_charges() {
    // --- acetate, from the oracle ---
    let acetate = CASES
        .iter()
        .find(|c| c.name == "acetate")
        .expect("the oracle carries acetate");
    let chain = Chain::run(acetate);
    let oxygens: Vec<f64> = acetate
        .elements
        .iter()
        .zip(&chain.charges)
        .filter(|(el, _)| **el == "O")
        .map(|(_, q)| *q)
        .collect();
    assert_eq!(oxygens.len(), 2, "acetate has two carboxylate oxygens");
    assert_eq!(
        oxygens[0].to_bits(),
        oxygens[1].to_bits(),
        "acetate's two carboxylate oxygens are equivalent by symmetry, so they carry the \
         SAME charge — the same BITS, because a class mean is one number shared out. Got \
         {:.6} and {:.6}. They once differed by 0.2014 e.",
        oxygens[0],
        oxygens[1]
    );

    // --- nitrate, built through the shared builder ---
    // Planar D3h NO3-: N at the origin, three O at 1.26 A, 120 degrees apart. Written
    // with ONE double bond, as any SDF would write it — which is the point: the Kekule
    // structure is an artefact of how the molecule was WRITTEN, not of what it IS.
    let r = 1.26_f64;
    let s = 0.866_025_403_784_438_6 * r; // sin(120 deg) * r
    let (nitrate, _) = build_molecule(
        &["N", "O", "O", "O"],
        &[
            [0.0, 0.0, 0.0],
            [r, 0.0, 0.0],
            [-0.5 * r, s, 0.0],
            [-0.5 * r, -s, 0.0],
        ],
        &[1, -1, -1, 0],
        &[(0, 1, 1.0, false), (0, 2, 1.0, false), (0, 3, 2.0, false)],
    );
    // Gasteiger needs no QM input, so nitrate can be a witness without an sqm run.
    let q = GasteigerModel
        .assign(&nitrate, None)
        .expect("nitrate types and charges");
    let o = [q[1], q[2], q[3]];
    assert!(
        o[0].to_bits() == o[1].to_bits() && o[1].to_bits() == o[2].to_bits(),
        "nitrate's three oxygens are equivalent by symmetry and must carry the same \
         charge. Got {:.6} / {:.6} / {:.6} — they were once split 6/9/9 by a bond-type \
         perception that believed the Kekule structure it was handed.",
        o[0],
        o[1],
        o[2]
    );
}

/// **The same molecule in a different conformation gets the SAME charges.**
///
/// That is what equivalencing is *for*. A charge model that reads geometry gives a
/// different answer for every rotamer, and the force field a user ships then depends on
/// which conformer they happened to have loaded.
///
/// A rigid rotation would not prove it — a geometric model is invariant under those
/// too. So the molecule is genuinely **bent**: every atom past the second is rotated
/// about x, which changes the internal geometry, not just the frame it is viewed in.
#[test]
fn charges_do_not_depend_on_the_conformation() {
    let bcc = BccModel::new(BccParameterSet::Bcc).expect("bcc table");

    for case in CASES {
        let (mol, ids) = build_case(case);

        let theta = 40.0_f64.to_radians();
        let (c, s) = (theta.cos(), theta.sin());
        let mut moved = mol.clone();
        for (i, aid) in ids.iter().enumerate().skip(2) {
            let [_, y, z] = case.xyz[i];
            moved.set_atom(*aid, keys::Y, y * c - z * s).expect("set y");
            moved.set_atom(*aid, keys::Z, y * s + z * c).expect("set z");
        }

        let q0 = bcc
            .assign(&mol, Some(case.am1_charges_raw))
            .unwrap_or_else(|e| panic!("{}: {e}", case.name));
        let q1 = bcc
            .assign(&moved, Some(case.am1_charges_raw))
            .unwrap_or_else(|e| panic!("{} (moved): {e}", case.name));

        assert!(
            q0.iter().zip(&q1).all(|(a, b)| a.to_bits() == b.to_bits()),
            "{}: the same molecule in a different conformation got different charges. BCC \
             reads TOPOLOGY — bond types, atom types, equivalence classes. A \
             geometry-dependent answer means something in the chain is reading coordinates \
             it has no business reading.",
            case.name
        );
    }
}

/// **A bare sulfur is REFUSED by BCC.**
///
/// Every `ATOMTYPE_*.DEF` ends in an unconditional catch-all row, so "no rule matched"
/// arrives as a *label* (`DU`), not an error. A **bonded** `DU` atom fails anyway, at
/// the parameter lookup. A **bondless** one does not — nothing looks its type up — so it
/// would come back with its charge silently left alone: exactly the plausible-looking
/// answer molrs's no-fallback-values rule exists to prevent.
///
/// Helium is not the witness — it types fine. molrs's own docs name the one element whose
/// BCC rules all require at least one bond, and it is sulfur.
#[test]
fn a_bare_sulfur_is_refused_by_the_bcc_charge_model() {
    let (sulfur, _) = build_molecule(&["S"], &[[0.0, 0.0, 0.0]], &[0], &[]);
    let err = BccModel::new(BccParameterSet::Bcc)
        .expect("bcc table")
        .assign(&sulfur, Some(&[0.0]))
        .expect_err(
            "a bare sulfur has no BCC atom type — every rule in ATOMTYPE_BCC.DEF requires \
             at least one bond — so the model must REFUSE it. Handing back its charge \
             unchanged is a fallback value, and a plausible one, which is the worst kind.",
        );
    assert!(
        matches!(err, ChargeError::MissingAtomType { .. }),
        "the refusal must name its cause (no atom type); got: {err}"
    );
}

/// The chain conserves total charge to ULP scale — and does **not** renormalize it away.
///
/// The input to the chain is `am1_charges_raw` (un-averaged sqm Mulliken). Two stages
/// act on it, and neither may move the total:
///
/// * **equivalencing** replaces each class by its mean — which conserves the sum in
///   exact arithmetic, and to ULP scale in f64 (a class mean is a rounded `f64`, so
///   `n·fl(Σq/n) ≠ Σq` unless `n` is a power of two; measured worst case 3.7e-16);
/// * **the BCC increments** are pairwise antisymmetric, so they conserve it exactly.
///
/// Hence `Σq_out == Σq_raw` to 1e-12 — which is the number that matters, because an
/// `f32` round-trip anywhere on this path would miss it by four orders of magnitude
/// while sailing straight through a 1e-4 comparison against a 6-decimal oracle.
///
/// The comparison is against the chain's own INPUT, not against the oracle's
/// `am1_charges` column: that column is antechamber's *equivalenced* charges printed to
/// six decimals, so comparing to it measures the rounding of a text file (1e-6) rather
/// than the arithmetic of the chain. And it is not compared to the integer net charge
/// either: antechamber carries the AM1 rounding residual through and does not
/// normalize it away, so neither does molrs — a chain that tidied the total to a round
/// number would look better and would no longer be antechamber's answer.
#[test]
fn the_chain_conserves_charge_without_renormalizing() {
    let mut fails = Vec::new();
    let mut renormalized = Vec::new();

    for case in CASES {
        let chain = Chain::run(case);
        let sum_in: f64 = case.am1_charges_raw.iter().sum();
        let sum_out: f64 = chain.charges.iter().sum();
        if (sum_out - sum_in).abs() > 1e-12 {
            fails.push(format!(
                "  {}: Σq_in = {sum_in:.15}, Σq_out = {sum_out:.15}, drift {:.3e}",
                case.name,
                (sum_out - sum_in).abs()
            ));
        }
        // The residual is CARRIED, not scrubbed. If a molecule whose input sum is not
        // the integer net charge comes back sitting exactly on it, something rescaled.
        let net = f64::from(case.net_charge);
        if (sum_in - net).abs() > 1e-9 && sum_out == net {
            renormalized.push(format!(
                "  {}: Σq_in = {sum_in:.9} but Σq_out is EXACTLY {net:.1}",
                case.name
            ));
        }
    }

    assert!(
        fails.is_empty(),
        "equivalencing conserves the sum to ULP and the BCC increments conserve it \
         exactly, so the chain cannot move the total charge by 1e-12:\n{}",
        fails.join("\n")
    );
    assert!(
        renormalized.is_empty(),
        "the chain RENORMALIZED the total charge onto the integer net charge:\n{}\n\
         antechamber carries its AM1 rounding residual through (`am1bcc.c` ends at the \
         increment loop). Scrubbing it makes a tidier number that is no longer the \
         oracle's.",
        renormalized.join("\n")
    );
}

/// Analytic forces match central finite differences, on every fixture the chain closes on.
///
/// A wrong sign, a missing term, a mis-indexed atom: each produces a plausible energy
/// and an impossible gradient.
#[test]
fn the_chain_forces_match_finite_differences() {
    let h = 1.0e-5;
    for name in mmff_fixtures() {
        let (_, forces, _) = mmff94_energy_forces(&name);
        let t = MMFF94Typifier::new();
        let (frame, coords) = mmff_frame(&t, &name);
        let pots = t.ff().to_potentials(&frame).expect("potentials");

        let mut worst = 0.0f64;
        for i in 0..coords.len() {
            let mut cp = coords.clone();
            cp[i] += h;
            let mut cm = coords.clone();
            cm[i] -= h;
            let fd = (pots.calc_energy_forces(&cp).0 - pots.calc_energy_forces(&cm).0) / (2.0 * h);
            worst = worst.max((forces[i] + fd).abs());
        }
        assert!(
            worst < 1.0e-5,
            "{name}: analytic force vs central difference, worst error {worst:.3e}"
        );
    }
}

/// How far the chain actually reaches, printed per molecule rather than compressed into
/// a boolean. (`cargo test --test end_to_end -- --nocapture`.)
#[test]
fn chain_coverage_report() {
    let mut closed = 0usize;
    let mut rows: BTreeMap<&str, String> = BTreeMap::new();
    for case in CASES {
        let chain = Chain::run(case);
        let row = match chain.energy_forces() {
            Some((e, _)) => {
                closed += 1;
                format!("E = {e:12.4} kcal/mol")
            }
            None => format!("no force field: {}", chain.ff.as_ref().unwrap_err()),
        };
        rows.insert(case.name, row);
    }
    for (name, row) in &rows {
        println!("{name:24} {row}");
    }
    println!("\nthe chain closes on {closed}/{} molecules", CASES.len());
}
