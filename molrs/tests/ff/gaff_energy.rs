//! GAFF electrostatic energy — the composition gate `gaff-electrostatics` adds.
//!
//! Every stage of the GAFF chain (perceive, ATD typing, BCC charges, parm tables)
//! already had its own oracle. Nothing had ever run
//!
//! ```text
//! Perceive → AtdTypifier → BccModel → gaff_forcefield → Potentials → energy
//! ```
//!
//! to a number. That is how a force field with **no Coulomb style at all** stayed
//! green on 37/37 typing and charge checks: the missing term was never asked for.
//!
//! # External oracle
//!
//! [`fixtures/gaff_sander_energy.json`](fixtures/gaff_sander_energy.json) is not
//! produced by molrs. `scripts/gen_gaff_energy_oracle.py` runs AmberTools25
//! (antechamber AM1-BCC/GAFF → parmchk2 → tleap → sander single-point) on the
//! **same geometries** the antechamber typing/charge oracle already carries, and
//! records sander's electrostatic energy plus the Coulomb constant recovered from
//!
//! ```text
//! k = (EEL + 1-4 EEL) / Σ scale(i,j) · qᵢqⱼ / rᵢⱼ
//! ```
//!
//! (1-2/1-3 excluded, 1-4 scaled by SCEE = 1.2). The three cases are the ions the
//! defect screams on — acetate (−1), methylammonium (+1), imidazolium (+1) — not a
//! neutral subset that could hide a missing Coulomb term.

use std::path::{Path, PathBuf};

use molrs::ff::charge::{BccModel, BccParameterSet};
use molrs::ff::forcefield::ForceField;
use molrs::ff::forcefield::gaff::{GaffParameterSet, gaff_forcefield};
use molrs::ff::potential::intramolecular_pairs;
use molrs::ff::typifier::Typifier;
use molrs::ff::typifier::atd::{AtdParameterSet, AtdTypifier};
use molrs::store::keys;
use molrs::{AtomId, Atomistic};
use serde_json::Value;

use crate::typifier::antechamber_oracle::{AntechamberCase, CASES};
use crate::typifier::oracle_mol::build_case;

/// AMBER's electrostatic conversion factor, kcal·Å·mol⁻¹·e⁻².
///
/// Measured from AmberTools25 sander on the three ion cases (see the fixture's
/// `recovered_coulomb` fields and `scripts/gen_gaff_energy_oracle.py`). It is
/// **not** CODATA's 332.06371 and **not** Halgren's 332.0716.
const AMBER_COULOMB: f64 = 332.052_217_29;

/// AMBER SCEE = 1.2 → 1-4 Coulomb weight.
const AMBER_COUL_14: f64 = 1.0 / 1.2;

/// Vacuum dielectric — the medium GAFF was parameterised in.
const VACUUM_DIELECTRIC: f64 = 1.0;

/// Per-ion electrostatic parity vs sander (kcal/mol).
///
/// Sander prints energies to 4 decimals. Charges from `BccModel` match antechamber
/// to ~1e-4 e; on these ions that moves the Coulomb term by well under 0.05 kcal/mol.
const ELE_TOL: f64 = 5.0e-2;

/// The ions that must be present. Hard-coded so a neutral-only fixture cannot
/// quietly satisfy every assertion below (the `["e_ethane"]` mistake).
const REQUIRED_IONS: [&str; 3] = ["acetate", "methylammonium", "imidazolium"];

fn fixture_path() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/ff/fixtures/gaff_sander_energy.json")
}

fn load_oracle() -> Value {
    let text = std::fs::read_to_string(fixture_path())
        .unwrap_or_else(|e| panic!("read {}: {e}", fixture_path().display()));
    serde_json::from_str(&text).expect("parse gaff_sander_energy.json")
}

fn case(name: &str) -> &'static AntechamberCase {
    CASES
        .iter()
        .find(|c| c.name == name)
        .unwrap_or_else(|| panic!("{name} missing from antechamber oracle"))
}

fn oracle_case<'a>(oracle: &'a Value, name: &str) -> &'a Value {
    oracle["cases"]
        .as_array()
        .expect("cases array")
        .iter()
        .find(|c| c["name"].as_str() == Some(name))
        .unwrap_or_else(|| panic!("{name} missing from gaff_sander_energy.json"))
}

/// Type with GAFF, correct charges with BCC, parameterise, lay charges, build pairs.
///
/// This is the composition the suite never ran: every stage was tested, the chain
/// to energy was not.
fn gaff_chain(case: &AntechamberCase) -> (ForceField, molrs::store::frame::Frame, Vec<f64>) {
    let (mol, ids) = build_case(case);
    let typed = AtdTypifier::new(AtdParameterSet::Gff)
        .typify(&mol)
        .unwrap_or_else(|e| panic!("{}: ATD typing failed: {e}", case.name));
    let charges = BccModel::new(BccParameterSet::Bcc)
        .expect("build BCC model")
        .correct(&mol, case.am1_charges)
        .unwrap_or_else(|e| panic!("{}: BCC correct failed: {e}", case.name));
    assert_eq!(
        charges.len(),
        ids.len(),
        "{}: charge count {} != atom count {}",
        case.name,
        charges.len(),
        ids.len()
    );

    let (mut labelled, ff) = gaff_forcefield(GaffParameterSet::Gaff, &typed)
        .unwrap_or_else(|e| panic!("{}: gaff_forcefield failed: {e}", case.name));
    lay_charges(&mut labelled, &ids, &charges);
    let mut frame = labelled.to_frame();
    frame.insert("pairs", intramolecular_pairs(&frame));
    let coords = flat_coords(case.xyz);
    (ff, frame, coords)
}

fn lay_charges(mol: &mut Atomistic, ids: &[AtomId], charges: &[f64]) {
    for (aid, q) in ids.iter().zip(charges) {
        mol.set_atom(*aid, keys::CHARGE, *q)
            .unwrap_or_else(|e| panic!("set charge on {aid:?}: {e}"));
    }
}

fn flat_coords(xyz: &[[f64; 3]]) -> Vec<f64> {
    xyz.iter().flat_map(|p| [p[0], p[1], p[2]]).collect()
}

/// Energy of the force field's `pair/coul/cut` style alone.
fn coulomb_energy(ff: &ForceField, frame: &molrs::store::frame::Frame, coords: &[f64]) -> f64 {
    let style = ff.get_style("pair", "coul/cut").unwrap_or_else(|| {
        panic!(
            "force field `{}` defines no `pair/coul/cut` — GAFF's electrostatics is the \
             unbuffered Coulomb form of that generic kernel, and without the style the term \
             is silently zero",
            ff.name
        )
    });
    let pot = style
        .to_potential(frame, ff.special_bonds())
        .unwrap_or_else(|e| panic!("pair/coul/cut to_potential failed: {e}"))
        .unwrap_or_else(|| {
            panic!(
                "pair/coul/cut compiled to nothing — the pairs block is empty, so there is no \
                 nonbonded topology for the Coulomb term to act on"
            )
        });
    pot.calc_energy_forces(coords).0
}

fn n_14_pairs(frame: &molrs::store::frame::Frame) -> usize {
    frame
        .get("pairs")
        .and_then(|b| b.get_bool("is_14"))
        .map(|c| c.iter().filter(|&&f| f).count())
        .unwrap_or(0)
}

fn total_abs_charge(frame: &molrs::store::frame::Frame) -> f64 {
    frame
        .get("atoms")
        .and_then(|b| b.get_float("charge"))
        .map(|c| c.iter().map(|q| q.abs()).sum())
        .unwrap_or_else(|| panic!("typed frame has no atoms/charge column"))
}

// ---------------------------------------------------------------------------
// ac-001 / ac-002 — the force field SPEAKS its electrostatics
// ---------------------------------------------------------------------------

/// ac-001 — `gaff_forcefield` declares `pair/coul/cut` with AMBER's numbers.
///
/// The kernel has always been in the registry (`pair/coul/cut` at δ = 0 is the
/// textbook Coulomb). What was missing was the force field defining the style.
/// A registered-but-unreferenced kernel is a term that does not exist.
#[test]
fn gaff_forcefield_declares_unbuffered_coulomb_style() {
    // Any multi-atom GAFF molecule will do for the style declaration; the ions
    // are reserved for the energy assertions that need non-zero net charge.
    let case = case("acetate");
    let (mol, _) = build_case(case);
    let typed = AtdTypifier::new(AtdParameterSet::Gff)
        .typify(&mol)
        .expect("type acetate");
    let (_, ff) = gaff_forcefield(GaffParameterSet::Gaff, &typed).expect("parameterise acetate");

    let style = ff.get_style("pair", "coul/cut").unwrap_or_else(|| {
        panic!(
            "gaff_forcefield did not declare `pair/coul/cut`. GAFF's electrostatics is the \
             unbuffered form of that generic kernel (δ = 0) — not a kernel of its own. The \
             charges were always on the Frame; the missing style is the defect this suite \
             exists to kill."
        )
    });

    for (key, want) in [
        ("coulomb", AMBER_COULOMB),
        ("dielectric", VACUUM_DIELECTRIC),
        ("delta", 0.0),
    ] {
        let got = style.params.get(key).unwrap_or_else(|| {
            panic!(
                "`pair/coul/cut` carries no `{key}` style param. A missing `coulomb` or \
                 `dielectric` is now an Err at compile time; a missing `delta` silently \
                 defaults to 0 (correct for AMBER) — but AMBER still has to SAY so. The \
                 force field must be the source of its own constants."
            )
        });
        assert!(
            (got - want).abs() < 1e-12,
            "`pair/coul/cut` {key} = {got}, want {want}"
        );
    }

    let sp = ff.special_bonds();
    assert!(
        (sp.coul[2] - AMBER_COUL_14).abs() < 1e-12,
        "special_bonds.coul[2] = {}, want {AMBER_COUL_14} (AMBER SCEE = 1.2). This is the \
         scale factor that used to be declared and consumed by NOTHING — a constant nobody \
         reads is the same smell as rows nobody loads.",
        sp.coul[2]
    );
    assert!(
        (sp.lj[2] - 0.5).abs() < 1e-12,
        "special_bonds.lj[2] = {}, want 0.5 (AMBER SCNB = 2)",
        sp.lj[2]
    );
    assert_eq!(sp.coul[0], 0.0, "1-2 Coulomb must be excluded");
    assert_eq!(sp.coul[1], 0.0, "1-3 Coulomb must be excluded");
}

/// ac-002 — the Coulomb constant is the measured AMBER value, not CODATA or Halgren.
///
/// The fixture records the constant recovered from sander on each ion. All three
/// must land on the same number the force field carries; if they diverge, either
/// the recovery geometry changed or someone "unified" the constant onto CODATA.
#[test]
fn measured_coulomb_constant_matches_sander_recovery() {
    let oracle = load_oracle();
    let fixture_k = oracle["coulomb"]
        .as_f64()
        .expect("fixture top-level coulomb");
    assert!(
        (fixture_k - AMBER_COULOMB).abs() < 1e-12,
        "fixture coulomb = {fixture_k}, code carries {AMBER_COULOMB} — they must be the same \
         measured number"
    );

    let cases = oracle["cases"].as_array().expect("cases");
    assert!(
        !cases.is_empty(),
        "gaff_sander_energy.json has no cases — every assertion below would be vacuous"
    );

    let mut fails = Vec::new();
    for entry in cases {
        let name = entry["name"].as_str().expect("name");
        let recovered = entry["recovered_coulomb"]
            .as_f64()
            .unwrap_or_else(|| panic!("{name}: missing recovered_coulomb"));
        // sander prints energies to 4 decimals; recovery is good to ~1e-3 absolute.
        let delta = (recovered - AMBER_COULOMB).abs();
        println!("{name:16} recovered={recovered:.10}  amber={AMBER_COULOMB}  d={delta:.3e}");
        if delta > 1.0e-3 {
            fails.push(format!(
                "{name}: recovered {recovered:.10} vs AMBER_COULOMB {AMBER_COULOMB} (d={delta:.3e})"
            ));
        }
    }
    assert!(
        fails.is_empty(),
        "sander recovery disagrees with the force field's Coulomb constant:\n  {}",
        fails.join("\n  ")
    );

    // The two numbers that are NOT interchangeable with AMBER's.
    let codata = molrs::units::constants::COULOMB_REAL;
    let mmff = 332.0716_f64;
    assert!(
        (AMBER_COULOMB - codata).abs() > 1e-4,
        "AMBER's Coulomb constant ({AMBER_COULOMB}) collapsed onto CODATA ({codata})"
    );
    assert!(
        (AMBER_COULOMB - mmff).abs() > 1e-4,
        "AMBER's Coulomb constant ({AMBER_COULOMB}) collapsed onto Halgren/MMFF ({mmff})"
    );
}

// ---------------------------------------------------------------------------
// ac-003 / ac-004 — the chain reaches a non-zero energy, checked against sander
// ---------------------------------------------------------------------------

/// The fixture still names exactly the three required ions — no silent drop.
#[test]
fn sander_oracle_names_the_required_ions() {
    let oracle = load_oracle();
    let names: Vec<&str> = oracle["cases"]
        .as_array()
        .expect("cases")
        .iter()
        .map(|c| c["name"].as_str().expect("name"))
        .collect();
    for ion in REQUIRED_IONS {
        assert!(
            names.contains(&ion),
            "gaff_sander_energy.json is missing required ion `{ion}` (present: {names:?}). A \
             neutral-only fixture would be the third replay of the `['e_ethane']` mistake."
        );
        assert_ne!(
            case(ion).net_charge,
            0,
            "{ion} must have non-zero net charge — that is why it is on this list"
        );
    }
}

/// ac-003 / ac-004 — each ion's Coulomb energy is non-zero and matches sander.
///
/// Chain: user molecule → ATD (GAFF types) → BCC charges from AM1 base →
/// `gaff_forcefield` → `pair/coul/cut` energy. Geometry and AM1 base charges come
/// from the antechamber oracle (same embed as the sander generator); the energy
/// reference is sander's `EEL + 1-4 EEL`, which molrs did not produce.
#[test]
fn ion_electrostatic_energy_matches_sander_oracle() {
    let oracle = load_oracle();
    let mut fails = Vec::new();

    for name in REQUIRED_IONS {
        let case = case(name);
        let ref_case = oracle_case(&oracle, name);
        let want = ref_case["sander_electrostatic_energy"]
            .as_f64()
            .unwrap_or_else(|| panic!("{name}: missing sander_electrostatic_energy"));
        let want_sum_abs = ref_case["sum_abs_charge"]
            .as_f64()
            .unwrap_or_else(|| panic!("{name}: missing sum_abs_charge"));
        let want_n14 = ref_case["n_14_pairs"]
            .as_u64()
            .unwrap_or_else(|| panic!("{name}: missing n_14_pairs"))
            as usize;

        let (ff, frame, coords) = gaff_chain(case);
        let sum_abs = total_abs_charge(&frame);
        let n14 = n_14_pairs(&frame);
        let got = coulomb_energy(&ff, &frame, &coords);

        println!(
            "{name:16} E_ele={got:12.6}  sander={want:12.6}  d={:.3e}  \
             sum|q|={sum_abs:.6} (oracle {want_sum_abs:.6})  n14={n14} (oracle {want_n14})",
            (got - want).abs()
        );

        if sum_abs < 0.5 {
            fails.push(format!(
                "{name}: sum|q| = {sum_abs:.6} — an ion fixture lost its charges"
            ));
        }
        // Fixture sum|q| is antechamber's printed charges; BccModel is within 1e-4/atom.
        if (sum_abs - want_sum_abs).abs() > 5.0e-3 {
            fails.push(format!(
                "{name}: sum|q| = {sum_abs:.6} vs oracle {want_sum_abs:.6}"
            ));
        }
        if n14 != want_n14 {
            fails.push(format!(
                "{name}: n_14_pairs = {n14}, oracle says {want_n14} — 1-4 scaling cannot be \
                 exercised if the neighbour list disagrees"
            ));
        }
        if got == 0.0 {
            fails.push(format!(
                "{name}: electrostatic energy is EXACTLY 0.0 with sum|q| = {sum_abs:.6} and \
                 net charge {}. That is the silent-drop defect.",
                case.net_charge
            ));
        }
        let delta = (got - want).abs();
        if delta > ELE_TOL {
            fails.push(format!(
                "{name}: E_ele = {got:.6} vs sander {want:.6} (d={delta:.3e} > {ELE_TOL:.0e})"
            ));
        }
    }

    assert!(
        fails.is_empty(),
        "GAFF electrostatic chain failed against sander:\n  {}",
        fails.join("\n  ")
    );
}

// ---------------------------------------------------------------------------
// ac-005 — special_bonds.coul[2] is actually CONSUMED
// ---------------------------------------------------------------------------

/// ac-005 — changing the 1-4 Coulomb weight changes the energy of a molecule
/// that has 1-4 pairs.
///
/// `special_bonds.coul[2] = 1/1.2` used to be declared and then consumed by
/// nothing, because there was no Coulomb style. A constant nobody reads is not
/// a scale factor.
#[test]
fn special_bonds_coul_14_is_consumed() {
    // Acetate has 6 1-4 pairs (fixture); any of the three ions works.
    let case = case("acetate");
    let (mut ff, frame, coords) = gaff_chain(case);
    let n14 = n_14_pairs(&frame);
    assert!(
        n14 > 0,
        "acetate must carry 1-4 pairs for this gate (got {n14})"
    );

    let e_amber = coulomb_energy(&ff, &frame, &coords);

    // Flip the 1-4 weight to 1.0 (no scale) — energy must move if the weight is live.
    let mut sp = *ff.special_bonds();
    sp.coul[2] = 1.0;
    ff.set_special_bonds(sp);
    let e_unscaled = coulomb_energy(&ff, &frame, &coords);

    // And to 0.0 (drop all 1-4 Coulomb) — a third distinct value.
    sp.coul[2] = 0.0;
    ff.set_special_bonds(sp);
    let e_dropped = coulomb_energy(&ff, &frame, &coords);

    println!(
        "acetate 1-4 scale: amber(1/1.2)={e_amber:.6}  unscaled(1.0)={e_unscaled:.6}  \
         dropped(0.0)={e_dropped:.6}  n14={n14}"
    );

    assert!(
        (e_amber - e_unscaled).abs() > 1e-6,
        "changing special_bonds.coul[2] from {AMBER_COUL_14} to 1.0 did not move acetate's \
         Coulomb energy ({e_amber:.9} vs {e_unscaled:.9}). The 1-4 weight is declared but not \
         consumed — the original tell that the Coulomb term was missing."
    );
    assert!(
        (e_amber - e_dropped).abs() > 1e-6,
        "dropping 1-4 Coulomb (coul[2]=0) did not move acetate's energy ({e_amber:.9} vs \
         {e_dropped:.9})"
    );
    assert!(
        (e_unscaled - e_dropped).abs() > 1e-6,
        "unscaled and fully-dropped 1-4 Coulomb gave the same energy ({e_unscaled:.9}) — the \
         1-4 pairs contribute nothing"
    );
}
