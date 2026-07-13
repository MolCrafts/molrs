//! The charge models against AmberTools25, over all 37 oracle molecules.
//!
//! `am1bcc_antechamber.rs` (in the typifier tree, next to the perception it grew
//! out of) owns the `-c bcc` ladder. This file owns what only exists once there is
//! a MODEL rather than a typifier:
//!
//! * **ABCG2** — the second correction family, through the same `BccModel` with
//!   nothing changed but the parameter set (ac-002);
//! * **charge conservation** — the BCC stage adds pairwise-antisymmetric increments,
//!   so it must not move the total; and it must not "fix" the total either (ac-005).
//!
//! The oracle carries `bcc_charges` and `abcg2_charges` as corrections of the SAME
//! `am1_charges` column — `scripts/gen_am1bcc_oracle.py` asserts, per molecule, that
//! the two antechamber runs consumed identical AM1 base charges — so the two columns
//! differ in exactly one thing: which BCCPARM table was read.

use molrs::ff::charge::{BccModel, BccParameterSet};
use molrs::perceive::equivalence::{EquivalenceOptions, average_charges, find_equivalence_classes};
use molrs::store::keys;
use molrs::{AtomId, Atomistic};

use crate::typifier::antechamber_oracle::CASES;
use crate::typifier::oracle_mol::{build_case, report};

const CHARGE_TOL: f64 = 1.0e-4;

/// antechamber does not renormalize, so the final charges do not sum to the integer
/// net charge — they sum to whatever the AM1 charges summed to. `sqm` prints its
/// Mulliken charges to 3 decimals, and that rounding residual is what is left over.
/// Measured over the 37 oracle molecules: worst 0.003997 e (trimethyl phosphate).
const NET_CHARGE_RESIDUAL_TOL: f64 = 0.005;

/// How many of the 37 molecules actually carry a residual. Pinned, because it is
/// what makes the tolerance assertion above non-vacuous: a model that renormalized
/// would drive every residual to exactly zero and sail through a `<= 0.005` bound.
const EXPECTED_MOLECULES_WITH_RESIDUAL: usize = 27;

/// A change larger than the last bits of an f64 sum.
const ULP_SCALE: f64 = 1.0e-14;

fn model(set: BccParameterSet) -> BccModel {
    BccModel::new(set).expect("build the charge model")
}

/// Sum in atom order — the same order the model produced the charges in.
///
/// Order matters at the scale this file asserts at: f64 addition is not associative,
/// so "the total charge" is only a well-defined number once you say how it was added
/// up. Both sides of every comparison below use this one function.
fn total(charges: &[f64]) -> f64 {
    charges.iter().sum()
}

// ── ac-002: ABCG2, through the same model ────────────────────────────────────

/// ABCG2 end-to-end: 37/37 against `antechamber -c abcg2`.
///
/// One engine, two parameter sets. The ONLY thing that differs from the AM1-BCC
/// gate next door is `BccParameterSet::Abcg2`, which selects `BCCPARM_ABCG2.DAT`
/// (439 rows, 36 `CORR` aliases) over `BCCPARM.DAT` (405 rows, no aliases) and
/// `ATOMTYPE_ABCG2.DEF` over `ATOMTYPE_BCC.DEF`. Same molecules, same AM1 base
/// charges, same code path.
#[test]
fn abcg2_charges_match_antechamber_end_to_end() {
    let abcg2 = model(BccParameterSet::Abcg2);
    let mut failures = Vec::new();
    for case in CASES {
        let (mol, _) = build_case(case);
        let got = match abcg2.correct(&mol, case.am1_charges) {
            Ok(q) => q,
            Err(e) => {
                failures.push(format!("  {:22} ERROR {e}", case.name));
                continue;
            }
        };
        let mut worst = 0.0f64;
        let mut wrong = Vec::new();
        for (k, (q, want)) in got.iter().zip(case.abcg2_charges).enumerate() {
            let d = (q - want).abs();
            worst = worst.max(d);
            if d > CHARGE_TOL {
                wrong.push(format!("{k}{}: {q:+.4} vs {want:+.4}", case.elements[k]));
            }
        }
        if !wrong.is_empty() {
            failures.push(format!(
                "  {:22} max|dq|={worst:.4}  {}",
                case.name,
                wrong.join(", ")
            ));
        }
    }
    report("ABCG2 end-to-end", &failures, CASES.len());
}

/// Guard the guard: the two families really do disagree.
///
/// If `BccModel` ignored its parameter set and always read `BCCPARM.DAT`, the ABCG2
/// test above would still have to fail — but only if the two oracle columns are
/// actually different. They are (methanol's oxygen: −0.5988 under BCC, −0.6610 under
/// ABCG2), and this pins that they stay so. A fixture regenerated with both columns
/// accidentally equal would turn ac-002 into a second copy of ac-001.
#[test]
fn bcc_and_abcg2_are_not_the_same_column() {
    let differing = CASES
        .iter()
        .filter(|c| {
            c.bcc_charges
                .iter()
                .zip(c.abcg2_charges)
                .any(|(a, b)| (a - b).abs() > CHARGE_TOL)
        })
        .count();
    assert!(
        differing >= 30,
        "only {differing}/37 molecules have different BCC and ABCG2 charges — the \
         two parameter sets have collapsed into one and ac-002 is testing nothing"
    );
}

// ── ac-005: the BCC stage conserves charge, and never corrects the total ─────

/// The corrections cancel: the total charge out is the total charge in.
///
/// Every increment is added to one atom and subtracted from another, so their sum is
/// exactly zero in exact arithmetic. In f64 it is zero to within the rounding of
/// `q[i] = am1[i] + delta[i]` — measured over the 37 oracle molecules, the worst
/// residual is 2.8e-16, i.e. ONE ulp of 1.0, and 16 of the 37 land bit-for-bit.
///
/// It is asserted at ulp scale rather than bitwise for the same reason
/// `equivalencing_conserves_total_charge` is: no f64 assignment that gives each atom
/// a rounded `am1[i] + delta[i]` can also reproduce the sum of the un-rounded values
/// bit-for-bit. Demanding bits here would be demanding that IEEE-754 addition be
/// associative. What the bound DOES exclude is any real leak — a dropped increment,
/// a single-ended application, a rescale — all of which are 1e-3 effects, twelve
/// orders of magnitude above this line.
#[test]
fn bcc_conserves_total_charge_through_the_corrections() {
    let bcc = model(BccParameterSet::Bcc);
    let mut worst = 0.0f64;
    for case in CASES {
        let (mol, _) = build_case(case);
        let q = bcc
            .correct(&mol, case.am1_charges)
            .unwrap_or_else(|e| panic!("{}: {e}", case.name));

        let before = total(case.am1_charges);
        let after = total(&q);
        let drift = (after - before).abs();
        worst = worst.max(drift);
        assert!(
            drift < ULP_SCALE,
            "{}: the BCC stage moved the total charge {before:+.17} -> {after:+.17} \
             (drift {drift:e}). The increments are pairwise antisymmetric; a drift \
             this size means one was applied to only one end of its bond.",
            case.name
        );
    }
    assert!(
        worst < ULP_SCALE,
        "residual is ULP-scale, not a leak: {worst:e}"
    );
}

/// The AM1 rounding residual is CARRIED THROUGH, never normalized away.
///
/// `am1bcc.c` ends at the increment loop: no rescale, no shift, no round. `sqm`
/// prints Mulliken charges to 3 decimals, so a neutral molecule's AM1 charges sum to
/// something like −0.002 rather than 0, and antechamber's final AM1-BCC charges sum
/// to that same −0.002. molrs must land in the same place.
///
/// The deleted `normalize_total_charge` did the opposite: it spread `(target − sum)/N`
/// over every atom. That is not a fix, it is three failures at once —
///
/// 1. it makes molrs DIVERGE from antechamber (by up to 0.004/N e per atom; on
///    trimethyl phosphate that is 2.4e-4, which is above the 1e-4 gate);
/// 2. it violates molrs's own no-fallback-values rule;
/// 3. it HIDES BUGS — an AM1 that failed to converge and returned +0.7 on a neutral
///    molecule would come back silently flattened into a plausible-looking answer.
///
/// So the integer net charge is asserted as a TOLERANCE and never as a target, and
/// the residual is asserted to be exactly antechamber's own.
#[test]
fn bcc_carries_the_am1_residual_and_never_renormalizes() {
    let bcc = model(BccParameterSet::Bcc);
    let mut with_residual = 0;
    for case in CASES {
        let (mol, _) = build_case(case);
        let q = bcc
            .correct(&mol, case.am1_charges)
            .unwrap_or_else(|e| panic!("{}: {e}", case.name));

        let net = f64::from(case.net_charge);
        let molrs_residual = total(&q) - net;
        let antechamber_residual = total(case.bcc_charges) - net;

        assert!(
            molrs_residual.abs() <= NET_CHARGE_RESIDUAL_TOL,
            "{}: the total charge is {molrs_residual:+.6} away from the net charge \
             {net:+} — that is beyond the AM1 rounding residual, so something is \
             genuinely wrong (do NOT make this pass by normalizing)",
            case.name
        );
        assert!(
            (molrs_residual - antechamber_residual).abs() < 1.0e-9,
            "{}: molrs's residual {molrs_residual:+.9} != antechamber's \
             {antechamber_residual:+.9}. If molrs's is zero, a renormalizer has been \
             reintroduced: antechamber does not renormalize, so matching it means \
             carrying its residual, not removing it.",
            case.name
        );
        if molrs_residual.abs() > 1.0e-9 {
            with_residual += 1;
        }
    }

    // Non-vacuity: the tolerance above is only a test if some molecules exercise it.
    // A renormalizing model would land every single one of these on exactly zero.
    assert_eq!(
        with_residual, EXPECTED_MOLECULES_WITH_RESIDUAL,
        "{with_residual}/37 molecules carry an AM1 rounding residual; antechamber \
         produces {EXPECTED_MOLECULES_WITH_RESIDUAL}. Zero would mean the charges are \
         being normalized to the integer net charge."
    );
}

// ── what `needs_equivalencing()` MEANS, spelled out as a pipeline ────────────

/// Raw sqm Mulliken -> equivalence -> BCC = antechamber, 37/37.
///
/// The BCC stage's input is not "the AM1 charges", it is "the AM1 charges AFTER
/// `-eq 1` averaging" — which is why `BccModel::correct` takes them as an argument
/// and `ChargeModel::needs_equivalencing()` returns true for this family. Here that
/// contract is executed by hand, out of molrs's own parts: perceive the classes,
/// average the raw charges over them, then correct. Landing on antechamber from the
/// RAW column proves the two halves compose, and pins what `assign` must be doing
/// internally (see `model::every_model_lands_on_its_own_oracle_column`).
#[test]
fn equivalencing_then_correcting_reproduces_antechamber_from_raw_mulliken() {
    let bcc = model(BccParameterSet::Bcc);
    let mut failures = Vec::new();
    for case in CASES {
        let (mut mol, ids) = build_case(case);
        for (aid, q) in ids.iter().zip(case.am1_charges_raw) {
            mol.set_atom(*aid, keys::CHARGE, *q).expect("set raw AM1");
        }

        let classes = find_equivalence_classes(&mol, EquivalenceOptions::bcc());
        let averaged = average_charges(&mol, &classes);
        let am1: Vec<f64> = charges_of(&averaged, &ids);

        let got = match bcc.correct(&averaged, &am1) {
            Ok(q) => q,
            Err(e) => {
                failures.push(format!("  {:22} ERROR {e}", case.name));
                continue;
            }
        };
        let mut worst = 0.0f64;
        let mut wrong = Vec::new();
        for (k, (q, want)) in got.iter().zip(case.bcc_charges).enumerate() {
            let d = (q - want).abs();
            worst = worst.max(d);
            if d > CHARGE_TOL {
                wrong.push(format!("{k}{}: {q:+.4} vs {want:+.4}", case.elements[k]));
            }
        }
        if !wrong.is_empty() {
            failures.push(format!(
                "  {:22} max|dq|={worst:.4}  {}",
                case.name,
                wrong.join(", ")
            ));
        }
    }
    report("raw Mulliken -> equivalence -> BCC", &failures, CASES.len());
}

/// The charges of `mol`, in `ids` order.
fn charges_of(mol: &Atomistic, ids: &[AtomId]) -> Vec<f64> {
    ids.iter()
        .map(|aid| {
            mol.get_atom(*aid)
                .expect("atom")
                .get_f64(keys::CHARGE)
                .expect("charge")
        })
        .collect()
}
