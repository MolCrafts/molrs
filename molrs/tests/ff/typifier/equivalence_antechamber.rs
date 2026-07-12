//! Charge-equivalencing regression tests against AmberTools25 `antechamber -eq 1`.
//!
//! The oracle carries **both halves** of the equivalencing step per molecule:
//!
//! * `am1_charges_raw` — the un-averaged `sqm` Mulliken charges, i.e. exactly what
//!   Atomiverse's AM1 hands molrs;
//! * `am1_charges` — antechamber's `ANTECHAMBER_AM1BCC_PRE.AC`, the same charges
//!   *after* it averaged them over its path-score equivalence classes.
//!
//! So the step can be tested end-to-end without running any AM1: average the raw
//! column with molrs' classes and it must land on antechamber's column. That is a
//! sharp test of the **classes** — a class too coarse smears two chemically
//! distinct atoms together, a class too fine leaves a `sqm` conformer artefact
//! (methanol's methyl H split 0.053 / 0.098 / 0.053) in the force field.
//!
//! Lives here, next to the oracle it consumes, for the same reason
//! `am1bcc_antechamber` does; the perception's own properties (path scores, not
//! automorphism orbits; exact score comparison) are pinned in the `perceive` test
//! target, which is where the algorithm lives.

use molrs::perceive::equivalence::{
    EquivalenceLevel, EquivalenceOptions, average_charges, find_equivalence_classes,
};
use molrs::store::keys;
use molrs::{AtomId, Atomistic};

use super::antechamber_oracle::{AntechamberCase, CASES};

/// antechamber writes charges with 6 decimals; `sqm`'s Mulliken output carries 3.
/// Anything above this is a real disagreement about the classes, not formatting.
const CHARGE_TOL: f64 = 1.0e-4;

/// The molecules whose charges `antechamber -eq 1` actually moves. The other 17 are
/// already symmetric out of `sqm` (or have no equivalent atoms at all), and are just
/// as load-bearing: over-merging would move them.
const EXPECTED_CHANGED: usize = 20;

/// A change larger than `sqm` prints. Below this, a "change" is the last-bit noise
/// of averaging values that were already equal, not equivalencing doing work.
const CHANGE_EPS: f64 = 1.0e-9;

/// Build the molecule with its RAW AM1 charges — what a molrs user has after AM1,
/// before any equivalencing.
fn build_with_raw_charges(case: &AntechamberCase) -> (Atomistic, Vec<AtomId>) {
    let mut mol = Atomistic::new();
    let ids: Vec<AtomId> = case
        .elements
        .iter()
        .zip(case.xyz)
        .map(|(el, [x, y, z])| mol.add_atom_xyz(el, *x, *y, *z))
        .collect();
    for (aid, fc) in ids.iter().zip(case.formal_charges) {
        mol.set_atom(*aid, "formal_charge", *fc)
            .expect("set formal_charge");
    }
    for (i, j, order, aromatic) in case.bonds {
        let bid = mol.add_bond(ids[*i], ids[*j]).expect("add bond");
        mol.set_bond_prop(bid, keys::ORDER, *order)
            .expect("set bond order");
        if *aromatic {
            mol.set_bond_prop(bid, "is_aromatic", true)
                .expect("set aromatic");
        }
    }
    for (aid, q) in ids.iter().zip(case.am1_charges_raw) {
        mol.set_atom(*aid, keys::CHARGE, *q)
            .expect("set raw am1 charge");
    }
    (mol, ids)
}

/// Average a case's raw charges over molrs' classes.
fn equivalenced(case: &AntechamberCase) -> Vec<f64> {
    let (mol, ids) = build_with_raw_charges(case);
    let classes = find_equivalence_classes(&mol, EquivalenceOptions::bcc());
    let averaged = average_charges(&mol, &classes);
    ids.iter()
        .map(|aid| {
            averaged
                .get_atom(*aid)
                .expect("atom")
                .get_f64(keys::CHARGE)
                .expect("charge")
        })
        .collect()
}

fn report(label: &str, failures: &[String], total: usize) {
    if failures.is_empty() {
        return;
    }
    panic!(
        "{label}: {}/{total} molecules disagree with antechamber\n{}",
        failures.len(),
        failures.join("\n")
    );
}

// ── ac-001: equivalencing reproduces antechamber from the raw Mulliken ───────
#[test]
fn equivalenced_charges_match_antechamber() {
    let mut failures = Vec::new();
    for case in CASES {
        let got = equivalenced(case);
        let mut worst = 0.0f64;
        let mut wrong = Vec::new();
        for (k, (g, want)) in got.iter().zip(case.am1_charges).enumerate() {
            let d = (g - want).abs();
            worst = worst.max(d);
            if d > CHARGE_TOL {
                wrong.push(format!("{k}{}: {g:+.4} vs {want:+.4}", case.elements[k]));
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
    report("AM1 equivalencing", &failures, CASES.len());
}

// ── ac-001 (second half): the classes must do work on exactly the molecules
// antechamber does work on. Reproducing the 17 already-symmetric molecules is easy
// — a no-op does it — so the test only bites if the 20 it *moves* are the same 20.
#[test]
fn equivalencing_changes_the_same_molecules_antechamber_changes() {
    let changed_by = |a: &[f64], b: &[f64]| {
        a.iter()
            .zip(b)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f64, f64::max)
            > CHANGE_EPS
    };

    let mut molrs_changed = Vec::new();
    let mut antechamber_changed = Vec::new();
    for case in CASES {
        if changed_by(&equivalenced(case), case.am1_charges_raw) {
            molrs_changed.push(case.name);
        }
        if changed_by(case.am1_charges, case.am1_charges_raw) {
            antechamber_changed.push(case.name);
        }
    }

    assert_eq!(
        antechamber_changed.len(),
        EXPECTED_CHANGED,
        "the oracle itself moved: antechamber changes {antechamber_changed:?}"
    );
    assert_eq!(
        molrs_changed, antechamber_changed,
        "molrs equivalences a different set of molecules than antechamber"
    );
}

// ── ac-004: averaging conserves total charge ─────────────────────────────────
// A class mean is a rounded f64, so broadcasting it perturbs the total by a few
// ULP — no assignment that gives every member of a class the *same* f64 can also
// reproduce the total bit-for-bit, and antechamber carries the identical residual
// (it does not renormalize, and neither do we). What must hold bit-for-bit is that
// nothing else moves: see `equivalencing_touches_only_multi_member_classes`.
#[test]
fn equivalencing_conserves_total_charge() {
    let mut worst = 0.0f64;
    for case in CASES {
        let before: f64 = case.am1_charges_raw.iter().sum();
        let after: f64 = equivalenced(case).iter().sum();
        let drift = (after - before).abs();
        worst = worst.max(drift);
        assert!(
            drift < 1.0e-12,
            "{}: equivalencing drifted the total charge {before:+.17} -> {after:+.17}",
            case.name
        );
    }
    // Measured over the 37 oracle molecules: the residual is pure rounding.
    assert!(
        worst < 1.0e-14,
        "residual is ULP-scale, not a leak: {worst:e}"
    );
}

// ── ac-004 (the bit-exact half): the ONLY atoms that move are the members of a
// class with something to average. Every singleton keeps its exact bits, and every
// member of a class carries bit-identical bits.
#[test]
fn equivalencing_touches_only_multi_member_classes() {
    for case in CASES {
        let (mol, ids) = build_with_raw_charges(case);
        let classes = find_equivalence_classes(&mol, EquivalenceOptions::bcc());
        let got = equivalenced(case);

        for (class, members) in classes.classes().enumerate() {
            let rows: Vec<usize> = members
                .iter()
                .map(|m| ids.iter().position(|id| id == m).expect("member"))
                .collect();
            if rows.len() == 1 {
                let k = rows[0];
                assert_eq!(
                    got[k].to_bits(),
                    case.am1_charges_raw[k].to_bits(),
                    "{}: atom {k} is alone in class {class} but its charge moved",
                    case.name
                );
            } else {
                for w in rows.windows(2) {
                    assert_eq!(
                        got[w[0]].to_bits(),
                        got[w[1]].to_bits(),
                        "{}: atoms {} and {} share class {class} but not a charge",
                        case.name,
                        w[0],
                        w[1]
                    );
                }
            }
        }
    }
}

// ── `-eq 2` is strictly FINER than `-eq 1`, never coarser ───────────────────
//
// Methyl methacrylate is the molecule that shows it. Its two vinyl hydrogens (10,
// 11 on the =CH2) are topologically identical — `-eq 1` merges them and averages
// their raw 0.139 / 0.125 to 0.132 — but they are not geometrically identical: one
// is cis to the ester, the other trans. `-eq 2` weights those path positions
// 0.99 / 1.01 and so keeps them apart.
//
// Verified against AmberTools25 on this molecule:
//
//   antechamber -c bcc -eq 1  ->  H = +0.1320  +0.1320   (merged)
//   antechamber -c bcc -eq 2  ->  H = +0.1390  +0.1250   (split, = the raw values)
//
// `-eq 2` is nobody's default (every charge method defaults to 0 or 1), so it has
// no oracle fixture; this is its pin.
#[test]
fn eq_two_splits_the_cis_trans_vinyl_hydrogens_eq_one_merges() {
    let case = CASES
        .iter()
        .find(|c| c.name == "methyl_methacrylate")
        .expect("methyl_methacrylate in oracle");
    let (mol, ids) = build_with_raw_charges(case);
    let (cis, trans) = (ids[10], ids[11]);
    assert_eq!(
        (case.am1_charges_raw[10], case.am1_charges_raw[11]),
        (0.139, 0.125),
        "the oracle's vinyl H moved"
    );

    let level1 = find_equivalence_classes(&mol, EquivalenceOptions::bcc());
    assert_eq!(
        level1.class_of(cis),
        level1.class_of(trans),
        "-eq 1 must merge the two vinyl H (antechamber averages them to 0.132)"
    );

    let level2 = find_equivalence_classes(
        &mol,
        EquivalenceOptions {
            level: EquivalenceLevel::PathsAndGeometry,
            max_path_length: None,
        },
    );
    assert_ne!(
        level2.class_of(cis),
        level2.class_of(trans),
        "-eq 2 must keep the cis and trans vinyl H apart"
    );

    // The three methyl H stay merged at BOTH levels: a methyl rotor has no E/Z.
    for level in [level1, level2] {
        let first = level.class_of(ids[12]).expect("class");
        assert_eq!(level.class_of(ids[13]), Some(first));
        assert_eq!(level.class_of(ids[14]), Some(first));
    }
}

// `-eq 2` refines `-eq 1`: every level-2 class is contained in a level-1 class, for
// every molecule in the oracle. It may split; it may never merge.
#[test]
fn eq_two_is_a_refinement_of_eq_one() {
    for case in CASES {
        let (mol, ids) = build_with_raw_charges(case);
        let level1 = find_equivalence_classes(&mol, EquivalenceOptions::bcc());
        let level2 = find_equivalence_classes(
            &mol,
            EquivalenceOptions {
                level: EquivalenceLevel::PathsAndGeometry,
                max_path_length: None,
            },
        );
        assert!(
            level2.n_classes() >= level1.n_classes(),
            "{}: -eq 2 came out COARSER than -eq 1 ({} < {} classes)",
            case.name,
            level2.n_classes(),
            level1.n_classes()
        );
        for (a, b) in ids.iter().zip(ids.iter().skip(1)) {
            if level2.class_of(*a) == level2.class_of(*b) {
                assert_eq!(
                    level1.class_of(*a),
                    level1.class_of(*b),
                    "{}: -eq 2 merged two atoms that -eq 1 keeps apart",
                    case.name
                );
            }
        }
    }
}

// ── ac-005: methanol's three methyl hydrogens ───────────────────────────────
// The canonical symptom. sqm splits them 0.053 / 0.098 / 0.053 purely because the
// conformer has one of them eclipsing the O-H; equivalencing must give all three
// exactly 0.068, the mean — not "0.068 within a tolerance", the same bits.
#[test]
fn methanol_methyl_hydrogens_become_identical() {
    let case = CASES
        .iter()
        .find(|c| c.name == "methanol")
        .expect("methanol in oracle");
    assert_eq!(
        case.am1_charges_raw[2..5],
        [0.053, 0.098, 0.053],
        "the oracle's raw methyl H moved"
    );

    let got = equivalenced(case);
    assert_eq!(
        got[2].to_bits(),
        got[3].to_bits(),
        "methyl H are not bit-identical: {got:?}"
    );
    assert_eq!(got[2].to_bits(), got[4].to_bits());
    for (k, (q, want)) in got[2..5].iter().zip(&case.am1_charges[2..5]).enumerate() {
        assert!(
            (q - 0.068).abs() < CHARGE_TOL,
            "methyl H {} is {q}, antechamber says 0.068",
            k + 2
        );
        assert!(
            (q - want).abs() < CHARGE_TOL,
            "methyl H {} disagrees with the oracle",
            k + 2
        );
    }
    // The hydroxyl H is NOT a methyl H: over-merging would drag it in.
    assert_eq!(got[5].to_bits(), 0.195_f64.to_bits());
}
