//! What BCC bond-type perception must deliver at the *charge* level.
//!
//! The perception itself is tested in the `perceive` target; the antechamber
//! oracle next door pins it to AmberTools over 37 molecules. This file asserts the
//! two things that are only visible once the correction table is applied:
//!
//! * **the Kekulé-invariance of the table** (`ac-002`) — the licence for not
//!   caring which of 7/8 a given ring bond gets;
//! * **the symmetry** that type 9 exists to produce — a carboxylate's two oxygens,
//!   and nitrate's two identical O⁻, must come out with the same charge.

use molrs::ff::params::BCC_CORRECTIONS;
use molrs::ff::typifier::Typifier;
use molrs::ff::typifier::am1bcc::AM1BCCTypifier;
use molrs::store::keys;
use molrs::{AtomId, Atomistic};
use std::collections::HashMap;

/// AM1 base charges are Atomiverse's job; here they are supplied, so that what is
/// under test is only the BCC half.
use crate::typifier::am1bcc_antechamber::SuppliedAM1;

/// Build a molecule from elements and `(i, j, order)` bonds.
fn mol(elements: &[&str], bonds: &[(usize, usize, f64)]) -> (Atomistic, Vec<AtomId>) {
    let mut g = Atomistic::new();
    let ids: Vec<AtomId> = elements
        .iter()
        .enumerate()
        .map(|(i, el)| g.add_atom_xyz(el, i as f64, 0.0, 0.0))
        .collect();
    for (i, j, order) in bonds {
        let bid = g.add_bond(ids[*i], ids[*j]).expect("add bond");
        g.set_bond_prop(bid, keys::ORDER, *order)
            .expect("set order");
    }
    (g, ids)
}

/// Final AM1-BCC charges, given supplied AM1 base charges.
fn charges(mol: &Atomistic, ids: &[AtomId], am1: &[f64]) -> Vec<f64> {
    let typed = AM1BCCTypifier::bcc(SuppliedAM1(am1.to_vec()))
        .expect("build AM1BCCTypifier")
        .typify(mol)
        .expect("typify");
    ids.iter()
        .map(|aid| {
            typed
                .get_atom(*aid)
                .expect("atom")
                .get_f64(keys::CHARGE)
                .expect("charge")
        })
        .collect()
}

// ── ac-002: the BCC increments are Kekulé-invariant ─────────────────────────

#[test]
fn bcc_corrections_are_identical_for_bond_types_7_8_and_10() {
    // Types 7 (aromatic single), 8 (aromatic double) and 10 (the unresolved
    // aromatic precursor) hold the SAME numbers in BCCPARM.DAT — deliberately, so
    // that an aromatic bond's charge correction cannot depend on which Kekulé
    // structure the perceiver happened to pick. Resonance is not allowed to be a
    // free parameter.
    //
    // This identity is what licenses treating a ring's 7/8 alternation as a
    // presentational detail. It is cheap to check and sharp: if a future table
    // edit broke it, every aromatic charge would silently become Kekulé-dependent.
    let mut by_type: HashMap<i32, HashMap<(&str, &str), f64>> = HashMap::new();
    for row in BCC_CORRECTIONS {
        by_type
            .entry(row.bond_type)
            .or_default()
            .insert((row.left, row.right), row.delta);
    }

    let mut compared = Vec::new();
    for (a, b) in [(7, 8), (7, 10), (8, 10)] {
        let left = by_type.get(&a).expect("aromatic rows exist");
        let right = by_type.get(&b).expect("aromatic rows exist");
        let common: Vec<&(&str, &str)> = left.keys().filter(|k| right.contains_key(*k)).collect();
        assert!(
            !common.is_empty(),
            "types {a} and {b} share no atom-type pair — the check would be vacuous"
        );
        for key in &common {
            let (l, r) = **key;
            assert_eq!(
                left[*key], right[*key],
                "BCCPARM {l}|{r}: type {a} = {} but type {b} = {} — aromatic \
                 corrections are no longer Kekulé-invariant",
                left[*key], right[*key]
            );
        }
        compared.push((a, b, common.len()));
    }

    // Guard the guard: these are the pair counts the table actually carries, so a
    // future table that quietly lost its type-10 rows cannot pass by comparing
    // nothing.
    assert_eq!(
        compared,
        vec![(7, 8, 15), (7, 10, 25), (8, 10, 15)],
        "the aromatic row coverage of BCCPARM changed"
    );
}

// ── ac-003: delocalization is what makes equivalent atoms equivalent ─────────

#[test]
fn acetate_carboxylate_oxygens_receive_identical_charges() {
    // The input draws one C=O and one C-O(-). Chemistry says they are the same
    // bond; the type-9 rule is how AM1-BCC says so. Without it the two oxygens
    // correct through DIFFERENT table rows and end up 0.20 e apart — a resonance
    // artefact, not a charge.
    //
    // AM1 base charges are antechamber's own (equivalenced), so the two oxygens
    // arrive equal; the assertion is that the BCC stage keeps them that way.
    let (g, ids) = mol(
        &["C", "C", "O", "O", "H", "H", "H"],
        &[
            (0, 1, 1.0),
            (1, 2, 2.0), // C=O
            (1, 3, 1.0), // C-O(-)
            (0, 4, 1.0),
            (0, 5, 1.0),
            (0, 6, 1.0),
        ],
    );
    let am1 = [
        -0.268, 0.321, -0.596, -0.596, 0.046_333, 0.046_333, 0.046_333,
    ];
    let q = charges(&g, &ids, &am1);

    assert!(
        (q[2] - q[3]).abs() < 1.0e-12,
        "acetate's two carboxylate oxygens must be indistinguishable, got {:+.4} and {:+.4}",
        q[2],
        q[3]
    );
    // And they land where antechamber lands.
    assert!((q[2] - -0.861_3).abs() < 1.0e-4, "got {:+.4}", q[2]);
}

#[test]
fn nitrate_identical_oxygens_receive_identical_charges() {
    // DELIBERATE DIVERGENCE FROM ANTECHAMBER, asserted as a requirement.
    //
    // Nitrate's O0 and O3 are indistinguishable: both terminal, both singly bonded
    // to the same N. AmberTools25 types their bonds 6 and 9 respectively — its
    // type-6 scan stops at the first non-partner neighbour, so which oxygen "wins"
    // depends on the order the bonds were listed — and the two rows it then reads
    // (23|31 at type 6 = +0.1317, at type 9 = -0.1500) drive the two oxygens
    // 0.28 e apart (-0.6997 vs -0.4180). molrs repairs the scan, so both bonds type
    // 9 and both oxygens correct identically.
    let (g, ids) = mol(
        &["O", "N", "O", "O"],
        &[
            (0, 1, 1.0), // N-O(-)
            (1, 2, 2.0), // N=O
            (1, 3, 1.0), // N-O(-)  — identical to bond 0
        ],
    );
    // Equivalent atoms arrive with equivalent AM1 charges (antechamber's `-eq`
    // averaging does this before the BCC stage); the three oxygens of nitrate are
    // one equivalence class.
    let am1 = [-0.633, 0.899, -0.633, -0.633];
    let q = charges(&g, &ids, &am1);

    assert!(
        (q[0] - q[3]).abs() < 1.0e-12,
        "nitrate's two identical O(-) must receive the same charge, got {:+.4} and {:+.4} \
         (antechamber: -0.6997 vs -0.4180)",
        q[0],
        q[3]
    );
    // The whole group is delocalized, so all three oxygens correct through the same
    // row and stay one equivalence class.
    assert!((q[0] - q[2]).abs() < 1.0e-12, "all three O must agree");
    assert!(
        (q.iter().sum::<f64>() - -1.0).abs() < 1.0e-9,
        "BCC must conserve the -1 net charge"
    );
}
