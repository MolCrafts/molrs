//! Charge-equivalence classes — the property the antechamber oracle cannot state.
//!
//! The classes themselves are pinned by the oracle in the `ff` test target
//! (`typifier::equivalence_antechamber`, 37 molecules). What lives here is the one
//! thing that oracle can only *imply*: **which algorithm** produced them.
//!
//! Reproducing antechamber's charges does not, by itself, prove the engine is a
//! path score — a graph-automorphism partition (Morgan / Weisfeiler-Leman /
//! `graph_hash`) reproduces most of the same molecules. This test finds a molecule
//! where the two disagree, and pins molrs to antechamber's side of it.

use std::collections::HashMap;

use molrs::perceive::Perceive;
use molrs::perceive::equivalence::{EquivalenceOptions, average_charges, find_equivalence_classes};
use molrs::store::keys;
use molrs::{AtomId, Atomistic};

/// Acetate, `CC(=O)[O-]`, exactly as the antechamber oracle carries it: a **Kekulé**
/// carboxylate, one `C=O` and one `C–O⁻`. Atom order: C0 (methyl), C1 (carboxyl),
/// O2 (double-bonded), O3 (anionic), H4, H5, H6.
fn acetate() -> (Atomistic, Vec<AtomId>) {
    let mut mol = Atomistic::new();
    let els = ["C", "C", "O", "O", "H", "H", "H"];
    let xyz = [
        [-0.6280, -0.0860, -0.0900],
        [0.8600, 0.1240, 0.1260],
        [1.1830, 1.2520, 0.5910],
        [1.5850, -0.8610, -0.1900],
        [-1.2020, 0.7990, 0.2000],
        [-0.8240, -0.2940, -1.1460],
        [-0.9740, -0.9340, 0.5090],
    ];
    let ids: Vec<AtomId> = els
        .iter()
        .zip(xyz)
        .map(|(el, [x, y, z])| mol.add_atom_xyz(el, x, y, z))
        .collect();
    for (aid, fc) in ids.iter().zip([0, 0, 0, -1, 0, 0, 0]) {
        mol.set_atom(*aid, "formal_charge", fc).expect("formal");
    }
    // The Kekulé structure: C1=O2 is a double bond, C1–O3 is a single bond.
    for (i, j, order) in [
        (0, 1, 1.0),
        (1, 2, 2.0),
        (1, 3, 1.0),
        (0, 4, 1.0),
        (0, 5, 1.0),
        (0, 6, 1.0),
    ] {
        let bid = mol.add_bond(ids[i], ids[j]).expect("bond");
        mol.set_bond_prop(bid, keys::ORDER, order).expect("order");
    }
    (mol, ids)
}

/// The raw `sqm` Mulliken charges for acetate, and antechamber's `-eq 1` answer.
/// From the AmberTools25 oracle (`tests/ff/typifier/antechamber_oracle.rs`); the two
/// oxygens come out of AM1 at **-0.595 / -0.597** and antechamber returns
/// **-0.596 / -0.596** — it merged them.
const ACETATE_RAW: [f64; 7] = [-0.268, 0.321, -0.595, -0.597, 0.047, 0.046, 0.046];
const ACETATE_ANTECHAMBER: [f64; 7] = [
    -0.268, 0.321, -0.596, -0.596, 0.046_333, 0.046_333, 0.046_333,
];

/// A bond-order- and charge-aware Weisfeiler-Leman refinement — the partition a
/// graph-hash / automorphism engine computes, and the one `graph_hash` computes
/// (element + degree + charge + aromatic flag, refined over bond-order-labeled
/// edges). WL colors are **coarser** than true automorphism orbits, so if WL already
/// separates two atoms, every orbit partition separates them too: this is a sound
/// witness that orbits are *strictly finer* than the path-score classes.
fn wl_orbits(mol: &Atomistic, ids: &[AtomId]) -> Vec<u64> {
    let index: HashMap<AtomId, usize> = ids
        .iter()
        .copied()
        .enumerate()
        .map(|(i, a)| (a, i))
        .collect();
    let mut adj: Vec<Vec<(usize, u64)>> = vec![Vec::new(); ids.len()];
    for (_, bond) in mol.bonds() {
        let (i, j) = (index[&bond.nodes[0]], index[&bond.nodes[1]]);
        let order = bond
            .props
            .get(keys::ORDER)
            .and_then(molrs::PropValue::as_f64)
            .unwrap_or(1.0);
        adj[i].push((j, order.to_bits()));
        adj[j].push((i, order.to_bits()));
    }
    // Initial color: element + formal charge (what `graph_hash::initial_color` folds in).
    let mut color: Vec<u64> = ids
        .iter()
        .map(|aid| {
            let atom = mol.get_atom(*aid).expect("atom");
            let el = atom.get_str(keys::ELEMENT).unwrap_or("").len() as u64 * 131
                + u64::from(
                    atom.get_str(keys::ELEMENT)
                        .and_then(|s| s.bytes().next())
                        .unwrap_or(0),
                );
            let charge = atom.get_int("formal_charge").unwrap_or(0);
            el.wrapping_mul(1_000_003).wrapping_add(charge as u64)
        })
        .collect();
    for _ in 0..ids.len() {
        let next: Vec<u64> = (0..ids.len())
            .map(|i| {
                let mut nbrs: Vec<(u64, u64)> =
                    adj[i].iter().map(|(j, ob)| (*ob, color[*j])).collect();
                nbrs.sort_unstable();
                let mut h = color[i].wrapping_mul(1_000_003);
                for (ob, c) in nbrs {
                    h = h
                        .wrapping_mul(31)
                        .wrapping_add(ob)
                        .wrapping_mul(31)
                        .wrapping_add(c);
                }
                h
            })
            .collect();
        if next == color {
            break;
        }
        color = next;
    }
    color
}

// ── ac-002: the classes come from path scores, NOT automorphism orbits ───────
//
// Acetate is the molecule where the two engines part company. The path score reads
// only atomic numbers and connectivity, so the two oxygens — topologically identical
// on the carboxyl carbon — are ONE class. An orbit / graph-hash partition reads the
// bond orders (2.0 vs 1.0) and the formal charge (0 vs -1) and puts them in TWO.
//
// antechamber merges them, and so must molrs.
#[test]
fn path_scores_merge_atoms_that_automorphism_orbits_split() {
    let (mol, ids) = acetate();
    let (o_double, o_anion) = (ids[2], ids[3]);

    // 1. Orbits are STRICTLY FINER: WL (already coarser than true orbits) splits
    //    the two oxygens.
    let orbits = wl_orbits(&mol, &ids);
    assert_ne!(
        orbits[2], orbits[3],
        "the fixture is wrong: a bond-order-aware refinement must split C=O from C-O-"
    );

    // 2. The path score MERGES them — as antechamber does.
    let classes = find_equivalence_classes(&mol, EquivalenceOptions::bcc());
    assert_eq!(
        classes.class_of(o_double),
        classes.class_of(o_anion),
        "the two carboxylate oxygens must be ONE class; splitting them is the orbit answer"
    );

    // 3. And the two engines really do give different charges. Path-score averaging
    //    lands on antechamber; orbit averaging would leave the sqm values untouched,
    //    keeping a symmetry-broken carboxylate.
    let mut with_raw = mol.clone();
    for (aid, q) in ids.iter().zip(ACETATE_RAW) {
        with_raw.set_atom(*aid, keys::CHARGE, q).expect("charge");
    }
    let averaged = average_charges(&with_raw, &classes);
    let q = |m: &Atomistic, id: AtomId| m.get_atom(id).unwrap().get_f64(keys::CHARGE).unwrap();

    assert_eq!(
        q(&averaged, o_double).to_bits(),
        q(&averaged, o_anion).to_bits(),
        "merged oxygens must carry bit-identical charge"
    );
    for (k, want) in ACETATE_ANTECHAMBER.iter().enumerate() {
        let got = q(&averaged, ids[k]);
        assert!(
            (got - want).abs() < 1.0e-4,
            "atom {k}: molrs {got:+.6}, antechamber {want:+.6}"
        );
    }
    // The divergence an orbit engine would ship: -0.595 / -0.597, still split.
    assert!(
        (ACETATE_RAW[2] - ACETATE_ANTECHAMBER[2]).abs() > 5.0e-4,
        "if the raw and equivalenced oxygens agreed, this test would prove nothing"
    );
}

// ── The methyl hydrogens are the same story on the other end: the path score
// merges the three H of a methyl group, which orbits also merge. That agreement is
// what makes acetate's oxygens the sharp case.
#[test]
fn methyl_hydrogens_are_one_class() {
    let (mol, ids) = acetate();
    let classes = find_equivalence_classes(&mol, EquivalenceOptions::bcc());
    let first = classes.class_of(ids[4]).expect("class");
    assert_eq!(classes.class_of(ids[5]), Some(first));
    assert_eq!(classes.class_of(ids[6]), Some(first));
    // 4 classes: methyl C | carboxyl C | both O | all three H.
    assert_eq!(classes.n_classes(), 4);
}

// ── The builder contract: graph in / graph out, non-mutating, class id on every
// atom (the same contract every other `find_*` honours).
#[test]
fn the_builder_annotates_a_clone_and_leaves_the_input_alone() {
    let (mol, ids) = acetate();
    let perceived = Perceive::new().find_equivalence_classes(&mol);

    assert_eq!(perceived.n_atoms(), mol.n_atoms());
    for id in &ids {
        assert!(
            mol.get_atom(*id).unwrap().get_int("equiv_class").is_none(),
            "the input graph was mutated"
        );
        assert!(
            perceived
                .get_atom(*id)
                .unwrap()
                .get_int("equiv_class")
                .is_some(),
            "every atom must carry a class id"
        );
    }
    // The two carboxylate oxygens share their id; the two carbons do not.
    let class = |id: AtomId| {
        perceived
            .get_atom(id)
            .unwrap()
            .get_int("equiv_class")
            .unwrap()
    };
    assert_eq!(class(ids[2]), class(ids[3]));
    assert_ne!(class(ids[0]), class(ids[1]));
}

// ── `-eq 0` is a real level, not a synonym for "compute anyway": every atom is its
// own class, so averaging cannot move a charge.
#[test]
fn eq_zero_leaves_every_atom_in_its_own_class() {
    let (mol, ids) = acetate();
    let classes = find_equivalence_classes(&mol, EquivalenceOptions::off());
    assert_eq!(classes.n_classes(), mol.n_atoms());
    assert_ne!(classes.class_of(ids[2]), classes.class_of(ids[3]));
}
