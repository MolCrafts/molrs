//! ac-007 (runtime half) — the labels come from the ONE correct resolver.
//!
//! `typifier/mmff/classify.rs` is a second implementation of MMFF's context
//! rules, and it is **wrong**. `ff/mmff/params.rs` (the moved `energy/params.rs`)
//! is the first one, and it is right — it is the RDKit-faithful port, and the
//! 11/11 fixture parity that mmff-orthogonal-01 established is parity with the
//! numbers *it* resolves.
//!
//! After the deletion, `frame_builder` derives every bond / angle / dihedral type
//! label from the resolver. The labels are then only provenance — the six
//! per-instance kernels read Frame *columns*, not labels — but "only provenance"
//! is not a licence to be wrong, and today they are wrong in a way that is easy
//! to state and easy to check.
//!
//! # What the deleted classifiers get wrong
//!
//! **`typify_bond` is backwards.** RDKit's `getMMFFBondType` returns 1 only for a
//! bond that is `Bond::SINGLE` *and* joins two `sbmb`/`arom` atom types. After
//! MMFF aromaticity perception an aromatic ring bond is `AROMATIC`, never
//! `SINGLE` — so an aromatic bond is bond type **0**. `classify.rs:10-12` returns
//! **1** for it, having tested the raw bond order for ~1.5. And on a *Kekulé*
//! input (which is what an SDF gives you) it does something even stranger: the
//! three "single" ring bonds of benzene take the `sbmb && sbmb` branch and come
//! out as type 1, while the three "double" ones come out as type 0. The same
//! six-fold-symmetric ring, labelled two different ways.
//!
//! **`typify_angle`'s signature cannot express the rule.** It takes
//! `(bt_ij, bt_jk)` — two bond types, and nothing else. RDKit's
//! `getMMFFAngleType` needs the *topology*: an angle inside a 3- or 4-membered
//! ring is promoted to angle type 3..8. Two integers cannot say whether an angle
//! is in a ring, so the function cannot be fixed without changing its signature —
//! which is another way of saying it was never right. Cyclopropane below is the
//! proof: every C-C-C angle in it is angle type **3**, and `classify.rs` can only
//! ever return 0, 1 or 2.
//!
//! **`typify_dihedral` has the same disease** (no 4-/5-ring torsion types).
//!
//! # Why these two molecules
//!
//! Benzene is the aromatic case, and its answer is *uniform*: every one of its
//! bonds, angles and dihedrals is type 0. Cyclopropane is the ring-rule case, and
//! its C-C-C angle type (3) is a value `classify.rs` has no way to produce.
//! Together they pin both defects with values taken from RDKit's rules, not from
//! molrs's own output.
//!
//! # The label grammar these tests depend on
//!
//! `frame_builder` documents it: bond `"{bond_type}_{ti}_{tj}"`, angle
//! `"{angle_type}_{ti}_{tj}_{tk}"`, dihedral `"{torsion_type}_{ti}_{tj}_{tk}_{tl}"`.
//! The assertions read the **first** `_`-separated field. Keeping the type code
//! first is therefore part of the contract these tests hold.

use molrs::Atomistic;
use molrs::ff::typifier::mmff::MMFF94Typifier;
use molrs::store::frame::Frame;
use molrs::system::molgraph::PropValue;
use molrs::types::F;
use molrs::{AtomId, Atomistic as Mol};

fn bond(mol: &mut Mol, a: AtomId, b: AtomId, order: F) {
    let bid = mol.add_bond(a, b).expect("add bond");
    mol.set_bond_prop(bid, "order", PropValue::F64(order))
        .expect("set bond order");
}

/// Benzene, Kekulé — alternating single / double ring bonds.
///
/// This is what every real input gives you: an SDF mol block, a SMILES parse, a
/// PDB with CONECT records. MMFF perceives aromaticity itself
/// (`set_mmff_aromaticity`), and after that the ring bonds are AROMATIC — which
/// is exactly the fact `classify.rs` never sees, because it is handed the raw
/// bond order instead of the perceived topology.
fn benzene() -> Mol {
    let mut mol = Mol::new();
    let r_c = 1.39;
    let r_h = 2.48;
    let carbons: Vec<AtomId> = (0..6)
        .map(|i| {
            let a = std::f64::consts::TAU * (i as f64) / 6.0;
            mol.add_atom_xyz("C", r_c * a.cos(), r_c * a.sin(), 0.0)
        })
        .collect();
    let hydrogens: Vec<AtomId> = (0..6)
        .map(|i| {
            let a = std::f64::consts::TAU * (i as f64) / 6.0;
            mol.add_atom_xyz("H", r_h * a.cos(), r_h * a.sin(), 0.0)
        })
        .collect();
    for i in 0..6 {
        // Kekulé: 1,2 / 3,4 / 5,6 double.
        let order = if i % 2 == 0 { 2.0 } else { 1.0 };
        bond(&mut mol, carbons[i], carbons[(i + 1) % 6], order);
        bond(&mut mol, carbons[i], hydrogens[i], 1.0);
    }
    mol
}

/// Cyclopropane (C3H6) — the smallest 3-membered ring.
///
/// Every C-C-C angle is inside a 3-ring, so `getMMFFAngleType` promotes it to
/// angle type 3. No combination of bond types can produce that, which is the
/// whole point.
fn cyclopropane() -> Mol {
    let mut mol = Mol::new();
    let r = 0.87; // C-C = 1.51 A -> circumradius
    let carbons: Vec<AtomId> = (0..3)
        .map(|i| {
            let a = std::f64::consts::TAU * (i as f64) / 3.0;
            mol.add_atom_xyz("C", r * a.cos(), r * a.sin(), 0.0)
        })
        .collect();
    for i in 0..3 {
        bond(&mut mol, carbons[i], carbons[(i + 1) % 3], 1.0);
        let a = std::f64::consts::TAU * (i as f64) / 3.0;
        // Two H per carbon, above and below the ring plane.
        for z in [0.89, -0.89] {
            let h = mol.add_atom_xyz("H", 1.62 * a.cos(), 1.62 * a.sin(), z);
            bond(&mut mol, carbons[i], h, 1.0);
        }
    }
    mol
}

fn typed(mol: &Atomistic) -> Frame {
    MMFF94Typifier::new()
        .typify(mol)
        .expect("typify")
        .to_frame()
}

/// The leading `_`-separated field of every `type` label in `block`.
fn type_codes(frame: &Frame, block: &str) -> Vec<u32> {
    let b = frame
        .get(block)
        .unwrap_or_else(|| panic!("typed frame has no `{block}` block"));
    let labels = b
        .get_string("type")
        .unwrap_or_else(|| panic!("`{block}` block has no `type` column"));
    assert!(
        !labels.is_empty(),
        "`{block}` block is empty — an assertion over zero rows proves nothing"
    );
    labels
        .iter()
        .map(|l| {
            l.split('_')
                .next()
                .unwrap_or_default()
                .parse::<u32>()
                .unwrap_or_else(|_| {
                    panic!(
                        "`{block}` type label {l:?} does not start with a numeric MMFF type code. \
                         The label grammar is `{{type_code}}_{{atom_types...}}` — these tests, \
                         and every human reading a typed frame, depend on the code coming first"
                    )
                })
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Benzene — the aromatic case. Every code is 0.
// ---------------------------------------------------------------------------

/// RDKit: an aromatic bond is **bond type 0**.
///
/// `getMMFFBondType` requires `Bond::SINGLE`; after MMFF aromaticity perception a
/// benzene ring bond is `AROMATIC`. molrs's own RDKit-faithful port says the same
/// thing, in as many words (`ff/mmff/charges.rs:32-38`):
///
/// > "After MMFF aromaticity perception, aromatic ring bonds have type AROMATIC
/// > (not SINGLE), so they never get bond type 1 — only a genuine single bond
/// > joining two sbmb/arom atoms does (e.g. the inter-ring bond in biphenyl)."
///
/// The C-H bonds are type 0 too (H is neither `sbmb` nor `arom`). So **all twelve
/// bonds of benzene are type 0**, and the label census is a single number.
///
/// RED today: `classify.rs` returns 1 for the three Kekulé single ring bonds
/// (both carbons are type 37, `sbmb = 1`) and 0 for the three double ones —
/// labelling a six-fold-symmetric ring two different ways.
#[test]
fn every_benzene_bond_is_mmff_bond_type_0() {
    let frame = typed(&benzene());
    let codes = type_codes(&frame, "bonds");
    assert_eq!(codes.len(), 12, "benzene has 6 C-C + 6 C-H bonds");

    let wrong: Vec<u32> = codes.iter().copied().filter(|&c| c != 0).collect();
    assert!(
        wrong.is_empty(),
        "benzene bond type codes are {codes:?}, want all 0.\n\
         RDKit's `getMMFFBondType` returns 1 only for a bond that is SINGLE *and* joins two \
         sbmb/arom types. After MMFF aromaticity perception a benzene ring bond is AROMATIC, so \
         it is bond type 0 — and the C-H bonds are 0 as well.\n\
         The {} non-zero code(s) come from the Kekulé input: `classify.rs` reads the raw bond \
         order, so the three ring bonds that happen to be drawn as single take the \
         `sbmb && sbmb` branch and come out as type 1, while the three drawn as double come out \
         as 0. Same ring, same six carbons, two different answers.",
        wrong.len()
    );
}

/// RDKit: every benzene angle is **angle type 0**.
///
/// `getMMFFAngleType`: `at = bt_ij + bt_jk` (both 0, per the bond test above),
/// then promoted only if the angle sits in a 3- or 4-membered ring. Benzene's is
/// a 6-ring, so nothing is promoted. C-C-C and C-C-H alike: type 0.
#[test]
fn every_benzene_angle_is_mmff_angle_type_0() {
    let frame = typed(&benzene());
    let codes = type_codes(&frame, "angles");

    let wrong: Vec<u32> = codes.iter().copied().filter(|&c| c != 0).collect();
    assert!(
        wrong.is_empty(),
        "benzene angle type codes are {codes:?}, want all 0.\n\
         `getMMFFAngleType` sums the two bond types (both 0 for an aromatic ring bond) and \
         promotes only inside a 3-/4-membered ring. Benzene has neither.\n\
         The {} non-zero code(s) are `classify.rs::typify_angle(bt_ij, bt_jk)` propagating the \
         bogus bond type 1 it invented for the Kekulé single bonds.",
        wrong.len()
    );
}

/// RDKit: every benzene dihedral is **torsion type 0**.
///
/// `getMMFFTorsionType`: `tt = bt_jk` (0, aromatic central bond); the `tt = 2`
/// branch needs the central bond to be SINGLE, which an aromatic bond is not; and
/// the 4-/5-ring promotions need a 4- or 5-membered ring, which benzene is not.
#[test]
fn every_benzene_dihedral_is_mmff_torsion_type_0() {
    let frame = typed(&benzene());
    let codes = type_codes(&frame, "dihedrals");

    let wrong: Vec<u32> = codes.iter().copied().filter(|&c| c != 0).collect();
    assert!(
        wrong.is_empty(),
        "benzene dihedral type codes are {codes:?}, want all 0.\n\
         The central bond of every benzene torsion is an aromatic ring bond: bond type 0, and \
         not SINGLE — so neither `tt = bt_jk` nor the `tt = 2` branch fires, and a 6-ring \
         triggers no 4-/5-ring promotion.\n\
         The {} non-zero code(s) are `classify.rs::typify_dihedral` keying off the same bogus \
         bond type 1.",
        wrong.len()
    );
}

// ---------------------------------------------------------------------------
// Cyclopropane — the ring rule the classifier's SIGNATURE cannot express
// ---------------------------------------------------------------------------

/// RDKit: a C-C-C angle inside a 3-membered ring is **angle type 3**.
///
/// `getMMFFAngleType` calls `isAngleInRingOfSize3or4`, and when the answer is 3
/// it *overwrites* the bond-type sum: `at = size`. Cyclopropane's three ring
/// angles are the canonical case.
///
/// This is the assertion that shows `typify_angle(bt_ij, bt_jk)` is not merely
/// buggy but **unfixable in its own signature**: its two arguments are bond types,
/// both 0 here, and no function of two zeroes can return 3. The ring membership is
/// simply not among its inputs. It can only ever answer 0, 1 or 2 — which is why
/// its match arms stop there.
///
/// The six H-C-C angles and three H-C-H angles are NOT in the ring (the two ring
/// carbons are not bonded to the hydrogen), so they stay type 0. Both halves are
/// asserted: a test that only demanded "some angle is type 3" would be satisfied
/// by a classifier that stamped 3 on everything.
#[test]
fn cyclopropane_ring_angles_are_mmff_angle_type_3() {
    let frame = typed(&cyclopropane());
    let angles = frame.get("angles").expect("angles block");
    let labels = angles.get_string("type").expect("angle type column");
    let atomi = angles.get_uint("atomi").expect("atomi");
    let atomj = angles.get_uint("atomj").expect("atomj");
    let atomk = angles.get_uint("atomk").expect("atomk");

    // Atom order: carbons are added first (indices 0,1,2), then hydrogens.
    let is_carbon = |i: u32| i < 3;

    let (mut ring, mut other) = (Vec::new(), Vec::new());
    for n in 0..labels.len() {
        let code: u32 = labels[n]
            .split('_')
            .next()
            .unwrap_or_default()
            .parse()
            .expect("numeric angle type code");
        if is_carbon(atomi[n]) && is_carbon(atomj[n]) && is_carbon(atomk[n]) {
            ring.push(code);
        } else {
            other.push(code);
        }
    }

    assert_eq!(
        ring.len(),
        3,
        "cyclopropane has exactly three C-C-C angles; found {}",
        ring.len()
    );
    assert!(
        ring.iter().all(|&c| c == 3),
        "cyclopropane C-C-C angle type codes are {ring:?}, want all 3.\n\
         RDKit's `getMMFFAngleType` calls `isAngleInRingOfSize3or4` and, on a hit, OVERWRITES \
         the bond-type sum with the ring size: `at = size` = 3.\n\
         `classify.rs::typify_angle(bt_ij, bt_jk)` cannot produce this value at any input. Its \
         arguments are two bond types — here both 0 — and ring membership is not among them. \
         Its match arms stop at 2 because they have to: the signature itself cannot see the \
         ring. This is not a bug to be fixed in place; it is a function that was never able to \
         be right."
    );
    assert!(
        other.iter().all(|&c| c == 0),
        "cyclopropane's H-C-C / H-C-H angle type codes are {other:?}, want all 0. They are NOT \
         in the ring (a hydrogen is bonded to one carbon only), so no promotion applies. If \
         these came out as 3 as well, the ring rule is being stamped on everything rather than \
         evaluated."
    );
}

/// The resolver's labels stay in step with the parameters it bakes.
///
/// A weaker but broader statement than the two molecules above, and it holds for
/// every fixture: the label a row carries and the numbers on that same row come
/// from the same resolver call, so a bond that resolved real parameters cannot be
/// carrying a label from a different rule set. Today the numbers already come
/// from `eparams` (`frame_builder.rs:115`) while the labels come from
/// `classify.rs` — two rule sets on the same row.
///
/// Benzene's C-C bonds all resolve `kb`/`r0` from the same aromatic-bond MMFF row,
/// so they must all carry the same label. Today three of them do not.
#[test]
fn benzene_symmetric_bonds_carry_symmetric_labels() {
    let frame = typed(&benzene());
    let bonds = frame.get("bonds").expect("bonds block");
    let labels = bonds.get_string("type").expect("bond type column");
    let kb = bonds.get_float("kb").expect("kb column");
    let atomi = bonds.get_uint("atomi").expect("atomi");
    let atomj = bonds.get_uint("atomj").expect("atomj");

    // The six C-C ring bonds. `benzene()` adds all six carbons first, so a bond
    // is a ring bond iff both its atoms are indices 0..=5.
    let ring: Vec<usize> = (0..labels.len())
        .filter(|&n| atomi[n] < 6 && atomj[n] < 6)
        .collect();
    assert_eq!(ring.len(), 6, "benzene has six C-C ring bonds");

    let kbs: Vec<f64> = ring.iter().map(|&n| kb[n]).collect();
    let first_kb = kbs[0];
    assert!(
        kbs.iter().all(|k| (k - first_kb).abs() < 1e-12),
        "benzene's six C-C bonds resolved DIFFERENT force constants {kbs:?} — the resolver \
         itself is broken, not just the labels"
    );

    let names: Vec<&String> = ring.iter().map(|&n| &labels[n]).collect();
    let first = names[0];
    assert!(
        names.iter().all(|l| *l == first),
        "benzene's six C-C ring bonds resolved the SAME force constant ({first_kb}) but carry \
         {} different labels: {names:?}.\n\
         The numbers come from `ff/mmff/params.rs` (the RDKit-faithful resolver) and the labels \
         come from `classify.rs` — two different rule sets stamped on the same row. That is the \
         duplication this spec deletes: after it, both come from the resolver, and a row whose \
         parameters are identical cannot claim to be a different type.",
        names
            .iter()
            .collect::<std::collections::BTreeSet<_>>()
            .len()
    );
}
