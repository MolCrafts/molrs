//! BCC atom typing and correction-table integration tests.
//!
//! What survived the push-API rework, and why it lives here rather than in
//! `ff/charge/`: this file tests the two pieces the BCC *typifier* owns — the
//! `ATOMTYPE_BCC.DEF` atom labels and the `BCCPARM.DAT` lookup table — through the
//! objects that own them ([`BCCAtomTypifier`], [`BCCCorrectionTable`],
//! [`BCCCorrector`]). The charge MODEL that composes them (`BccModel`, the push
//! API) is tested in `ff/charge/`, next to the source it now lives in.
//!
//! The `FakeAM1Backend` that used to open this file is gone with the trait it
//! implemented. It was the tell: to reach the BCC table, a test — and, worse,
//! `molrs-cxxapi` in production — had to dress a `Vec<f64>` up as an "AM1 backend"
//! and smuggle it through a `compute_am1_charges(&mol)` whose `mol` it discarded.
//! Every assertion that shim carried is preserved below or in `ff/charge/`, on an
//! object a caller can actually build.

use molrs::ff::params::BccCorrectionRow;
use molrs::ff::typifier::am1bcc::{BCCAtomTypifier, BCCCorrectionTable, BCCCorrector};
use molrs::perceive::bond_type::BCC_BOND_TYPE;
use molrs::store::keys;
use molrs::{AtomId, Atomistic, Element};

fn methane() -> (Atomistic, AtomId, Vec<AtomId>) {
    let mut mol = Atomistic::new();
    let c = mol.add_atom_xyz("C", 0.0, 0.0, 0.0);
    let mut hs = Vec::new();
    for [x, y, z] in [
        [0.63, 0.63, 0.63],
        [-0.63, -0.63, 0.63],
        [-0.63, 0.63, -0.63],
        [0.63, -0.63, -0.63],
    ] {
        let h = mol.add_atom_xyz("H", x, y, z);
        mol.add_bond(c, h).expect("add C-H");
        hs.push(h);
    }
    (mol, c, hs)
}

/// Methane already carrying BCC labels — the input [`BCCCorrector`] contracts for.
///
/// The corrector is the LOW-level half of the model: it applies a table to a
/// molecule that is already typed, so the types are its argument, not its job.
/// (`BccModel::correct` is the high-level half, and types the molecule itself —
/// which is why it must never be handed a molecule pre-typed like this one.)
fn bcc_typed_methane() -> (Atomistic, AtomId, Vec<AtomId>) {
    let (mut mol, c, hs) = methane();
    mol.set_atom(c, keys::TYPE, "11")
        .expect("set carbon BCC type");
    for h in &hs {
        mol.set_atom(*h, keys::TYPE, "91")
            .expect("set hydrogen BCC type");
    }
    let bond_ids: Vec<_> = mol.bonds().map(|(bid, _)| bid).collect();
    for bid in bond_ids {
        mol.set_bond_prop(bid, BCC_BOND_TYPE, 1.0)
            .expect("set BCC bond type");
    }
    (mol, c, hs)
}

/// The AM1 base charges an atom-ordered `Atomistic` carries, in atom order.
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

/// A one-row synthetic correction table: C (`11`) – H (`91`) single bond, δ = +0.01 e.
///
/// The δ is deliberately **not** the embedded table's 0.0393: a corrector that
/// quietly fell back to `BCCPARM.DAT` instead of applying the table it was handed
/// would fail the charge assertions below rather than pass them.
///
/// Applied to methane, carbon is the `left` of four such bonds (+4 × 0.01 = +0.04 e)
/// and each hydrogen the `right` of one (−0.01 e); the molecule stays neutral.
fn synthetic_ch_table() -> BCCCorrectionTable {
    BCCCorrectionTable::from_rows(&[BccCorrectionRow {
        left: "11",
        right: "91",
        bond_type: 1,
        delta: 0.01,
    }])
}

/// A table built from explicit rows holds exactly those rows — nothing else.
///
/// `from_rows` is the constructor that replaces the empty-`HashMap`
/// `BCCCorrectionTable::new()` (ac-004), so the property that matters is that it
/// *names its content*: what goes in comes out, and no parameter set leaks in
/// behind the caller's back.
#[test]
fn from_rows_builds_exactly_the_rows_it_is_given() {
    let table = synthetic_ch_table();

    assert_eq!(table.len(), 1, "one row in, one row out");
    let dq = table
        .correction("11", "91", 1)
        .expect("the row the table was built from");
    assert!((dq - 0.01).abs() < 1e-12, "{dq}");

    let reversed = table
        .correction("91", "11", 1)
        .expect("a reversed bond is the same row");
    assert!((reversed + 0.01).abs() < 1e-12, "{reversed}");

    assert!(
        table.correction("11", "31", 1).is_none(),
        "C–O is a row of the embedded BCC table, not of this one — `from_rows` \
         must not fall back to a parameter set the caller never named"
    );
}

/// The corrector applies the table it was handed — all of it, and only it.
///
/// This is what `fake_backend_plus_explicit_bcc_table_writes_charges` asserted,
/// with the fake backend (which existed only to inject `vec![0.0; 5]`) replaced by
/// setting those base charges directly. The numbers are unchanged: with a base of
/// zero and a synthetic +0.01 C–H row, carbon is the `left` of four bonds (+0.04)
/// and each hydrogen the `right` of one (−0.01), and the molecule stays neutral.
#[test]
fn corrector_applies_the_explicit_table_it_was_given() {
    let (mut mol, c, hs) = bcc_typed_methane();
    let ids: Vec<AtomId> = mol.atoms().map(|(aid, _)| aid).collect();
    for aid in &ids {
        mol.set_atom(*aid, keys::CHARGE, 0.0).expect("set am1 base");
    }

    BCCCorrector::new(synthetic_ch_table())
        .apply(&mut mol)
        .expect("apply the synthetic table");

    let c_atom = mol.get_atom(c).expect("carbon");
    assert_eq!(
        c_atom.get_str(keys::TYPE),
        Some("11"),
        "the corrector must not relabel the atoms it was handed"
    );
    assert!((c_atom.get_f64(keys::CHARGE).unwrap() - 0.04).abs() < 1e-12);
    for h in hs {
        let h_atom = mol.get_atom(h).expect("hydrogen");
        assert_eq!(h_atom.get_str(keys::TYPE), Some("91"));
        assert!((h_atom.get_f64(keys::CHARGE).unwrap() + 0.01).abs() < 1e-12);
    }
    let total: f64 = charges_of(&mol, &ids).iter().sum();
    assert!(
        total.abs() < 1e-12,
        "pairwise-antisymmetric increments conserve the total charge: {total}"
    );
}

/// A bond whose row is absent from the table is an ERROR — never a silent 0.0.
///
/// The table here is **populated** and real (two rows copied verbatim from
/// `BCCPARM.DAT`); it simply does not cover the only bond methane has. It knows
/// C–C and C–O single bonds, and methane is all C–H.
///
/// This used to be spelled with an *empty* table, which conflated two different
/// failures: "this model was built with no parameter set" (the ac-004 footgun —
/// now unconstructible) and "this row is not in the parameter set" (a real,
/// permanent error path). Only the latter is pinned here, and it is pinned on the
/// object a user can actually build. Its typed-error twin — the same failure
/// reaching a caller as `ChargeError::MissingCorrection` rather than a `String` —
/// is in `ff/charge/bcc.rs`.
#[test]
fn missing_bcc_correction_row_is_an_error_not_zero() {
    let (mut mol, _, _) = bcc_typed_methane();
    let ids: Vec<AtomId> = mol.atoms().map(|(aid, _)| aid).collect();
    for aid in &ids {
        mol.set_atom(*aid, keys::CHARGE, 0.0).expect("set am1 base");
    }
    let table = BCCCorrectionTable::from_rows(&[
        BccCorrectionRow {
            left: "11",
            right: "11",
            bond_type: 1,
            delta: 0.0000,
        },
        BccCorrectionRow {
            left: "11",
            right: "31",
            bond_type: 1,
            delta: 0.0718,
        },
    ]);
    assert!(
        !table.is_empty(),
        "the premise of this test is a populated table that lacks ONE row; an \
         empty table would test the deleted footgun instead"
    );

    let err = BCCCorrector::new(table)
        .apply(&mut mol)
        .expect_err("an uncorrectable bond must be rejected, not silently left at 0.0");
    assert!(err.contains("missing BCC correction"), "{err}");
    assert!(
        err.contains("11|91|1") || err.contains("91|11|1"),
        "the error must name the C–H bond it could not correct, got: {err}"
    );
}

#[test]
fn atom_typifier_assigns_bcc_types_to_untyped_methane() {
    let (mol, c, hs) = methane();
    let typed = BCCAtomTypifier::bcc()
        .typify(&mol)
        .expect("typify methane atoms");

    assert_eq!(typed.get_atom(c).unwrap().get_str(keys::TYPE), Some("11"));
    for h in hs {
        assert_eq!(typed.get_atom(h).unwrap().get_str(keys::TYPE), Some("91"));
    }
    for (_, bond) in typed.bonds() {
        assert_eq!(
            bond.props.get(BCC_BOND_TYPE).and_then(|v| v.as_f64()),
            Some(1.0)
        );
    }
}

#[test]
fn atom_typifier_covers_reference_element_rows_through_mt() {
    for z in 1..=109 {
        if z == 16 {
            continue;
        }
        let element = Element::by_number(z).expect("element");
        let mut mol = Atomistic::new();
        let aid = mol.add_atom_xyz(element.symbol(), 0.0, 0.0, 0.0);
        let typed = BCCAtomTypifier::bcc()
            .typify(&mol)
            .unwrap_or_else(|e| panic!("z={z} {}: {e}", element.symbol()));
        assert!(typed.get_atom(aid).unwrap().get_str(keys::TYPE).is_some());
    }
}
