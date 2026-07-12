//! AM1-BCC typifier integration tests.

use molrs::ff::charge::ChargeAssigner;
use molrs::ff::params::BccCorrectionRow;
use molrs::ff::typifier::Typifier;
use molrs::ff::typifier::am1bcc::{
    AM1BCCTypifier, AM1ChargeBackend, AM1ChargeResult, AM1ChargeTypifier, BCCAtomTypifier,
    BCCCorrectionTable, UnavailableAM1Backend,
};
use molrs::store::keys;
use molrs::{AtomId, Atomistic, Element};

#[derive(Clone)]
struct FakeAM1Backend {
    charges: Vec<f64>,
}

impl AM1ChargeBackend for FakeAM1Backend {
    fn compute_am1_charges(&self, _mol: &Atomistic) -> Result<AM1ChargeResult, String> {
        Ok(AM1ChargeResult {
            charges: self.charges.clone(),
            total_charge: None,
            heat_of_formation_kcal_mol: None,
            reference: "test fake backend".to_owned(),
        })
    }
}

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
        mol.set_bond_prop(bid, keys::TYPE, 1.0)
            .expect("set BCC bond type");
    }
    (mol, c, hs)
}

fn methanol() -> Atomistic {
    let mut mol = Atomistic::new();
    let c = mol.add_atom_xyz("C", 0.0, 0.0, 0.0);
    let o = mol.add_atom_xyz("O", 1.43, 0.0, 0.0);
    let h1 = mol.add_atom_xyz("H", -0.63, 0.63, 0.63);
    let h2 = mol.add_atom_xyz("H", -0.63, -0.63, 0.63);
    let h3 = mol.add_atom_xyz("H", -0.63, 0.0, -0.89);
    let ho = mol.add_atom_xyz("H", 1.76, 0.89, 0.0);
    for (a, b) in [(c, o), (c, h1), (c, h2), (c, h3), (o, ho)] {
        mol.add_bond(a, b).expect("add bond");
    }
    mol
}

#[allow(dead_code)] // WIP fixture — kept for the upcoming parity cases
fn bcc_typed_methanol() -> Atomistic {
    let mut mol = methanol();
    let atoms: Vec<_> = mol.atoms().map(|(aid, _)| aid).collect();
    for (aid, bcc_type) in [
        (atoms[0], "11"),
        (atoms[1], "31"),
        (atoms[2], "91"),
        (atoms[3], "91"),
        (atoms[4], "91"),
        (atoms[5], "91"),
    ] {
        mol.set_atom(aid, keys::TYPE, bcc_type)
            .expect("set BCC atom type");
    }
    let bond_ids: Vec<_> = mol.bonds().map(|(bid, _)| bid).collect();
    for bid in bond_ids {
        mol.set_bond_prop(bid, keys::TYPE, 1.0)
            .expect("set BCC bond type");
    }
    mol
}

fn chloromethane() -> Atomistic {
    let mut mol = Atomistic::new();
    let c = mol.add_atom_xyz("C", 0.0, 0.0, 0.0);
    let cl = mol.add_atom_xyz("Cl", 1.78, 0.0, 0.0);
    let h1 = mol.add_atom_xyz("H", -0.63, 0.63, 0.63);
    let h2 = mol.add_atom_xyz("H", -0.63, -0.63, 0.63);
    let h3 = mol.add_atom_xyz("H", -0.63, 0.0, -0.89);
    for (a, b) in [(c, cl), (c, h1), (c, h2), (c, h3)] {
        mol.add_bond(a, b).expect("add bond");
    }
    mol
}

#[allow(dead_code)] // WIP fixture — kept for the upcoming parity cases
fn bcc_typed_chloromethane() -> Atomistic {
    let mut mol = chloromethane();
    let atoms: Vec<_> = mol.atoms().map(|(aid, _)| aid).collect();
    for (aid, bcc_type) in [
        (atoms[0], "11"),
        (atoms[1], "72"),
        (atoms[2], "91"),
        (atoms[3], "91"),
        (atoms[4], "91"),
    ] {
        mol.set_atom(aid, keys::TYPE, bcc_type)
            .expect("set BCC atom type");
    }
    let bond_ids: Vec<_> = mol.bonds().map(|(bid, _)| bid).collect();
    for bid in bond_ids {
        mol.set_bond_prop(bid, keys::TYPE, 1.0)
            .expect("set BCC bond type");
    }
    mol
}

/// A one-row synthetic correction table: C (`11`) – H (`91`) single bond, δ = +0.01 e.
///
/// The δ is deliberately **not** the embedded table's 0.0393: a typifier that
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

/// No AM1 backend configured → a clear error, never an invented charge.
///
/// This was spelled `AM1BCCTypifier::default()`, which asserted the behaviour of a
/// typifier with *neither* a backend *nor* a parameter set. The missing parameter
/// set was the ac-004 footgun; the missing backend is the real guarantee, and it
/// stands on its own — here the correction table is the full embedded BCC set, so
/// the backend guard is the only thing that can fire.
#[test]
fn am1bcc_without_an_am1_backend_is_an_error() {
    let (mol, _, _) = methane();
    let typifier = AM1BCCTypifier::bcc(UnavailableAM1Backend).expect("load embedded BCC table");

    let err = typifier
        .typify(&mol)
        .expect_err("no AM1 backend is configured");
    assert!(
        err.contains("AM1ChargeBackend"),
        "the error must name the backend that is missing, got: {err}"
    );
    assert!(
        !err.contains("missing BCC correction"),
        "the table is fully populated — the backend guard is what must fire, got: {err}"
    );
}

#[test]
fn am1_charge_typifier_writes_base_charges_without_bcc() {
    let (mol, c, hs) = methane();
    let typifier = AM1ChargeTypifier::new(FakeAM1Backend {
        charges: vec![-0.2, 0.05, 0.05, 0.05, 0.05],
    });

    let typed = typifier.typify(&mol).expect("typify AM1 charges");
    assert!((typed.get_atom(c).unwrap().get_f64(keys::CHARGE).unwrap() + 0.2).abs() < 1e-12);
    for h in hs {
        assert!((typed.get_atom(h).unwrap().get_f64(keys::CHARGE).unwrap() - 0.05).abs() < 1e-12);
    }
}

#[test]
fn fake_backend_plus_explicit_bcc_table_writes_charges() {
    let (mol, c, hs) = bcc_typed_methane();
    let typifier = AM1BCCTypifier::new(
        FakeAM1Backend {
            charges: vec![0.0; 5],
        },
        synthetic_ch_table(),
    );

    let typed = typifier.typify(&mol).expect("typify methane");
    let c_atom = typed.get_atom(c).expect("carbon");
    assert_eq!(c_atom.get_str(keys::TYPE), Some("11"));
    assert!((c_atom.get_f64(keys::CHARGE).unwrap() - 0.04).abs() < 1e-12);
    for h in hs {
        let h_atom = typed.get_atom(h).expect("hydrogen");
        assert_eq!(h_atom.get_str(keys::TYPE), Some("91"));
        assert!((h_atom.get_f64(keys::CHARGE).unwrap() + 0.01).abs() < 1e-12);
    }
    let total: f64 = typed
        .atoms()
        .map(|(_, a)| a.get_f64(keys::CHARGE).unwrap())
        .sum();
    assert!(total.abs() < 1e-12);
}

#[test]
fn charge_assigner_runs_struct_typifier_and_reports_result() {
    let (mut mol, _, _) = bcc_typed_methane();
    let typifier = AM1BCCTypifier::new(
        FakeAM1Backend {
            charges: vec![0.0; 5],
        },
        synthetic_ch_table(),
    );

    let result = ChargeAssigner::new(typifier)
        .with_method("AM1-BCC")
        .assign_atomistic(&mut mol)
        .expect("assign charges");
    assert_eq!(result.method, "AM1-BCC");
    assert_eq!(result.charges.len(), 5);
    assert!(result.total_charge.abs() < 1e-12);
    for (_, atom) in mol.atoms() {
        assert!(atom.get_f64(keys::CHARGE).is_some());
    }
}

#[test]
fn embedded_bcc_table_applies_known_methane_row() {
    let (mol, c, hs) = bcc_typed_methane();
    let typifier = AM1BCCTypifier::bcc(FakeAM1Backend {
        charges: vec![0.0; 5],
    })
    .expect("load embedded BCC table");

    let typed = typifier.typify(&mol).expect("typify methane");
    assert!((typed.get_atom(c).unwrap().get_f64(keys::CHARGE).unwrap() - 0.1572).abs() < 1e-12);
    for h in hs {
        assert!((typed.get_atom(h).unwrap().get_f64(keys::CHARGE).unwrap() + 0.0393).abs() < 1e-12);
    }
}

#[test]
fn embedded_table_reproduces_reference25_oracle_vectors() {
    let cases = [
        (
            methane().0,
            vec![-0.266000, 0.066000, 0.066000, 0.066000, 0.066000],
            vec![-0.108800, 0.026700, 0.026700, 0.026700, 0.026700],
        ),
        (
            methanol(),
            vec![-0.073000, -0.326000, 0.068000, 0.068000, 0.068000, 0.195000],
            vec![0.116700, -0.598800, 0.028700, 0.028700, 0.028700, 0.396000],
        ),
        (
            chloromethane(),
            vec![-0.177000, -0.117000, 0.098000, 0.098000, 0.098000],
            vec![0.014300, -0.190400, 0.058700, 0.058700, 0.058700],
        ),
    ];

    for (mol, pre_bcc, expected) in cases {
        let typifier = AM1BCCTypifier::bcc(FakeAM1Backend { charges: pre_bcc })
            .expect("load embedded BCC table");
        let typed = typifier.typify(&mol).expect("apply BCC corrections");
        let actual: Vec<_> = typed
            .atoms()
            .map(|(_, atom)| atom.get_f64(keys::CHARGE).unwrap())
            .collect();
        assert_eq!(actual.len(), expected.len());
        for (q, q_ref) in actual.iter().zip(expected.iter()) {
            assert!((q - q_ref).abs() < 5.0e-7, "{q} != {q_ref}");
        }
    }
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
            bond.props.get(keys::TYPE).and_then(|v| v.as_f64()),
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
/// object a user can actually build.
#[test]
fn missing_bcc_correction_row_is_an_error_not_zero() {
    let (mol, _, _) = bcc_typed_methane();
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
    let typifier = AM1BCCTypifier::new(
        FakeAM1Backend {
            charges: vec![0.0; 5],
        },
        table,
    );

    let err = typifier
        .typify(&mol)
        .expect_err("an uncorrectable bond must be rejected, not silently left at 0.0");
    assert!(err.contains("missing BCC correction"), "{err}");
    assert!(
        err.contains("11|91|1") || err.contains("91|11|1"),
        "the error must name the C–H bond it could not correct, got: {err}"
    );
}
