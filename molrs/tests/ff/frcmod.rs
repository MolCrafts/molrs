//! Tests for the public `ff::frcmod` format object.

use molrs::ff::Frcmod;

const FRCMOD_TEXT: &str = r#"
benzyl missing parameters

MASS
ca        12.010000
ha         1.008000

BOND
ca-ha    367.000000    1.080000

ANGLE
ca-ca-ha  48.000000  120.010000

DIHE
X -ca-ca-X  9  1.400000  180.000000  2.000000

IMPROPER
ca-ca-ca-ha  1.100000  180.000000  2.000000

NONBON
ca   1.908000  0.086000
ha   1.459000  0.015000
"#;

#[test]
fn parses_all_frcmod_sections_into_structs() {
    let frcmod = Frcmod::parse_str(FRCMOD_TEXT).expect("parse frcmod");

    assert_eq!(frcmod.title, "benzyl missing parameters");
    assert_eq!(frcmod.masses.len(), 2);
    assert_eq!(frcmod.bonds.len(), 1);
    assert_eq!(frcmod.angles.len(), 1);
    assert_eq!(frcmod.dihedrals.len(), 1);
    assert_eq!(frcmod.impropers.len(), 1);
    assert_eq!(frcmod.nonbonded.len(), 2);

    assert_eq!(frcmod.bonds[0].i, "ca");
    assert_eq!(frcmod.bonds[0].j, "ha");
    assert_eq!(frcmod.bonds[0].force_constant, 367.0);
    assert_eq!(frcmod.bonds[0].equilibrium, 1.08);

    assert_eq!(frcmod.angles[0].i, "ca");
    assert_eq!(frcmod.angles[0].j, "ca");
    assert_eq!(frcmod.angles[0].k, "ha");
    assert_eq!(frcmod.angles[0].equilibrium_deg, 120.01);

    assert_eq!(frcmod.dihedrals[0].i, "X");
    assert_eq!(frcmod.dihedrals[0].j, "ca");
    assert_eq!(frcmod.dihedrals[0].k, "ca");
    assert_eq!(frcmod.dihedrals[0].l, "X");
    assert_eq!(frcmod.dihedrals[0].divisor, 9.0);
    assert_eq!(frcmod.dihedrals[0].periodicity, 2.0);

    assert_eq!(frcmod.impropers[0].barrier, 1.1);
    assert_eq!(frcmod.nonbonded[0].atom_type, "ca");
    assert_eq!(frcmod.nonbonded[0].radius, 1.908);
}

#[test]
fn frcmod_write_round_trips_through_parser() {
    let first = Frcmod::parse_str(FRCMOD_TEXT).expect("parse first");
    let text = first.write_string();
    let second = Frcmod::parse_str(&text).expect("parse generated frcmod");

    assert_eq!(second, first);
}

#[test]
fn malformed_rows_return_errors() {
    let err = Frcmod::parse_str(
        r#"
bad
BOND
ca-ha not-a-number 1.08
"#,
    )
    .expect_err("bad BOND row should error");

    assert!(err.contains("invalid BOND row"), "{err}");
}
