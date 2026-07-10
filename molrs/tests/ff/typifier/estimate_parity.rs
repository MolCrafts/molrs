//! Hardcoded parmchk2-style oracle coverage for the [`ParameterEstimator`].
//!
//! The project test suite never shells out to external chemistry programs. Any
//! external parity data belongs here as a literal oracle that is consumed through
//! the same public structs users call.

use molrs::ff::Frcmod;

const AC009_FRCMOD_ORACLE: &str = r#"
ac-009 missing GAFF parameters

BOND
c3-os   317.000000  1.410000

ANGLE
c3-os-h1   49.300000  109.500000

DIHE
X -c3-os-X   1  0.155000  0.000000  3.000000
"#;

#[test]
fn hardcoded_parmchk2_oracle_parses_as_frcmod_struct() {
    let frcmod = Frcmod::parse_str(AC009_FRCMOD_ORACLE).expect("parse oracle frcmod");

    assert_eq!(frcmod.title, "ac-009 missing GAFF parameters");
    assert_eq!(frcmod.bonds.len(), 1);
    assert_eq!(frcmod.angles.len(), 1);
    assert_eq!(frcmod.dihedrals.len(), 1);

    let bond = &frcmod.bonds[0];
    assert_eq!((&bond.i, &bond.j), (&"c3".to_owned(), &"os".to_owned()));
    assert!((bond.force_constant - 317.0).abs() < 1.0e-12);
    assert!((bond.equilibrium - 1.41).abs() < 1.0e-12);

    let angle = &frcmod.angles[0];
    assert_eq!(
        (&angle.i, &angle.j, &angle.k),
        (&"c3".to_owned(), &"os".to_owned(), &"h1".to_owned())
    );
    assert!((angle.force_constant - 49.3).abs() < 1.0e-12);
    assert!((angle.equilibrium_deg - 109.5).abs() < 1.0e-12);

    let dihedral = &frcmod.dihedrals[0];
    assert_eq!(
        (&dihedral.i, &dihedral.j, &dihedral.k, &dihedral.l),
        (
            &"X".to_owned(),
            &"c3".to_owned(),
            &"os".to_owned(),
            &"X".to_owned()
        )
    );
    assert!((dihedral.barrier - 0.155).abs() < 1.0e-12);
    assert!((dihedral.periodicity - 3.0).abs() < 1.0e-12);
}
