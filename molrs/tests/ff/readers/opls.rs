//! Integration test for the OPLS-AA / GROMACS XML reader against the *real*
//! `oplsaa.xml` bundled with molpy.
//!
//! Per the workspace IO rule, happy-path format coverage uses a real file, not a
//! fabricated string (the conversion/edge-case unit tests live inline in
//! `src/forcefield/readers/opls.rs`). The full molpy↔molrs energy parity
//! (spec criterion #4) runs in the `bm-molrs-molpy` harness; here we assert the
//! reader digests the entire real file into a well-formed, non-empty ForceField.
//!
//! The file is located via `$MOLPY_OPLSAA_XML` or a few sibling-repo candidates.
//! It is deliberately `#[ignore]`: the input belongs to molpy and is not vendored
//! here, so an ordinary molrs checkout cannot run this integration test.  Marking
//! that external dependency explicitly keeps CI honest instead of returning green
//! without executing an assertion.

use molrs::ff::{ForceFieldReader, OplsXmlReader};

/// Resolve the real `oplsaa.xml`, or `None` if it isn't present on this machine.
fn locate_oplsaa() -> Option<std::path::PathBuf> {
    if let Ok(p) = std::env::var("MOLPY_OPLSAA_XML") {
        let p = std::path::PathBuf::from(p);
        if p.is_file() {
            return Some(p);
        }
    }
    // CARGO_MANIFEST_DIR = .../molrs/molrs-ff; molpy is a sibling of molrs.
    let manifest = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let candidates = [
        manifest.join("../../molpy/src/molpy/data/forcefield/oplsaa.xml"),
        manifest.join("../molpy/src/molpy/data/forcefield/oplsaa.xml"),
    ];
    candidates.into_iter().find(|p| p.is_file())
}

#[test]
#[ignore = "requires molpy's non-vendored oplsaa.xml; set MOLPY_OPLSAA_XML and run explicitly"]
fn reads_real_molpy_oplsaa() {
    let path = locate_oplsaa().expect(
        "molpy oplsaa.xml not found; set MOLPY_OPLSAA_XML before running this ignored test",
    );

    let ff = OplsXmlReader::new()
        .read(path.to_str().unwrap())
        .expect("real oplsaa.xml should parse");

    // Every modeled section produced a non-empty style.
    assert!(
        !ff.get_styles("bond").is_empty(),
        "expected a bond style from HarmonicBondForce"
    );
    assert!(
        !ff.get_styles("angle").is_empty(),
        "expected an angle style"
    );
    assert!(
        ff.get_style("dihedral", "opls").is_some(),
        "expected an opls dihedral style from RBTorsionForce"
    );
    assert!(
        ff.get_style("pair", "lj/cut").is_some() && ff.get_style("pair", "coul/cut").is_some(),
        "expected lj/cut + coul/cut pair styles from NonbondedForce"
    );
    assert!(
        ff.get_style("atom", "full").is_some(),
        "expected an atom style carrying mass + charge"
    );

    // Atom and nonbonded vocabularies are populated (real file has 825 types).
    assert!(
        ff.get_atomtypes().len() > 100,
        "expected many opls_NNN atom types, got {}",
        ff.get_atomtypes().len()
    );

    // Unit sanity on a known row: OW-HW bond, length 0.09572 nm → 0.9572 Å.
    let bond = ff.get_style("bond", "harmonic").unwrap();
    let ow_hw = bond.get_bondtype("OW", "HW").expect("OW-HW bond present");
    assert!(
        (ow_hw.params.get("r0").unwrap() - 0.9572).abs() < 1e-6,
        "OW-HW r0 should be 0.9572 Å"
    );
}
