//! End-to-end tests for `forcefield` (mod + xml): parse / build a `ForceField`
//! and turn it into evaluable `Potentials` via `to_potentials` against a Frame.

use crate::helpers::{atoms_frame, flat_coords, pairs_block, topo_block, typed_atoms_block};
use molrs::ff::ForceField;
use molrs::ff::potential::extract_coords;
use molrs::ff::read_forcefield_xml_str;
use molrs::ff::typifier::mmff::MMFF94Typifier;
use molrs::store::frame::Frame;
use molrs::types::F;

// ---------------------------------------------------------------------------
// In-code ForceField → to_potentials → calc
// ---------------------------------------------------------------------------

#[test]
fn compile_bond_then_eval_matches_analytical_energy() {
    // E = 0.5 * k0 * (r - r0)^2 ; r = 2.0, r0 = 1.5, k0 = 300 -> 0.5*300*0.25 = 37.5
    let mut ff = ForceField::new("test");
    ff.def_bondstyle("harmonic")
        .def_type("CT-CT", &[("k", 300.0), ("r0", 1.5)]);

    let mut frame = atoms_frame(&[[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]);
    frame.insert(
        "bonds",
        topo_block(&[("atomi", &[0]), ("atomj", &[1])], &["CT-CT"]),
    );

    let pots = ff.to_potentials(&frame).unwrap();
    let coords = extract_coords(&frame).unwrap();
    let (e, _) = pots.calc_energy_forces(&coords);
    assert!((e - 37.5).abs() < 1e-9, "energy {e}");
}

#[test]
fn compile_multi_style_sums_independent_contributions() {
    // A bond + a pair style compiled together; total energy is the sum.
    let mut ff = ForceField::new("multi");
    ff.def_bondstyle("harmonic")
        .def_type("A-A", &[("k", 100.0), ("r0", 1.0)]);
    ff.def_pairstyle("lj/cut", &[("cutoff", 10.0)])
        .def_type("A", &[("epsilon", 1.0), ("sigma", 1.0)]);

    // Two atoms at r = 2.0; per-atom `type` for lj/cut, `is_14` neighbour list.
    let coords_xyz = [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]];
    let mut frame = Frame::new();
    frame.insert("atoms", typed_atoms_block(&coords_xyz, &["A", "A"], None));
    frame.insert(
        "bonds",
        topo_block(&[("atomi", &[0]), ("atomj", &[1])], &["A-A"]),
    );
    frame.insert("pairs", pairs_block(&[0], &[1], &[false]));

    let pots = ff.to_potentials(&frame).unwrap();
    assert_eq!(pots.len(), 2);
    let coords = flat_coords(&coords_xyz);
    let (e, _) = pots.calc_energy_forces(&coords);

    // bond: 0.5*100*(2-1)^2 = 50.0
    let e_bond: F = 50.0;
    // LJ: 4*1*((1/2)^12 - (1/2)^6) = 4*(1/4096 - 1/64)
    let e_lj: F = 4.0 * (1.0 / 4096.0 - 1.0 / 64.0);
    assert!((e - (e_bond + e_lj)).abs() < 1e-9, "total {e}");
}

#[test]
fn atom_style_is_skipped() {
    // An atom style carries types/charges, not a pairwise kernel — to_potentials
    // skips it rather than erroring.
    let ff = ForceField::new("test").with_atomstyle("full");
    let frame = atoms_frame(&[[0.0, 0.0, 0.0]]);
    let pots = ff.to_potentials(&frame).unwrap();
    assert_eq!(pots.len(), 0);
}

#[test]
fn compile_skips_absent_topology_block() {
    // A style whose topology block is absent from the frame contributes nothing
    // (the molecule has no interactions of that kind) — it is skipped, not an
    // error. (A *present* block with an unknown type label still errors.)
    let mut ff = ForceField::new("test");
    ff.def_pairstyle("lj/cut", &[("cutoff", 10.0)])
        .def_type("A", &[("epsilon", 1.0), ("sigma", 1.0)]);
    // No "pairs" block in the frame.
    let frame = atoms_frame(&[[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]);
    let pots = ff.to_potentials(&frame).unwrap();
    assert_eq!(pots.len(), 0);
}

// ---------------------------------------------------------------------------
// XML parse path (generic layout) → compile → eval
// ---------------------------------------------------------------------------

#[test]
fn parse_generic_xml_then_compile_and_eval() {
    // Generic <ForceField> layout (the documented public schema for
    // read_forcefield_xml_str). This is a STRUCTURE definition, not file-format
    // sample data being round-tripped — it is the API's declarative input.
    let xml = r#"
        <ForceField name="demo">
          <BondStyle name="harmonic">
            <Type name="CT-CT" k="300.0" r0="1.5" />
          </BondStyle>
        </ForceField>
    "#;
    let ff = read_forcefield_xml_str(xml).expect("parse generic xml");
    assert_eq!(ff.name, "demo");
    assert!(ff.get_style("bond", "harmonic").is_some());

    let mut frame = atoms_frame(&[[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]);
    frame.insert(
        "bonds",
        topo_block(&[("atomi", &[0]), ("atomj", &[1])], &["CT-CT"]),
    );
    let pots = ff.to_potentials(&frame).unwrap();
    let coords = extract_coords(&frame).unwrap();
    assert!((pots.calc_energy(&coords) - 37.5).abs() < 1e-9);
}

#[test]
fn parse_xml_rejects_wrong_root() {
    let err = read_forcefield_xml_str("<NotAForceField/>").expect_err("bad root");
    assert!(err.contains("ForceField"), "{err}");
}

/// The shipped MMFF94 set is a full, multi-style `ForceField` — however it is stored.
///
/// This used to read `molrs::data::MMFF94_XML` through `read_forcefield_xml_str`.
/// chem-perceive-14 ac-001 deletes that const (the tables are compiled Rust now), and
/// the fact worth asserting never depended on the storage: what matters is that the
/// parameter set molrs ships arrives as many styles with real bond types, not that it
/// arrives via a particular parser. So it goes through the front door, and survives the
/// representation change untouched.
///
/// The XML *parser* keeps its own coverage above — `parse_generic_xml_then_compile_and_eval`
/// and `parse_xml_rejects_wrong_root` — on caller-supplied XML, which is what the reader
/// is for once molrs ships no XML of its own.
#[test]
fn shipped_mmff94_parameter_set_yields_many_styles() {
    let typifier = MMFF94Typifier::new();
    let ff = typifier.ff();
    // MMFF tables compile into bond/angle/dihedral/improper/pair styles.
    assert!(
        !ff.styles().is_empty(),
        "MMFF94 should yield at least one style"
    );
    assert!(!ff.get_styles("bond").is_empty(), "expected bond styles");
}
