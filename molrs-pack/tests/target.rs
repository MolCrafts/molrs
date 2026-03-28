//! Tests for Target builder: construction, natoms/count, fixed_at,
//! centering modes, constraint attachment, and hook validation.

use molrs_pack::{
    F, InsideBoxConstraint, InsideSphereConstraint, MoleculeConstraint, Molpack, Restraint, Target,
};

// ── helpers ────────────────────────────────────────────────────────────────

fn water_positions() -> Vec<[F; 3]> {
    vec![
        [0.0, 0.0, 0.0],    // O
        [0.96, 0.0, 0.0],   // H
        [-0.24, 0.93, 0.0], // H
    ]
}

fn water_radii() -> Vec<F> {
    vec![1.52, 1.20, 1.20]
}

// ── construction ───────────────────────────────────────────────────────────

#[test]
fn from_coords_basic() {
    let t = Target::from_coords(&water_positions(), &water_radii(), 5);
    assert_eq!(t.natoms(), 3);
    assert_eq!(t.count, 5);
    assert!(t.name.is_none());
    assert!(t.fixed_at.is_none());
}

#[test]
fn with_name() {
    let t = Target::from_coords(&water_positions(), &water_radii(), 1).with_name("water");
    assert_eq!(t.name, Some("water".to_string()));
}

#[test]
fn ref_coords_are_centered() {
    let coords = vec![[10.0, 20.0, 30.0], [12.0, 20.0, 30.0]];
    let t = Target::from_coords(&coords, &[1.0, 1.0], 1);
    // Center should be at (11, 20, 30), so ref_coords are [-1, 0, 0] and [1, 0, 0]
    let cx: F = t.ref_coords.iter().map(|p| p[0]).sum::<F>() / t.natoms() as F;
    let cy: F = t.ref_coords.iter().map(|p| p[1]).sum::<F>() / t.natoms() as F;
    let cz: F = t.ref_coords.iter().map(|p| p[2]).sum::<F>() / t.natoms() as F;
    assert!(cx.abs() < 1e-6, "ref_coords x center should be 0");
    assert!(cy.abs() < 1e-6, "ref_coords y center should be 0");
    assert!(cz.abs() < 1e-6, "ref_coords z center should be 0");
}

#[test]
fn input_coords_preserved() {
    let coords = vec![[10.0, 20.0, 30.0]];
    let t = Target::from_coords(&coords, &[1.0], 1);
    assert!((t.input_coords[0][0] - 10.0).abs() < 1e-6);
    assert!((t.input_coords[0][1] - 20.0).abs() < 1e-6);
}

#[test]
fn new_uses_geometric_center_even_when_elements_are_known() {
    use molrs::block::Block;
    use ndarray::Array1;

    let mut atoms = Block::new();
    atoms
        .insert("x", Array1::from_vec(vec![0.0, 1.0]).into_dyn())
        .expect("insert x");
    atoms
        .insert("y", Array1::from_vec(vec![0.0, 0.0]).into_dyn())
        .expect("insert y");
    atoms
        .insert("z", Array1::from_vec(vec![0.0, 0.0]).into_dyn())
        .expect("insert z");
    atoms
        .insert(
            "element",
            Array1::from_vec(vec!["O".to_string(), "H".to_string()]).into_dyn(),
        )
        .expect("insert element");

    let mut frame = molrs::Frame::new();
    frame.insert("atoms", atoms);

    let t = Target::new(frame, 1);
    let arithmetic_center = (t.ref_coords[0][0] + t.ref_coords[1][0]) / 2.0;
    assert!(
        arithmetic_center.abs() < 1e-6,
        "geometry center should be zero"
    );
}

// ── constraints ────────────────────────────────────────────────────────────

#[test]
fn with_constraint() {
    let c =
        MoleculeConstraint::new().add(Restraint::inside_box([0.0, 0.0, 0.0], [20.0, 20.0, 20.0]));
    let t = Target::from_coords(&water_positions(), &water_radii(), 5).with_constraint(c);
    assert_eq!(t.molecule_constraint.restraints.len(), 1);
}

#[test]
fn with_constraint_chained() {
    let t = Target::from_coords(&water_positions(), &water_radii(), 5)
        .with_constraint(InsideBoxConstraint::new(
            [0.0, 0.0, 0.0],
            [20.0, 20.0, 20.0],
        ))
        .with_constraint(InsideSphereConstraint::new(50.0, [10.0, 10.0, 10.0]));
    assert_eq!(t.molecule_constraint.restraints.len(), 2);
}

#[test]
fn with_constraint_for_atoms() {
    let t = Target::from_coords(&water_positions(), &water_radii(), 5).with_constraint_for_atoms(
        &[1, 2], // Packmol 1-based → internally 0-based [0, 1]
        InsideSphereConstraint::new(5.0, [0.0, 0.0, 0.0]),
    );
    assert_eq!(t.atom_constraints.len(), 1);
    // Indices converted to 0-based
    assert_eq!(t.atom_constraints[0].atom_indices, vec![0, 1]);
}

// ── fixed placement ────────────────────────────────────────────────────────

#[test]
fn fixed_at_sets_count_to_1() {
    let t = Target::from_coords(&water_positions(), &water_radii(), 5).fixed_at([0.0, 0.0, 0.0]);
    assert_eq!(t.count, 1);
    assert!(t.fixed_at.is_some());
    let fp = t.fixed_at.unwrap();
    assert!((fp.position[0]).abs() < 1e-6);
    assert!((fp.euler[0]).abs() < 1e-6);
}

#[test]
fn fixed_at_with_euler() {
    let t = Target::from_coords(&water_positions(), &water_radii(), 5)
        .fixed_at_with_euler([1.0, 2.0, 3.0], [0.1, 0.2, 0.3]);
    assert_eq!(t.count, 1);
    let fp = t.fixed_at.unwrap();
    assert!((fp.position[0] - 1.0).abs() < 1e-6);
    assert!((fp.euler[2] - 0.3).abs() < 1e-6);
}

#[test]
fn fixed_target_auto_centering_disabled() {
    // When fixed_at is used with Auto centering (default), the fixed molecule
    // should NOT be centered — its input coords are used directly.
    let free = Target::from_coords(&[[0.0, 0.0, 0.0]], &[1.0], 1).with_constraint(
        MoleculeConstraint::new().add(Restraint::inside_box([-5.0, -5.0, -5.0], [5.0, 5.0, 5.0])),
    );
    let fixed = Target::from_coords(&[[10.0, 0.0, 0.0], [12.0, 0.0, 0.0]], &[1.0, 1.0], 1)
        .fixed_at([0.0, 0.0, 0.0]);

    let result = Molpack::new()
        .pack(&[free, fixed], 5, Some(1))
        .expect("pack should succeed");

    // Fixed atoms follow free atoms in output.
    assert!((result.positions()[1][0] - 10.0).abs() < 1e-6);
    assert!((result.positions()[2][0] - 12.0).abs() < 1e-6);
}

#[test]
fn fixed_target_centered() {
    let free = Target::from_coords(&[[0.0, 0.0, 0.0]], &[1.0], 1).with_constraint(
        MoleculeConstraint::new().add(Restraint::inside_box([-5.0, -5.0, -5.0], [5.0, 5.0, 5.0])),
    );
    let fixed = Target::from_coords(&[[10.0, 0.0, 0.0], [12.0, 0.0, 0.0]], &[1.0, 1.0], 1)
        .with_center()
        .fixed_at([0.0, 0.0, 0.0]);

    let result = Molpack::new()
        .pack(&[free, fixed], 5, Some(1))
        .expect("pack should succeed");

    // COM of [10,12] = 11. After centering, ref_coords = [-1, +1].
    // Placed at origin → positions = [-1, +1].
    assert!((result.positions()[1][0] + 1.0).abs() < 1e-6);
    assert!((result.positions()[2][0] - 1.0).abs() < 1e-6);
}

// ── centering modes ────────────────────────────────────────────────────────

#[test]
fn with_center_mode() {
    use molrs_pack::CenteringMode;
    let t = Target::from_coords(&water_positions(), &water_radii(), 1).with_center();
    assert_eq!(t.centering, CenteringMode::Center);
}

#[test]
fn without_centering_mode() {
    use molrs_pack::CenteringMode;
    let t = Target::from_coords(&water_positions(), &water_radii(), 1).without_centering();
    assert_eq!(t.centering, CenteringMode::None);
}

// ── rotation constraints ───────────────────────────────────────────────────

#[test]
fn constrain_rotation() {
    let t = Target::from_coords(&[[0.0, 0.0, 0.0]], &[1.0], 1)
        .constrain_rotation_x(0.0, 10.0)
        .constrain_rotation_y(90.0, 5.0)
        .constrain_rotation_z(180.0, 15.0);
    // x → teta (index 2), y → beta (index 0), z → gama (index 1)
    assert!(t.constrain_rotation[0].is_some()); // beta (y)
    assert!(t.constrain_rotation[1].is_some()); // gama (z)
    assert!(t.constrain_rotation[2].is_some()); // teta (x)
}

// ── maxmove ────────────────────────────────────────────────────────────────

#[test]
fn with_maxmove() {
    let t = Target::from_coords(&[[0.0, 0.0, 0.0]], &[1.0], 1).with_maxmove(5);
    assert_eq!(t.maxmove, Some(5));
}

// ── default element ────────────────────────────────────────────────────────

#[test]
fn default_elements_are_x() {
    let t = Target::from_coords(&[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], &[1.0, 1.0], 1);
    assert_eq!(t.elements, vec!["X", "X"]);
}

// ── panics ─────────────────────────────────────────────────────────────────

#[test]
#[should_panic(expected = "positions and radii must have the same length")]
fn mismatched_coords_and_radii_panics() {
    Target::from_coords(&[[0.0, 0.0, 0.0]], &[1.0, 2.0], 1);
}
