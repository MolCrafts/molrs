//! End-to-end packer tests: small molecule packing, PBC, multi-target,
//! fixed target, error cases.

use molrs_pack::{
    F, InsideBoxConstraint, InsideSphereConstraint, MoleculeConstraint, Molpack, NullHandler,
    OutsideSphereConstraint, PackError, RegionConstraint, Restraint, Target,
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

fn single_atom() -> (Vec<[F; 3]>, Vec<F>) {
    (vec![[0.0, 0.0, 0.0]], vec![1.0])
}

// ── basic packing ──────────────────────────────────────────────────────────

#[test]
fn pack_single_water_in_box() {
    let c =
        MoleculeConstraint::new().add(Restraint::inside_box([0.0, 0.0, 0.0], [40.0, 40.0, 40.0]));
    let target = Target::from_coords(&water_positions(), &water_radii(), 1).with_constraint(c);
    let result = Molpack::new().pack(&[target], 5, Some(42));
    assert!(result.is_ok());
    assert_eq!(result.unwrap().natoms(), 3);
}

#[test]
fn pack_three_waters_in_box() {
    let c =
        MoleculeConstraint::new().add(Restraint::inside_box([0.0, 0.0, 0.0], [40.0, 40.0, 40.0]));
    let target = Target::from_coords(&water_positions(), &water_radii(), 3).with_constraint(c);
    let result = Molpack::new().pack(&[target], 5, Some(42));
    assert!(result.is_ok());
    assert_eq!(result.unwrap().natoms(), 9, "3 water * 3 atoms = 9");
}

#[test]
fn pack_in_sphere() {
    let target = Target::from_coords(&water_positions(), &water_radii(), 3)
        .with_constraint(InsideSphereConstraint::new(20.0, [0.0, 0.0, 0.0]));
    let result = Molpack::new().pack(&[target], 5, Some(42));
    assert!(result.is_ok());
    assert_eq!(result.unwrap().natoms(), 9);
}

// ── multi-target ───────────────────────────────────────────────────────────

#[test]
fn pack_two_targets_same_box() {
    let box_c = InsideBoxConstraint::new([0.0, 0.0, 0.0], [40.0, 40.0, 40.0]);
    let (coords, radii) = single_atom();
    let t1 = Target::from_coords(&coords, &radii, 3)
        .with_name("A")
        .with_constraint(box_c.clone());
    let t2 = Target::from_coords(&coords, &radii, 2)
        .with_name("B")
        .with_constraint(box_c);
    let result = Molpack::new().pack(&[t1, t2], 5, Some(42));
    assert!(result.is_ok());
    assert_eq!(result.unwrap().natoms(), 5); // 3 + 2
}

#[test]
fn pack_mixed_free_and_fixed() {
    let (coords, radii) = single_atom();
    let free = Target::from_coords(&coords, &radii, 2).with_constraint(InsideBoxConstraint::new(
        [0.0, 0.0, 0.0],
        [20.0, 20.0, 20.0],
    ));
    let fixed = Target::from_coords(&coords, &radii, 1).fixed_at([10.0, 10.0, 10.0]);
    let result = Molpack::new()
        .pack(&[free, fixed], 5, Some(42))
        .expect("should succeed");
    assert_eq!(result.natoms(), 3); // 2 free + 1 fixed
    // Fixed atom should be at its placement position
    let fixed_pos = result.positions()[2]; // fixed atoms come after free
    assert!((fixed_pos[0] - 10.0).abs() < 1e-6);
    assert!((fixed_pos[1] - 10.0).abs() < 1e-6);
}

// ── PBC ────────────────────────────────────────────────────────────────────

#[test]
fn pbc_box_packing() {
    let c =
        MoleculeConstraint::new().add(Restraint::inside_box([0.0, 0.0, 0.0], [30.0, 30.0, 30.0]));
    let target = Target::from_coords(&water_positions(), &water_radii(), 2).with_constraint(c);
    let result = Molpack::new()
        .pbc_box([30.0, 30.0, 30.0])
        .pack(&[target], 5, Some(42));
    assert!(result.is_ok());
    assert_eq!(result.unwrap().natoms(), 6);
}

#[test]
fn invalid_pbc_box_rejected() {
    let c =
        MoleculeConstraint::new().add(Restraint::inside_box([0.0, 0.0, 0.0], [40.0, 40.0, 40.0]));
    let target = Target::from_coords(&water_positions(), &water_radii(), 1).with_constraint(c);
    let result = Molpack::new()
        .pbc([0.0, 0.0, 0.0], [10.0, 0.0, 10.0])
        .pack(&[target], 5, Some(7));
    assert!(
        matches!(result, Err(PackError::InvalidPBCBox { .. })),
        "expected InvalidPBCBox, got: {result:?}"
    );
}

// ── error cases ────────────────────────────────────────────────────────────

#[test]
fn empty_targets_returns_error() {
    let result = Molpack::new().pack(&[], 5, Some(42));
    assert!(
        matches!(result, Err(PackError::NoTargets)),
        "expected NoTargets, got: {result:?}"
    );
}

// ── handler integration ────────────────────────────────────────────────────

#[test]
fn null_handler_accepted() {
    let target = Target::from_coords(&water_positions(), &water_radii(), 2).with_constraint(
        InsideBoxConstraint::new([0.0, 0.0, 0.0], [40.0, 40.0, 40.0]),
    );
    let result = Molpack::new()
        .add_handler(NullHandler)
        .pack(&[target], 5, Some(42));
    assert!(result.is_ok());
}

// ── deterministic seed ─────────────────────────────────────────────────────

#[test]
fn same_seed_same_result() {
    let make_target = || {
        Target::from_coords(&water_positions(), &water_radii(), 3).with_constraint(
            InsideBoxConstraint::new([0.0, 0.0, 0.0], [40.0, 40.0, 40.0]),
        )
    };
    let r1 = Molpack::new().pack(&[make_target()], 5, Some(42)).unwrap();
    let r2 = Molpack::new().pack(&[make_target()], 5, Some(42)).unwrap();
    for (a, b) in r1.positions().iter().zip(r2.positions().iter()) {
        assert!((a[0] - b[0]).abs() < 1e-6);
        assert!((a[1] - b[1]).abs() < 1e-6);
        assert!((a[2] - b[2]).abs() < 1e-6);
    }
}

// ── builder chain ──────────────────────────────────────────────────────────

#[test]
fn builder_precision_and_tolerance() {
    let target = Target::from_coords(&water_positions(), &water_radii(), 2).with_constraint(
        InsideBoxConstraint::new([0.0, 0.0, 0.0], [40.0, 40.0, 40.0]),
    );
    let result =
        Molpack::new()
            .precision(0.1)
            .tolerance(3.0)
            .maxit(10)
            .pack(&[target], 5, Some(42));
    assert!(result.is_ok());
}

// ── composite constraint ───────────────────────────────────────────────────

#[test]
fn pack_with_composite_constraint() {
    let (coords, radii) = single_atom();
    let c = InsideBoxConstraint::new([-20.0, -20.0, -20.0], [20.0, 20.0, 20.0])
        .and(OutsideSphereConstraint::new(2.0, [0.0, 0.0, 0.0]));
    let target = Target::from_coords(&coords, &radii, 3).with_constraint(c);
    let result = Molpack::new().pack(&[target], 10, Some(42));
    assert!(result.is_ok());
    assert_eq!(result.unwrap().natoms(), 3);
}
