//! Tests for every constraint type: value=0 when satisfied, value>0 when
//! violated, and gradient points in the correct direction.

use molrs_pack::{
    AbovePlaneConstraint, BelowPlaneConstraint, F, InsideBoxConstraint, InsideSphereConstraint,
    MoleculeConstraint, OutsideSphereConstraint, RegionConstraint, Restraint,
};

const TOL: F = 1e-6;
const SCALE: F = 1.0;
const SCALE2: F = 0.01;

// ── helpers ────────────────────────────────────────────────────────────────

/// Assert that the gradient at `pos` pushes the atom toward the satisfied
/// region (each nonzero gradient component opposes the violation).
fn assert_gradient_opposes_violation(r: &Restraint, pos: &[F; 3], label: &str) {
    let h: F = 1e-4;
    let mut g = [0.0 as F; 3];
    r.gradient(pos, SCALE, SCALE2, &mut g);

    for axis in 0..3 {
        if g[axis].abs() < TOL {
            continue;
        }
        // A small step along -gradient should reduce the penalty.
        let mut pos_step = *pos;
        pos_step[axis] -= h * g[axis].signum();
        let v_before = r.value(pos, SCALE, SCALE2);
        let v_after = r.value(&pos_step, SCALE, SCALE2);
        assert!(
            v_after <= v_before + TOL,
            "{label}: stepping along -gradient on axis {axis} should reduce penalty \
             (before={v_before}, after={v_after}, g={:?})",
            g
        );
    }
}

// ── inside box (type 3) ────────────────────────────────────────────────────

#[test]
fn inside_box_satisfied() {
    let r = Restraint::inside_box([0.0, 0.0, 0.0], [10.0, 10.0, 10.0]);
    assert_eq!(r.value(&[5.0, 5.0, 5.0], SCALE, SCALE2), 0.0);
    assert_eq!(r.value(&[0.0, 0.0, 0.0], SCALE, SCALE2), 0.0);
    assert_eq!(r.value(&[10.0, 10.0, 10.0], SCALE, SCALE2), 0.0);
}

#[test]
fn inside_box_violated() {
    let r = Restraint::inside_box([0.0, 0.0, 0.0], [10.0, 10.0, 10.0]);
    assert!(r.value(&[11.0, 5.0, 5.0], SCALE, SCALE2) > 0.0);
    assert!(r.value(&[-1.0, 5.0, 5.0], SCALE, SCALE2) > 0.0);
    assert!(r.value(&[5.0, -1.0, 5.0], SCALE, SCALE2) > 0.0);
}

#[test]
fn inside_box_gradient_direction() {
    let r = Restraint::inside_box([0.0, 0.0, 0.0], [10.0, 10.0, 10.0]);
    // Atom above xmax → gradient pushes it back (positive gx)
    let mut g = [0.0 as F; 3];
    r.gradient(&[11.0, 5.0, 5.0], SCALE, SCALE2, &mut g);
    assert!(g[0] > 0.0, "gradient should push x back toward box");
    // Atom below xmin → negative gx
    let mut g = [0.0 as F; 3];
    r.gradient(&[-1.0, 5.0, 5.0], SCALE, SCALE2, &mut g);
    assert!(g[0] < 0.0, "gradient should push x into box");

    assert_gradient_opposes_violation(&r, &[11.0, 5.0, 5.0], "inside_box_above");
    assert_gradient_opposes_violation(&r, &[-1.0, -1.0, -1.0], "inside_box_below");
}

// ── inside cube (type 2) ───────────────────────────────────────────────────

#[test]
fn inside_cube_satisfied_and_violated() {
    let r = Restraint::inside_cube([0.0, 0.0, 0.0], 10.0);
    assert_eq!(r.value(&[5.0, 5.0, 5.0], SCALE, SCALE2), 0.0);
    assert!(r.value(&[11.0, 5.0, 5.0], SCALE, SCALE2) > 0.0);
    assert!(r.value(&[-1.0, 5.0, 5.0], SCALE, SCALE2) > 0.0);
    assert_gradient_opposes_violation(&r, &[11.0, 5.0, 5.0], "inside_cube");
}

// ── inside sphere (type 4) ─────────────────────────────────────────────────

#[test]
fn inside_sphere_satisfied() {
    let r = Restraint::inside_sphere([5.0, 5.0, 5.0], 10.0);
    assert_eq!(r.value(&[5.0, 5.0, 5.0], SCALE, SCALE2), 0.0);
    assert_eq!(r.value(&[0.0, 0.0, 0.0], SCALE, SCALE2), 0.0);
}

#[test]
fn inside_sphere_violated() {
    let r = Restraint::inside_sphere([5.0, 5.0, 5.0], 10.0);
    assert!(r.value(&[100.0, 0.0, 0.0], SCALE, SCALE2) > 0.0);
}

#[test]
fn inside_sphere_gradient_direction() {
    let r = Restraint::inside_sphere([0.0, 0.0, 0.0], 5.0);
    assert_gradient_opposes_violation(&r, &[7.0, 0.0, 0.0], "inside_sphere");
    assert_gradient_opposes_violation(&r, &[0.0, -7.0, 0.0], "inside_sphere_y");
}

// ── outside sphere (type 8) ────────────────────────────────────────────────

#[test]
fn outside_sphere_satisfied() {
    let r = Restraint::outside_sphere([0.0, 0.0, 0.0], 5.0);
    assert_eq!(r.value(&[10.0, 0.0, 0.0], SCALE, SCALE2), 0.0);
}

#[test]
fn outside_sphere_violated() {
    let r = Restraint::outside_sphere([0.0, 0.0, 0.0], 5.0);
    assert!(r.value(&[1.0, 0.0, 0.0], SCALE, SCALE2) > 0.0);
}

#[test]
fn outside_sphere_gradient_direction() {
    let r = Restraint::outside_sphere([0.0, 0.0, 0.0], 5.0);
    assert_gradient_opposes_violation(&r, &[1.0, 0.0, 0.0], "outside_sphere");
}

// ── inside ellipsoid (type 5) ──────────────────────────────────────────────

#[test]
fn inside_ellipsoid_satisfied_and_violated() {
    let r = Restraint::inside_ellipsoid([0.0, 0.0, 0.0], [10.0, 5.0, 3.0], 1.0);
    // Inside: x/10 < 1
    assert_eq!(r.value(&[0.0, 0.0, 0.0], SCALE, SCALE2), 0.0);
    // Outside along x (x=15 → (15/10)^2 = 2.25 > 1)
    assert!(r.value(&[15.0, 0.0, 0.0], SCALE, SCALE2) > 0.0);
    assert_gradient_opposes_violation(&r, &[15.0, 0.0, 0.0], "inside_ellipsoid");
}

// ── outside ellipsoid (type 9) ─────────────────────────────────────────────

#[test]
fn outside_ellipsoid_satisfied_and_violated() {
    let r = Restraint::outside_ellipsoid([0.0, 0.0, 0.0], [5.0, 5.0, 5.0], 1.0);
    // Outside: far away
    assert_eq!(r.value(&[10.0, 0.0, 0.0], SCALE, SCALE2), 0.0);
    // Inside: origin
    assert!(r.value(&[0.0, 0.0, 0.0], SCALE, SCALE2) > 0.0);
    assert_gradient_opposes_violation(&r, &[1.0, 0.0, 0.0], "outside_ellipsoid");
}

// ── outside box (type 7) ───────────────────────────────────────────────────

#[test]
fn outside_box_satisfied_and_violated() {
    let r = Restraint::outside_box([0.0, 0.0, 0.0], [10.0, 10.0, 10.0]);
    // Outside → no penalty
    assert_eq!(r.value(&[15.0, 5.0, 5.0], SCALE, SCALE2), 0.0);
    // Inside → penalty
    assert!(r.value(&[5.0, 5.0, 5.0], SCALE, SCALE2) > 0.0);
}

// ── outside cube (type 6) ──────────────────────────────────────────────────

#[test]
fn outside_cube_satisfied_and_violated() {
    let r = Restraint::outside_cube([0.0, 0.0, 0.0], 10.0);
    assert_eq!(r.value(&[15.0, 5.0, 5.0], SCALE, SCALE2), 0.0);
    assert!(r.value(&[5.0, 5.0, 5.0], SCALE, SCALE2) > 0.0);
}

// ── above plane (type 10) ──────────────────────────────────────────────────

#[test]
fn above_plane_satisfied() {
    let r = Restraint::above_plane([0.0, 0.0, 1.0], 5.0);
    assert_eq!(r.value(&[0.0, 0.0, 6.0], SCALE, SCALE2), 0.0);
    assert_eq!(r.value(&[0.0, 0.0, 5.0], SCALE, SCALE2), 0.0);
}

#[test]
fn above_plane_violated() {
    let r = Restraint::above_plane([0.0, 0.0, 1.0], 5.0);
    assert!(r.value(&[0.0, 0.0, 4.0], SCALE, SCALE2) > 0.0);
}

#[test]
fn above_plane_gradient_direction() {
    let r = Restraint::above_plane([0.0, 0.0, 1.0], 5.0);
    let mut g = [0.0 as F; 3];
    r.gradient(&[0.0, 0.0, 4.0], SCALE, SCALE2, &mut g);
    // Gradient should push z upward (positive gz)
    assert!(g[2] < 0.0, "above_plane gradient pushes z toward plane");
    assert_gradient_opposes_violation(&r, &[0.0, 0.0, 4.0], "above_plane");
}

// ── below plane (type 11) ──────────────────────────────────────────────────

#[test]
fn below_plane_satisfied() {
    let r = Restraint::below_plane([0.0, 0.0, 1.0], 5.0);
    assert_eq!(r.value(&[0.0, 0.0, 4.0], SCALE, SCALE2), 0.0);
    assert_eq!(r.value(&[0.0, 0.0, 5.0], SCALE, SCALE2), 0.0);
}

#[test]
fn below_plane_violated() {
    let r = Restraint::below_plane([0.0, 0.0, 1.0], 5.0);
    assert!(r.value(&[0.0, 0.0, 6.0], SCALE, SCALE2) > 0.0);
}

#[test]
fn below_plane_gradient_direction() {
    let r = Restraint::below_plane([0.0, 0.0, 1.0], 5.0);
    assert_gradient_opposes_violation(&r, &[0.0, 0.0, 6.0], "below_plane");
}

// ── inside cylinder (type 12) ──────────────────────────────────────────────

#[test]
fn inside_cylinder_satisfied() {
    // Cylinder along z-axis, center=[0,0,0], radius=5, length=10
    let r = Restraint::inside_cylinder([0.0, 0.0, 0.0], [0.0, 0.0, 1.0], 5.0, 10.0);
    assert_eq!(r.value(&[0.0, 0.0, 5.0], SCALE, SCALE2), 0.0);
}

#[test]
fn inside_cylinder_violated_radial() {
    let r = Restraint::inside_cylinder([0.0, 0.0, 0.0], [0.0, 0.0, 1.0], 5.0, 10.0);
    // Atom outside radially
    assert!(r.value(&[10.0, 0.0, 5.0], SCALE, SCALE2) > 0.0);
}

#[test]
fn inside_cylinder_violated_axial() {
    let r = Restraint::inside_cylinder([0.0, 0.0, 0.0], [0.0, 0.0, 1.0], 5.0, 10.0);
    // Atom outside axially (negative end)
    assert!(r.value(&[0.0, 0.0, -5.0], SCALE, SCALE2) > 0.0);
}

#[test]
fn inside_cylinder_gradient_direction() {
    let r = Restraint::inside_cylinder([0.0, 0.0, 0.0], [0.0, 0.0, 1.0], 5.0, 10.0);
    assert_gradient_opposes_violation(&r, &[10.0, 0.0, 5.0], "inside_cylinder_radial");
}

// ── outside cylinder (type 13) ─────────────────────────────────────────────

#[test]
fn outside_cylinder_satisfied_and_violated() {
    let r = Restraint::outside_cylinder([0.0, 0.0, 0.0], [0.0, 0.0, 1.0], 5.0, 10.0);
    // Outside radially → satisfied
    assert_eq!(r.value(&[10.0, 0.0, 5.0], SCALE, SCALE2), 0.0);
    // Inside → violated
    assert!(r.value(&[1.0, 0.0, 5.0], SCALE, SCALE2) > 0.0);
}

// ── above gaussian (type 14) ───────────────────────────────────────────────

#[test]
fn above_gaussian_satisfied_and_violated() {
    // Gaussian: center=(0,0), sigma=(5,5), z0=0, height=10
    let r = Restraint::above_gaussian(0.0, 0.0, 5.0, 5.0, 0.0, 10.0);
    // Atom well above the surface
    assert_eq!(r.value(&[0.0, 0.0, 20.0], SCALE, SCALE2), 0.0);
    // Atom below the surface at center (surface z ≈ 10)
    assert!(r.value(&[0.0, 0.0, 5.0], SCALE, SCALE2) > 0.0);
}

// ── below gaussian (type 15) ───────────────────────────────────────────────

#[test]
fn below_gaussian_satisfied_and_violated() {
    let r = Restraint::below_gaussian(0.0, 0.0, 5.0, 5.0, 0.0, 10.0);
    // Atom below the surface
    assert_eq!(r.value(&[0.0, 0.0, -5.0], SCALE, SCALE2), 0.0);
    // Atom above the surface at center (surface z ≈ 10)
    assert!(r.value(&[0.0, 0.0, 15.0], SCALE, SCALE2) > 0.0);
}

// ── high-level constraint builders ─────────────────────────────────────────

#[test]
fn inside_box_constraint_builder() {
    let c = InsideBoxConstraint::new([0.0, 0.0, 0.0], [10.0, 10.0, 10.0]);
    let mc: MoleculeConstraint = c.into();
    assert_eq!(mc.restraints.len(), 1);
    assert_eq!(mc.restraints[0].kind, 3);
}

#[test]
fn inside_sphere_constraint_builder() {
    let c = InsideSphereConstraint::new(10.0, [0.0, 0.0, 0.0]);
    let mc: MoleculeConstraint = c.into();
    assert_eq!(mc.restraints.len(), 1);
    assert_eq!(mc.restraints[0].kind, 4);
}

#[test]
fn outside_sphere_constraint_builder() {
    let c = OutsideSphereConstraint::new(10.0, [0.0, 0.0, 0.0]);
    let mc: MoleculeConstraint = c.into();
    assert_eq!(mc.restraints.len(), 1);
    assert_eq!(mc.restraints[0].kind, 8);
}

#[test]
fn plane_constraint_builders() {
    let above: MoleculeConstraint = AbovePlaneConstraint::new([0.0, 0.0, 1.0], 5.0).into();
    let below: MoleculeConstraint = BelowPlaneConstraint::new([0.0, 0.0, 1.0], 5.0).into();
    assert_eq!(above.restraints[0].kind, 10);
    assert_eq!(below.restraints[0].kind, 11);
}

#[test]
fn constraint_and_chaining() {
    let c = InsideBoxConstraint::new([0.0, 0.0, 0.0], [10.0, 10.0, 10.0])
        .and(OutsideSphereConstraint::new(2.0, [5.0, 5.0, 5.0]));
    assert_eq!(c.restraints.len(), 2);
    assert_eq!(c.restraints[0].kind, 3);
    assert_eq!(c.restraints[1].kind, 8);
}

#[test]
fn molecule_constraint_add_and_and() {
    let mc = MoleculeConstraint::new()
        .add(Restraint::inside_box([0.0, 0.0, 0.0], [10.0, 10.0, 10.0]))
        .and(InsideSphereConstraint::new(20.0, [5.0, 5.0, 5.0]));
    assert_eq!(mc.restraints.len(), 2);
}

// ── gradient is zero when satisfied ────────────────────────────────────────

#[test]
fn gradient_zero_when_inside_box() {
    let r = Restraint::inside_box([0.0, 0.0, 0.0], [10.0, 10.0, 10.0]);
    let mut g = [0.0 as F; 3];
    r.gradient(&[5.0, 5.0, 5.0], SCALE, SCALE2, &mut g);
    assert!(g[0].abs() < TOL);
    assert!(g[1].abs() < TOL);
    assert!(g[2].abs() < TOL);
}

#[test]
fn gradient_zero_when_inside_sphere() {
    let r = Restraint::inside_sphere([0.0, 0.0, 0.0], 10.0);
    let mut g = [0.0 as F; 3];
    r.gradient(&[1.0, 1.0, 1.0], SCALE, SCALE2, &mut g);
    assert!(g[0].abs() < TOL);
    assert!(g[1].abs() < TOL);
    assert!(g[2].abs() < TOL);
}

#[test]
fn gradient_zero_when_outside_sphere() {
    let r = Restraint::outside_sphere([0.0, 0.0, 0.0], 5.0);
    let mut g = [0.0 as F; 3];
    r.gradient(&[10.0, 0.0, 0.0], SCALE, SCALE2, &mut g);
    assert!(g[0].abs() < TOL);
    assert!(g[1].abs() < TOL);
    assert!(g[2].abs() < TOL);
}

// ── gradient accumulates (does not overwrite) ──────────────────────────────

#[test]
fn gradient_accumulates() {
    let r = Restraint::inside_box([0.0, 0.0, 0.0], [10.0, 10.0, 10.0]);
    let mut g = [100.0 as F; 3];
    r.gradient(&[11.0, 5.0, 5.0], SCALE, SCALE2, &mut g);
    // g[0] should be > 100 (accumulated), g[1] and g[2] unchanged
    assert!(g[0] > 100.0, "gradient should accumulate, not overwrite");
    assert!((g[1] - 100.0).abs() < TOL);
    assert!((g[2] - 100.0).abs() < TOL);
}
