//! Tests for every restraint type: `f()` = 0 when satisfied, `f()` > 0 when
//! violated, and gradient points in the correct direction.

use molrs_pack::{
    AboveGaussianRestraint, AbovePlaneRestraint, BelowGaussianRestraint, BelowPlaneRestraint, F,
    InsideBoxRestraint, InsideCubeRestraint, InsideCylinderRestraint, InsideEllipsoidRestraint,
    InsideSphereRestraint, OutsideBoxRestraint, OutsideCubeRestraint, OutsideCylinderRestraint,
    OutsideEllipsoidRestraint, OutsideSphereRestraint, Restraint,
};

const TOL: F = 1e-6;
const SCALE: F = 1.0;
const SCALE2: F = 0.01;

// ── helpers ────────────────────────────────────────────────────────────────

/// Accumulate gradient only (ignore returned value).
fn grad(r: &dyn Restraint, pos: &[F; 3], g: &mut [F; 3]) {
    let _ = r.fg(pos, SCALE, SCALE2, g);
}

/// Assert that the gradient at `pos` pushes the atom toward the satisfied
/// region (each nonzero gradient component opposes the violation).
fn assert_gradient_opposes_violation(r: &dyn Restraint, pos: &[F; 3], label: &str) {
    let h: F = 1e-4;
    let mut g = [0.0 as F; 3];
    grad(r, pos, &mut g);

    for axis in 0..3 {
        if g[axis].abs() < TOL {
            continue;
        }
        let mut pos_step = *pos;
        pos_step[axis] -= h * g[axis].signum();
        let v_before = r.f(pos, SCALE, SCALE2);
        let v_after = r.f(&pos_step, SCALE, SCALE2);
        assert!(
            v_after <= v_before + TOL,
            "{label}: stepping along -gradient on axis {axis} should reduce penalty \
             (before={v_before}, after={v_after}, g={:?})",
            g
        );
    }
}

// ── inside box (kind 3) ────────────────────────────────────────────────────

#[test]
fn inside_box_satisfied() {
    let r = InsideBoxRestraint::new([0.0, 0.0, 0.0], [10.0, 10.0, 10.0]);
    assert_eq!(r.f(&[5.0, 5.0, 5.0], SCALE, SCALE2), 0.0);
    assert_eq!(r.f(&[0.0, 0.0, 0.0], SCALE, SCALE2), 0.0);
    assert_eq!(r.f(&[10.0, 10.0, 10.0], SCALE, SCALE2), 0.0);
}

#[test]
fn inside_box_violated() {
    let r = InsideBoxRestraint::new([0.0, 0.0, 0.0], [10.0, 10.0, 10.0]);
    assert!(r.f(&[11.0, 5.0, 5.0], SCALE, SCALE2) > 0.0);
    assert!(r.f(&[-1.0, 5.0, 5.0], SCALE, SCALE2) > 0.0);
    assert!(r.f(&[5.0, -1.0, 5.0], SCALE, SCALE2) > 0.0);
}

#[test]
fn inside_box_gradient_direction() {
    let r = InsideBoxRestraint::new([0.0, 0.0, 0.0], [10.0, 10.0, 10.0]);
    let mut g = [0.0 as F; 3];
    grad(&r, &[11.0, 5.0, 5.0], &mut g);
    assert!(g[0] > 0.0);
    let mut g = [0.0 as F; 3];
    grad(&r, &[-1.0, 5.0, 5.0], &mut g);
    assert!(g[0] < 0.0);
    assert_gradient_opposes_violation(&r, &[11.0, 5.0, 5.0], "inside_box_above");
    assert_gradient_opposes_violation(&r, &[-1.0, -1.0, -1.0], "inside_box_below");
}

#[test]
fn inside_box_cube_from_origin() {
    let r = InsideBoxRestraint::cube_from_origin([0.0, 0.0, 0.0], 10.0);
    assert_eq!(r.min, [0.0, 0.0, 0.0]);
    assert_eq!(r.max, [10.0, 10.0, 10.0]);
}

// ── inside cube (kind 2) ───────────────────────────────────────────────────

#[test]
fn inside_cube_satisfied_and_violated() {
    let r = InsideCubeRestraint::new([0.0, 0.0, 0.0], 10.0);
    assert_eq!(r.f(&[5.0, 5.0, 5.0], SCALE, SCALE2), 0.0);
    assert!(r.f(&[11.0, 5.0, 5.0], SCALE, SCALE2) > 0.0);
    assert!(r.f(&[-1.0, 5.0, 5.0], SCALE, SCALE2) > 0.0);
    assert_gradient_opposes_violation(&r, &[11.0, 5.0, 5.0], "inside_cube");
}

// ── inside sphere (kind 4) ─────────────────────────────────────────────────

#[test]
fn inside_sphere_satisfied() {
    let r = InsideSphereRestraint::new([5.0, 5.0, 5.0], 10.0);
    assert_eq!(r.f(&[5.0, 5.0, 5.0], SCALE, SCALE2), 0.0);
    assert_eq!(r.f(&[0.0, 0.0, 0.0], SCALE, SCALE2), 0.0);
}

#[test]
fn inside_sphere_violated() {
    let r = InsideSphereRestraint::new([5.0, 5.0, 5.0], 10.0);
    assert!(r.f(&[100.0, 0.0, 0.0], SCALE, SCALE2) > 0.0);
}

#[test]
fn inside_sphere_gradient_direction() {
    let r = InsideSphereRestraint::new([0.0, 0.0, 0.0], 5.0);
    assert_gradient_opposes_violation(&r, &[7.0, 0.0, 0.0], "inside_sphere");
    assert_gradient_opposes_violation(&r, &[0.0, -7.0, 0.0], "inside_sphere_y");
}

// ── outside sphere (kind 8) ────────────────────────────────────────────────

#[test]
fn outside_sphere_satisfied() {
    let r = OutsideSphereRestraint::new([0.0, 0.0, 0.0], 5.0);
    assert_eq!(r.f(&[10.0, 0.0, 0.0], SCALE, SCALE2), 0.0);
}

#[test]
fn outside_sphere_violated() {
    let r = OutsideSphereRestraint::new([0.0, 0.0, 0.0], 5.0);
    assert!(r.f(&[1.0, 0.0, 0.0], SCALE, SCALE2) > 0.0);
}

#[test]
fn outside_sphere_gradient_direction() {
    let r = OutsideSphereRestraint::new([0.0, 0.0, 0.0], 5.0);
    assert_gradient_opposes_violation(&r, &[1.0, 0.0, 0.0], "outside_sphere");
}

// ── inside ellipsoid (kind 5) ──────────────────────────────────────────────

#[test]
fn inside_ellipsoid_satisfied_and_violated() {
    let r = InsideEllipsoidRestraint::new([0.0, 0.0, 0.0], [10.0, 5.0, 3.0], 1.0);
    assert_eq!(r.f(&[0.0, 0.0, 0.0], SCALE, SCALE2), 0.0);
    assert!(r.f(&[15.0, 0.0, 0.0], SCALE, SCALE2) > 0.0);
    assert_gradient_opposes_violation(&r, &[15.0, 0.0, 0.0], "inside_ellipsoid");
}

// ── outside ellipsoid (kind 9) ─────────────────────────────────────────────

#[test]
fn outside_ellipsoid_satisfied_and_violated() {
    let r = OutsideEllipsoidRestraint::new([0.0, 0.0, 0.0], [5.0, 5.0, 5.0], 1.0);
    assert_eq!(r.f(&[10.0, 0.0, 0.0], SCALE, SCALE2), 0.0);
    assert!(r.f(&[0.0, 0.0, 0.0], SCALE, SCALE2) > 0.0);
    assert_gradient_opposes_violation(&r, &[1.0, 0.0, 0.0], "outside_ellipsoid");
}

// ── outside box (kind 7) ───────────────────────────────────────────────────

#[test]
fn outside_box_satisfied_and_violated() {
    let r = OutsideBoxRestraint::new([0.0, 0.0, 0.0], [10.0, 10.0, 10.0]);
    assert_eq!(r.f(&[15.0, 5.0, 5.0], SCALE, SCALE2), 0.0);
    assert!(r.f(&[5.0, 5.0, 5.0], SCALE, SCALE2) > 0.0);
}

// ── outside cube (kind 6) ──────────────────────────────────────────────────

#[test]
fn outside_cube_satisfied_and_violated() {
    let r = OutsideCubeRestraint::new([0.0, 0.0, 0.0], 10.0);
    assert_eq!(r.f(&[15.0, 5.0, 5.0], SCALE, SCALE2), 0.0);
    assert!(r.f(&[5.0, 5.0, 5.0], SCALE, SCALE2) > 0.0);
}

// ── above plane (kind 10) ──────────────────────────────────────────────────

#[test]
fn above_plane_satisfied() {
    let r = AbovePlaneRestraint::new([0.0, 0.0, 1.0], 5.0);
    assert_eq!(r.f(&[0.0, 0.0, 6.0], SCALE, SCALE2), 0.0);
    assert_eq!(r.f(&[0.0, 0.0, 5.0], SCALE, SCALE2), 0.0);
}

#[test]
fn above_plane_violated() {
    let r = AbovePlaneRestraint::new([0.0, 0.0, 1.0], 5.0);
    assert!(r.f(&[0.0, 0.0, 4.0], SCALE, SCALE2) > 0.0);
}

#[test]
fn above_plane_gradient_direction() {
    let r = AbovePlaneRestraint::new([0.0, 0.0, 1.0], 5.0);
    let mut g = [0.0 as F; 3];
    grad(&r, &[0.0, 0.0, 4.0], &mut g);
    assert!(g[2] < 0.0);
    assert_gradient_opposes_violation(&r, &[0.0, 0.0, 4.0], "above_plane");
}

// ── below plane (kind 11) ──────────────────────────────────────────────────

#[test]
fn below_plane_satisfied() {
    let r = BelowPlaneRestraint::new([0.0, 0.0, 1.0], 5.0);
    assert_eq!(r.f(&[0.0, 0.0, 4.0], SCALE, SCALE2), 0.0);
    assert_eq!(r.f(&[0.0, 0.0, 5.0], SCALE, SCALE2), 0.0);
}

#[test]
fn below_plane_violated() {
    let r = BelowPlaneRestraint::new([0.0, 0.0, 1.0], 5.0);
    assert!(r.f(&[0.0, 0.0, 6.0], SCALE, SCALE2) > 0.0);
}

#[test]
fn below_plane_gradient_direction() {
    let r = BelowPlaneRestraint::new([0.0, 0.0, 1.0], 5.0);
    assert_gradient_opposes_violation(&r, &[0.0, 0.0, 6.0], "below_plane");
}

// ── inside cylinder (kind 12) ──────────────────────────────────────────────

#[test]
fn inside_cylinder_satisfied() {
    let r = InsideCylinderRestraint::new([0.0, 0.0, 0.0], [0.0, 0.0, 1.0], 5.0, 10.0);
    assert_eq!(r.f(&[0.0, 0.0, 5.0], SCALE, SCALE2), 0.0);
}

#[test]
fn inside_cylinder_violated_radial() {
    let r = InsideCylinderRestraint::new([0.0, 0.0, 0.0], [0.0, 0.0, 1.0], 5.0, 10.0);
    assert!(r.f(&[10.0, 0.0, 5.0], SCALE, SCALE2) > 0.0);
}

#[test]
fn inside_cylinder_violated_axial() {
    let r = InsideCylinderRestraint::new([0.0, 0.0, 0.0], [0.0, 0.0, 1.0], 5.0, 10.0);
    assert!(r.f(&[0.0, 0.0, -5.0], SCALE, SCALE2) > 0.0);
}

#[test]
fn inside_cylinder_gradient_direction() {
    let r = InsideCylinderRestraint::new([0.0, 0.0, 0.0], [0.0, 0.0, 1.0], 5.0, 10.0);
    assert_gradient_opposes_violation(&r, &[10.0, 0.0, 5.0], "inside_cylinder_radial");
}

// ── outside cylinder (kind 13) ─────────────────────────────────────────────

#[test]
fn outside_cylinder_satisfied_and_violated() {
    let r = OutsideCylinderRestraint::new([0.0, 0.0, 0.0], [0.0, 0.0, 1.0], 5.0, 10.0);
    assert_eq!(r.f(&[10.0, 0.0, 5.0], SCALE, SCALE2), 0.0);
    assert!(r.f(&[1.0, 0.0, 5.0], SCALE, SCALE2) > 0.0);
}

// ── above gaussian (kind 14) ───────────────────────────────────────────────

#[test]
fn above_gaussian_satisfied_and_violated() {
    let r = AboveGaussianRestraint::new(0.0, 0.0, 5.0, 5.0, 0.0, 10.0);
    assert_eq!(r.f(&[0.0, 0.0, 20.0], SCALE, SCALE2), 0.0);
    assert!(r.f(&[0.0, 0.0, 5.0], SCALE, SCALE2) > 0.0);
}

// ── below gaussian (kind 15) ───────────────────────────────────────────────

#[test]
fn below_gaussian_satisfied_and_violated() {
    let r = BelowGaussianRestraint::new(0.0, 0.0, 5.0, 5.0, 0.0, 10.0);
    assert_eq!(r.f(&[0.0, 0.0, -5.0], SCALE, SCALE2), 0.0);
    assert!(r.f(&[0.0, 0.0, 15.0], SCALE, SCALE2) > 0.0);
}

// ── gradient is zero when satisfied ────────────────────────────────────────

#[test]
fn gradient_zero_when_inside_box() {
    let r = InsideBoxRestraint::new([0.0, 0.0, 0.0], [10.0, 10.0, 10.0]);
    let mut g = [0.0 as F; 3];
    grad(&r, &[5.0, 5.0, 5.0], &mut g);
    assert!(g[0].abs() < TOL);
    assert!(g[1].abs() < TOL);
    assert!(g[2].abs() < TOL);
}

#[test]
fn gradient_zero_when_inside_sphere() {
    let r = InsideSphereRestraint::new([0.0, 0.0, 0.0], 10.0);
    let mut g = [0.0 as F; 3];
    grad(&r, &[1.0, 1.0, 1.0], &mut g);
    assert!(g[0].abs() < TOL);
    assert!(g[1].abs() < TOL);
    assert!(g[2].abs() < TOL);
}

#[test]
fn gradient_zero_when_outside_sphere() {
    let r = OutsideSphereRestraint::new([0.0, 0.0, 0.0], 5.0);
    let mut g = [0.0 as F; 3];
    grad(&r, &[10.0, 0.0, 0.0], &mut g);
    assert!(g[0].abs() < TOL);
    assert!(g[1].abs() < TOL);
    assert!(g[2].abs() < TOL);
}

// ── gradient accumulates (does not overwrite) ──────────────────────────────

#[test]
fn gradient_accumulates() {
    let r = InsideBoxRestraint::new([0.0, 0.0, 0.0], [10.0, 10.0, 10.0]);
    let mut g = [100.0 as F; 3];
    grad(&r, &[11.0, 5.0, 5.0], &mut g);
    assert!(g[0] > 100.0, "gradient should accumulate, not overwrite");
    assert!((g[1] - 100.0).abs() < TOL);
    assert!((g[2] - 100.0).abs() < TOL);
}

// ── Phase B.6 acceptance: user-plugin type equality + scope equivalence ────

/// User-defined `Restraint` — identical in shape to the 14 built-ins.
/// Demonstrates direction-3: no ceremony to plug in your own geometry.
#[derive(Debug, Clone, Copy)]
struct MyHalfSpaceRestraint {
    /// Quadratic penalty below `z_min`; zero above.
    z_min: F,
}

impl Restraint for MyHalfSpaceRestraint {
    fn f(&self, pos: &[F; 3], _scale: F, scale2: F) -> F {
        let v = (self.z_min - pos[2]).max(0.0);
        scale2 * v * v
    }
    fn fg(&self, pos: &[F; 3], scale: F, scale2: F, g: &mut [F; 3]) -> F {
        let v = (self.z_min - pos[2]).max(0.0);
        if v > 0.0 {
            g[2] += -2.0 * scale2 * v;
        }
        self.f(pos, scale, scale2)
    }
}

#[test]
fn user_plugin_restraint_is_type_equal_to_builtin() {
    // The same `with_restraint` call accepts both a built-in and a user type
    // — there is no second-class citizenship for plugins.
    use molrs_pack::Target;
    let t = Target::from_coords(&[[0.0; 3]], &[1.0], 1)
        .with_restraint(InsideBoxRestraint::new([0.0; 3], [10.0; 3]))
        .with_restraint(MyHalfSpaceRestraint { z_min: 5.0 });
    assert_eq!(t.molecule_restraints.len(), 2);
}

#[test]
fn user_plugin_restraint_gradient_matches_fd() {
    let r = MyHalfSpaceRestraint { z_min: 5.0 };
    let x = [0.0, 0.0, 2.0]; // below z_min → violated
    let mut g = [0.0; 3];
    let _ = r.fg(&x, 1.0, 1.0, &mut g);

    let h: F = 1e-5;
    for k in 0..3 {
        let mut xp = x;
        xp[k] += h;
        let mut xm = x;
        xm[k] -= h;
        let fd = (r.f(&xp, 1.0, 1.0) - r.f(&xm, 1.0, 1.0)) / (2.0 * h);
        assert!(
            (g[k] - fd).abs() < 1e-4,
            "user restraint gradient axis {k}: analytic={} fd={fd}",
            g[k]
        );
    }
}
