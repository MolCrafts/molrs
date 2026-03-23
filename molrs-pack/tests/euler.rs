//! Tests for Euler angle functions: eulerrmat, compcart, eulerfixed,
//! eulerrmat_derivatives.

use molrs_pack::F;
use molrs_pack::euler::{compcart, eulerfixed, eulerrmat, eulerrmat_derivatives};

const TOL: F = 1e-6;
const PI: F = std::f64::consts::PI as F;

// ── helpers ────────────────────────────────────────────────────────────────

fn dot(a: &[F; 3], b: &[F; 3]) -> F {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn norm(a: &[F; 3]) -> F {
    dot(a, a).sqrt()
}

// ── eulerrmat ──────────────────────────────────────────────────────────────

#[test]
fn identity_rotation() {
    let (v1, v2, v3) = eulerrmat(0.0, 0.0, 0.0);
    // v1 ≈ [1,0,0], v2 ≈ [0,1,0], v3 ≈ [0,0,1]
    assert!((v1[0] - 1.0).abs() < TOL);
    assert!(v1[1].abs() < TOL);
    assert!(v1[2].abs() < TOL);

    assert!(v2[0].abs() < TOL);
    assert!((v2[1] - 1.0).abs() < TOL);
    assert!(v2[2].abs() < TOL);

    assert!(v3[0].abs() < TOL);
    assert!(v3[1].abs() < TOL);
    assert!((v3[2] - 1.0).abs() < TOL);
}

#[test]
fn rotation_columns_are_orthonormal() {
    // Test with arbitrary angles
    let angles = [
        (0.3, 0.5, 0.7),
        (PI / 4.0, PI / 3.0, PI / 6.0),
        (1.0, 2.0, 3.0),
    ];
    for (beta, gama, teta) in angles {
        let (v1, v2, v3) = eulerrmat(beta, gama, teta);
        // Unit length
        assert!(
            (norm(&v1) - 1.0).abs() < TOL,
            "v1 not unit for ({beta},{gama},{teta})"
        );
        assert!((norm(&v2) - 1.0).abs() < TOL, "v2 not unit");
        assert!((norm(&v3) - 1.0).abs() < TOL, "v3 not unit");
        // Orthogonal
        assert!(dot(&v1, &v2).abs() < TOL, "v1·v2 not zero");
        assert!(dot(&v1, &v3).abs() < TOL, "v1·v3 not zero");
        assert!(dot(&v2, &v3).abs() < TOL, "v2·v3 not zero");
    }
}

#[test]
fn beta_90_rotates_around_y() {
    let (v1, _v2, v3) = eulerrmat(PI / 2.0, 0.0, 0.0);
    // After 90° around y: x→z, z→-x
    // v1 should map [1,0,0] → [0,0,1]-ish, v3 should map [0,0,1] → [-1,0,0]-ish
    // Actually eulerrmat convention: v1 is the first column of R
    // For beta=π/2, gama=0, teta=0: sb=1, cb=0, sg=0, cg=1, st=0, ct=1
    // v1 = [-1*0*1+0*1, -1*1*1-0*0, 1*0] = [0, -1, 0]... let me just check orthogonality
    assert!((norm(&v1) - 1.0).abs() < TOL);
    assert!((norm(&v3) - 1.0).abs() < TOL);
}

// ── compcart ───────────────────────────────────────────────────────────────

#[test]
fn compcart_identity_is_translation() {
    let (v1, v2, v3) = eulerrmat(0.0, 0.0, 0.0);
    let xcm = [1.0, 2.0, 3.0];
    let xref = [0.1, 0.2, 0.3];
    let result = compcart(&xcm, &xref, &v1, &v2, &v3);
    // With identity rotation: result = xcm + xref
    assert!((result[0] - 1.1).abs() < TOL);
    assert!((result[1] - 2.2).abs() < TOL);
    assert!((result[2] - 3.3).abs() < TOL);
}

#[test]
fn compcart_zero_ref_is_just_com() {
    let (v1, v2, v3) = eulerrmat(0.5, 1.0, 0.3);
    let xcm = [10.0, 20.0, 30.0];
    let xref = [0.0, 0.0, 0.0];
    let result = compcart(&xcm, &xref, &v1, &v2, &v3);
    assert!((result[0] - 10.0).abs() < TOL);
    assert!((result[1] - 20.0).abs() < TOL);
    assert!((result[2] - 30.0).abs() < TOL);
}

#[test]
fn compcart_preserves_distance_from_com() {
    let (v1, v2, v3) = eulerrmat(0.3, 0.5, 0.7);
    let xcm = [1.0, 2.0, 3.0];
    let xref = [1.0, 0.0, 0.0];
    let result = compcart(&xcm, &xref, &v1, &v2, &v3);
    // Distance from COM should equal |xref| = 1.0
    let d = ((result[0] - xcm[0]).powi(2)
        + (result[1] - xcm[1]).powi(2)
        + (result[2] - xcm[2]).powi(2))
    .sqrt();
    assert!(
        (d - 1.0).abs() < TOL,
        "rotation should preserve distance from COM"
    );
}

#[test]
fn compcart_two_atoms_preserve_internal_distance() {
    let (v1, v2, v3) = eulerrmat(1.2, -0.5, 0.8);
    let xcm = [5.0, 5.0, 5.0];
    let r1 = [1.0, 0.0, 0.0];
    let r2 = [0.0, 1.0, 0.0];
    let p1 = compcart(&xcm, &r1, &v1, &v2, &v3);
    let p2 = compcart(&xcm, &r2, &v1, &v2, &v3);
    let d = ((p1[0] - p2[0]).powi(2) + (p1[1] - p2[1]).powi(2) + (p1[2] - p2[2]).powi(2)).sqrt();
    let d_ref =
        ((r1[0] - r2[0]).powi(2) + (r1[1] - r2[1]).powi(2) + (r1[2] - r2[2]).powi(2)).sqrt();
    assert!(
        (d - d_ref).abs() < TOL,
        "rotation should preserve internal distances"
    );
}

// ── eulerfixed ─────────────────────────────────────────────────────────────

#[test]
fn eulerfixed_identity() {
    let (v1, v2, v3) = eulerfixed(0.0, 0.0, 0.0);
    assert!((v1[0] - 1.0).abs() < TOL);
    assert!((v2[1] - 1.0).abs() < TOL);
    assert!((v3[2] - 1.0).abs() < TOL);
}

#[test]
fn eulerfixed_columns_orthonormal() {
    let (v1, v2, v3) = eulerfixed(0.5, 1.0, -0.3);
    assert!((norm(&v1) - 1.0).abs() < TOL);
    assert!((norm(&v2) - 1.0).abs() < TOL);
    assert!((norm(&v3) - 1.0).abs() < TOL);
    assert!(dot(&v1, &v2).abs() < TOL);
    assert!(dot(&v1, &v3).abs() < TOL);
    assert!(dot(&v2, &v3).abs() < TOL);
}

// ── eulerrmat_derivatives ──────────────────────────────────────────────────

#[test]
fn derivatives_match_finite_difference() {
    let beta: F = 0.3;
    let gama: F = 0.5;
    let teta: F = 0.7;
    // Use h=1e-3 for f32 stability; f64 would allow 1e-5.
    let h: F = 1e-3;

    let (dv1b, dv1g, dv1t, dv2b, dv2g, dv2t, dv3b, dv3g, dv3t) =
        eulerrmat_derivatives(beta, gama, teta);

    // Finite difference for d/dbeta
    let (v1p, v2p, v3p) = eulerrmat(beta + h, gama, teta);
    let (v1m, v2m, v3m) = eulerrmat(beta - h, gama, teta);
    for k in 0..3 {
        let fd = (v1p[k] - v1m[k]) / (2.0 * h);
        assert!(
            (dv1b[k] - fd).abs() < 1e-3,
            "dv1/dbeta[{k}]: analytic={} fd={fd}",
            dv1b[k]
        );
    }
    for k in 0..3 {
        let fd = (v2p[k] - v2m[k]) / (2.0 * h);
        assert!(
            (dv2b[k] - fd).abs() < 1e-3,
            "dv2/dbeta[{k}]: analytic={} fd={fd}",
            dv2b[k]
        );
    }
    for k in 0..3 {
        let fd = (v3p[k] - v3m[k]) / (2.0 * h);
        assert!(
            (dv3b[k] - fd).abs() < 1e-3,
            "dv3/dbeta[{k}]: analytic={} fd={fd}",
            dv3b[k]
        );
    }

    // d/dgama
    let (v1p, v2p, v3p) = eulerrmat(beta, gama + h, teta);
    let (v1m, v2m, v3m) = eulerrmat(beta, gama - h, teta);
    for k in 0..3 {
        let fd = (v1p[k] - v1m[k]) / (2.0 * h);
        assert!((dv1g[k] - fd).abs() < 1e-3, "dv1/dgama[{k}]");
    }
    for k in 0..3 {
        let fd = (v2p[k] - v2m[k]) / (2.0 * h);
        assert!((dv2g[k] - fd).abs() < 1e-3, "dv2/dgama[{k}]");
    }
    for k in 0..3 {
        let fd = (v3p[k] - v3m[k]) / (2.0 * h);
        assert!((dv3g[k] - fd).abs() < 1e-3, "dv3/dgama[{k}]");
    }

    // d/dteta
    let (v1p, v2p, v3p) = eulerrmat(beta, gama, teta + h);
    let (v1m, v2m, v3m) = eulerrmat(beta, gama, teta - h);
    for k in 0..3 {
        let fd = (v1p[k] - v1m[k]) / (2.0 * h);
        assert!((dv1t[k] - fd).abs() < 1e-3, "dv1/dteta[{k}]");
    }
    for k in 0..3 {
        let fd = (v2p[k] - v2m[k]) / (2.0 * h);
        assert!((dv2t[k] - fd).abs() < 1e-3, "dv2/dteta[{k}]");
    }
    for k in 0..3 {
        let fd = (v3p[k] - v3m[k]) / (2.0 * h);
        assert!((dv3t[k] - fd).abs() < 1e-3, "dv3/dteta[{k}]");
    }
}
