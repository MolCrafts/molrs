//! Legendre reorientation TCF integration tests — analytic constant-rate
//! rotation, the static limit, multi-origin exactness, and edge cases. Also
//! pins that `LegendreReorientation` (vector observable) is a separate type
//! from `order::RotationalAutocorrelation` (quaternion observable): both are
//! imported and constructed here.

use molrs::Frame;
use molrs::compute::order::{LegendreReorientation, RotationalAutocorrelation};
use molrs::compute::traits::Compute;
use molrs::store::block::Block;
use molrs::types::F;
use ndarray::Array1;

/// Build a frame with two atoms: atom 0 at the origin, atom 1 at `dir`.
fn bond_frame(dir: [F; 3]) -> Frame {
    let mut block = Block::new();
    block
        .insert("x", Array1::from_vec(vec![0.0, dir[0]]).into_dyn())
        .unwrap();
    block
        .insert("y", Array1::from_vec(vec![0.0, dir[1]]).into_dyn())
        .unwrap();
    block
        .insert("z", Array1::from_vec(vec![0.0, dir[2]]).into_dyn())
        .unwrap();
    let mut frame = Frame::new();
    frame.insert("atoms", block);
    frame
}

/// ac-003: a unit vector rotating at constant rate ω yields C_1(t)=cos(ωt) and
/// C_2(t)=(3cos²(ωt)−1)/2 to 1e-9.
#[test]
fn constant_rotation_matches_legendre_analytic() {
    let dtheta: F = 0.1; // ω·Δt, radians per frame
    let t_frames = 30;
    let frames_owned: Vec<Frame> = (0..t_frames)
        .map(|k| {
            let a = k as F * dtheta;
            bond_frame([a.cos(), a.sin(), 0.0])
        })
        .collect();
    let refs: Vec<&Frame> = frames_owned.iter().collect();

    let res = LegendreReorientation::new(15)
        .compute(&refs, &[(0u32, 1u32)])
        .unwrap();

    for (t, &lag) in res.lags.iter().enumerate() {
        let phi = lag as F * dtheta;
        let c1 = phi.cos();
        let c2 = 0.5 * (3.0 * c1 * c1 - 1.0);
        assert!(
            (res.c1[t] - c1).abs() < 1e-9,
            "C1[{lag}]={} vs {c1}",
            res.c1[t]
        );
        assert!(
            (res.c2[t] - c2).abs() < 1e-9,
            "C2[{lag}]={} vs {c2}",
            res.c2[t]
        );
    }
}

/// ac-003: a static vector yields C_1 = C_2 = 1 for all lags.
#[test]
fn static_vector_gives_unity() {
    let frames_owned: Vec<Frame> = (0..10).map(|_| bond_frame([0.0, 0.0, 1.0])).collect();
    let refs: Vec<&Frame> = frames_owned.iter().collect();
    let res = LegendreReorientation::new(5)
        .compute(&refs, &[(0u32, 1u32)])
        .unwrap();
    for t in 0..res.lags.len() {
        assert!((res.c1[t] - 1.0).abs() < 1e-12);
        assert!((res.c2[t] - 1.0).abs() < 1e-12);
    }
}

/// ac-004: multi-origin averaging is exact for uniform rotation — striding the
/// origins leaves C_ℓ unchanged.
#[test]
fn multi_origin_exact_for_uniform_rotation() {
    let dtheta: F = 0.07;
    let frames_owned: Vec<Frame> = (0..40)
        .map(|k| {
            let a = k as F * dtheta;
            bond_frame([a.cos(), a.sin(), 0.0])
        })
        .collect();
    let refs: Vec<&Frame> = frames_owned.iter().collect();
    let pairs = [(0u32, 1u32)];

    let a = LegendreReorientation::new(10)
        .compute(&refs, &pairs)
        .unwrap();
    let b = LegendreReorientation::new(10)
        .with_stride(3)
        .compute(&refs, &pairs)
        .unwrap();
    for t in 0..a.lags.len() {
        assert!((a.c1[t] - b.c1[t]).abs() < 1e-9);
        assert!((a.c2[t] - b.c2[t]).abs() < 1e-9);
    }
}

/// ac-006 (edge): a zero-length reorientation vector (coincident atoms) returns
/// a typed ComputeError rather than a silent NaN.
#[test]
fn zero_length_vector_is_error() {
    let frames_owned = [bond_frame([0.0, 0.0, 0.0])];
    let refs: Vec<&Frame> = frames_owned.iter().collect();
    let res = LegendreReorientation::new(0).compute(&refs, &[(0u32, 1u32)]);
    assert!(res.is_err(), "coincident atoms should error, got {res:?}");
}

/// ac-005: the two reorientation analyzers are distinct, independently usable
/// types — `LegendreReorientation` consumes a molecular vector while
/// `RotationalAutocorrelation` consumes quaternions. Constructing both here is
/// the compile-time + runtime witness of that separation.
#[test]
fn legendre_and_quaternion_acf_are_distinct_types() {
    let _legendre = LegendreReorientation::new(4);
    let _quat = RotationalAutocorrelation::new(2);
    // Legendre runs on a vector trajectory:
    let frames_owned: Vec<Frame> = (0..6).map(|_| bond_frame([1.0, 0.0, 0.0])).collect();
    let refs: Vec<&Frame> = frames_owned.iter().collect();
    let res = _legendre.compute(&refs, &[(0u32, 1u32)]).unwrap();
    assert_eq!(res.c1.len(), 5);
}
