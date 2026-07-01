//! End-to-end tests for the geometric distribution functions (ADF / DDF /
//! distance DF), asserting against closed-form geometry and the PBC contract.
//!
//! Covers acceptance criteria ac-001..ac-006 of
//! `travis-parity-01-geometric-distributions`.

use molrs::Frame;
use molrs::compute::distribution::{
    AngleObservable, AtomGroups, DihedralObservable, DistanceObservable, DistributionFunction,
    Observable,
};
use molrs::compute::traits::Compute;
use molrs::spatial::region::simbox::SimBox;
use molrs::store::block::Block;
use molrs::types::F;
use ndarray::{Array1, array};

const PI: F = std::f64::consts::PI;

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------

fn frame(coords: &[[F; 3]], box_len: Option<F>) -> Frame {
    let n = coords.len();
    let xs = Array1::from_iter(coords.iter().map(|c| c[0]));
    let ys = Array1::from_iter(coords.iter().map(|c| c[1]));
    let zs = Array1::from_iter(coords.iter().map(|c| c[2]));
    let mut block = Block::new();
    block.insert("x", xs.into_dyn()).unwrap();
    block.insert("y", ys.into_dyn()).unwrap();
    block.insert("z", zs.into_dyn()).unwrap();
    let mut f = Frame::new();
    f.insert("atoms", block);
    if let Some(l) = box_len {
        f.simbox = Some(SimBox::cube(l, array![0.0 as F, 0.0, 0.0], [true, true, true]).unwrap());
    }
    let _ = n;
    f
}

// ---------------------------------------------------------------------------
// ac-001 — ADF reproduces a known 60° angle
// ---------------------------------------------------------------------------

#[test]
fn adf_peaks_at_sixty_degrees() {
    // Vertex j at origin; i along +x; k at 60° in the xy-plane.
    let i = [1.0, 0.0, 0.0];
    let j = [0.0, 0.0, 0.0];
    let k = [
        (60.0_f64).to_radians().cos(),
        (60.0_f64).to_radians().sin(),
        0.0,
    ];
    let f = frame(&[i, j, k], None);
    let groups = AtomGroups::triples(&[(0, 1, 2)]);

    // Raw sample is exactly π/3 (ac-001: f64 within 1e-9 before binning).
    let raw = AngleObservable.sample(&f, &groups).unwrap();
    assert!((raw[0] - PI / 3.0).abs() < 1e-9, "angle = {}", raw[0]);

    let df = DistributionFunction::over_natural_range(AngleObservable, 180).unwrap();
    let res = df.compute(&[&f], &groups).unwrap();
    let peak = res
        .density
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .unwrap()
        .0;
    // Peak bin center within one bin-width of 60°.
    assert!(
        (res.bin_centers[peak] - PI / 3.0).abs() <= res.bin_width,
        "peak at {} rad, expected π/3",
        res.bin_centers[peak]
    );
}

// ---------------------------------------------------------------------------
// ac-002 — dihedral matches IUPAC-signed reference torsions
// ---------------------------------------------------------------------------

#[test]
fn dihedral_matches_signed_reference() {
    // Central bond j→k along +x. i fixed in +y half-plane; l swept around the
    // x-axis: l = k + (0, cos φ, sin φ) ⇒ signed torsion = φ.
    let j = [0.0, 0.0, 0.0];
    let k = [1.0, 0.0, 0.0];
    let i = [0.0, 1.0, 0.0];
    let groups = AtomGroups::quads(&[(0, 1, 2, 3)]);

    for deg in [0.0_f64, 90.0, 180.0, -90.0] {
        let phi = deg.to_radians();
        let l = [1.0, phi.cos(), phi.sin()];
        let f = frame(&[i, j, k, l], None);
        let s = DihedralObservable.sample(&f, &groups).unwrap();
        // 180° and −180° are the same point; compare on the circle.
        let diff = ((s[0] - phi + PI).rem_euclid(2.0 * PI)) - PI;
        assert!(diff.abs() < 1e-9, "φ={deg}°: got {} rad", s[0]);
    }
}

// ---------------------------------------------------------------------------
// ac-003 — distance DF honors the minimum image
// ---------------------------------------------------------------------------

#[test]
fn distance_uses_minimum_image() {
    let box_len = 10.0;
    // Straddles the boundary: raw separation 9.0, minimum image 1.0.
    let f = frame(&[[0.5, 0.0, 0.0], [9.5, 0.0, 0.0]], Some(box_len));
    let groups = AtomGroups::pairs(&[(0, 1)]);
    let d = DistanceObservable.sample(&f, &groups).unwrap();
    assert!((d[0] - 1.0).abs() < 1e-9, "min-image distance = {}", d[0]);

    // Without a box, the same pair is the raw 9.0 separation.
    let f_free = frame(&[[0.5, 0.0, 0.0], [9.5, 0.0, 0.0]], None);
    let d_free = DistanceObservable.sample(&f_free, &groups).unwrap();
    assert!((d_free[0] - 9.0).abs() < 1e-9);
}

// ---------------------------------------------------------------------------
// ac-004 — normalization
// ---------------------------------------------------------------------------

#[test]
fn density_integrates_to_one_and_counts_sum() {
    // Distance DF over 3 pairs across 2 frames, all distances inside [0, 5].
    let f1 = frame(
        &[
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [3.5, 0.0, 0.0],
        ],
        None,
    );
    let f2 = frame(
        &[
            [0.0, 0.0, 0.0],
            [1.5, 0.0, 0.0],
            [2.5, 0.0, 0.0],
            [3.0, 0.0, 0.0],
        ],
        None,
    );
    let groups = AtomGroups::pairs(&[(0, 1), (1, 2), (2, 3)]);
    let df = DistributionFunction::new(DistanceObservable, 50, 0.0, 5.0).unwrap();
    let res = df.compute(&[&f1, &f2], &groups).unwrap();

    let integral: F = res.density.iter().map(|&p| p * res.bin_width).sum();
    assert!((integral - 1.0).abs() < 1e-6, "integral = {integral}");

    let total_counts: F = res.counts.iter().sum();
    assert_eq!(
        total_counts,
        (3 * 2) as F,
        "raw counts = n_samples × n_frames"
    );
}

// ---------------------------------------------------------------------------
// ac-005 — raw vs sin θ-corrected ADF both available and distinct
// ---------------------------------------------------------------------------

#[test]
fn adf_raw_and_sin_corrected_distinct() {
    // A non-uniform set of angles (clustered away from 90°).
    let mut coords = vec![[0.0, 0.0, 0.0]]; // vertex j = atom 0
    let mut triples = Vec::new();
    let i_idx = 1;
    coords.push([1.0, 0.0, 0.0]); // i = atom 1, along +x
    for (n, deg) in [20.0_f64, 25.0, 30.0, 35.0, 40.0].iter().enumerate() {
        let a = deg.to_radians();
        coords.push([a.cos(), a.sin(), 0.0]);
        triples.push((i_idx, 0u32, (2 + n) as u32));
    }
    let f = frame(&coords, None);
    let groups = AtomGroups::triples(&triples);
    let df = DistributionFunction::over_natural_range(AngleObservable, 180).unwrap();
    let res = df.compute(&[&f], &groups).unwrap();

    let corrected = res
        .density_sin_corrected
        .as_ref()
        .expect("ADF must expose a sin θ-corrected density");
    // Distinct from the raw density.
    let max_diff = res
        .density
        .iter()
        .zip(corrected.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);
    assert!(max_diff > 1e-6, "raw and corrected ADF should differ");

    // Corrected == raw/sin θ renormalized: recompute independently.
    let mut w = res.density.clone();
    for (idx, c) in res.bin_centers.iter().enumerate() {
        let s = c.sin();
        w[idx] = if s.abs() > 1e-12 {
            res.density[idx] / s
        } else {
            0.0
        };
    }
    let total: F = w.iter().sum::<F>() * res.bin_width;
    let expect = w.mapv(|x| x / total);
    for (a, b) in corrected.iter().zip(expect.iter()) {
        assert!((a - b).abs() < 1e-9, "corrected ADF mismatch: {a} vs {b}");
    }
}

// ---------------------------------------------------------------------------
// ac-006 — degenerate geometry: no NaN, typed error, no panic
// ---------------------------------------------------------------------------

#[test]
fn collinear_triple_is_zero_or_pi_no_nan() {
    // i, j, k collinear (k opposite i across vertex j) → angle π.
    let f = frame(&[[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [-1.0, 0.0, 0.0]], None);
    let groups = AtomGroups::triples(&[(0, 1, 2)]);
    let s = AngleObservable.sample(&f, &groups).unwrap();
    assert!(s[0].is_finite());
    assert!((s[0] - PI).abs() < 1e-9, "collinear angle = {}", s[0]);

    // Same point i and k on the same side → angle 0.
    let f0 = frame(&[[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [2.0, 0.0, 0.0]], None);
    let s0 = AngleObservable.sample(&f0, &groups).unwrap();
    assert!(s0[0].is_finite() && s0[0].abs() < 1e-9);
}

#[test]
fn zero_length_vector_is_typed_error() {
    // Vertex j coincides with i → zero-length bond.
    let f = frame(&[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], None);
    let groups = AtomGroups::triples(&[(0, 1, 2)]);
    let err = AngleObservable.sample(&f, &groups).unwrap_err();
    assert!(
        matches!(err, molrs::compute::error::ComputeError::NonFinite { .. }),
        "expected typed NonFinite error, got {err:?}"
    );
}

#[test]
fn empty_groups_yield_empty_result() {
    let f = frame(&[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], None);
    let groups = AtomGroups::pairs(&[]);
    let df = DistributionFunction::new(DistanceObservable, 10, 0.0, 5.0).unwrap();
    let res = df.compute(&[&f], &groups).unwrap();
    assert_eq!(res.counts.iter().sum::<F>(), 0.0);
    assert_eq!(res.n_raw_samples, 0);
    // density falls back to zeros, no NaN, no panic.
    assert!(res.density.iter().all(|&p| p == 0.0));
}
