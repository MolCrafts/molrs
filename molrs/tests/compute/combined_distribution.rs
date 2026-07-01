//! End-to-end tests for Combined Distribution Functions (CDF).
//!
//! Covers acceptance criteria ac-001..ac-005 of
//! `travis-parity-02-combined-distribution-functions`: marginal consistency
//! with the link-01 1-D distributions, correlation-vs-independence resolution,
//! joint normalization (2-D and 3-D), sample-count validation, and the
//! free-energy floor.

use molrs::Frame;
use molrs::compute::distribution::{
    AngleObservable, AnyObservable, AtomGroups, AxisSpec, CombinedDistribution, DihedralObservable,
    DistanceObservable, DistributionFunction,
};
use molrs::compute::traits::Compute;
use molrs::store::block::Block;
use molrs::types::F;
use ndarray::Array1;

const PI: F = std::f64::consts::PI;

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------

fn frame_from(coords: &[[F; 3]]) -> Frame {
    let xs = Array1::from_iter(coords.iter().map(|c| c[0]));
    let ys = Array1::from_iter(coords.iter().map(|c| c[1]));
    let zs = Array1::from_iter(coords.iter().map(|c| c[2]));
    let mut block = Block::new();
    block.insert("x", xs.into_dyn()).unwrap();
    block.insert("y", ys.into_dyn()).unwrap();
    block.insert("z", zs.into_dyn()).unwrap();
    let mut f = Frame::new();
    f.insert("atoms", block);
    f
}

/// A 3-atom frame realizing distance `r` (pair 0–1) and angle `θ` (triple
/// 1–0–2, vertex 0). Vertex at origin, atom 1 along +x at distance `r`, atom 2
/// at radius `r` rotated by `θ` in the xy-plane.
fn dist_angle_frame(r: F, theta: F) -> Frame {
    frame_from(&[
        [0.0, 0.0, 0.0],
        [r, 0.0, 0.0],
        [r * theta.cos(), r * theta.sin(), 0.0],
    ])
}

fn dist_groups() -> AtomGroups {
    AtomGroups::pairs(&[(0, 1)])
}
fn angle_groups() -> AtomGroups {
    AtomGroups::triples(&[(1, 0, 2)])
}

const RMIN: F = 1.0;
const RMAX: F = 3.0;
const NX: usize = 10;
const NY: usize = 12;

// ---------------------------------------------------------------------------
// ac-001 — 2-D CDF marginals equal the link-01 1-D distributions
// ---------------------------------------------------------------------------

#[test]
fn marginals_equal_link01_distributions() {
    // A spread of (r, θ) samples, all strictly in range so no tuple is skipped
    // on either axis (then the joint marginals match the 1-D histograms exactly).
    let mut frames = Vec::new();
    for a in 0..7 {
        for b in 0..9 {
            let r = RMIN + (a as F + 0.5) / 7.0 * (RMAX - RMIN);
            let theta = (b as F + 0.5) / 9.0 * PI;
            frames.push(dist_angle_frame(r, theta));
        }
    }
    let refs: Vec<&Frame> = frames.iter().collect();
    let dg = dist_groups();
    let ag = angle_groups();

    let dist_df = DistributionFunction::new(DistanceObservable, NX, RMIN, RMAX).unwrap();
    let adf = DistributionFunction::new(AngleObservable, NY, 0.0, PI).unwrap();
    let d1 = dist_df.compute(&refs, &dg).unwrap();
    let a1 = adf.compute(&refs, &ag).unwrap();

    let cdf = CombinedDistribution::new(
        vec![DistanceObservable.into(), AngleObservable.into()],
        vec![
            AxisSpec::new(NX, RMIN, RMAX).unwrap(),
            AxisSpec::new(NY, 0.0, PI).unwrap().with_sin_weight(true),
        ],
    )
    .unwrap();
    let groups = [dg.clone(), ag.clone()];
    let joint = cdf.compute(&refs, &groups).unwrap();

    let m0 = joint.marginal(0);
    let m1 = joint.marginal(1);
    for i in 0..NX {
        assert!(
            (m0.density[i] - d1.density[i]).abs() < 1e-6,
            "distance marginal bin {i}: {} vs {}",
            m0.density[i],
            d1.density[i]
        );
    }
    for j in 0..NY {
        assert!(
            (m1.density[j] - a1.density[j]).abs() < 1e-6,
            "angle marginal bin {j}: {} vs {}",
            m1.density[j],
            a1.density[j]
        );
    }
}

// ---------------------------------------------------------------------------
// ac-002 — correlation vs independence
// ---------------------------------------------------------------------------

#[test]
fn correlated_data_lies_on_diagonal_band() {
    // θ is a deterministic function of r: as r sweeps its range, θ sweeps its
    // range monotonically. The joint density must be confined to a diagonal band.
    let mut frames = Vec::new();
    let m = 200;
    for s in 0..m {
        let frac = (s as F + 0.5) / m as F;
        let r = RMIN + frac * (RMAX - RMIN);
        let theta = frac * PI; // angle determined by distance
        frames.push(dist_angle_frame(r, theta));
    }
    let refs: Vec<&Frame> = frames.iter().collect();
    let cdf = CombinedDistribution::new(
        vec![DistanceObservable.into(), AngleObservable.into()],
        vec![
            AxisSpec::new(NX, RMIN, RMAX).unwrap(),
            AxisSpec::new(NX, 0.0, PI).unwrap(),
        ],
    )
    .unwrap();
    let groups = [dist_groups(), angle_groups()];
    let joint = cdf.compute(&refs, &groups).unwrap();

    // frac↦(r,θ) are both linear with identical fractional bin position, so
    // each sample's cloud-in-cell deposit lands in the 2×2 block {ip,ip+1}²:
    // occupied cells lie on the diagonal BAND |ix−iy| ≤ 1 (TRAVIS bilinear CIC
    // widens a sharp diagonal by one bin). Cells further off-diagonal are
    // exactly empty — the correlation is still resolved as a tight band.
    for ix in 0..NX {
        for iy in 0..NX {
            let c = joint.counts[joint.flat_index(&[ix, iy])];
            if ix.abs_diff(iy) >= 2 {
                assert_eq!(c, 0.0, "far-off-band cell ({ix},{iy}) = {c}");
            }
        }
    }
    // All mass stays within the diagonal band and totals to m (CIC conserves
    // weight: each sample deposits total weight 1).
    let band: F = (0..NX)
        .flat_map(|ix| (0..NX).map(move |iy| (ix, iy)))
        .filter(|&(ix, iy)| ix.abs_diff(iy) <= 1)
        .map(|(ix, iy)| joint.counts[joint.flat_index(&[ix, iy])])
        .sum();
    assert!((band - m as F).abs() < 1e-9, "band mass {band} != {m}");
}

#[test]
fn independent_data_factorizes_into_outer_product() {
    // Enumerate the full grid of (r_i, θ_j) once each: the joint is uniform, so
    // it equals the outer product of its (uniform) marginals exactly.
    let mut frames = Vec::new();
    for i in 0..NX {
        for j in 0..NY {
            let r = RMIN + (i as F + 0.5) / NX as F * (RMAX - RMIN);
            let theta = (j as F + 0.5) / NY as F * PI;
            frames.push(dist_angle_frame(r, theta));
        }
    }
    let refs: Vec<&Frame> = frames.iter().collect();
    let cdf = CombinedDistribution::new(
        vec![DistanceObservable.into(), AngleObservable.into()],
        vec![
            AxisSpec::new(NX, RMIN, RMAX).unwrap(),
            AxisSpec::new(NY, 0.0, PI).unwrap(),
        ],
    )
    .unwrap();
    let groups = [dist_groups(), angle_groups()];
    let joint = cdf.compute(&refs, &groups).unwrap();

    let m0 = joint.marginal(0);
    let m1 = joint.marginal(1);
    // For independent observables the joint density factorizes:
    // p_joint(x, y) == p_x(x) · p_y(y) (consistent units, 1/(Å·rad)).
    for ix in 0..NX {
        for iy in 0..NY {
            let pj = joint.density[joint.flat_index(&[ix, iy])];
            let outer = m0.density[ix] * m1.density[iy];
            assert!(
                (pj - outer).abs() < 1e-6,
                "cell ({ix},{iy}): joint {pj} vs outer {outer}"
            );
        }
    }
}

// ---------------------------------------------------------------------------
// ac-003 — joint density normalized (2-D and 3-D)
// ---------------------------------------------------------------------------

#[test]
fn joint_density_integrates_to_one_2d() {
    let mut frames = Vec::new();
    for s in 0..150 {
        let frac = (s as F + 0.5) / 150.0;
        frames.push(dist_angle_frame(RMIN + frac * (RMAX - RMIN), frac * PI));
    }
    let refs: Vec<&Frame> = frames.iter().collect();
    let cdf = CombinedDistribution::new(
        vec![DistanceObservable.into(), AngleObservable.into()],
        vec![
            AxisSpec::new(NX, RMIN, RMAX).unwrap(),
            AxisSpec::new(NY, 0.0, PI).unwrap(),
        ],
    )
    .unwrap();
    let groups = [dist_groups(), angle_groups()];
    let joint = cdf.compute(&refs, &groups).unwrap();
    let cell = joint.bin_width_product();
    let integral: F = joint.density.iter().map(|&p| p * cell).sum();
    assert!((integral - 1.0).abs() < 1e-6, "∬ = {integral}");
}

/// A 5-atom frame for a distance×angle×dihedral 3-D CDF.
/// Atoms: 0,1 give distance; 1-0-2 give angle; 0-1-2-3 give a dihedral.
fn three_obs_frame(r: F, theta: F, phi: F) -> Frame {
    // Build a standard dihedral geometry for atoms i=0,j=1,k=2,l=3.
    let i = [0.0, 0.0, 0.0];
    let j = [1.5, 0.0, 0.0];
    let k = [1.5 + 1.5 * theta.cos(), 1.5 * theta.sin(), 0.0];
    // l rotated by phi about the j-k axis (here approximated about +x for a
    // monotone phi sweep — sufficient for normalization, not a parity check).
    let l = [
        k[0] + 1.0 * phi.cos(),
        k[1] + 1.0 * phi.sin() * theta.cos(),
        1.0 * phi.sin() * theta.sin() + 0.3 * r,
    ];
    frame_from(&[i, j, k, l])
}

#[test]
fn joint_density_integrates_to_one_3d() {
    let mut frames = Vec::new();
    for s in 0..120 {
        let frac = (s as F + 0.5) / 120.0;
        frames.push(three_obs_frame(
            RMIN + frac * (RMAX - RMIN),
            0.3 + frac * 2.0,
            frac * PI,
        ));
    }
    let refs: Vec<&Frame> = frames.iter().collect();
    let cdf = CombinedDistribution::new(
        vec![
            DistanceObservable.into(),
            AngleObservable.into(),
            DihedralObservable.into(),
        ],
        vec![
            AxisSpec::new(6, 0.0, 5.0).unwrap(),
            AxisSpec::new(6, 0.0, PI).unwrap(),
            AxisSpec::new(6, -PI, PI).unwrap(),
        ],
    )
    .unwrap();
    let groups = [
        AtomGroups::pairs(&[(0, 1)]),
        AtomGroups::triples(&[(1, 0, 2)]),
        AtomGroups::quads(&[(0, 1, 2, 3)]),
    ];
    let joint = cdf.compute(&refs, &groups).unwrap();
    assert_eq!(joint.ndim(), 3);
    let cell = joint.bin_width_product();
    let integral: F = joint.density.iter().map(|&p| p * cell).sum();
    // Some samples may fall out of the (0,5) distance / dihedral ranges; the
    // normalization is over what was binned, so it must still integrate to 1.
    assert!(joint.binned > 0.0);
    assert!((integral - 1.0).abs() < 1e-6, "∭ = {integral}");
}

// ---------------------------------------------------------------------------
// ac-004 — mismatched observable sample counts rejected
// ---------------------------------------------------------------------------

#[test]
fn mismatched_sample_counts_error() {
    let f = dist_angle_frame(2.0, PI / 3.0);
    let cdf = CombinedDistribution::new(
        vec![DistanceObservable.into(), AngleObservable.into()],
        vec![
            AxisSpec::new(NX, RMIN, RMAX).unwrap(),
            AxisSpec::new(NY, 0.0, PI).unwrap(),
        ],
    )
    .unwrap();
    // 2 distance pairs but only 1 angle triple → unequal per-axis sample counts.
    let groups = [AtomGroups::pairs(&[(0, 1), (0, 2)]), angle_groups()];
    let err = cdf.compute(&[&f], &groups).unwrap_err();
    assert!(
        matches!(
            err,
            molrs::compute::error::ComputeError::DimensionMismatch { .. }
        ),
        "expected DimensionMismatch, got {err:?}"
    );
}

// ---------------------------------------------------------------------------
// ac-005 — free_energy finite on populated bins, floored on empty
// ---------------------------------------------------------------------------

#[test]
fn free_energy_is_finite_and_floored() {
    // Diagonal-band data leaves the off-band cells empty.
    let mut frames = Vec::new();
    for s in 0..100 {
        let frac = (s as F + 0.5) / 100.0;
        frames.push(dist_angle_frame(RMIN + frac * (RMAX - RMIN), frac * PI));
    }
    let refs: Vec<&Frame> = frames.iter().collect();
    let cdf = CombinedDistribution::new(
        vec![DistanceObservable.into(), AngleObservable.into()],
        vec![
            AxisSpec::new(NX, RMIN, RMAX).unwrap(),
            AxisSpec::new(NX, 0.0, PI).unwrap(),
        ],
    )
    .unwrap();
    let groups = [dist_groups(), angle_groups()];
    let joint = cdf.compute(&refs, &groups).unwrap();

    let g = joint.free_energy(300.0);
    // Every value finite (no -inf / NaN).
    assert!(
        g.iter().all(|v| v.is_finite()),
        "free energy has non-finite"
    );

    // Populated bins below the floor; empty bins exactly at the floor.
    let floor = g.iter().cloned().fold(F::NEG_INFINITY, F::max);
    for (i, &p) in joint.density.iter().enumerate() {
        if p > 0.0 {
            assert!(g[i] <= floor + 1e-12);
        } else {
            assert!((g[i] - floor).abs() < 1e-12, "empty bin {i} not at floor");
        }
    }
}

// ---------------------------------------------------------------------------
// Validation: ndim bounds + empty input
// ---------------------------------------------------------------------------

#[test]
fn rejects_unsupported_dimensionality() {
    let one = CombinedDistribution::new(
        vec![DistanceObservable.into()],
        vec![AxisSpec::new(NX, RMIN, RMAX).unwrap()],
    );
    assert!(one.is_err(), "1-D CDF must be rejected");
    let four: Vec<AnyObservable> = vec![
        DistanceObservable.into(),
        DistanceObservable.into(),
        DistanceObservable.into(),
        DistanceObservable.into(),
    ];
    let axes: Vec<AxisSpec> = (0..4)
        .map(|_| AxisSpec::new(4, 0.0, 4.0).unwrap())
        .collect();
    assert!(
        CombinedDistribution::new(four, axes).is_err(),
        "4-D rejected"
    );
}

#[test]
fn empty_frames_error() {
    let cdf = CombinedDistribution::new(
        vec![DistanceObservable.into(), AngleObservable.into()],
        vec![
            AxisSpec::new(NX, RMIN, RMAX).unwrap(),
            AxisSpec::new(NY, 0.0, PI).unwrap(),
        ],
    )
    .unwrap();
    let groups = [dist_groups(), angle_groups()];
    let refs: Vec<&Frame> = Vec::new();
    assert!(cdf.compute(&refs, &groups).is_err());
}
