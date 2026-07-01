//! End-to-end tests for the Spatial Distribution Function (SDF) and its native
//! Kabsch front-end.
//!
//! Covers acceptance criteria ac-001..ac-006 of
//! `travis-parity-03-spatial-distribution-function`.

use molrs::Frame;
use molrs::compute::density::kabsch::{det3, kabsch};
use molrs::compute::density::{GridSpec, SpatialDistribution};
use molrs::compute::traits::Compute;
use molrs::spatial::region::simbox::SimBox;
use molrs::store::block::Block;
use molrs::types::F;
use ndarray::{Array1, Array2, array};
use rand::RngExt;
use rand::SeedableRng;
use rand::rngs::StdRng;

// ---------------------------------------------------------------------------
// Geometry helpers
// ---------------------------------------------------------------------------

type M3 = [[F; 3]; 3];

fn rot_y(a: F) -> M3 {
    let (c, s) = (a.cos(), a.sin());
    [[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]]
}
fn rot_z(a: F) -> M3 {
    let (c, s) = (a.cos(), a.sin());
    [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]
}
fn matmul(a: &M3, b: &M3) -> M3 {
    let mut r = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            for k in 0..3 {
                r[i][j] += a[i][k] * b[k][j];
            }
        }
    }
    r
}
fn apply(r: &M3, v: [F; 3]) -> [F; 3] {
    [
        r[0][0] * v[0] + r[0][1] * v[1] + r[0][2] * v[2],
        r[1][0] * v[0] + r[1][1] * v[1] + r[1][2] * v[2],
        r[2][0] * v[0] + r[2][1] * v[1] + r[2][2] * v[2],
    ]
}

/// A non-degenerate rigid reference template (spans 3-D, centroid ≈ 0).
fn template() -> Array2<F> {
    // Centroid-subtracted so COM translation maps cleanly.
    let raw = [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ];
    let mut c = [0.0; 3];
    for r in &raw {
        for d in 0..3 {
            c[d] += r[d] / 4.0;
        }
    }
    let mut a = Array2::<F>::zeros((4, 3));
    for (i, r) in raw.iter().enumerate() {
        for d in 0..3 {
            a[[i, d]] = r[d] - c[d];
        }
    }
    a
}

fn frame(coords: &[[F; 3]], box_len: Option<F>) -> Frame {
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
    f
}

// ---------------------------------------------------------------------------
// ac-001 — native Kabsch recovers a known rotation, proper det, reflection guard
// ---------------------------------------------------------------------------

#[test]
fn kabsch_recovers_rotation_and_guards_reflection() {
    let templ = template();
    let r_true = matmul(&rot_z(0.6), &rot_y(-0.9));
    let t = [4.0, -3.0, 2.0];
    let mut fr = templ.clone();
    for i in 0..templ.nrows() {
        let v = [templ[[i, 0]], templ[[i, 1]], templ[[i, 2]]];
        let rv = apply(&r_true, v);
        for d in 0..3 {
            fr[[i, d]] = rv[d] + t[d];
        }
    }
    let (r, rmsd) = kabsch(templ.view(), fr.view()).unwrap();
    assert!(rmsd < 1e-9, "rmsd = {rmsd}");
    assert!((det3(&r) - 1.0).abs() < 1e-9, "det = {}", det3(&r));

    // Reflection guard: a mirrored copy still yields a proper (det +1) rotation.
    let mut mirror = templ.clone();
    for i in 0..templ.nrows() {
        mirror[[i, 0]] = -templ[[i, 0]];
    }
    let (rm, _) = kabsch(templ.view(), mirror.view()).unwrap();
    assert!((det3(&rm) - 1.0).abs() < 1e-9, "mirror det = {}", det3(&rm));
}

// ---------------------------------------------------------------------------
// ac-002 — SDF is frame-invariant: a body-fixed target → one sharp voxel
// ---------------------------------------------------------------------------

#[test]
fn sdf_is_frame_invariant_single_voxel() {
    let templ = template();
    // Mid-voxel offset (each component maps to a half-integer voxel coordinate,
    // so tiny numerical noise never flips floor() across a voxel boundary).
    let offset = [1.5, 0.7, -0.5]; // fixed body-frame position of the target
    let grid = GridSpec {
        n: [30, 30, 30],
        extent: [6.0, 6.0, 6.0],
    };

    // Tumble the whole assembly arbitrarily each frame; the body-frame relation
    // between reference and target never changes.
    let mut frames = Vec::new();
    for k in 0..20 {
        let a = 0.31 * k as F;
        let b = -0.21 * k as F + 0.5;
        let r_t = matmul(&rot_z(a), &rot_y(b));
        let trans = [10.0 + 0.7 * k as F, -5.0, 3.0];
        let mut coords = Vec::new();
        for i in 0..templ.nrows() {
            let v = [templ[[i, 0]], templ[[i, 1]], templ[[i, 2]]];
            let rv = apply(&r_t, v);
            coords.push([rv[0] + trans[0], rv[1] + trans[1], rv[2] + trans[2]]);
        }
        let tv = apply(&r_t, offset);
        coords.push([tv[0] + trans[0], tv[1] + trans[1], tv[2] + trans[2]]);
        frames.push(frame(&coords, None));
    }
    let refs: Vec<&Frame> = frames.iter().collect();

    let sdf = SpatialDistribution::new(vec![0, 1, 2, 3], templ, vec![4], grid).unwrap();
    let res = sdf.compute(&refs, ()).unwrap();

    // All counts land in the single voxel for `offset`; nowhere else.
    let total: F = res.counts.iter().sum();
    assert_eq!(total, 20.0);
    let peak = res.counts.iter().cloned().fold(0.0_f64, F::max);
    assert_eq!(
        peak, 20.0,
        "all frames must share one voxel (got peak {peak})"
    );
}

// ---------------------------------------------------------------------------
// ac-003 — bulk-normalized g_SDF → 1 for an ideal-gas target
// ---------------------------------------------------------------------------

#[test]
fn g_sdf_approaches_one_for_ideal_gas() {
    let templ = template();
    let grid = GridSpec {
        n: [10, 10, 10],
        extent: [10.0, 10.0, 10.0],
    };
    let region = 10.0_f64.powi(3); // extent volume
    let n_target = 4000usize;
    let bulk = n_target as F / region;

    // Reference fixed at the box centre (identity orientation); targets
    // uniformly fill the grid region each frame.
    let mut rng = StdRng::seed_from_u64(7);
    let mut frames = Vec::new();
    for _ in 0..6 {
        let mut coords = Vec::new();
        for i in 0..templ.nrows() {
            coords.push([
                templ[[i, 0]] + 25.0,
                templ[[i, 1]] + 25.0,
                templ[[i, 2]] + 25.0,
            ]);
        }
        for _ in 0..n_target {
            coords.push([
                25.0 + (rng.random::<F>() - 0.5) * 10.0,
                25.0 + (rng.random::<F>() - 0.5) * 10.0,
                25.0 + (rng.random::<F>() - 0.5) * 10.0,
            ]);
        }
        frames.push(frame(&coords, Some(50.0)));
    }
    let refs: Vec<&Frame> = frames.iter().collect();
    let target: Vec<usize> = (4..4 + n_target).collect();

    let sdf = SpatialDistribution::new(vec![0, 1, 2, 3], templ, target, grid)
        .unwrap()
        .with_bulk_density(bulk);
    let res = sdf.compute(&refs, ()).unwrap();
    let g = res.g_sdf.as_ref().expect("g_sdf present");

    // Mean over all voxels should be ≈ 1 (uniform target).
    let mean: F = g.iter().sum::<F>() / g.len() as F;
    assert!(
        (mean - 1.0).abs() < 0.05,
        "mean g_SDF = {mean}, expected ≈ 1"
    );
}

// ---------------------------------------------------------------------------
// ac-004 — target unwrapping honors PBC about the reference COM
// ---------------------------------------------------------------------------

#[test]
fn target_uses_minimum_image_about_com() {
    let templ = template();
    let l = 10.0;
    let grid = GridSpec {
        n: [20, 20, 20],
        extent: [4.0, 4.0, 4.0],
    };

    // Reference COM near the origin corner; target just across the +x boundary,
    // i.e. at x = 9.5 which is min-image −0.5 relative to a COM at x ≈ 0.
    let mut coords = Vec::new();
    for i in 0..templ.nrows() {
        coords.push([templ[[i, 0]], templ[[i, 1]], templ[[i, 2]]]);
    }
    // COM of the (centroid-zero) template is (0,0,0). Target across the edge:
    coords.push([9.5, 0.0, 0.0]);
    let f = frame(&coords, Some(l));

    let sdf = SpatialDistribution::new(vec![0, 1, 2, 3], templ, vec![4], grid).unwrap();
    let res = sdf.compute(&[&f], ()).unwrap();

    // Min-image offset is (−0.5, 0, 0): voxel index along x =
    // floor((−0.5 + 2)/0.2) = floor(7.5) = 7; raw (no MIC) would be x=9.5 → out
    // of the ±2 Å grid → no count. So exactly one count, at ix = 7.
    let total: F = res.counts.iter().sum();
    assert_eq!(total, 1.0, "min-image target must be inside the grid");
    let vs = grid.extent[0] / grid.n[0] as F;
    let ix = ((-0.5 + 0.5 * grid.extent[0]) / vs).floor() as usize;
    let mut found = None;
    for (idx, &c) in res.counts.indexed_iter() {
        if c > 0.0 {
            found = Some(idx);
        }
    }
    assert_eq!(
        found.unwrap().0,
        ix,
        "count must sit at the min-image voxel"
    );
}

// ---------------------------------------------------------------------------
// ac-005 — orientation field: fixed body vector → norm 1; random → norm 0
// ---------------------------------------------------------------------------

#[test]
fn orientation_field_fixed_vs_random() {
    let templ = template();
    // Mid-voxel offset so all frames deposit into a single voxel (see the
    // frame-invariance test for the boundary rationale).
    let offset = [1.3, 0.7, -0.5];
    let grid = GridSpec {
        n: [30, 30, 30],
        extent: [6.0, 6.0, 6.0],
    };

    // Build frames where the target sits at `offset` (body frame) and carries a
    // vector that is FIXED in the body frame (head = target + R·body_vec).
    let body_vec = [0.0, 1.0, 0.0];
    let make = |fixed: bool, seed: u64| {
        let mut rng = StdRng::seed_from_u64(seed);
        let mut frames = Vec::new();
        for k in 0..40 {
            let r_t = matmul(&rot_z(0.27 * k as F), &rot_y(0.13 * k as F + 0.2));
            let mut coords = Vec::new();
            for i in 0..templ.nrows() {
                let v = [templ[[i, 0]], templ[[i, 1]], templ[[i, 2]]];
                coords.push(apply(&r_t, v));
            }
            let tpos = apply(&r_t, offset);
            coords.push(tpos); // target (index 4)
            // head atom (index 5) defines the vector tail=4 → head=5.
            let lab_vec = if fixed {
                apply(&r_t, body_vec)
            } else {
                let u = [
                    rng.random::<F>() - 0.5,
                    rng.random::<F>() - 0.5,
                    rng.random::<F>() - 0.5,
                ];
                let n = (u[0] * u[0] + u[1] * u[1] + u[2] * u[2]).sqrt();
                [u[0] / n, u[1] / n, u[2] / n]
            };
            coords.push([
                tpos[0] + lab_vec[0],
                tpos[1] + lab_vec[1],
                tpos[2] + lab_vec[2],
            ]);
            frames.push(frame(&coords, None));
        }
        frames
    };

    // Fixed-vector case → per-voxel mean has norm ≈ 1.
    let frames = make(true, 1);
    let refs: Vec<&Frame> = frames.iter().collect();
    let sdf = SpatialDistribution::new(vec![0, 1, 2, 3], templ.clone(), vec![4], grid)
        .unwrap()
        .with_orientation(vec![(4, 5)]);
    let res = sdf.compute(&refs, ()).unwrap();
    let orient = res.orientation.as_ref().unwrap();
    let mut max_norm = 0.0_f64;
    for ix in 0..grid.n[0] {
        for iy in 0..grid.n[1] {
            for iz in 0..grid.n[2] {
                let v = [
                    orient[[ix, iy, iz, 0]],
                    orient[[ix, iy, iz, 1]],
                    orient[[ix, iy, iz, 2]],
                ];
                let nrm = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
                max_norm = max_norm.max(nrm);
            }
        }
    }
    assert!(
        (max_norm - 1.0).abs() < 1e-9,
        "fixed body vector → mean norm 1 (got {max_norm})"
    );

    // Random-vector case → mean norm → 0.
    let frames_r = make(false, 99);
    let refs_r: Vec<&Frame> = frames_r.iter().collect();
    let sdf_r = SpatialDistribution::new(vec![0, 1, 2, 3], templ, vec![4], grid)
        .unwrap()
        .with_orientation(vec![(4, 5)]);
    let res_r = sdf_r.compute(&refs_r, ()).unwrap();
    let orient_r = res_r.orientation.as_ref().unwrap();
    let mut rnorm = 0.0_f64;
    for ix in 0..grid.n[0] {
        for iy in 0..grid.n[1] {
            for iz in 0..grid.n[2] {
                let v = [
                    orient_r[[ix, iy, iz, 0]],
                    orient_r[[ix, iy, iz, 1]],
                    orient_r[[ix, iy, iz, 2]],
                ];
                rnorm = rnorm.max((v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt());
            }
        }
    }
    assert!(rnorm < 0.4, "random vector → mean norm → 0 (got {rnorm})");
}

// ---------------------------------------------------------------------------
// ac-006 — degenerate reference rejected
// ---------------------------------------------------------------------------

#[test]
fn degenerate_reference_is_rejected() {
    // Fewer than 3 reference atoms → constructor error.
    let two = Array2::<F>::zeros((2, 3));
    assert!(
        SpatialDistribution::new(
            vec![0, 1],
            two,
            vec![2],
            GridSpec {
                n: [4, 4, 4],
                extent: [2.0, 2.0, 2.0]
            }
        )
        .is_err()
    );

    // Three COLLINEAR reference atoms → constructor ok, but compute errors when
    // Kabsch sees the rank-deficient set.
    let line = array![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]];
    let coords = [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [5.0, 5.0, 5.0],
    ];
    let f = frame(&coords, None);
    let sdf = SpatialDistribution::new(
        vec![0, 1, 2],
        line,
        vec![3],
        GridSpec {
            n: [4, 4, 4],
            extent: [4.0, 4.0, 4.0],
        },
    )
    .unwrap();
    assert!(sdf.compute(&[&f], ()).is_err());
}
