//! Van Hove integration tests — the two physical bridges (RDF at t=0, MSD as
//! the self second moment) plus multi-origin stability and edge cases.

use molrs::Frame;
use molrs::compute::msd::MSD;
use molrs::compute::rdf::RDF;
use molrs::compute::traits::Compute;
use molrs::compute::van_hove::VanHove;
use molrs::spatial::neighbors::{LinkCell, NbListAlgo, NeighborList};
use molrs::spatial::region::simbox::SimBox;
use molrs::store::block::Block;
use molrs::types::F;
use ndarray::{Array1, Array2, array};
use rand::RngExt;
use rand::SeedableRng;
use rand::rngs::StdRng;

fn frame_from(xs: Vec<F>, ys: Vec<F>, zs: Vec<F>, box_len: Option<F>) -> Frame {
    let mut block = Block::new();
    block.insert("x", Array1::from_vec(xs).into_dyn()).unwrap();
    block.insert("y", Array1::from_vec(ys).into_dyn()).unwrap();
    block.insert("z", Array1::from_vec(zs).into_dyn()).unwrap();
    let mut frame = Frame::new();
    frame.insert("atoms", block);
    if let Some(l) = box_len {
        frame.simbox =
            Some(SimBox::cube(l, array![0.0 as F, 0.0, 0.0], [true, true, true]).unwrap());
    }
    frame
}

fn random_frame(n: usize, box_len: F, seed: u64) -> Frame {
    let mut rng = StdRng::seed_from_u64(seed);
    let xs = (0..n).map(|_| rng.random::<F>() * box_len).collect();
    let ys = (0..n).map(|_| rng.random::<F>() * box_len).collect();
    let zs = (0..n).map(|_| rng.random::<F>() * box_len).collect();
    frame_from(xs, ys, zs, Some(box_len))
}

fn positions(frame: &Frame) -> Array2<F> {
    let atoms = frame.get("atoms").unwrap();
    let xs = atoms.get_float("x").unwrap();
    let ys = atoms.get_float("y").unwrap();
    let zs = atoms.get_float("z").unwrap();
    let n = xs.len();
    let mut p = Array2::<F>::zeros((n, 3));
    for i in 0..n {
        p[[i, 0]] = xs[i];
        p[[i, 1]] = ys[i];
        p[[i, 2]] = zs[i];
    }
    p
}

fn nlist(frame: &Frame, r_max: F) -> NeighborList {
    let pos = positions(frame);
    let mut lc = LinkCell::new().cutoff(r_max);
    lc.build(pos.view(), frame.simbox.as_ref().unwrap());
    lc.query().clone()
}

/// ac-001: G_d(r,0) = ρ·g(r) from the existing RDF, and G_s(r,0) is a spike in
/// the first r-bin.
#[test]
fn distinct_at_zero_lag_equals_rho_g_of_r() {
    let n = 300;
    let box_len: F = 10.0;
    let r_max: F = 4.0;
    let n_bins = 40;

    let frame = random_frame(n, box_len, 7);

    let rdf = RDF::new(n_bins, r_max, 0.0).unwrap();
    let rdf_res = rdf.compute(&[&frame], &vec![nlist(&frame, r_max)]).unwrap();

    let vh = VanHove::new(n_bins, r_max, vec![0]).unwrap();
    let vh_res = vh.compute(&[&frame], ()).unwrap();

    let rho = n as F / (box_len * box_len * box_len);
    // Interior bins (skip the empty short-range bins and the r_max edge bin).
    for k in 5..n_bins - 2 {
        let expected = rho * rdf_res.rdf[k];
        let got = vh_res.g_distinct[[0, k]];
        if expected > 1e-6 {
            let rel = (got - expected).abs() / expected;
            assert!(
                rel < 1e-3,
                "bin {k}: G_d={got:.6} vs ρg(r)={expected:.6} (rel {rel:.2e})"
            );
        }
    }

    // G_s(r,0): all displacements are zero → all density in bin 0.
    assert!(vh_res.g_self[[0, 0]] > 0.0);
    for k in 1..n_bins {
        assert_eq!(vh_res.g_self[[0, k]], 0.0, "self part leaked into bin {k}");
    }
}

/// ac-002: the self second moment ∫ r² G_s(r,t) dr equals the MSD from
/// compute::msd at each lag (within 2%), and the width grows like √t.
#[test]
fn self_second_moment_tracks_msd() {
    // Seeded 3-D Gaussian random walk, no periodic box (free displacement),
    // so VanHove self-displacements and MSD see identical raw motion.
    let n = 400;
    let t_frames = 40;
    let step: F = 0.1; // per-component std per frame
    let mut rng = StdRng::seed_from_u64(2024);

    let mut x = vec![0.0; n];
    let mut y = vec![0.0; n];
    let mut z = vec![0.0; n];
    let mut frames_owned: Vec<Frame> = Vec::with_capacity(t_frames);
    for _ in 0..t_frames {
        frames_owned.push(frame_from(x.clone(), y.clone(), z.clone(), None));
        for i in 0..n {
            // Box–Muller-free: uniform sum approximates Gaussian well enough.
            let g = |rng: &mut StdRng| -> F { (0..12).map(|_| rng.random::<F>()).sum::<F>() - 6.0 };
            x[i] += g(&mut rng) * step;
            y[i] += g(&mut rng) * step;
            z[i] += g(&mut rng) * step;
        }
    }
    let refs: Vec<&Frame> = frames_owned.iter().collect();

    let lags = vec![2usize, 5, 10];
    let r_max: F = 6.0;
    let vh = VanHove::new(2000, r_max, lags.clone()).unwrap();
    let vh_res = vh.compute(&refs, ()).unwrap();

    let msd = MSD::windowed().compute(&refs, ()).unwrap();

    for (li, &lag) in vh_res.lags.iter().enumerate() {
        let m2 = vh_res.self_second_moment(li);
        let msd_lag = msd.data[lag].mean;
        let rel = (m2 - msd_lag).abs() / msd_lag;
        assert!(
            rel < 0.02,
            "lag {lag}: ∫r²G_s={m2:.5} vs MSD={msd_lag:.5} (rel {rel:.3})"
        );
    }

    // Width ∝ √t: second moment at lag 10 ≈ 2× that at lag 5 (Fickian).
    let m5 = vh_res.self_second_moment(1);
    let m10 = vh_res.self_second_moment(2);
    let ratio = m10 / m5;
    assert!(
        (ratio - 2.0).abs() < 0.15,
        "diffusive width ratio {ratio:.3} ≉ 2"
    );
}

/// ac-004: multi-origin averaging is stable — doubling the origin stride
/// (halving origin count) changes the self second moment by < 5%.
#[test]
fn multi_origin_is_stable() {
    let n = 300;
    let t_frames = 40;
    let step: F = 0.1;
    let mut rng = StdRng::seed_from_u64(99);
    let mut x = vec![0.0; n];
    let mut y = vec![0.0; n];
    let mut z = vec![0.0; n];
    let mut frames_owned: Vec<Frame> = Vec::with_capacity(t_frames);
    for _ in 0..t_frames {
        frames_owned.push(frame_from(x.clone(), y.clone(), z.clone(), None));
        for i in 0..n {
            let g = |rng: &mut StdRng| -> F { (0..12).map(|_| rng.random::<F>()).sum::<F>() - 6.0 };
            x[i] += g(&mut rng) * step;
            y[i] += g(&mut rng) * step;
            z[i] += g(&mut rng) * step;
        }
    }
    let refs: Vec<&Frame> = frames_owned.iter().collect();

    let a = VanHove::new(2000, 6.0, vec![5])
        .unwrap()
        .compute(&refs, ())
        .unwrap();
    let b = VanHove::new(2000, 6.0, vec![5])
        .unwrap()
        .with_stride(2)
        .compute(&refs, ())
        .unwrap();
    let ma = a.self_second_moment(0);
    let mb = b.self_second_moment(0);
    let rel = (ma - mb).abs() / ma;
    assert!(
        rel < 0.05,
        "origin stride changed second moment by {rel:.3}"
    );
}

/// ac-006 (edge): a single-frame input yields only the t=0 row.
#[test]
fn single_frame_yields_only_zero_lag() {
    let frame = random_frame(50, 10.0, 1);
    let vh = VanHove::new(20, 4.0, vec![0, 3, 7]).unwrap();
    let res = vh.compute(&[&frame], ()).unwrap();
    assert_eq!(res.lags, vec![0]);
    assert_eq!(res.g_self.shape()[0], 1);
}

/// ac-006 (edge): no SimBox → distinct part is skipped (not a panic), self part
/// still computed.
#[test]
fn missing_simbox_skips_distinct() {
    let frame = frame_from(vec![0.0, 1.0], vec![0.0, 0.0], vec![0.0, 0.0], None);
    let vh = VanHove::new(10, 4.0, vec![0]).unwrap();
    let res = vh.compute(&[&frame], ()).unwrap();
    assert!(!res.has_distinct);
    assert!(res.g_distinct.iter().all(|&v| v == 0.0));
}

/// Empty trajectory → typed EmptyInput.
#[test]
fn empty_frames_is_error() {
    let vh = VanHove::new(10, 4.0, vec![0]).unwrap();
    let frames: Vec<&Frame> = Vec::new();
    assert!(vh.compute(&frames, ()).is_err());
}
