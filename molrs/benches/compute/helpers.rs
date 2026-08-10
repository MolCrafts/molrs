//! Shared fixtures + regression sizing for the `molrs` compute benches.
//!
//! These are **regression** benchmarks, not peak-performance sweeps: every
//! kernel bench runs ONE small representative input ([`REG_N`] atoms over
//! [`REG_FRAMES`] frames) with a reduced criterion sample budget
//! ([`configure`]). The goal is "does it still run + catch a perf regression",
//! not throughput scaling. Every per-kernel bench file imports its fixtures
//! from here so the inputs stay consistent and the files stay short.

use std::time::Duration;

use criterion::BenchmarkGroup;
use criterion::measurement::Measurement;
use molrs::spatial::neighbors::{NeighborList, Neighbors, NeighborsStorage};
use molrs::spatial::simbox::SimBox;
use molrs::store::block::Block;
use molrs::store::frame::Frame;
use molrs::types::F;
use ndarray::{Array2, ArrayD, IxDyn, array};
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};

use molrs::compute::cluster::{Cluster, ClusterResult};
use molrs::compute::shape::center_of_mass::{COMResult, CenterOfMass};
use molrs::compute::shape::cluster_centers::{ClusterCenters, ClusterCentersResult};
use molrs::compute::traits::Compute;

// --- regression sizing ----------------------------------------------------

/// Neighbor cutoff shared by every fixture so nlists are reusable.
pub const CUTOFF: F = 4.0;
/// Liquid-like number density (atoms / Å³). Keeps the per-particle neighbor
/// count roughly constant.
pub const DENSITY: F = 0.03;
/// Bin count for radial kernels (RDF, van Hove, correlation function).
pub const RDF_BINS: usize = 100;

/// Single representative particle count for every regression bench (~1k atoms).
pub const REG_N: usize = 1_000;
/// Frames per trajectory fixture. A handful — enough for time-series kernels
/// (MSD, van Hove lags) without turning a regression check into a sweep.
pub const REG_FRAMES: usize = 3;

// --- box / fixture helpers ------------------------------------------------

/// Cubic box length that yields the target [`DENSITY`] for `n` atoms.
pub fn box_for_density(n: usize) -> F {
    (n as F / DENSITY).cbrt()
}

/// Regression sampling: small sample count + short measurement window so each
/// bench completes in well under a second while still catching a regression.
pub fn configure<M: Measurement>(group: &mut BenchmarkGroup<'_, M>) {
    group.warm_up_time(Duration::from_millis(200));
    group.measurement_time(Duration::from_millis(500));
    group.sample_size(10);
}

pub fn random_positions(n: usize, box_size: F, seed: u64) -> Array2<F> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut pts = Array2::<F>::zeros((n, 3));
    for i in 0..n {
        pts[[i, 0]] = rng.random::<F>() * box_size;
        pts[[i, 1]] = rng.random::<F>() * box_size;
        pts[[i, 2]] = rng.random::<F>() * box_size;
    }
    pts
}

/// Build a Frame with an `"atoms"` block holding `x`/`y`/`z` columns
/// plus a PBC simbox. Positions are the rows of `pts` (shape `[n, 3]`).
pub fn frame_from_positions(pts: &Array2<F>, simbox: SimBox) -> Frame {
    let n = pts.nrows();
    let col = |axis: usize| -> ArrayD<F> {
        let mut v = Vec::with_capacity(n);
        for i in 0..n {
            v.push(pts[[i, axis]]);
        }
        ArrayD::from_shape_vec(IxDyn(&[n]), v).expect("column shape")
    };
    let mut atoms = Block::new();
    atoms.insert("x", col(0)).expect("insert x");
    atoms.insert("y", col(1)).expect("insert y");
    atoms.insert("z", col(2)).expect("insert z");

    let mut frame = Frame::new();
    frame.insert("atoms", atoms);
    frame.simbox = Some(simbox);
    frame
}

/// Build a self-query [`Neighbors`] table for the given positions using the
/// cell-list backend, with every column present.
pub fn build_nlist(pts: &Array2<F>, simbox: &SimBox, cutoff: F) -> Neighbors {
    let mut nl = NeighborList::new(cutoff);
    nl.build(pts.view(), simbox);
    nl.neighbors(NeighborsStorage::FULL)
}

/// Build `n_frames` independent frames + neighbor lists at the requested
/// particle count and constant [`DENSITY`].
pub fn build_pool(n: usize, n_frames: usize, base_seed: u64) -> (Vec<Frame>, Vec<Neighbors>) {
    let box_len = box_for_density(n);
    let simbox = SimBox::cube(
        box_len,
        array![0.0 as F, 0.0 as F, 0.0 as F],
        [true, true, true],
    )
    .expect("invalid box");
    let mut frames = Vec::with_capacity(n_frames);
    let mut nlists = Vec::with_capacity(n_frames);
    for t in 0..n_frames {
        let pts = random_positions(n, box_len, base_seed + t as u64);
        let nl = build_nlist(&pts, &simbox, CUTOFF);
        let frame = frame_from_positions(&pts, simbox.clone());
        frames.push(frame);
        nlists.push(nl);
    }
    (frames, nlists)
}

/// The standard single-input regression pool: [`REG_N`] atoms over
/// [`REG_FRAMES`] frames.
pub fn reg_pool() -> (Vec<Frame>, Vec<Neighbors>) {
    build_pool(REG_N, REG_FRAMES, 42)
}

/// Precomputed upstream results reused by dependent kernel benches.
pub struct Deps {
    pub cluster: Vec<ClusterResult>,
    pub com: Vec<COMResult>,
    pub centers: Vec<ClusterCentersResult>,
}

/// Compute Cluster / COM / ClusterCenters once so kernel benches measure
/// only the kernel under test.
pub fn build_deps(frames: &[&Frame], nlists: &Vec<Neighbors>) -> Deps {
    let cluster = Cluster::new(2).compute(frames, nlists).unwrap();
    let com = CenterOfMass::new().compute(frames, &cluster).unwrap();
    let centers = ClusterCenters::new().compute(frames, &cluster).unwrap();
    Deps {
        cluster,
        com,
        centers,
    }
}
