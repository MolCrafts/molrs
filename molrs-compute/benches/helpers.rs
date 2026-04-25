//! Shared fixtures + sweep constants for `molrs-compute` benches.
//!
//! Every per-kernel bench file (`rdf.rs`, `cluster.rs`, …) imports the same
//! sweep axes from here so the matrix stays consistent and the individual
//! files stay short.

use std::time::Duration;

use criterion::BenchmarkGroup;
use criterion::measurement::Measurement;
use molrs::block::Block;
use molrs::frame::Frame;
use molrs::neighbors::{LinkCell, NbList, NeighborList};
use molrs::region::simbox::SimBox;
use molrs::types::F;
use ndarray::{Array2, ArrayD, IxDyn, array};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use molrs_compute::center_of_mass::{COMResult, CenterOfMass};
use molrs_compute::cluster::{Cluster, ClusterResult};
use molrs_compute::cluster_centers::{ClusterCenters, ClusterCentersResult};
use molrs_compute::traits::Compute;

// --- sweep constants ------------------------------------------------------

/// Neighbor cutoff shared by every fixture so nlists are reusable.
pub const CUTOFF: F = 4.0;
/// Liquid-like number density (atoms / Å³). Keeps the per-particle neighbor
/// count roughly constant across `N`.
pub const DENSITY: F = 0.03;
/// Bin count for the RDF sweeps.
pub const RDF_BINS: usize = 100;

/// Atom counts for the size-sweep axis (single-threaded scaling).
pub const SIZES: &[usize] = &[100, 500, 2_000, 10_000, 50_000];
/// Frames per size-sweep point. Small enough that setup doesn't dominate,
/// large enough that rayon stays engaged.
pub const SIZE_SWEEP_FRAMES: usize = 4;

/// Atom count for the frame-sweep axis (parallel scaling).
pub const FRAME_SWEEP_N: usize = 5_000;
/// Frame counts for the frame-sweep axis.
pub const FRAME_COUNTS: &[usize] = &[1, 2, 4, 8, 16, 32, 64];
/// Max frames in the frame-sweep pool (= last entry of FRAME_COUNTS).
pub const MAX_FRAMES: usize = 64;

// --- box / fixture helpers ------------------------------------------------

/// Cubic box length that yields the target `DENSITY` for `n` atoms.
pub fn box_for_density(n: usize) -> F {
    (n as F / DENSITY).cbrt()
}

/// Apply the standard warm-up / sample-count tuning to a criterion group.
pub fn configure<M: Measurement>(group: &mut BenchmarkGroup<'_, M>) {
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs(2));
    group.sample_size(15);
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
/// plus a PBC simbox. Positions are the columns of `pts` (shape `[n, 3]`).
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

/// Build a self-query [`NeighborList`] for the given positions using LinkCell.
pub fn build_nlist(pts: &Array2<F>, simbox: &SimBox, cutoff: F) -> NeighborList {
    let mut nl = NbList(LinkCell::new().cutoff(cutoff));
    nl.build(pts.view(), simbox);
    nl.query().clone()
}

/// Build `n_frames` independent frames + neighbor lists at the requested
/// particle count and constant [`DENSITY`].
pub fn build_pool(n: usize, n_frames: usize, base_seed: u64) -> (Vec<Frame>, Vec<NeighborList>) {
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

/// Precomputed upstream results reused by dependent kernel benches.
pub struct Deps {
    pub cluster: Vec<ClusterResult>,
    pub com: Vec<COMResult>,
    pub centers: Vec<ClusterCentersResult>,
}

/// Compute Cluster / COM / ClusterCenters once so kernel benches measure
/// only the kernel under test.
pub fn build_deps(frames: &[&Frame], nlists: &Vec<NeighborList>) -> Deps {
    let cluster = Cluster::new(2).compute(frames, nlists).unwrap();
    let com = CenterOfMass::new().compute(frames, &cluster).unwrap();
    let centers = ClusterCenters::new().compute(frames, &cluster).unwrap();
    Deps {
        cluster,
        com,
        centers,
    }
}

// --- legacy narrow-bench fixtures (kept for graph benches) ---------------

pub const BOX_SIZE: F = 30.0;

pub fn pbc_simbox(size: F) -> SimBox {
    SimBox::cube(
        size,
        array![0.0 as F, 0.0 as F, 0.0 as F],
        [true, true, true],
    )
    .expect("invalid box length")
}

/// One-shot single-frame fixture used by the Graph overhead bench.
#[allow(dead_code)]
pub struct Fixture {
    pub positions: Array2<F>,
    pub simbox: SimBox,
    pub frame: Frame,
    pub nlist: NeighborList,
}

pub fn fixture(n: usize, seed: u64) -> Fixture {
    let positions = random_positions(n, BOX_SIZE, seed);
    let simbox = pbc_simbox(BOX_SIZE);
    let nlist = build_nlist(&positions, &simbox, CUTOFF);
    let frame = frame_from_positions(&positions, simbox.clone());
    Fixture {
        positions,
        simbox,
        frame,
        nlist,
    }
}
