//! Shared test fixtures + regression sizing for the core benchmarks.
//!
//! These are **regression** benchmarks: each bench runs ONE small
//! representative input ([`REG_N`]) with a reduced criterion sample budget
//! ([`configure`]). The goal is "does it still run + catch a perf regression",
//! not scaling.

use std::time::Duration;

use criterion::BenchmarkGroup;
use criterion::measurement::Measurement;
use molrs::spatial::simbox::SimBox;
use molrs::types::F;
use ndarray::{Array1, Array2, array};
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};

/// Single representative element count for the core regression benches.
pub const REG_N: usize = 1_000;

/// Regression sampling: small sample count + short measurement window so each
/// bench completes in well under a second while still catching a regression.
pub fn configure<M: Measurement>(group: &mut BenchmarkGroup<'_, M>) {
    group.warm_up_time(Duration::from_millis(200));
    group.measurement_time(Duration::from_millis(500));
    group.sample_size(10);
}

/// Generate N random points inside a cubic box using the crate float type.
pub fn random_points(n: usize, box_size: F, seed: u64) -> Array2<F> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut pts = Array2::<F>::zeros((n, 3));
    for i in 0..n {
        pts[[i, 0]] = rng.random::<F>() * box_size;
        pts[[i, 1]] = rng.random::<F>() * box_size;
        pts[[i, 2]] = rng.random::<F>() * box_size;
    }
    pts
}

/// Create a PBC cubic SimBox.
pub fn make_pbc_simbox(size: F) -> SimBox {
    SimBox::cube(
        size,
        array![0.0 as F, 0.0 as F, 0.0 as F],
        [true, true, true],
    )
    .expect("invalid box length")
}

/// Generate N random `[F; 3]` points inside a cubic box (native Vec layout).
pub fn random_points_native(n: usize, box_size: F, seed: u64) -> Vec<[F; 3]> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..n)
        .map(|_| {
            [
                rng.random::<F>() * box_size,
                rng.random::<F>() * box_size,
                rng.random::<F>() * box_size,
            ]
        })
        .collect()
}

/// Generate a 1D `Vec<f32>` of length `n` with values in [0, 1).
pub fn random_1d_vec(n: usize, seed: u64) -> Vec<f32> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..n).map(|_| rng.random::<f32>()).collect()
}

/// Generate a 1D `Array1<f32>` of length `n` with values in [0, 1).
pub fn random_1d_ndarray(n: usize, seed: u64) -> Array1<f32> {
    let mut rng = StdRng::seed_from_u64(seed);
    Array1::from_iter((0..n).map(|_| rng.random::<f32>()))
}
