//! Shared test fixtures for benchmarks.

use molrs_core::region::simbox::SimBox;
use molrs_core::types::F;
use ndarray::{Array1, Array2, array};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

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
