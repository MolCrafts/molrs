//! k-means clustering with k-means++ initialization.
//!
//! Paired with [`super::pca`] in the dataset-explorer MVP — PCA produces the
//! 2D embedding, k-means colors it. The algorithm is deterministic given a
//! fixed `seed`: all RNG draws go through
//! [`rand::rngs::StdRng::seed_from_u64`].
//!
//! # Example
//!
//! ```
//! use molrs_compute::kmeans::KMeans;
//!
//! let coords: Vec<f64> = vec![
//!     0.0, 0.0, 0.1, 0.05, 0.02, -0.03, // cluster A
//!     5.0, 5.0, 5.1, 5.05, 4.98, 5.02,  // cluster B
//! ];
//! let km = KMeans::new(2, 100, 42).unwrap();
//! let labels = km.fit(&coords, 6, 2).unwrap();
//! assert_eq!(labels.len(), 6);
//! ```
//!
//! # Algorithm
//!
//! 1. **k-means++ init** — pick the first centroid uniformly at random, then
//!    pick each subsequent centroid with probability proportional to the
//!    squared distance from the nearest existing centroid.
//! 2. **Lloyd's iterations** — assign each point to its nearest centroid
//!    (squared Euclidean), then recompute each centroid as the mean of its
//!    assigned points. Repeat until total centroid L2 movement drops below
//!    `1e-8` or `max_iter` iterations have passed.

use crate::error::ComputeError;
use molrs::types::F;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

/// Configuration handle for k-means clustering.
///
/// Clone / copy freely — there's no heap state; everything lives on
/// [`fit`](Self::fit).
#[derive(Debug, Clone, Copy)]
pub struct KMeans {
    k: usize,
    max_iter: usize,
    seed: u64,
}

/// Convergence threshold on the sum of squared centroid displacements.
/// Equivalent to L2 movement of `sqrt(1e-16) = 1e-8`.
const CENTROID_MOVE_SQ: F = 1e-16;

impl KMeans {
    /// Create a new k-means configuration.
    ///
    /// # Arguments
    ///
    /// * `k` — number of clusters. Must be `>= 1`.
    /// * `max_iter` — maximum Lloyd's iterations. Must be `>= 1`.
    /// * `seed` — RNG seed for deterministic k-means++ initialization.
    ///
    /// # Errors
    ///
    /// Returns [`ComputeError::Invalid`] if `k == 0` or `max_iter == 0`.
    pub fn new(k: usize, max_iter: usize, seed: u64) -> Result<Self, ComputeError> {
        if k == 0 {
            return Err(ComputeError::Invalid("KMeans: k must be >= 1".into()));
        }
        if max_iter == 0 {
            return Err(ComputeError::Invalid(
                "KMeans: max_iter must be >= 1".into(),
            ));
        }
        Ok(Self { k, max_iter, seed })
    }

    /// Cluster `n_rows` points of dimension `n_dims`.
    ///
    /// # Arguments
    ///
    /// * `coords` — row-major `n_rows × n_dims` point matrix.
    /// * `n_rows` — number of points.
    /// * `n_dims` — dimensionality of each point.
    ///
    /// # Returns
    ///
    /// Cluster labels in `0..k`, one per row. The mapping from label integers
    /// to physical clusters is arbitrary (it depends on k-means++ ordering);
    /// same seed → identical labels.
    ///
    /// # Errors
    ///
    /// Returns [`ComputeError::Invalid`] on any of:
    /// - `k > n_rows` (cannot have more clusters than points)
    /// - `n_dims == 0`
    /// - `coords.len() != n_rows * n_dims`
    /// - any non-finite element in `coords`
    pub fn fit(
        &self,
        coords: &[F],
        n_rows: usize,
        n_dims: usize,
    ) -> Result<Vec<i32>, ComputeError> {
        if n_dims == 0 {
            return Err(ComputeError::Invalid("KMeans: n_dims must be >= 1".into()));
        }
        if self.k > n_rows {
            return Err(ComputeError::Invalid(format!(
                "KMeans: k ({}) must be <= n_rows ({})",
                self.k, n_rows
            )));
        }
        if coords.len() != n_rows * n_dims {
            return Err(ComputeError::Invalid(format!(
                "KMeans: coords length {} does not match n_rows * n_dims = {}",
                coords.len(),
                n_rows * n_dims
            )));
        }
        for (i, &v) in coords.iter().enumerate() {
            if !v.is_finite() {
                return Err(ComputeError::Invalid(format!(
                    "KMeans: non-finite coordinate at flat index {i}: {v}"
                )));
            }
        }

        let mut rng = StdRng::seed_from_u64(self.seed);
        let mut centroids = kmeans_pp_init(coords, n_rows, n_dims, self.k, &mut rng);
        let mut labels = vec![0i32; n_rows];

        for _ in 0..self.max_iter {
            assign_labels(coords, n_rows, n_dims, &centroids, self.k, &mut labels);
            let new_centroids =
                recompute_centroids(coords, n_rows, n_dims, &labels, self.k, &centroids);
            let move_sq = centroid_move_sq(&centroids, &new_centroids);
            centroids = new_centroids;
            if move_sq < CENTROID_MOVE_SQ {
                break;
            }
        }
        // Final label refresh against the last centroid update. This ensures
        // labels match the converged centroids even if the loop exits on the
        // movement check before re-assigning.
        assign_labels(coords, n_rows, n_dims, &centroids, self.k, &mut labels);

        Ok(labels)
    }
}

/// k-means++ initialization. Picks the first centroid uniformly at random,
/// then each subsequent centroid with probability proportional to the squared
/// distance to the nearest existing centroid.
fn kmeans_pp_init(
    coords: &[F],
    n_rows: usize,
    n_dims: usize,
    k: usize,
    rng: &mut StdRng,
) -> Vec<F> {
    let mut centroids = Vec::with_capacity(k * n_dims);

    // First centroid: uniform over points.
    let first = rng.random_range(0..n_rows);
    centroids.extend_from_slice(&coords[first * n_dims..(first + 1) * n_dims]);

    // Distance-to-nearest-centroid cache, one entry per point.
    let mut min_sq = vec![F::INFINITY; n_rows];
    update_min_sq(coords, n_rows, n_dims, &centroids[0..n_dims], &mut min_sq);

    for _c in 1..k {
        // Pick next centroid with probability proportional to min_sq.
        let total: F = min_sq.iter().sum();
        let next_idx = if total > 0.0 {
            let mut target: F = rng.random::<F>() * total;
            let mut chosen = n_rows - 1;
            for (i, &d) in min_sq.iter().enumerate() {
                target -= d;
                if target <= 0.0 {
                    chosen = i;
                    break;
                }
            }
            chosen
        } else {
            // All points coincide with existing centroids; fall back to uniform.
            rng.random_range(0..n_rows)
        };
        let start = centroids.len();
        centroids.extend_from_slice(&coords[next_idx * n_dims..(next_idx + 1) * n_dims]);
        update_min_sq(
            coords,
            n_rows,
            n_dims,
            &centroids[start..start + n_dims],
            &mut min_sq,
        );
    }

    centroids
}

/// Refresh `min_sq[i]` = min over centroids of squared Euclidean distance,
/// given a **newly added** centroid slice.
fn update_min_sq(coords: &[F], n_rows: usize, n_dims: usize, new_centroid: &[F], min_sq: &mut [F]) {
    for i in 0..n_rows {
        let p = &coords[i * n_dims..(i + 1) * n_dims];
        let d2 = sq_dist(p, new_centroid);
        if d2 < min_sq[i] {
            min_sq[i] = d2;
        }
    }
}

/// Assign each row to its nearest centroid (squared Euclidean).
fn assign_labels(
    coords: &[F],
    n_rows: usize,
    n_dims: usize,
    centroids: &[F],
    k: usize,
    labels: &mut [i32],
) {
    for i in 0..n_rows {
        let p = &coords[i * n_dims..(i + 1) * n_dims];
        let mut best = 0usize;
        let mut best_d2 = F::INFINITY;
        for c in 0..k {
            let cc = &centroids[c * n_dims..(c + 1) * n_dims];
            let d2 = sq_dist(p, cc);
            if d2 < best_d2 {
                best_d2 = d2;
                best = c;
            }
        }
        labels[i] = best as i32;
    }
}

/// Recompute centroids as the mean of points assigned to each cluster.
/// Empty clusters retain their previous centroid (a conservative choice — the
/// alternative is to re-seed from the farthest point, but for MVP sizes this
/// is rare and the stable fallback is simpler to test).
fn recompute_centroids(
    coords: &[F],
    n_rows: usize,
    n_dims: usize,
    labels: &[i32],
    k: usize,
    prev: &[F],
) -> Vec<F> {
    let mut sums = vec![0.0 as F; k * n_dims];
    let mut counts = vec![0usize; k];
    for i in 0..n_rows {
        let lab = labels[i] as usize;
        counts[lab] += 1;
        let start = lab * n_dims;
        for d in 0..n_dims {
            sums[start + d] += coords[i * n_dims + d];
        }
    }
    let mut out = vec![0.0 as F; k * n_dims];
    for (c, &count) in counts.iter().enumerate() {
        let start = c * n_dims;
        if count == 0 {
            out[start..start + n_dims].copy_from_slice(&prev[start..start + n_dims]);
        } else {
            let inv = 1.0 / count as F;
            for d in 0..n_dims {
                out[start + d] = sums[start + d] * inv;
            }
        }
    }
    out
}

/// Sum of squared per-coordinate differences between old and new centroid sets.
fn centroid_move_sq(a: &[F], b: &[F]) -> F {
    debug_assert_eq!(a.len(), b.len());
    a.iter().zip(b.iter()).map(|(x, y)| (x - y).powi(2)).sum()
}

/// Squared Euclidean distance between two equal-length slices.
fn sq_dist(a: &[F], b: &[F]) -> F {
    debug_assert_eq!(a.len(), b.len());
    a.iter().zip(b.iter()).map(|(x, y)| (x - y).powi(2)).sum()
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};
    use std::collections::HashSet;

    /// Same Gaussian-blob fixture as PCA tests — repeated here to avoid
    /// coupling the modules; test utilities are cheap.
    fn box_muller(rng: &mut StdRng) -> F {
        loop {
            let u1: F = rng.random();
            let u2: F = rng.random();
            if u1 > 0.0 {
                let r = (-2.0 * u1.ln()).sqrt();
                let theta = 2.0 * std::f64::consts::PI * u2;
                return r * theta.cos();
            }
        }
    }

    fn three_blobs(n_per_cluster: usize, seed: u64) -> (Vec<F>, usize) {
        let centers = [(0.0, 0.0), (10.0, 0.0), (5.0, 10.0)];
        let sigma: F = 0.5;
        let mut rng = StdRng::seed_from_u64(seed);
        let total = 3 * n_per_cluster;
        let mut data = Vec::with_capacity(total * 2);
        for (cx, cy) in centers.iter().copied() {
            for _ in 0..n_per_cluster {
                data.push(cx + sigma * box_muller(&mut rng));
                data.push(cy + sigma * box_muller(&mut rng));
            }
        }
        (data, total)
    }

    #[test]
    fn new_rejects_zero_k() {
        assert!(KMeans::new(0, 100, 42).is_err());
    }

    #[test]
    fn new_rejects_zero_max_iter() {
        assert!(KMeans::new(3, 0, 42).is_err());
    }

    #[test]
    fn three_blobs_produce_three_clusters() {
        let (coords, n_rows) = three_blobs(20, 7);
        let km = KMeans::new(3, 100, 42).unwrap();
        let labels = km.fit(&coords, n_rows, 2).unwrap();

        assert_eq!(labels.len(), n_rows);
        let unique: HashSet<i32> = labels.iter().copied().collect();
        assert_eq!(unique.len(), 3, "expected 3 unique labels, got {unique:?}");

        // Roughly equal cluster sizes (within ±5 of n/3).
        let expected = n_rows as i32 / 3;
        for c in 0..3i32 {
            let count = labels.iter().filter(|&&l| l == c).count() as i32;
            assert!(
                (count - expected).abs() <= 5,
                "cluster {c} size {count} not within ±5 of {expected}"
            );
        }
    }

    #[test]
    fn same_seed_identical_labels() {
        let (coords, n_rows) = three_blobs(20, 7);
        let km = KMeans::new(3, 100, 42).unwrap();
        let a = km.fit(&coords, n_rows, 2).unwrap();
        let b = km.fit(&coords, n_rows, 2).unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn different_seed_still_three_clusters() {
        let (coords, n_rows) = three_blobs(20, 7);
        let km_a = KMeans::new(3, 100, 42).unwrap();
        let km_b = KMeans::new(3, 100, 99).unwrap();
        let a = km_a.fit(&coords, n_rows, 2).unwrap();
        let b = km_b.fit(&coords, n_rows, 2).unwrap();

        let unique_a: HashSet<i32> = a.iter().copied().collect();
        let unique_b: HashSet<i32> = b.iter().copied().collect();
        assert_eq!(unique_a.len(), 3);
        assert_eq!(unique_b.len(), 3);
    }

    #[test]
    fn err_when_k_exceeds_rows() {
        let coords = vec![0.0, 0.0, 1.0, 1.0];
        let km = KMeans::new(5, 10, 42).unwrap();
        let err = km.fit(&coords, 2, 2).unwrap_err();
        assert!(matches!(err, ComputeError::Invalid(_)));
    }

    #[test]
    fn err_on_zero_dims() {
        let km = KMeans::new(2, 10, 42).unwrap();
        let err = km.fit(&[], 5, 0).unwrap_err();
        assert!(matches!(err, ComputeError::Invalid(_)));
    }

    #[test]
    fn err_on_nan_input() {
        let mut coords = vec![0.0 as F; 20];
        coords[3] = F::NAN;
        let km = KMeans::new(2, 10, 42).unwrap();
        let err = km.fit(&coords, 10, 2).unwrap_err();
        assert!(matches!(err, ComputeError::Invalid(_)));
    }

    #[test]
    fn err_on_length_mismatch() {
        let coords = vec![0.0 as F; 7];
        let km = KMeans::new(2, 10, 42).unwrap();
        let err = km.fit(&coords, 4, 2).unwrap_err();
        assert!(matches!(err, ComputeError::Invalid(_)));
    }
}
