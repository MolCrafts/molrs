//! k-means clustering with k-means++ initialization.
//!
//! [`KMeans`] is a [`Compute`] consuming an upstream [`PcaResult`]. The 2D
//! PCA scores are interpreted as a row-major `[n_rows, 2]` matrix and
//! clustered via Lloyd's algorithm with k-means++ init. Deterministic given a
//! fixed seed.

use molrs::frame_access::FrameAccess;
use molrs::types::F;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use crate::error::ComputeError;
use crate::pca::PcaResult;
use crate::result::ComputeResult;
use crate::traits::Compute;

/// Per-row cluster labels in `0..k`.
#[derive(Debug, Clone, Default)]
pub struct KMeansResult(pub Vec<i32>);

impl ComputeResult for KMeansResult {}

/// Configuration handle for k-means clustering.
#[derive(Debug, Clone, Copy)]
pub struct KMeans {
    k: usize,
    max_iter: usize,
    seed: u64,
}

const CENTROID_MOVE_SQ: F = 1e-16;

impl KMeans {
    pub fn new(k: usize, max_iter: usize, seed: u64) -> Result<Self, ComputeError> {
        if k == 0 {
            return Err(ComputeError::OutOfRange {
                field: "KMeans::k",
                value: k.to_string(),
            });
        }
        if max_iter == 0 {
            return Err(ComputeError::OutOfRange {
                field: "KMeans::max_iter",
                value: max_iter.to_string(),
            });
        }
        Ok(Self { k, max_iter, seed })
    }

    fn fit_coords(
        &self,
        coords: &[F],
        n_rows: usize,
        n_dims: usize,
    ) -> Result<Vec<i32>, ComputeError> {
        if n_dims == 0 {
            return Err(ComputeError::OutOfRange {
                field: "KMeans::n_dims",
                value: n_dims.to_string(),
            });
        }
        if self.k > n_rows {
            return Err(ComputeError::OutOfRange {
                field: "KMeans::k",
                value: format!("k={} exceeds n_rows={}", self.k, n_rows),
            });
        }
        if coords.len() != n_rows * n_dims {
            return Err(ComputeError::DimensionMismatch {
                expected: n_rows * n_dims,
                got: coords.len(),
                what: "KMeans coords length",
            });
        }
        for (i, &v) in coords.iter().enumerate() {
            if !v.is_finite() {
                return Err(ComputeError::NonFinite {
                    where_: "KMeans coords",
                    index: i,
                });
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
        assign_labels(coords, n_rows, n_dims, &centroids, self.k, &mut labels);

        Ok(labels)
    }
}

impl Compute for KMeans {
    type Args<'a> = &'a PcaResult;
    type Output = KMeansResult;

    fn compute<'a, FA: FrameAccess + Sync + 'a>(
        &self,
        _frames: &[&'a FA],
        pca: &'a PcaResult,
    ) -> Result<KMeansResult, ComputeError> {
        let n_dims = 2usize;
        if pca.coords.len() % n_dims != 0 {
            return Err(ComputeError::BadShape {
                expected: "coords length divisible by 2".into(),
                got: format!("len = {}", pca.coords.len()),
            });
        }
        let n_rows = pca.coords.len() / n_dims;
        let labels = self.fit_coords(&pca.coords, n_rows, n_dims)?;
        Ok(KMeansResult(labels))
    }
}

fn kmeans_pp_init(
    coords: &[F],
    n_rows: usize,
    n_dims: usize,
    k: usize,
    rng: &mut StdRng,
) -> Vec<F> {
    let mut centroids = Vec::with_capacity(k * n_dims);
    let first = rng.random_range(0..n_rows);
    centroids.extend_from_slice(&coords[first * n_dims..(first + 1) * n_dims]);

    let mut min_sq = vec![F::INFINITY; n_rows];
    update_min_sq(coords, n_rows, n_dims, &centroids[0..n_dims], &mut min_sq);

    for _c in 1..k {
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

fn update_min_sq(coords: &[F], n_rows: usize, n_dims: usize, new_centroid: &[F], min_sq: &mut [F]) {
    for i in 0..n_rows {
        let p = &coords[i * n_dims..(i + 1) * n_dims];
        let d2 = sq_dist(p, new_centroid);
        if d2 < min_sq[i] {
            min_sq[i] = d2;
        }
    }
}

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

fn centroid_move_sq(a: &[F], b: &[F]) -> F {
    debug_assert_eq!(a.len(), b.len());
    a.iter().zip(b.iter()).map(|(x, y)| (x - y).powi(2)).sum()
}

fn sq_dist(a: &[F], b: &[F]) -> F {
    debug_assert_eq!(a.len(), b.len());
    a.iter().zip(b.iter()).map(|(x, y)| (x - y).powi(2)).sum()
}

#[cfg(test)]
mod tests {
    use super::*;
    use molrs::Frame;
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};
    use std::collections::HashSet;

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

    fn three_blobs(n_per_cluster: usize, seed: u64) -> PcaResult {
        let centers = [(0.0, 0.0), (10.0, 0.0), (5.0, 10.0)];
        let sigma: F = 0.5;
        let mut rng = StdRng::seed_from_u64(seed);
        let total = 3 * n_per_cluster;
        let mut coords = Vec::with_capacity(total * 2);
        for (cx, cy) in centers.iter().copied() {
            for _ in 0..n_per_cluster {
                coords.push(cx + sigma * box_muller(&mut rng));
                coords.push(cy + sigma * box_muller(&mut rng));
            }
        }
        PcaResult {
            coords,
            variance: [1.0, 1.0],
        }
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
        let pca = three_blobs(20, 7);
        let km = KMeans::new(3, 100, 42).unwrap();
        let frame = Frame::new();
        let labels = km.compute(&[&frame], &pca).unwrap();

        assert_eq!(labels.0.len(), pca.coords.len() / 2);
        let unique: HashSet<i32> = labels.0.iter().copied().collect();
        assert_eq!(unique.len(), 3);

        let expected = (pca.coords.len() / 2) as i32 / 3;
        for c in 0..3i32 {
            let count = labels.0.iter().filter(|&&l| l == c).count() as i32;
            assert!(
                (count - expected).abs() <= 5,
                "cluster {c} size {count} not within ±5 of {expected}"
            );
        }
    }

    #[test]
    fn same_seed_identical_labels() {
        let pca = three_blobs(20, 7);
        let km = KMeans::new(3, 100, 42).unwrap();
        let frame = Frame::new();
        let a = km.compute(&[&frame], &pca).unwrap();
        let b = km.compute(&[&frame], &pca).unwrap();
        assert_eq!(a.0, b.0);
    }

    #[test]
    fn err_when_k_exceeds_rows() {
        let pca = PcaResult {
            coords: vec![0.0, 0.0, 1.0, 1.0],
            variance: [1.0, 1.0],
        };
        let km = KMeans::new(5, 10, 42).unwrap();
        let frame = Frame::new();
        let err = km.compute(&[&frame], &pca).unwrap_err();
        assert!(matches!(err, ComputeError::OutOfRange { .. }));
    }

    #[test]
    fn err_on_nan_input() {
        let pca = PcaResult {
            coords: vec![0.0, F::NAN, 1.0, 1.0, 2.0, 2.0],
            variance: [1.0, 1.0],
        };
        let km = KMeans::new(2, 10, 42).unwrap();
        let frame = Frame::new();
        let err = km.compute(&[&frame], &pca).unwrap_err();
        assert!(matches!(err, ComputeError::NonFinite { .. }));
    }
}
