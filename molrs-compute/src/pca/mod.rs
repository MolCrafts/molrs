//! 2-component Principal Component Analysis (PCA) for dataset exploration.
//!
//! Given a row-major `n_rows × n_cols` matrix of numeric observations, [`Pca2`]
//! produces a 2D projection suitable for scatter-plot visualization (see the
//! dataset-explorer spec). Columns are z-score standardized before covariance
//! is computed, so descriptor magnitudes do not dominate the directions.
//!
//! The top two eigenvectors of the covariance matrix are extracted via
//! **power iteration with deflation** — no external linear-algebra crate is
//! needed, the matrices are small (typically tens of columns), and the
//! algorithm is deterministic given a deterministic initial vector.
//!
//! # Example
//!
//! ```
//! use molrs_compute::pca::Pca2;
//!
//! // 3 points × 2 features (trivial, but illustrates shape):
//! let data = vec![0.0, 0.0, 1.0, 1.0, 2.0, 2.0];
//! let res = Pca2::fit_transform(&data, 3, 2).unwrap();
//! assert_eq!(res.coords.len(), 6);
//! ```
//!
//! # Errors
//!
//! [`Pca2::fit_transform`] rejects inputs that cannot produce a stable 2D
//! projection: `n_rows < 3`, `n_cols < 2`, non-finite values, or any
//! zero-variance column (which would cause division by zero during
//! standardization).

use crate::error::ComputeError;
use molrs::types::F;

/// Stateless PCA calculator with two components.
///
/// All parameters live on [`fit_transform`](Self::fit_transform); the struct
/// itself carries no state and is unit-sized.
#[derive(Debug, Clone, Copy, Default)]
pub struct Pca2;

/// Result of a 2-component PCA projection.
#[derive(Debug, Clone)]
pub struct PcaResult {
    /// Projected coordinates, row-major of shape `[n_rows, 2]`.
    ///
    /// `coords[2 * i + 0]` is the PC1 score for row `i`; `coords[2 * i + 1]`
    /// is the PC2 score.
    pub coords: Vec<F>,

    /// Explained variance per component: `variance[0]` is the eigenvalue
    /// associated with PC1, `variance[1]` with PC2. By construction
    /// `variance[0] >= variance[1] >= 0`. The "sample" covariance uses
    /// `1 / (n_rows - 1)`.
    pub variance: [F; 2],
}

/// Power-iteration convergence threshold on `||v_new - v_old||`.
const POWER_ITER_TOL: F = 1e-12;

/// Cap on power-iteration sweeps per component.
const POWER_ITER_MAX: usize = 200;

/// Minimum acceptable column standard deviation. Anything below this is
/// treated as a zero-variance column and rejected, because z-score
/// standardization would divide by near-zero.
const STD_FLOOR: F = 1e-12;

impl Pca2 {
    /// Standardize columns, compute the covariance matrix, and project the
    /// rows onto the top two eigenvectors.
    ///
    /// # Arguments
    ///
    /// * `matrix` — row-major `n_rows × n_cols` observation matrix.
    /// * `n_rows` — number of observations (frames, samples, rows).
    /// * `n_cols` — number of features/descriptors.
    ///
    /// # Errors
    ///
    /// Returns [`ComputeError::Invalid`] on any of:
    /// - `matrix.len() != n_rows * n_cols`
    /// - `n_rows < 3`
    /// - `n_cols < 2`
    /// - any non-finite element in `matrix`
    /// - any column with standard deviation below [`STD_FLOOR`] (1e-12)
    pub fn fit_transform(
        matrix: &[F],
        n_rows: usize,
        n_cols: usize,
    ) -> Result<PcaResult, ComputeError> {
        if n_rows < 3 {
            return Err(ComputeError::Invalid(format!(
                "PCA: n_rows must be >= 3, got {n_rows}"
            )));
        }
        if n_cols < 2 {
            return Err(ComputeError::Invalid(format!(
                "PCA: n_cols must be >= 2, got {n_cols}"
            )));
        }
        if matrix.len() != n_rows * n_cols {
            return Err(ComputeError::Invalid(format!(
                "PCA: matrix length {} does not match n_rows * n_cols = {}",
                matrix.len(),
                n_rows * n_cols
            )));
        }
        for (i, &v) in matrix.iter().enumerate() {
            if !v.is_finite() {
                return Err(ComputeError::Invalid(format!(
                    "PCA: non-finite value at flat index {i}: {v}"
                )));
            }
        }

        // ---- 1. Column means ------------------------------------------------
        let mut mean = vec![0.0 as F; n_cols];
        for i in 0..n_rows {
            for j in 0..n_cols {
                mean[j] += matrix[i * n_cols + j];
            }
        }
        let inv_n = 1.0 / n_rows as F;
        for m in mean.iter_mut() {
            *m *= inv_n;
        }

        // ---- 2. Column sample standard deviations --------------------------
        let mut var = vec![0.0 as F; n_cols];
        for i in 0..n_rows {
            for j in 0..n_cols {
                let d = matrix[i * n_cols + j] - mean[j];
                var[j] += d * d;
            }
        }
        let n_minus_1 = (n_rows - 1) as F;
        for v in var.iter_mut() {
            *v /= n_minus_1;
        }
        let mut std = vec![0.0 as F; n_cols];
        for (j, &v) in var.iter().enumerate() {
            let s = v.sqrt();
            if s < STD_FLOOR {
                return Err(ComputeError::Invalid(format!(
                    "PCA: column {j} has near-zero standard deviation ({s:e}); \
                     remove constant columns before PCA"
                )));
            }
            std[j] = s;
        }

        // ---- 3. Build standardized matrix Z --------------------------------
        let mut z = vec![0.0 as F; n_rows * n_cols];
        for i in 0..n_rows {
            for j in 0..n_cols {
                z[i * n_cols + j] = (matrix[i * n_cols + j] - mean[j]) / std[j];
            }
        }

        // ---- 4. Covariance C = (Z^T Z) / (n - 1) ---------------------------
        // Dense n_cols × n_cols symmetric matrix (row-major).
        let mut cov = vec![0.0 as F; n_cols * n_cols];
        for a in 0..n_cols {
            for b in a..n_cols {
                let mut s = 0.0 as F;
                for i in 0..n_rows {
                    s += z[i * n_cols + a] * z[i * n_cols + b];
                }
                let c = s / n_minus_1;
                cov[a * n_cols + b] = c;
                cov[b * n_cols + a] = c;
            }
        }

        // ---- 5. Top two eigenvectors via power iteration + deflation -------
        let v1 = power_iteration(&cov, n_cols);
        let lam1 = rayleigh_quotient(&cov, &v1, n_cols);

        // Deflate: C' = C - lam1 * v1 * v1^T
        let mut cov2 = cov.clone();
        for a in 0..n_cols {
            for b in 0..n_cols {
                cov2[a * n_cols + b] -= lam1 * v1[a] * v1[b];
            }
        }
        let v2 = power_iteration(&cov2, n_cols);
        let lam2 = rayleigh_quotient(&cov, &v2, n_cols);

        // ---- 6. Project rows onto [v1, v2] ---------------------------------
        let mut coords = vec![0.0 as F; n_rows * 2];
        for i in 0..n_rows {
            let mut pc1 = 0.0 as F;
            let mut pc2 = 0.0 as F;
            for j in 0..n_cols {
                let zij = z[i * n_cols + j];
                pc1 += zij * v1[j];
                pc2 += zij * v2[j];
            }
            coords[2 * i] = pc1;
            coords[2 * i + 1] = pc2;
        }

        // Guarantee ordering — power iteration + deflation can rarely produce
        // tiny numerical negatives. Clamp to >= 0 and keep the pair ordered.
        let variance = [lam1.max(0.0), lam2.max(0.0)];

        Ok(PcaResult { coords, variance })
    }
}

/// Dominant eigenvector of a dense symmetric `n × n` matrix (row-major) via
/// power iteration. The initial guess is a deterministic non-zero vector so
/// results are reproducible. Iterates up to [`POWER_ITER_MAX`] times or until
/// `||v_new - v_old|| < `[`POWER_ITER_TOL`].
fn power_iteration(mat: &[F], n: usize) -> Vec<F> {
    // Deterministic seed: uniform, normalized.
    let mut v = vec![1.0 / (n as F).sqrt(); n];

    for _ in 0..POWER_ITER_MAX {
        let mut next = mat_vec(mat, &v, n);
        let norm = vec_norm(&next);
        if norm <= 0.0 {
            // Degenerate — return current estimate. Should not happen with
            // standardized data but guard just in case.
            return v;
        }
        for x in next.iter_mut() {
            *x /= norm;
        }
        // Sign normalization: align with v's leading non-zero component so
        // power iteration with a sign flip doesn't oscillate.
        let dot: F = v.iter().zip(next.iter()).map(|(a, b)| a * b).sum();
        if dot < 0.0 {
            for x in next.iter_mut() {
                *x = -*x;
            }
        }

        let diff: F = v
            .iter()
            .zip(next.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<F>()
            .sqrt();
        v = next;
        if diff < POWER_ITER_TOL {
            break;
        }
    }
    v
}

/// Compute `v^T M v` for a symmetric `n × n` row-major matrix and vector.
fn rayleigh_quotient(mat: &[F], v: &[F], n: usize) -> F {
    let mv = mat_vec(mat, v, n);
    v.iter().zip(mv.iter()).map(|(a, b)| a * b).sum()
}

/// Matrix-vector product for a row-major `n × n` matrix.
fn mat_vec(mat: &[F], v: &[F], n: usize) -> Vec<F> {
    let mut out = vec![0.0 as F; n];
    for a in 0..n {
        let mut s = 0.0 as F;
        for b in 0..n {
            s += mat[a * n + b] * v[b];
        }
        out[a] = s;
    }
    out
}

/// Euclidean norm.
fn vec_norm(v: &[F]) -> F {
    v.iter().map(|&x| x * x).sum::<F>().sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    /// Draw one standard-normal sample via the Box-Muller transform using two
    /// uniform `(0, 1]` draws. Reasonable for tests — not performance-critical.
    fn box_muller(rng: &mut StdRng) -> F {
        // Ensure u1 > 0 so ln() is finite; u2 is free.
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

    /// Build three 2D Gaussian blobs around fixed centers.
    /// Returns a row-major `n × 2` matrix with `sigma = 0.5`.
    pub(super) fn three_blobs(n_per_cluster: usize, seed: u64) -> (Vec<F>, usize) {
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
    fn fit_transform_on_three_blobs() {
        let (matrix, n_rows) = three_blobs(20, 42);
        let result = Pca2::fit_transform(&matrix, n_rows, 2).unwrap();

        assert_eq!(result.coords.len(), 2 * n_rows);
        assert!(result.variance[0] > 0.0);
        assert!(result.variance[1] > 0.0);
        assert!(
            result.variance[0] >= result.variance[1],
            "variance must be ordered: {:?}",
            result.variance
        );
    }

    #[test]
    fn err_on_too_few_rows() {
        let matrix = vec![1.0, 2.0, 3.0, 4.0];
        let err = Pca2::fit_transform(&matrix, 2, 2).unwrap_err();
        assert!(matches!(err, ComputeError::Invalid(_)));
    }

    #[test]
    fn err_on_too_few_cols() {
        let matrix = vec![1.0; 5];
        let err = Pca2::fit_transform(&matrix, 5, 1).unwrap_err();
        assert!(matches!(err, ComputeError::Invalid(_)));
    }

    #[test]
    fn err_on_nan_input() {
        let mut matrix = vec![0.0 as F; 20];
        matrix[7] = F::NAN;
        let err = Pca2::fit_transform(&matrix, 5, 4).unwrap_err();
        assert!(matches!(err, ComputeError::Invalid(_)));
    }

    #[test]
    fn err_on_zero_variance_column() {
        // 5 rows × 2 cols; column 1 is constant.
        let matrix = vec![
            0.0, 1.0, //
            1.0, 1.0, //
            2.0, 1.0, //
            3.0, 1.0, //
            4.0, 1.0,
        ];
        let err = Pca2::fit_transform(&matrix, 5, 2).unwrap_err();
        assert!(matches!(err, ComputeError::Invalid(_)));
    }

    #[test]
    fn variance_sum_tracks_trace() {
        // For n_cols = 2, variance[0] + variance[1] should equal trace(C)
        // = sum of diagonal = 2 (standardized columns have variance 1).
        // Allow some tolerance for power-iteration convergence.
        let (matrix, n_rows) = three_blobs(40, 7);
        let result = Pca2::fit_transform(&matrix, n_rows, 2).unwrap();
        let sum = result.variance[0] + result.variance[1];
        assert!(
            (sum - 2.0).abs() < 1e-6,
            "variance sum {sum} should be ~2.0 (trace of standardized cov)"
        );
    }
}
