//! Metrization distance-matrix sampling + 4D eigenvalue embedding.
//!
//! Port of RDKit's `DistGeom::pickRandomDistMat` + `DistGeom::computeInitialCoords`
//! (`$RDBASE/Code/DistGeom/DistGeomUtils.cpp`, BSD-3, Copyright (C) 2004-2025
//! Greg Landrum and other RDKit contributors).
//!
//! Given a smoothed bounds matrix we sample one distance matrix uniformly
//! between the lower/upper bound of every pair, build the metric (Gram) matrix
//! relative to the centroid, eigen-decompose it, and read the top `dim`
//! eigenpairs as coordinates. ETKDG embeds in `dim = 4`; the fourth dimension
//! is later squeezed out by `FourthDimContribs` during minimization (see
//! `etmin`). Negative eigenvalues are replaced with small random jitter when
//! `rand_neg_eig` is set (RDKit `randNegEig`), matching `computeInitialCoords`.

use rand::RngExt;

use crate::conformer::distgeom::BoundsMatrix;

/// Eigenvalue tolerance (RDKit `EIGVAL_TOL`).
const EIGVAL_TOL: f64 = 0.001;

/// Sample a random distance matrix between the lower and upper bounds.
///
/// RDKit `pickRandomDistMat`: for every pair `(i, j)` draws
/// `d = lb + r·(ub − lb)` with `r ∈ [0, 1)`. Returns the full symmetric
/// `n×n` distance matrix (row-major).
pub fn pick_random_dist_mat<R: RngExt + ?Sized>(bounds: &BoundsMatrix, rng: &mut R) -> Vec<f64> {
    let n = bounds.len();
    let mut dist = vec![0.0; n * n];
    for i in 1..n {
        for j in 0..i {
            let ub = bounds.upper(i, j);
            let lb = bounds.lower(i, j);
            let r: f64 = rng.random::<f64>();
            let d = lb + r * (ub - lb);
            dist[i * n + j] = d;
            dist[j * n + i] = d;
        }
    }
    dist
}

/// Compute initial coordinates in `dim` dimensions from a sampled distance
/// matrix via metric-matrix eigen-decomposition (RDKit `computeInitialCoords`).
///
/// Returns a flat `n*dim` coordinate vector, or `None` when the embedding is
/// degenerate (too many near-zero / negative eigenvalues — RDKit's
/// `numZeroFail` / `randNegEig == false` early returns).
///
/// `dist` is the full symmetric `n×n` distance matrix from
/// [`pick_random_dist_mat`]. `dim` is the embedding dimension (4 for ETKDG).
pub fn compute_initial_coords<R: RngExt + ?Sized>(
    dist: &[f64],
    n: usize,
    dim: usize,
    rng: &mut R,
    rand_neg_eig: bool,
    num_zero_fail: usize,
) -> Option<Vec<f64>> {
    // Squared distances and global mean of squared distances.
    let mut sq = vec![0.0; n * n];
    let mut sum_sq = 0.0;
    for k in 0..n * n {
        sq[k] = dist[k] * dist[k];
        sum_sq += sq[k];
    }
    // RDKit accumulates its `SymmMatrix`'s stored triangle once, so its
    // `sumSqD2` is half of this full-matrix sum: the centroid term is
    // (1/(2N²)) Σ_jk D²_jk. Subtracting the full sum shifted every sqD0i down
    // by that much, which tripped the small-sqD0i refusal for atoms near the
    // centroid and put a spurious negative eigenvalue on the Gram matrix.
    sum_sq /= 2.0 * (n * n) as f64;

    // sqD0i[i] = mean_j sq[i][j] − sum_sq   (RDKit sqD0i).
    let mut sq_d0i = vec![0.0; n];
    for i in 0..n {
        let mut acc = 0.0;
        for j in 0..n {
            acc += sq[i * n + j];
        }
        acc /= n as f64;
        acc -= sum_sq;
        if acc < EIGVAL_TOL && n > 3 {
            return None;
        }
        sq_d0i[i] = acc;
    }

    // Metric (Gram) matrix T[i][j] = 0.5·(sqD0i[i] + sqD0i[j] − sq[i][j]).
    let mut t = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..n {
            t[i * n + j] = 0.5 * (sq_d0i[i] + sq_d0i[j] - sq[i * n + j]);
        }
    }

    let (eigvals, eigvecs) = jacobi_eigen(&t, n);
    // Take the `dim` largest eigenvalues (descending).
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| eigvals[b].partial_cmp(&eigvals[a]).unwrap());

    let mut found_neg = false;
    let mut zero_eigs = 0usize;
    // Per-dimension scale = sqrt(eigval) (0 for ~zero, flagged for negative).
    let mut scale = vec![0.0; dim];
    let mut neg_dim = vec![false; dim];
    for d in 0..dim {
        let ev = if d < n { eigvals[order[d]] } else { 0.0 };
        if ev > EIGVAL_TOL {
            scale[d] = ev.sqrt();
        } else if ev.abs() < EIGVAL_TOL {
            scale[d] = 0.0;
            zero_eigs += 1;
        } else {
            found_neg = true;
            neg_dim[d] = true;
        }
    }
    if found_neg && !rand_neg_eig {
        return None;
    }
    if zero_eigs >= num_zero_fail && n > 3 {
        return None;
    }

    let mut coords = vec![0.0; n * dim];
    for i in 0..n {
        for d in 0..dim {
            if !neg_dim[d] {
                let vec_comp = if d < n {
                    eigvecs[order[d] * n + i]
                } else {
                    0.0
                };
                coords[i * dim + d] = scale[d] * vec_comp;
            } else {
                // RDKit fills negative-eigenvalue dims with random jitter.
                coords[i * dim + d] = 1.0 - 2.0 * rng.random::<f64>();
            }
        }
    }
    Some(coords)
}

/// Random box coordinates fallback (RDKit `computeRandomCoords`): every
/// component uniform in `[−boxSize/2, boxSize/2)`. Used by the
/// `useRandomCoords` retry path.
pub fn compute_random_coords<R: RngExt + ?Sized>(
    n: usize,
    dim: usize,
    box_size: f64,
    rng: &mut R,
) -> Vec<f64> {
    let mut coords = vec![0.0; n * dim];
    for c in coords.iter_mut() {
        *c = box_size * (rng.random::<f64>() - 0.5);
    }
    coords
}

/// Symmetric eigen-decomposition via the cyclic Jacobi method.
///
/// Returns `(eigvalues[n], eigvectors_row_major[n*n])` where row `k`
/// (`eigvecs[k*n + i]`) is the eigenvector for `eigvalues[k]`. The input `a`
/// is an `n×n` symmetric matrix (row-major). Deterministic — no RNG — so the
/// whole embedding stage is reproducible under a fixed seed.
fn jacobi_eigen(a: &[f64], n: usize) -> (Vec<f64>, Vec<f64>) {
    let mut m = a.to_vec();
    // Eigenvector accumulator (identity), row-major; column i is eigenvector i.
    let mut v = vec![0.0; n * n];
    for i in 0..n {
        v[i * n + i] = 1.0;
    }
    if n == 0 {
        return (Vec::new(), v);
    }

    let max_sweeps = 100;
    for _ in 0..max_sweeps {
        // Off-diagonal magnitude.
        let mut off = 0.0;
        for p in 0..n {
            for q in (p + 1)..n {
                off += m[p * n + q] * m[p * n + q];
            }
        }
        if off < 1e-30 {
            break;
        }
        for p in 0..n {
            for q in (p + 1)..n {
                let apq = m[p * n + q];
                if apq.abs() < 1e-300 {
                    continue;
                }
                let app = m[p * n + p];
                let aqq = m[q * n + q];
                let theta = (aqq - app) / (2.0 * apq);
                let t = theta.signum() / (theta.abs() + (theta * theta + 1.0).sqrt());
                let c = 1.0 / (t * t + 1.0).sqrt();
                let s = t * c;
                // Rotate rows/cols p and q.
                for k in 0..n {
                    let akp = m[k * n + p];
                    let akq = m[k * n + q];
                    m[k * n + p] = c * akp - s * akq;
                    m[k * n + q] = s * akp + c * akq;
                }
                for k in 0..n {
                    let apk = m[p * n + k];
                    let aqk = m[q * n + k];
                    m[p * n + k] = c * apk - s * aqk;
                    m[q * n + k] = s * apk + c * aqk;
                }
                // Accumulate eigenvectors (column p, q).
                for k in 0..n {
                    let vkp = v[k * n + p];
                    let vkq = v[k * n + q];
                    v[k * n + p] = c * vkp - s * vkq;
                    v[k * n + q] = s * vkp + c * vkq;
                }
            }
        }
    }

    let eigvals: Vec<f64> = (0..n).map(|i| m[i * n + i]).collect();
    // Repack eigenvectors into row-major "row k = eigenvector k" layout
    // (currently stored as column k = eigenvector k).
    let mut eigvecs = vec![0.0; n * n];
    for k in 0..n {
        for i in 0..n {
            eigvecs[k * n + i] = v[i * n + k];
        }
    }
    (eigvals, eigvecs)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    /// Bounds that pin an equilateral triangle of side 1.
    fn triangle() -> BoundsMatrix {
        let mut b = BoundsMatrix::new(3, 0.0);
        for (i, j) in [(0, 1), (1, 2), (0, 2)] {
            b.set_lower(i, j, 1.0);
            b.set_upper(i, j, 1.0);
        }
        b
    }

    #[test]
    fn sampled_distances_stay_inside_their_bounds_and_are_symmetric() {
        let mut b = BoundsMatrix::new(4, 0.0);
        for i in 1..4 {
            for j in 0..i {
                b.set_lower(i, j, 1.0 + j as f64);
                b.set_upper(i, j, 2.0 + j as f64);
            }
        }
        let mut rng = StdRng::seed_from_u64(7);
        let d = pick_random_dist_mat(&b, &mut rng);
        for i in 1..4 {
            for j in 0..i {
                assert_eq!(d[i * 4 + j], d[j * 4 + i]);
                assert!(d[i * 4 + j] >= b.lower(i, j) && d[i * 4 + j] < b.upper(i, j));
            }
        }
        for i in 0..4 {
            assert_eq!(d[i * 4 + i], 0.0);
        }
    }

    #[test]
    fn the_embedding_reproduces_the_pinned_distances() {
        let b = triangle();
        let mut rng = StdRng::seed_from_u64(1);
        let d = pick_random_dist_mat(&b, &mut rng);
        let coords = compute_initial_coords(&d, 3, 3, &mut rng, false, 2).expect("embeds");
        assert_eq!(coords.len(), 9);
        for (i, j) in [(0, 1), (1, 2), (0, 2)] {
            let mut s = 0.0;
            for k in 0..3 {
                let diff = coords[i * 3 + k] - coords[j * 3 + k];
                s += diff * diff;
            }
            assert!((s.sqrt() - 1.0).abs() < 1e-9, "|{i}-{j}| = {}", s.sqrt());
        }
    }

    #[test]
    fn random_coordinates_fill_the_box_symmetrically() {
        let mut rng = StdRng::seed_from_u64(3);
        let coords = compute_random_coords(50, 4, 2.0, &mut rng);
        assert_eq!(coords.len(), 200);
        assert!(coords.iter().all(|c| (-1.0..1.0).contains(c)));
        assert!(coords.iter().any(|&c| c < 0.0) && coords.iter().any(|&c| c > 0.0));
    }
}
