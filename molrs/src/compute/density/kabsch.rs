//! Native, BLAS-free Kabsch superposition via Horn's quaternion method.
//!
//! Given a canonical `template` and an observed `frame` copy of the same rigid
//! point set (row-matched, `M × 3` each), [`kabsch`] returns the proper
//! rotation `R` (`det = +1`) that best maps the *frame* points onto the
//! *template* points (least-squares / minimum-RMSD), together with that RMSD.
//!
//! # Why the quaternion form
//!
//! The classic Kabsch algorithm takes the SVD of the `3 × 3` cross-covariance
//! and patches the sign of the smallest singular vector to forbid an improper
//! (`det = −1`) reflection. molrs ships no SVD, but it does ship a symmetric
//! `4 × 4` Jacobi eigensolver
//! ([`eigh_largest_sym_4x4`](molrs::math::diagonalize::eigh_largest_sym_4x4)).
//! Horn's method (Horn, *J. Opt. Soc. Am. A* **1987**, 4, 629) builds a `4 × 4`
//! key matrix `N` from the cross-covariance whose largest-eigenvalue
//! eigenvector *is* the optimal rotation quaternion. A unit quaternion always
//! encodes a proper rotation, so the reflection guard is automatic — there is
//! no `det = −1` branch to get wrong.
//!
//! This is a deliberate, documented deviation from TRAVIS's own alignment
//! bookkeeping in `CSDFMap` (`src/sdfmap.cpp`): TRAVIS aligns onto a reference
//! atom triple by explicit axis construction, whereas molrs uses the
//! mathematically-equivalent quaternion least-squares fit so it can reuse the
//! existing eigensolver and stay BLAS-free / WASM-clean. The accumulated SDF
//! grid and its normalization (below, in `spatial.rs`) follow `CSDFMap`.

use molrs::math::diagonalize::{eigh_largest_sym_4x4, eigvals_sym_3x3};
use molrs::types::{F, F3x3};
use ndarray::ArrayView2;

use crate::compute::error::ComputeError;

/// Relative tolerance below which the template's second spatial extent is
/// treated as collinear (rank-deficient → orientation undetermined).
const COLLINEAR_REL_TOL: F = 1e-10;

/// Best-fit proper rotation mapping `frame` onto `template`.
///
/// Both inputs are `M × 3` with matched rows (atom `i` of `frame` corresponds
/// to atom `i` of `template`). Returns `(R, rmsd)` where `R · (frameᵢ − c_f)`
/// best matches `(templateᵢ − c_t)` in the least-squares sense, `c_*` being the
/// respective centroids, and `det(R) = +1`.
///
/// # Errors
///
/// - [`ComputeError::DimensionMismatch`] if the row counts differ.
/// - [`ComputeError::OutOfRange`] if fewer than 3 points are given, or the
///   template points are collinear (a line cannot fix a 3-D orientation).
pub fn kabsch(template: ArrayView2<F>, frame: ArrayView2<F>) -> Result<(F3x3, F), ComputeError> {
    let m = template.nrows();
    if frame.nrows() != m {
        return Err(ComputeError::DimensionMismatch {
            expected: m,
            got: frame.nrows(),
            what: "kabsch row count",
        });
    }
    if m < 3 {
        return Err(ComputeError::OutOfRange {
            field: "kabsch::n_reference_atoms",
            value: m.to_string(),
        });
    }

    // Centroids.
    let ct = centroid(template);
    let cf = centroid(frame);

    // Centered coordinates.
    let mut x = vec![[0.0_f64; 3]; m]; // template (target)
    let mut p = vec![[0.0_f64; 3]; m]; // frame (moving)
    for i in 0..m {
        for d in 0..3 {
            x[i][d] = template[[i, d]] - ct[d];
            p[i][d] = frame[[i, d]] - cf[d];
        }
    }

    // Reject a collinear reference: its centered covariance must have a
    // second eigenvalue well above zero (rank ≥ 2 → spans a plane).
    if collinear(&x) {
        return Err(ComputeError::OutOfRange {
            field: "kabsch::reference",
            value: "collinear (fewer than 3 non-collinear atoms)".into(),
        });
    }

    // Cross-covariance S = Σ pᵢ ⊗ xᵢ  (3×3).
    let mut s = [[0.0_f64; 3]; 3];
    for i in 0..m {
        for a in 0..3 {
            for b in 0..3 {
                s[a][b] += p[i][a] * x[i][b];
            }
        }
    }

    // Horn's symmetric 4×4 key matrix N from the cross-covariance.
    let (sxx, sxy, sxz) = (s[0][0], s[0][1], s[0][2]);
    let (syx, syy, syz) = (s[1][0], s[1][1], s[1][2]);
    let (szx, szy, szz) = (s[2][0], s[2][1], s[2][2]);
    let mut n = [[0.0_f64; 4]; 4];
    n[0][0] = sxx + syy + szz;
    n[1][1] = sxx - syy - szz;
    n[2][2] = -sxx + syy - szz;
    n[3][3] = -sxx - syy + szz;
    n[0][1] = syz - szy;
    n[0][2] = szx - sxz;
    n[0][3] = sxy - syx;
    n[1][2] = sxy + syx;
    n[1][3] = szx + sxz;
    n[2][3] = syz + szy;
    // Symmetrize.
    n[1][0] = n[0][1];
    n[2][0] = n[0][2];
    n[3][0] = n[0][3];
    n[2][1] = n[1][2];
    n[3][1] = n[1][3];
    n[3][2] = n[2][3];

    let (lambda_max, q) = eigh_largest_sym_4x4(&n);

    // Quaternion (w, x, y, z) → proper rotation matrix.
    let r = quat_to_rotation(q);

    // RMSD from the eigenvalue identity:
    //   M·rmsd² = Σ|pᵢ|² + Σ|xᵢ|² − 2 λ_max.
    let mut g: F = 0.0;
    for i in 0..m {
        for d in 0..3 {
            g += p[i][d] * p[i][d] + x[i][d] * x[i][d];
        }
    }
    let msd = ((g - 2.0 * lambda_max) / m as F).max(0.0);
    Ok((r, msd.sqrt()))
}

fn centroid(pts: ArrayView2<F>) -> [F; 3] {
    let m = pts.nrows().max(1) as F;
    let mut c = [0.0_f64; 3];
    for row in pts.rows() {
        for d in 0..3 {
            c[d] += row[d];
        }
    }
    for cd in &mut c {
        *cd /= m;
    }
    c
}

/// True if the centered points span fewer than 2 dimensions (a line or point).
fn collinear(centered: &[[F; 3]]) -> bool {
    // Centered covariance (3×3 symmetric).
    let mut cov = F3x3::zeros((3, 3));
    for v in centered {
        for a in 0..3 {
            for b in 0..3 {
                cov[[a, b]] += v[a] * v[b];
            }
        }
    }
    let mut ev = eigvals_sym_3x3(&cov).to_vec();
    ev.sort_by(|a, b| b.partial_cmp(a).unwrap());
    // Largest defines the scale; second must be non-trivial for rank ≥ 2.
    let lead = ev[0].abs();
    if lead == 0.0 {
        return true;
    }
    ev[1].abs() < COLLINEAR_REL_TOL * lead
}

/// Unit quaternion `(w, x, y, z)` → `3×3` proper rotation matrix.
fn quat_to_rotation(q: [F; 4]) -> F3x3 {
    let norm = (q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]).sqrt();
    let (w, x, y, z) = (q[0] / norm, q[1] / norm, q[2] / norm, q[3] / norm);
    let mut r = F3x3::zeros((3, 3));
    r[[0, 0]] = 1.0 - 2.0 * (y * y + z * z);
    r[[0, 1]] = 2.0 * (x * y - w * z);
    r[[0, 2]] = 2.0 * (x * z + w * y);
    r[[1, 0]] = 2.0 * (x * y + w * z);
    r[[1, 1]] = 1.0 - 2.0 * (x * x + z * z);
    r[[1, 2]] = 2.0 * (y * z - w * x);
    r[[2, 0]] = 2.0 * (x * z - w * y);
    r[[2, 1]] = 2.0 * (y * z + w * x);
    r[[2, 2]] = 1.0 - 2.0 * (x * x + y * y);
    r
}

/// Apply a `3×3` rotation to a 3-vector.
pub(crate) fn rotate(r: &F3x3, v: [F; 3]) -> [F; 3] {
    [
        r[[0, 0]] * v[0] + r[[0, 1]] * v[1] + r[[0, 2]] * v[2],
        r[[1, 0]] * v[0] + r[[1, 1]] * v[1] + r[[1, 2]] * v[2],
        r[[2, 0]] * v[0] + r[[2, 1]] * v[1] + r[[2, 2]] * v[2],
    ]
}

/// Determinant of a `3×3` matrix (reflection check; `+1` for a proper rotation).
pub fn det3(r: &F3x3) -> F {
    r[[0, 0]] * (r[[1, 1]] * r[[2, 2]] - r[[1, 2]] * r[[2, 1]])
        - r[[0, 1]] * (r[[1, 0]] * r[[2, 2]] - r[[1, 2]] * r[[2, 0]])
        + r[[0, 2]] * (r[[1, 0]] * r[[2, 1]] - r[[1, 1]] * r[[2, 0]])
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    fn rot_z(theta: F) -> F3x3 {
        let (c, s) = (theta.cos(), theta.sin());
        array![[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]
    }

    #[test]
    fn recovers_known_rotation_with_proper_det() {
        // A non-degenerate template (spans 3-D).
        let template = array![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ];
        let r_true = rot_z(0.7);
        // frame = R_true · template + t.
        let t = [3.0, -2.0, 5.0];
        let mut frame = template.clone();
        for i in 0..template.nrows() {
            let v = [template[[i, 0]], template[[i, 1]], template[[i, 2]]];
            let rv = rotate(&r_true, v);
            for d in 0..3 {
                frame[[i, d]] = rv[d] + t[d];
            }
        }
        let (r, rmsd) = kabsch(template.view(), frame.view()).unwrap();
        assert!(rmsd < 1e-9, "rmsd = {rmsd}");
        assert!((det3(&r) - 1.0).abs() < 1e-9, "det = {}", det3(&r));
        // R maps frame back onto template ⇒ R = R_trueᵀ.
        let rt = r_true.t().to_owned();
        for a in 0..3 {
            for b in 0..3 {
                assert!((r[[a, b]] - rt[[a, b]]).abs() < 1e-9);
            }
        }
    }

    #[test]
    fn reflection_guard_keeps_det_plus_one() {
        // A chiral template; its mirror image cannot be reached by a proper
        // rotation, so kabsch must still return det = +1 (best proper fit),
        // never a det = −1 reflection.
        let template = array![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ];
        let mut mirror = template.clone();
        for i in 0..template.nrows() {
            mirror[[i, 2]] = -template[[i, 2]]; // reflect through z = 0
        }
        let (r, _rmsd) = kabsch(template.view(), mirror.view()).unwrap();
        assert!((det3(&r) - 1.0).abs() < 1e-9, "det = {}", det3(&r));
    }

    #[test]
    fn collinear_reference_is_rejected() {
        let line = array![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]];
        assert!(matches!(
            kabsch(line.view(), line.view()),
            Err(ComputeError::OutOfRange { .. })
        ));
    }

    #[test]
    fn too_few_atoms_rejected() {
        let two = array![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]];
        assert!(matches!(
            kabsch(two.view(), two.view()),
            Err(ComputeError::OutOfRange { .. })
        ));
    }
}
