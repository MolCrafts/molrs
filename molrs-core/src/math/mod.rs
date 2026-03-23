//! Small linear algebra helpers with optional BLAS-backed implementation.
//!
//! Default: hand-written 3x3 routines (WASM-friendly, zero external deps).
//! Feature `blas`: use ndarray-linalg for determinant and inverse (requires LAPACK backend).

use ndarray::{Array2, ArrayView2, array};

use crate::types::{F, F3, F3x3};

#[inline]
pub fn norm3(v: &F3) -> F {
    v.dot(v).sqrt()
}

#[inline]
pub fn mat3_mul_vec(m: &F3x3, v: &F3) -> F3 {
    m.dot(v)
}

/// Cross product of two 3-element vectors.
#[inline]
pub fn cross3(a: &F3, b: &F3) -> F3 {
    array![
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

#[cfg(not(feature = "blas"))]
pub fn det3(m: &F3x3) -> F {
    let m = |r: usize, c: usize| m[[r, c]];
    m(0, 0) * (m(1, 1) * m(2, 2) - m(1, 2) * m(2, 1))
        - m(0, 1) * (m(1, 0) * m(2, 2) - m(1, 2) * m(2, 0))
        + m(0, 2) * (m(1, 0) * m(2, 1) - m(1, 1) * m(2, 0))
}

#[cfg(not(feature = "blas"))]
pub fn inv3(m: &F3x3) -> Option<F3x3> {
    let m = |r: usize, c: usize| m[[r, c]];
    let c00 = m(1, 1) * m(2, 2) - m(1, 2) * m(2, 1);
    let c01 = -(m(1, 0) * m(2, 2) - m(1, 2) * m(2, 0));
    let c02 = m(1, 0) * m(2, 1) - m(1, 1) * m(2, 0);

    let c10 = -(m(0, 1) * m(2, 2) - m(0, 2) * m(2, 1));
    let c11 = m(0, 0) * m(2, 2) - m(0, 2) * m(2, 0);
    let c12 = -(m(0, 0) * m(2, 1) - m(0, 1) * m(2, 0));

    let c20 = m(0, 1) * m(1, 2) - m(0, 2) * m(1, 1);
    let c21 = -(m(0, 0) * m(1, 2) - m(0, 2) * m(1, 0));
    let c22 = m(0, 0) * m(1, 1) - m(0, 1) * m(1, 0);

    let det = m(0, 0) * c00 + m(0, 1) * c01 + m(0, 2) * c02;
    let eps: F = 1e-8;
    if det.abs() <= eps {
        return None;
    }
    let inv_det = 1.0 / det;
    Some(array![
        [c00 * inv_det, c10 * inv_det, c20 * inv_det],
        [c01 * inv_det, c11 * inv_det, c21 * inv_det],
        [c02 * inv_det, c12 * inv_det, c22 * inv_det],
    ])
}

#[cfg(feature = "blas")]
pub fn det3(m: &F3x3) -> F {
    use ndarray_linalg::Determinant;
    m.det()
        .expect("Matrix determinant calculation failed (singular matrix)")
}

#[cfg(feature = "blas")]
pub fn inv3(m: &F3x3) -> Option<F3x3> {
    use ndarray_linalg::Inverse;
    m.inv().ok()
}

/// General matrix multiplication: C = A x B
/// - A: (m x k) view
/// - B: (k x n) owned or borrowable
/// - Returns C: (m x n) owned
///
/// Path selection:
/// - With feature `blas` enabled: use ndarray's optimized dot (BLAS/LAPACK backend if linked)
/// - Else if feature `rayon` enabled: parallel row-by-row multiply
/// - Else: serial multiply
pub fn matmul(a: ArrayView2<F>, b: &Array2<F>) -> Array2<F> {
    let (_m, k_a) = a.dim();
    let (k_b, _n) = b.dim();
    assert_eq!(
        k_a, k_b,
        "matmul: inner dims must match: got {} vs {}",
        k_a, k_b
    );

    #[cfg(feature = "blas")]
    {
        // ndarray's .dot is backed by optimized kernels; with proper backend it will leverage BLAS
        a.dot(b)
    }

    #[cfg(all(not(feature = "blas"), feature = "rayon"))]
    {
        use rayon::prelude::*;
        let mut c = Array2::<F>::zeros((_m, _n));
        let c_slice = c.as_slice_mut().expect("c must be contiguous");
        c_slice.par_chunks_mut(_n).enumerate().for_each(|(i, row)| {
            for j in 0.._n {
                let mut sum: F = 0.0;
                for k in 0..k_a {
                    sum += a[[i, k]] * b[[k, j]];
                }
                row[j] = sum;
            }
        });
        c
    }

    #[cfg(all(not(feature = "blas"), not(feature = "rayon")))]
    {
        let mut c = Array2::<F>::zeros((_m, _n));
        for i in 0.._m {
            for j in 0.._n {
                let mut sum: F = 0.0;
                for k in 0..k_a {
                    sum += a[[i, k]] * b[[k, j]];
                }
                c[[i, j]] = sum;
            }
        }
        c
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{F, F3, F3x3};
    use ndarray::{Array2, array};

    const TOL: F = 1e-5;

    // ---------- norm3 ----------

    #[test]
    fn test_norm3() {
        // Unit vectors have norm 1
        let ex: F3 = array![1.0, 0.0, 0.0];
        assert!((norm3(&ex) - 1.0).abs() < TOL);

        let ey: F3 = array![0.0, 1.0, 0.0];
        assert!((norm3(&ey) - 1.0).abs() < TOL);

        let ez: F3 = array![0.0, 0.0, 1.0];
        assert!((norm3(&ez) - 1.0).abs() < TOL);

        // Zero vector has norm 0
        let zero: F3 = array![0.0, 0.0, 0.0];
        assert!((norm3(&zero) - 0.0).abs() < TOL);

        // Known vector: (3, 4, 0) -> norm = 5
        let v: F3 = array![3.0, 4.0, 0.0];
        assert!((norm3(&v) - 5.0).abs() < TOL);
    }

    // ---------- cross3 ----------

    #[test]
    fn test_cross3() {
        let i: F3 = array![1.0, 0.0, 0.0];
        let j: F3 = array![0.0, 1.0, 0.0];
        let k: F3 = array![0.0, 0.0, 1.0];

        // i x j = k
        let result = cross3(&i, &j);
        assert!((result[0] - k[0]).abs() < TOL);
        assert!((result[1] - k[1]).abs() < TOL);
        assert!((result[2] - k[2]).abs() < TOL);

        // j x k = i
        let result = cross3(&j, &k);
        assert!((result[0] - i[0]).abs() < TOL);
        assert!((result[1] - i[1]).abs() < TOL);
        assert!((result[2] - i[2]).abs() < TOL);

        // k x i = j
        let result = cross3(&k, &i);
        assert!((result[0] - j[0]).abs() < TOL);
        assert!((result[1] - j[1]).abs() < TOL);
        assert!((result[2] - j[2]).abs() < TOL);

        // Parallel vectors produce zero vector
        let a: F3 = array![2.0, 3.0, 4.0];
        let b: F3 = array![4.0, 6.0, 8.0]; // 2 * a
        let result = cross3(&a, &b);
        assert!((result[0]).abs() < TOL);
        assert!((result[1]).abs() < TOL);
        assert!((result[2]).abs() < TOL);
    }

    // ---------- det3 ----------

    #[test]
    fn test_det3() {
        // Identity matrix has determinant 1
        let eye: F3x3 = array![[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        assert!((det3(&eye) - 1.0).abs() < TOL);

        // Known matrix:
        // | 1  2  3 |
        // | 0  1  4 |
        // | 5  6  0 |
        // det = 1*(0-24) - 2*(0-20) + 3*(0-5) = -24 + 40 - 15 = 1
        let m: F3x3 = array![[1.0, 2.0, 3.0], [0.0, 1.0, 4.0], [5.0, 6.0, 0.0]];
        assert!((det3(&m) - 1.0).abs() < TOL);

        // Singular matrix (row 3 = row 1 + row 2) has det ~ 0
        let singular: F3x3 = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [5.0, 7.0, 9.0]];
        assert!((det3(&singular)).abs() < TOL);
    }

    // ---------- inv3 ----------

    #[test]
    fn test_inv3_identity() {
        let eye: F3x3 = array![[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let inv = inv3(&eye).expect("Identity should be invertible");
        for r in 0..3 {
            for c in 0..3 {
                let expected = if r == c { 1.0 } else { 0.0 };
                assert!(
                    (inv[[r, c]] - expected).abs() < TOL,
                    "inv(I)[{},{}] = {} (expected {})",
                    r,
                    c,
                    inv[[r, c]],
                    expected,
                );
            }
        }
    }

    #[test]
    fn test_inv3_known() {
        // Use the matrix with det=1 from above; verify A * inv(A) ~ I
        let a: F3x3 = array![[1.0, 2.0, 3.0], [0.0, 1.0, 4.0], [5.0, 6.0, 0.0]];
        let a_inv = inv3(&a).expect("Matrix should be invertible");

        // Compute product A * A_inv
        let product = matmul(a.view(), &a_inv);
        for r in 0..3 {
            for c in 0..3 {
                let expected = if r == c { 1.0 } else { 0.0 };
                assert!(
                    (product[[r, c]] - expected).abs() < TOL,
                    "A*inv(A)[{},{}] = {} (expected {})",
                    r,
                    c,
                    product[[r, c]],
                    expected,
                );
            }
        }
    }

    #[test]
    fn test_inv3_singular() {
        let singular: F3x3 = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [5.0, 7.0, 9.0]];
        assert!(
            inv3(&singular).is_none(),
            "Singular matrix should return None"
        );
    }

    // ---------- mat3_mul_vec ----------

    #[test]
    fn test_mat3_mul_vec() {
        let eye: F3x3 = array![[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let v: F3 = array![3.0, 7.0, -2.0];

        // I * v = v
        let result = mat3_mul_vec(&eye, &v);
        for idx in 0..3 {
            assert!(
                (result[idx] - v[idx]).abs() < TOL,
                "I*v[{}] = {} (expected {})",
                idx,
                result[idx],
                v[idx],
            );
        }

        // Known product:
        // | 1  2  3 |   | 1 |   | 1+4+9  |   | 14 |
        // | 4  5  6 | * | 2 | = | 4+10+18| = | 32 |
        // | 7  8  9 |   | 3 |   | 7+16+27|   | 50 |
        let m: F3x3 = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]];
        let u: F3 = array![1.0, 2.0, 3.0];
        let result = mat3_mul_vec(&m, &u);
        assert!((result[0] - 14.0).abs() < TOL);
        assert!((result[1] - 32.0).abs() < TOL);
        assert!((result[2] - 50.0).abs() < TOL);
    }

    // ---------- matmul ----------

    #[test]
    fn test_matmul_identity() {
        let eye: Array2<F> = array![[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let a: Array2<F> = array![[2.0, 3.0, 4.0], [5.0, 6.0, 7.0], [8.0, 9.0, 10.0]];

        let result = matmul(eye.view(), &a);
        for r in 0..3 {
            for c in 0..3 {
                assert!(
                    (result[[r, c]] - a[[r, c]]).abs() < TOL,
                    "I*A[{},{}] = {} (expected {})",
                    r,
                    c,
                    result[[r, c]],
                    a[[r, c]],
                );
            }
        }
    }

    #[test]
    fn test_matmul_known() {
        // (2x3) * (3x2) -> (2x2)
        // | 1 2 3 |   | 7  8  |   | 1*7+2*9+3*11   1*8+2*10+3*12  |   | 58   64  |
        // | 4 5 6 | * | 9  10 | = | 4*7+5*9+6*11   4*8+5*10+6*12  | = | 139  154 |
        //              | 11 12 |
        let a: Array2<F> = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
        let b: Array2<F> = array![[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]];

        let result = matmul(a.view(), &b);
        assert_eq!(result.dim(), (2, 2));
        assert!((result[[0, 0]] - 58.0).abs() < TOL);
        assert!((result[[0, 1]] - 64.0).abs() < TOL);
        assert!((result[[1, 0]] - 139.0).abs() < TOL);
        assert!((result[[1, 1]] - 154.0).abs() < TOL);
    }

    #[test]
    #[should_panic(expected = "inner dims must match")]
    fn test_matmul_dimension_mismatch() {
        // (2x3) * (2x2) should panic because inner dims 3 != 2
        let a: Array2<F> = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
        let b: Array2<F> = array![[1.0, 2.0], [3.0, 4.0]];
        let _result = matmul(a.view(), &b);
    }
}
