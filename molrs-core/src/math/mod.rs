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
