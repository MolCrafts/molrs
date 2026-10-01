//! Scalar and fixed-size array aliases shared by the `op` kernels.
//!
//! Two families live here:
//!
//! - The **F-prefix family** of ndarray-backed aliases over the scalar [`F`]
//!   (always `f64`): `F3`, `F3x3`, `FN`, `FNx3` and their views. These
//!   are the API types of the crate's column stores; `core::types` re-exports
//!   them, so the crate-root `types` module stays their canonical spelling for
//!   downstream code.
//! - The **stack aliases** [`Vec3`], [`Mat3`] and [`Quat`] that the `op`
//!   kernels compute on. They have exactly one path, this module, and are never
//!   re-exported. [`to_vec3`] and [`to_mat3`] are where ndarray meets them.

use ndarray::{Array1, Array2, ArrayView1, ArrayView2};

/// Primary floating-point scalar type — always `f64`.
///
/// Scientific algorithms (potentials, optimizers, coordinate transforms) require
/// double precision.  Lower precision is only used in accelerator hot-paths
/// (GPU kernels) or estimation algorithms, and those are handled locally, not
/// through this project-wide alias.
pub type F = f64;

// ---- Fixed-size 3D types ----

/// 3-element vector (position, velocity, force, displacement).
pub type F3 = Array1<F>;

/// 3×3 matrix (box matrix, rotation, stress tensor).
pub type F3x3 = Array2<F>;

// ---- Variable-size types ----

/// N-element vector.
pub type FN = Array1<F>;

/// N×3 matrix (collection of 3D vectors).
pub type FNx3 = Array2<F>;

// ---- Views ----

/// Borrowed view of a 3-element vector.
pub type F3View<'a> = ArrayView1<'a, F>;

/// Borrowed N×3 view.
pub type FNx3View<'a> = ArrayView2<'a, F>;

// ---- Stack types of the op kernels ----

/// A 3-vector on the stack: a point, a displacement or a direction.
pub type Vec3 = [F; 3];

/// A 3×3 matrix on the stack, row-major: `m[row][col]`.
pub type Mat3 = [[F; 3]; 3];

/// A quaternion `(w, x, y, z)`: scalar part first, then the vector part.
///
/// A quaternion is a four-component number `w + x i + y j + z k` whose
/// imaginary units multiply as `i² = j² = k² = ijk = −1`; unit-length
/// quaternions encode 3D rotations without the gimbal problems of angles.
///
/// This is the crate's one quaternion type. Products are Hamilton products
/// (`i ⊗ j = k`); a unit quaternion `q` rotates the vector `v` as `q v q*`,
/// where `v` is written as the quaternion `(0, v)` and
/// `q* = (w, −x, −y, −z)` is the conjugate. The rotation angle `θ` about the
/// unit axis `n` is encoded as `q = (cos θ/2, sin θ/2 · n)`.
pub type Quat = [F; 4];

/// Copy a 3-element ndarray view onto the stack.
///
/// # Panics
///
/// If `v` does not have exactly 3 elements.
pub fn to_vec3(v: ArrayView1<'_, F>) -> Vec3 {
    assert_eq!(v.len(), 3, "to_vec3 expects 3 elements, got {}", v.len());
    [v[0], v[1], v[2]]
}

/// Copy a 3×3 ndarray view onto the stack, row-major.
///
/// # Panics
///
/// If `m` is not 3×3.
pub fn to_mat3(m: ArrayView2<'_, F>) -> Mat3 {
    assert_eq!(
        m.dim(),
        (3, 3),
        "to_mat3 expects a 3×3 matrix, got {:?}",
        m.dim()
    );
    let mut out = [[0.0; 3]; 3];
    for (r, row) in out.iter_mut().enumerate() {
        for (c, x) in row.iter_mut().enumerate() {
            *x = m[[r, c]];
        }
    }
    out
}
