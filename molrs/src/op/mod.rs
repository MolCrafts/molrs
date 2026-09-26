//! Pure numeric primitives beneath `core`: stack vectors, small linear algebra, rigid motions, superposition and SO(3) sampling.
//!
//! Everything here computes on plain fixed-size arrays (`[f64; 3]` points,
//! row-major 3×3 matrices, `(w, x, y, z)` quaternions) and names no other
//! molrs module. Coordinates are in whatever length unit the caller uses; in
//! molrs that is Å, because every coordinate reader normalises to Å.
//!
//! | Module | What it provides |
//! |---|---|
//! | [`types`] | the scalar `F = f64`, the ndarray aliases, and the stack types [`Vec3`](types::Vec3), [`Mat3`](types::Mat3), [`Quat`](types::Quat) |
//! | [`vec3`] | vector arithmetic: sum, difference, dot and cross products, normalisation |
//! | [`linalg`] | 3×3 determinant and inverse; eigenvalues and eigenvectors of symmetric 3×3 / 4×4 matrices |
//! | [`rigid`] | a **rigid motion** — a rotation followed by a translation, which moves a body without deforming it — and the quaternion kernels |
//! | [`superpose`] | **superposition**: the rigid motion that best lays one set of matched points onto another (least squares), and the weighted centroid |
//! | [`so3`] | uniform random sampling of rotations and directions |
//!
//! **SO(3)** is the set of all 3D rotations (3×3 orthogonal matrices of
//! determinant +1). A "uniform" random rotation is uniform with respect to the
//! **Haar measure**, the unique rotation-invariant way of spreading
//! probability over SO(3): rotating every sample by a fixed rotation leaves the
//! distribution unchanged.
pub mod linalg;
pub mod rigid;
pub mod so3;
pub mod superpose;
pub mod types;
pub mod vec3;
