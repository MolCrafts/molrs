//! Pure numeric primitives beneath `core`: stack vectors, small linear algebra, rigid motions, superposition and uniform direction sampling.
//!
//! Everything here computes on plain fixed-size arrays (`[f64; 3]` points,
//! row-major 3×3 matrices, `(w, x, y, z)` quaternions) and names no other
//! molrs module. Coordinates are in whatever length unit the caller uses; in
//! molrs that is Å, because every coordinate reader normalises to Å.
//!
//! | Module | What it provides |
//! |---|---|
//! | [`types`] | the scalar `F = f64`, the ndarray aliases, and the stack types [`Vec3`](types::Vec3), [`Mat3`](types::Mat3), [`Quat`](types::Quat) |
//! | [`vec3`] | vector arithmetic: sum, difference, dot and cross products, normalisation; the internal coordinates (bond angle, dihedral) |
//! | [`linalg`] | 3×3 determinant and inverse; eigenvalues and eigenvectors of symmetric 3×3 / 4×4 matrices |
//! | [`rigid`] | a **rigid motion** — a rotation followed by a translation, which moves a body without deforming it — the quaternion kernels, and NeRF placement of a point from internal coordinates |
//! | [`superpose`] | **superposition**: the rigid motion that best lays one set of matched points onto another (least squares), and the weighted centroid |
//! | [`so3`] | uniform sampling of directions on S² |
//! | [`random`] | random variates (the standard normal) over a caller-seeded RNG |
pub mod linalg;
pub mod random;
pub mod rigid;
pub mod so3;
pub mod superpose;
pub mod types;
pub mod vec3;
