//! Pure numeric primitives beneath `core`: stack vectors, small linear algebra, rigid motions, superposition and uniform direction sampling.
//!
//! Everything here computes on plain fixed-size arrays (`[f64; 3]` points,
//! row-major 3×3 matrices, `(w, x, y, z)` quaternions) and names no other
//! molrs module. Coordinates are in whatever length unit the caller uses; in
//! molrs that is Å, because every coordinate reader normalises to Å.
//!
//! The namespace is flat — `molrs::op::<name>`, the same path as Python's
//! `molrs.op.<name>` — except [`vec3`], which stays a namespace because its
//! names (`add`, `sub`, `scale`, `dot`, …) only read right qualified.
//!
//! | Kind | What it provides |
//! |---|---|
//! | numeric aliases | the scalar [`F`] `= f64`, the ndarray aliases, and the stack types [`Vec3`], [`Mat3`], [`Quat`] |
//! | [`vec3`] | vector arithmetic: sum, difference, dot and cross products, normalisation; the internal coordinates (bond angle, dihedral) |
//! | linear algebra | 3×3 determinant and inverse; eigenvalues and eigenvectors of symmetric 3×3 / 4×4 matrices |
//! | rigid motions | [`Rigid`] — a rotation followed by a translation, which moves a body without deforming it — the quaternion kernels, and NeRF placement of a point from internal coordinates ([`place_from_internal_coords`]) |
//! | superposition | [`superpose`] → [`Superposition`]: the rigid motion that best lays one set of matched points onto another (least squares), and the weighted [`centroid`] |
//! | molecule geometry | [`translate`], [`scale`], [`rotate`] and [`center`] on a `MolGraph`'s coordinates |
//! | sampling | uniform directions on S² ([`unit_vector_from_uniform`]) and the standard normal ([`standard_normal`]) |
mod geometry;
mod linalg;
mod numeric;
mod random;
mod rigid;
mod so3;
mod superpose;
pub mod vec3;

pub use geometry::{CenterError, center, rotate, scale, translate};
pub use linalg::{det3, eigh_sym_3x3, eigh_sym_4x4, inv3};
pub use numeric::{
    F, F3, F3View, Fnx3, Fnx3View, I, Idx, Mat3, Pbc3, Quat, Vec3, to_mat3, to_vec3,
};
pub use random::standard_normal;
pub use rigid::{
    Rigid, alignment_axis_angle, axis_angle, compose_rigid, orthonormal_frame,
    place_from_internal_coords, quat_conj, quat_dot, quat_mul, quat_norm, quat_to_matrix,
    rotate_by_quat, rotation_about, transform_point, transform_points,
};
pub use so3::unit_vector_from_uniform;
pub use superpose::{
    DEFAULT_GAP_TOL, Freedom, Superposition, SuperpositionError, centroid, superpose,
};
