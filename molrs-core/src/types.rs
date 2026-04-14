//! Common numeric and geometry type aliases used across the crate.
//!
//! The **F-prefix family** provides a consistent naming convention for
//! ndarray-backed types parameterized by the float precision [`F`].

use ndarray::{Array1, Array2, ArrayView1, ArrayView2};

/// Primary floating-point scalar type — always `f64`.
///
/// Scientific algorithms (potentials, optimizers, coordinate transforms) require
/// double precision.  Lower precision is only used in accelerator hot-paths
/// (GPU kernels) or estimation algorithms, and those are handled locally, not
/// through this project-wide alias.
///
/// The former `f64` Cargo feature is deprecated and ignored.
pub type F = f64;

/// Primary signed integer scalar type — always `i32`.
///
/// The former `i64` Cargo feature is deprecated and ignored.
pub type I = i32;

/// Primary unsigned integer scalar type — always `u32`.
///
/// The former `u64` Cargo feature is deprecated and ignored.
pub type U = u32;

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

// ---- Non-float ----

/// Per-axis periodic boundary condition flags.
pub type Pbc3 = [bool; 3];
