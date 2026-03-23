//! Common numeric and geometry type aliases used across the crate.
//!
//! The **F-prefix family** provides a consistent naming convention for
//! ndarray-backed types parameterized by the float precision [`F`].

use ndarray::{Array1, Array2, ArrayView1, ArrayView2};

/// Primary floating-point scalar type.
///
/// Defaults to `f32`. Enable the `f64` feature for double precision,
/// which is recommended for packing (GENCAN optimizer needs >6 significant
/// digits in finite-difference Hessian-vector products).
#[cfg(not(feature = "f64"))]
pub type F = f32;
#[cfg(feature = "f64")]
pub type F = f64;

/// Primary signed integer scalar type.
///
/// Defaults to `i32`. Enable the `i64` feature for 64-bit integers.
#[cfg(not(feature = "i64"))]
pub type I = i32;
#[cfg(feature = "i64")]
pub type I = i64;

/// Primary unsigned integer scalar type.
///
/// Defaults to `u32`. Enable the `u64` feature for 64-bit unsigned integers.
#[cfg(not(feature = "u64"))]
pub type U = u32;
#[cfg(feature = "u64")]
pub type U = u64;

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
