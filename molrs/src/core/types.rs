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
pub type F = f64;

/// Primary signed integer scalar type — always `i32`.
pub type I = i32;

/// An index into a block, or a stable entity identifier.
///
/// Named for what it *means*, not for what it *is*. The retired alias `U` was
/// named after a type, so one name carried two unrelated jobs: the width a
/// column stores at, and the type a domain value happens to be. Those pull
/// opposite ways -- a formal charge wants to be small, an identifier wants to
/// be wide -- and one name could not serve both.
///
/// Every column this appears in is identity: `id`, `mol_id`, `type_id`,
/// `res_id`, and the `atomi`/`atomj`/`atomk`/`atoml` relation endpoints.
/// Sixty-four bits because an identifier that wraps is not an identifier:
/// a value past `u32::MAX` used to be truncated rather than refused.
///
/// `U` is also uranium. A text-level rename of the old alias once rewrote
/// `Element::U`, `symbol: "U"` and the GAFF/BCC/ABCG2 `atom_type: "U"` rows
/// along with the type references, and nothing caught it. Rename this through
/// the compiler -- it points only at type positions -- never through a regex.
pub type Idx = u64;

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
