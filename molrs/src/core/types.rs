//! Common numeric and geometry type aliases used across the crate.
//!
//! The **F-prefix family** of ndarray-backed aliases parameterized by the
//! float precision [`F`] is defined in [`crate::op::types`] and re-exported
//! here; the non-float aliases ([`I`], [`Idx`], [`Pbc3`]) are defined here.

/// The float aliases live in [`crate::op::types`]; this re-export keeps
/// `molrs::types::*` the canonical crate-root spelling (permanent policy). The
/// stack aliases `Vec3` / `Mat3` / `Quat` are deliberately not re-exported.
pub use crate::op::types::{F, F3, F3View, F3x3, FN, FNx3, FNx3View};

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

// ---- Non-float ----

/// Per-axis periodic boundary condition flags.
pub type Pbc3 = [bool; 3];
