//! Spatial primitives: simulation box ([`SimBox`](simbox::SimBox)), geometric
//! regions, neighbor-list algorithms, and geometry utilities.
//!
//! ## Layout
//!
//! - [`simbox`] — periodic/triclinic simulation cell (`SimBox`, MIC, wrap)
//! - [`region`] — geometric containment predicates (`Region`, `Cuboid`,
//!   `Parallelepiped`, spheres, Boolean composition)
//! - [`neighbors`] — neighbor search algorithms
//! - [`geometry`] — free geometric helpers

pub mod geometry;
pub mod neighbors;
pub mod region;
pub mod simbox;

pub use simbox::{BoxError, BoxKind, Mic, SimBox};
