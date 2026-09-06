//! Spatial primitives: simulation box ([`SimBox`](simbox::SimBox)), geometric
//! regions, neighbor-list algorithms, and geometry utilities.
//!
//! ## Layout
//!
//! - [`simbox`] — periodic/triclinic simulation cell (`SimBox`, MIC, wrap)
//! - [`region`] — geometric containment predicates (`Region`, `Cuboid`,
//!   `Parallelepiped`, spheres, Boolean composition)
//! - [`mesh`] — triangle surfaces (`TriMesh`), what an STL reads into
//! - [`neighbors`] — neighbor search algorithms
//! - [`geometry`] — free geometric helpers

pub mod geometry;
pub mod mesh;
pub mod neighbors;
pub mod region;
pub mod simbox;

pub use mesh::TriMesh;
pub use simbox::{BoxError, BoxKind, Mic, SimBox};
