//! Spatial primitives: simulation box ([`SimBox`]), geometric
//! regions, neighbor-list algorithms, and geometry utilities.
//!
//! ## Layout
//!
//! - [`simbox`] — periodic/triclinic simulation cell (`SimBox`, MIC, wrap)
//! - [`region`] — solids with a signed distance (`Region`, `Sphere`, `Cuboid`,
//!   `Parallelepiped`, Boolean composition)
//! - [`mesh`] — triangle surfaces (`TriMesh`), what an STL reads into
//! - [`neighbors`] — neighbor search algorithms
//! - [`geometry`] — free geometric helpers

pub(crate) mod bvh;
pub mod geometry;
pub mod mesh;
pub mod neighbors;
pub mod region;
pub mod simbox;
pub(crate) mod vec3;

pub use mesh::TriMesh;
pub use simbox::{BoxError, BoxKind, Mic, SimBox};
