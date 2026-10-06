//! Spatial primitives: simulation box ([`SimBox`]), geometric
//! regions, neighbor-list algorithms, and geometry utilities.
//!
//! ## Layout
//!
//! - [`SimBox`] — periodic/triclinic simulation cell (MIC, wrap)
//! - [`region`] — solids with a signed distance (`Region`, `Sphere`, `Cuboid`,
//!   `Parallelepiped`, Boolean composition)
//! - [`TriMesh`] — triangle surfaces, what an STL reads into
//! - [`neighbors`] — neighbor search algorithms
//! - [`GhostSet`] — ghost atoms for the MD force path
//! - [`translate`], [`scale`], [`rotate`] — whole-graph transforms, and the
//!   node-group centre query [`center`]
//! - [`Trace`] — an ordered path of 3D points with no chemistry

pub(crate) mod bvh;
mod geometry;
mod mesh;
pub mod neighbors;
mod periodic;
pub mod region;
mod simbox;
mod trace;

pub use geometry::{CenterError, center, rotate, scale, translate};
pub use mesh::{DEGENERATE_AREA2, TriMesh};
pub use periodic::{GhostError, GhostSet, ImageRange};
pub use simbox::{BoxError, BoxKind, Mic, SimBox};
pub use trace::Trace;
