//! Spatial types for JavaScript — the WASM face of `molrs::spatial`.
//!
//! - [`simbox`] — the periodic cell, exported to JS as `Box`.
//! - [`region`] — the geometric solids and their composition.
//! - [`mesh`] — triangle surfaces (`Mesh`).
//! - [`neighbors`] — neighbor search: `NeighborList` (self), `NeighborQuery`
//!   (cross) and the `Neighbors` pair table they produce.

pub(crate) mod mesh;
pub(crate) mod neighbors;
pub(crate) mod region;
pub(crate) mod simbox;

pub use mesh::Mesh;
pub use neighbors::{NeighborList, NeighborQuery, Neighbors};
pub use region::*;
pub use simbox::Box;
