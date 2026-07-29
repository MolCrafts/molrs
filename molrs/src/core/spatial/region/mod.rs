//! Geometric regions and spatial containment predicates.
//!
//! This module is **only** for boolean geometry (`contains` / `bounds`). The
//! periodic simulation cell lives in [`crate::spatial::simbox`] — it is not a
//! region type and must not be imported from here.

#[allow(clippy::module_inception)]
pub mod region;

pub use crate::types::FNx3;
pub use region::{
    AndRegion, Cuboid, HollowSphere, NotRegion, OrRegion, Parallelepiped, Region, Sphere,
};
