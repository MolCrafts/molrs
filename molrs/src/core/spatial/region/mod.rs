//! Geometric regions: solids with a signed distance to their boundary.
//!
//! Every shape ([`Sphere`], [`Cuboid`], [`Parallelepiped`], [`HalfSpace`],
//! [`Cylinder`], [`Ellipsoid`], [`Polyhedron`], [`SphereUnion`]) describes its inside; outside, shells and
//! voids are Boolean compositions ([`NotRegion`], [`AndRegion`], [`OrRegion`]). The trait's
//! one required method beyond [`Region::bounds`] is [`Region::distance`]
//! (negative inside, positive outside); containment and the gradient follow
//! from it. The periodic simulation cell lives in [`crate::spatial::simbox`]
//! — it is not a region type and must not be imported from here.

pub mod cylinder;
pub mod ellipsoid;
pub mod half_space;
pub mod polyhedron;
#[allow(clippy::module_inception)]
pub mod region;
pub mod sphere_union;

pub use crate::types::FNx3;
pub use cylinder::Cylinder;
pub use ellipsoid::Ellipsoid;
pub use half_space::HalfSpace;
pub use polyhedron::{Polyhedron, PolyhedronError};
pub use region::{AndRegion, Cuboid, NotRegion, OrRegion, Parallelepiped, Region, Sphere};
pub use sphere_union::{SphereUnion, SphereUnionError};
