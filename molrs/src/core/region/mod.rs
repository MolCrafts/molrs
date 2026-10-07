//! Geometric regions: solids with a signed distance to their boundary.
//!
//! Every shape ([`Sphere`], [`Cuboid`], [`Parallelepiped`], [`HalfSpace`],
//! [`Cylinder`], [`Ellipsoid`], [`Polyhedron`], [`SphereUnion`]) describes its inside; outside, shells and
//! voids are Boolean compositions ([`NotRegion`], [`AndRegion`], [`OrRegion`]). The trait's
//! one required method beyond [`Region::bounds`] is [`Region::distance`]
//! (negative inside, positive outside); containment and the gradient follow
//! from it. The periodic simulation cell lives in [`crate::core::SimBox`]
//! — it is not a region type and must not be imported from here.

mod cylinder;
mod ellipsoid;
mod half_space;
mod polyhedron;
mod primitives;
mod sphere_union;

pub use cylinder::Cylinder;
pub use ellipsoid::Ellipsoid;
pub use half_space::HalfSpace;
pub use polyhedron::{Polyhedron, PolyhedronError};
pub use primitives::{AndRegion, Cuboid, NotRegion, OrRegion, Parallelepiped, Region, Sphere};
pub use sphere_union::{SphereUnion, SphereUnionError};
