//! Structure builders — produce molecular graphs / paths from parameters.
//!
//! This is the inverse direction of `crate::compute` (Frame → analysis; a
//! separate feature, so not linked here).
//! Everything that *constructs* structure from a few scalars lives here:
//!
//! | Builder | Output |
//! |---------|--------|
//! | [`GrapheneBuilder`] | flat honeycomb sheet [`Frame`] |
//! | [`CarbonTubeBuilder`] | rolled SWCNT [`Frame`] (exact graphene quotient) |
//! | [`SelfAvoidingWalk`] | multi-chain [`Trace`](crate::spatial::Trace)s + [`SimBox`](crate::spatial::simbox::SimBox) (no chemistry) |
//! | [`SitePlacer`], [`GrowthPlacer`] (each a [`Placer`]) | one pose ([`Rigid`](crate::op::rigid::Rigid)) per site: centre of mass on the site, or grown onto the parent's port |
//! | [`AxisOrienter`] (an [`Orienter`]) | one rotation per site, about the template's centre of mass: chain units onto the site axis and bond line, branch units by port-direction fit |
//! | [`Assembler`] | one placed, linked world graph (any graph type, chosen by the caller) from a site graph (`frag_id` per site, `mol_id` per connected component) |
//!
//! The SARW path generator is a clean-room port of the kernel from the CAVS
//! LAMMPS tutorial `mc_gen.c` (Mark A. Tschopp & Don K. Ward), with chemistry
//! and file I/O stripped; FCC is one [`GrowthStrategy`] among others.

mod assemble;
mod carbon_tube;
mod graphene;
mod occupancy;
mod orient;
mod place;
mod strategy;
mod walk;

pub use assemble::{AssembleError, Assembler};
pub use carbon_tube::{CarbonTubeBuilder, CarbonTubeError};
pub use graphene::{GrapheneBuilder, GrapheneError};
pub use occupancy::OccupancyMode;
pub use orient::{AxisOrienter, OrientError, Orienter, SiteLink, SiteView};
pub use place::{GrowthPlacer, ParentJoin, PlaceError, PlaceSite, Placer, SitePlacer};
pub use strategy::{FccLattice, OffLattice};
pub use walk::{GrowthStrategy, SelfAvoidingWalk, WalkError, WalkOutput};
