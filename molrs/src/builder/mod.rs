//! Structure builders — produce molecular graphs / paths from parameters.
//!
//! This is the inverse direction of [`crate::compute`] (Frame → analysis).
//! Everything that *constructs* structure from a few scalars lives here:
//!
//! | Builder | Output |
//! |---------|--------|
//! | [`GrapheneBuilder`] | flat honeycomb sheet [`Frame`] |
//! | [`CarbonTubeBuilder`] | rolled SWCNT [`Frame`] (exact graphene quotient) |
//! | [`SelfAvoidingWalk`] | multi-chain [`Trace`](crate::spatial::Trace)s + [`SimBox`](crate::spatial::simbox::SimBox) (no chemistry) |
//! | [`TracePlacer`] (a [`Placer`]) | fragments grown out of their parents at bonding range, optionally along a [`Trace`](crate::spatial::Trace) |
//! | [`SiteMap`] | marks the atoms a reaction may bind |
//! | [`LineOrienter`] / [`TangOrienter`] (an [`Orienter`]) | body axis onto a direction |
//!
//! The SARW path generator is a clean-room port of the kernel from the CAVS
//! LAMMPS tutorial `mc_gen.c` (Mark A. Tschopp & Don K. Ward), with chemistry
//! and file I/O stripped; FCC is one [`GrowthStrategy`] among others.

mod carbon_tube;
mod graphene;
mod occupancy;
mod place;
mod sites;
mod strategy;
mod walk;

pub use carbon_tube::{CarbonTubeBuilder, CarbonTubeError};
pub use graphene::{GrapheneBuilder, GrapheneError};
pub use occupancy::OccupancyMode;
pub use place::{LineOrienter, Orienter, PlaceError, Placer, TangOrienter, TracePlacer};
pub use sites::{PRE_REACTION_CHARGE_KEY, SITE_KEY, SiteError, SiteMap};
pub use strategy::{FccLattice, OffLattice};
pub use walk::{GrowthStrategy, SelfAvoidingWalk, WalkError, WalkOutput};
