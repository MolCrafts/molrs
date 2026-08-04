//! Structure builders — produce molecular graphs / paths from parameters.
//!
//! This is the inverse direction of [`crate::compute`] (Frame → analysis).
//! Everything that *constructs* structure from a few scalars lives here:
//!
//! | Builder | Output |
//! |---------|--------|
//! | [`GrapheneBuilder`] | flat honeycomb sheet [`Frame`] |
//! | [`CarbonTubeBuilder`] | rolled SWCNT [`Frame`] (exact graphene quotient) |
//! | [`SelfAvoidingWalk`] | multi-chain paths + [`SimBox`](crate::spatial::simbox::SimBox) (no chemistry) |
//!
//! The SARW path generator is a clean-room port of the kernel from the CAVS
//! LAMMPS tutorial `mc_gen.c` (Mark A. Tschopp & Don K. Ward), with chemistry
//! and file I/O stripped; FCC is one [`GrowthStrategy`] among others.

mod carbon_tube;
mod graphene;
mod occupancy;
mod strategy;
mod walk;

pub use carbon_tube::{CarbonTubeBuilder, CarbonTubeError};
pub use graphene::{GrapheneBuilder, GrapheneError};
pub use occupancy::OccupancyMode;
pub use strategy::{FccLattice, OffLattice};
pub use walk::{GrowthStrategy, SelfAvoidingWalk, WalkError, WalkOutput};
