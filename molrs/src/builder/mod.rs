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
//! | [`TracePlacer`] (a [`Placer`]) | one translation-only [`Rigid`](crate::op::rigid::Rigid) per trace point |
//! | [`Assembler`] | one placed, linked world [`Fragment`](crate::system::fragment::Fragment) from traces and unit names (`frag_id` per unit, `mol_id` per trace) |
//!
//! The SARW path generator is a clean-room port of the kernel from the CAVS
//! LAMMPS tutorial `mc_gen.c` (Mark A. Tschopp & Don K. Ward), with chemistry
//! and file I/O stripped; FCC is one [`GrowthStrategy`] among others.

mod assemble;
mod carbon_tube;
mod graphene;
mod occupancy;
mod place;
mod strategy;
mod walk;

pub use assemble::{AssembleError, Assembler};
pub use carbon_tube::{CarbonTubeBuilder, CarbonTubeError};
pub use graphene::{GrapheneBuilder, GrapheneError};
pub use occupancy::OccupancyMode;
pub use place::{PlaceError, Placer, TracePlacer};
pub use strategy::{FccLattice, OffLattice};
pub use walk::{GrowthStrategy, SelfAvoidingWalk, WalkError, WalkOutput};
