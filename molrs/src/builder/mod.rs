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
//! | [`FragLibrary`] | named templates; `map` covers a coarse graph with them |
//! | [`TracePlacer`] | one rigid [`Rigid`](crate::op::rigid::Rigid) per unit: a fragment superposed on its trace points |
//! | [`NullOrienter`] / [`RandomOrienter`] / [`HintOrienter`] | completes an under-determined fit's rotation |
//! | [`Reacter`] / [`PortReacter`] | joins two ports: deletes the handle branches, folds their charge, bonds the anchors |
//! | [`Assembler`] | a [`FragGraph`](crate::system::frag_graph::FragGraph) placed, replicated and linked into one world [`Fragment`](crate::system::fragment::Fragment) |
//! | [`Finalizer`] | completes angles, dihedrals and (optionally) impropers of an assembled molecule |
//!
//! The SARW path generator is a clean-room port of the kernel from the CAVS
//! LAMMPS tutorial `mc_gen.c` (Mark A. Tschopp & Don K. Ward), with chemistry
//! and file I/O stripped; FCC is one [`GrowthStrategy`] among others.

mod assemble;
mod carbon_tube;
mod finalize;
mod graphene;
mod library;
mod occupancy;
mod orient;
mod place;
mod react;
mod strategy;
mod walk;

pub use assemble::{AssembleError, Assembler};
pub use carbon_tube::{CarbonTubeBuilder, CarbonTubeError};
pub use finalize::Finalizer;
pub use graphene::{GrapheneBuilder, GrapheneError};
pub use library::{FragLibrary, FragLibraryError};
pub use occupancy::OccupancyMode;
pub use orient::{BodyAxis, HintOrienter, NullOrienter, OrientError, Orienter, RandomOrienter};
pub use place::{PlaceError, Placer, TracePlacer};
pub use react::{PairError, PortReacter, ReactError, Reacter};
pub use strategy::{FccLattice, OffLattice};
pub use walk::{GrowthStrategy, SelfAvoidingWalk, WalkError, WalkOutput};
