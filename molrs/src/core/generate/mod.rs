//! Structure-generating algorithms — the inverse direction of `compute`
//! (which maps a `Frame` to analysis quantities; generators produce
//! coordinates/structure from parameters).
//!
//! The first inhabitant is a periodic, fixed-bond-length **self-avoiding random
//! walk (SARW)** multi-chain path generator: see [`SelfAvoidingWalk`]. It emits
//! only *paths* — `Vec<Vec<F3>>` plus the [`SimBox`](crate::spatial::region::SimBox) it
//! used — and deliberately knows nothing about topology, chemistry (mass / atom
//! types / potentials), or file IO. Downstream consumers (e.g. molpy's
//! `PolymerBuilder`) attach bonds and atom types to the returned paths.
//!
//! The algorithm is a clean-room port of the path-generation kernel from the
//! CAVS LAMMPS tutorial `mc_gen.c` (Mark A. Tschopp & Don K. Ward), with the
//! chemistry, the LAMMPS-data IO, and the FCC-only assumption all stripped: the
//! FCC lattice is demoted to one [`GrowthStrategy`] among others.

mod occupancy;
mod strategy;
mod walk;

pub use occupancy::OccupancyMode;
pub use strategy::{FccLattice, OffLattice};
pub use walk::{GrowthStrategy, SelfAvoidingWalk, WalkError, WalkOutput};
