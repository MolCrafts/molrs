//! Hydrogen-bond analysis: geometric detection, network topology, and lifetimes.
//!
//! Ported from TRAVIS (`src/hbond.cpp`, `src/aggrtopo.cpp`). Three pieces:
//!
//! 1. [`HBonds`] — per-frame geometric D–H···A detection over donor `(D, H)` and
//!    acceptor selections, using the existing [`NeighborQuery`] candidate search
//!    and minimum-image geometry, gated by an [`HBondCriterion`]
//!    (Luzar–Chandler defaults).
//! 2. [`hbond_components`] — connected-component sizes of the per-frame bond graph
//!    via the native `core::Topology` (no petgraph).
//! 3. [`hbond_lifetimes`] — continuous `S_HB(t)` and intermittent `C_HB(t)`
//!    lifetime TCFs over the geometric presence series.
//!
//! [`NeighborQuery`]: molrs::spatial::neighbors::NeighborQuery
//!
//! Layer: `compute` → `core` only; WASM-clean; no new dependency; **no petgraph**.

mod criterion;
mod detect;
mod lifetime;
mod network;

pub use criterion::{DistKind, HBondCriterion};
pub use detect::{HBond, HBonds, HBondsResult};
pub use lifetime::{LifetimeResult, hbond_lifetimes, presence_from_hbonds};
pub use network::{NetworkResult, hbond_components};
