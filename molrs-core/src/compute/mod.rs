//! Analysis compute modules for molrs molecular simulation.
//!
//! Provides a trait-based framework for post-simulation analysis,
//! inspired by [freud-analysis](https://freud.readthedocs.io/).
//!
//! # Two-layer trait design
//!
//! - [`Compute`] — analyses that only need a `Frame`
//! - [`PairCompute`] — analyses that also need a pre-built `NeighborList`
//!
//! # Accumulation
//!
//! [`Reducer`] + [`Accumulator`] / [`PairAccumulator`] compose single-frame
//! computes with trajectory-level reduction (sum, concat, etc.).
//!
//! # Available analyses
//!
//! | Module | Trait | Description |
//! |--------|-------|-------------|
//! | [`rdf`] | `PairCompute` | Radial distribution function g(r) |
//! | [`msd`] | `Compute` | Mean squared displacement |
//! | [`cluster`] | `PairCompute` | Distance-based cluster analysis (BFS) |

pub mod accumulator;
pub mod cluster;
pub mod error;
pub mod msd;
pub mod rdf;
pub mod reducer;
pub mod traits;
pub mod util;

// Re-exports
pub use accumulator::{Accumulator, PairAccumulator};
pub use cluster::{Cluster, ClusterProperties, ClusterPropsResult, ClusterResult};
pub use error::ComputeError;
pub use msd::{MSD, MSDResult};
pub use rdf::{RDF, RDFResult};
pub use reducer::{ConcatReducer, Reducer, SumReducer};
pub use traits::{Compute, PairCompute};
