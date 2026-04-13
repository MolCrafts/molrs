//! Analysis compute modules for molrs molecular simulation.
//!
//! Provides a trait-based framework for post-simulation analysis,
//! inspired by [freud-analysis](https://freud.readthedocs.io/).
//!
//! # Unified trait
//!
//! [`Compute`] — all analyses implement this trait with a GAT `Args` type:
//!
//! - `Args = ()` for frame-only analyses (MSD)
//! - `Args = &NeighborList` for pair-based analyses (RDF, Cluster)
//! - `Args = &ClusterResult` for cluster property analyses
//!
//! # Accumulation
//!
//! [`Reducer`] + [`Accumulator`] compose single-frame computes with
//! trajectory-level reduction (sum, concat, etc.).
//!
//! # Available analyses
//!
//! | Module | Args | Description |
//! |--------|------|-------------|
//! | [`rdf`] | `&NeighborList` | Radial distribution function g(r) |
//! | [`msd`] | `()` | Mean squared displacement |
//! | [`cluster`] | `&NeighborList` | Distance-based cluster analysis (BFS) |
//! | [`cluster_centers`] | `&ClusterResult` | Geometric centers (MIC-aware) |
//! | [`center_of_mass`] | `&ClusterResult` | Mass-weighted centers |
//! | [`gyration_tensor`] | `&ClusterResult` | Gyration tensor per cluster |
//! | [`inertia_tensor`] | `&ClusterResult` | Moment of inertia tensor |
//! | [`radius_of_gyration`] | `&ClusterResult` | Radius of gyration |

pub mod accumulator;
pub mod center_of_mass;
pub mod cluster;
pub mod cluster_centers;
pub mod error;
pub mod gyration_tensor;
pub mod inertia_tensor;
pub mod msd;
pub mod radius_of_gyration;
pub mod rdf;
pub mod reducer;
pub mod traits;
pub mod util;

// Re-exports
pub use accumulator::Accumulator;
pub use center_of_mass::{CenterOfMass, CenterOfMassResult};
pub use cluster::{Cluster, ClusterResult};
pub use cluster_centers::ClusterCenters;
pub use error::ComputeError;
pub use gyration_tensor::GyrationTensor;
pub use inertia_tensor::InertiaTensor;
pub use msd::{MSD, MSDResult};
pub use radius_of_gyration::RadiusOfGyration;
pub use rdf::{RDF, RDFResult};
pub use reducer::{ConcatReducer, Reducer, SumReducer};
pub use traits::Compute;
