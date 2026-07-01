//! Integration tests for `molrs-compute`.
//!
//! The module tree below mirrors the crate's `src/` layout: one test module per
//! source area. Unlike `molrs-io`, compute does no file I/O — every test builds
//! its frames / point sets in code and asserts against the PUBLIC API. Inline
//! `#[cfg(test)]` unit tests inside `src/**` stay there; these integration tests
//! exercise cross-module guarantees (the `Graph` DAG) and end-to-end analyses
//! against known analytical results (RDF of a lattice, MSD of linear motion).

#[path = "compute/center_of_mass.rs"]
mod center_of_mass;
#[path = "compute/combined_distribution.rs"]
mod combined_distribution;
#[path = "compute/distribution.rs"]
mod distribution;
#[path = "compute/hbond.rs"]
mod hbond;
#[path = "compute/msd.rs"]
mod msd;
#[path = "compute/rdf.rs"]
mod rdf;
#[path = "compute/reorientation.rs"]
mod reorientation;
#[path = "compute/spatial_distribution.rs"]
mod spatial_distribution;
#[path = "compute/spectra_chiral.rs"]
mod spectra_chiral;
#[path = "compute/van_hove.rs"]
mod van_hove;
#[cfg(feature = "voronoi")]
#[path = "compute/voronoi.rs"]
mod voronoi;
#[cfg(feature = "voronoi")]
#[path = "compute/voronoi_integration.rs"]
mod voronoi_integration;
