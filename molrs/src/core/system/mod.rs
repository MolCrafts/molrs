//! Molecular system representations: atomistic and coarse-grained entities,
//! topology, the [`MolGraph`](molgraph::MolGraph) molecular graph, element
//! data, and CG mapping.

pub mod atomistic;
pub mod coarsegrain;
pub mod element;
pub mod entity_table;
pub mod extract;
pub mod graph_hash;
pub mod mapping;
pub mod molgraph;
pub mod topology;

pub use extract::{ExtractedBall, InducedSubgraph};
