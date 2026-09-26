//! Molecular system representations: the domain-agnostic
//! [`MolGraph`](molgraph::MolGraph) and its three newtype leaves — all-atom
//! [`Atomistic`](atomistic::Atomistic), coarse-grained
//! [`CoarseGrain`](coarsegrain::CoarseGrain) and
//! [`Fragment`], a graph with named attachment points —
//! plus the bond vocabulary, connectivity [`Topology`](topology::Topology),
//! subgraph extraction, graph hashing and element data.

pub mod atomistic;
pub mod bond;
pub mod bond_weights;
pub mod coarsegrain;
pub(crate) mod element;
pub mod entity_table;
pub mod extract;
pub mod frag_graph;
pub mod fragment;
pub mod graph_hash;
// Its only non-test caller, `FragLibrary::map`, is behind `builder`.
#[cfg(any(test, feature = "builder"))]
pub(crate) mod graph_match;
pub mod mapping;
pub mod molgraph;
pub mod topology;

pub use bond::{BondNumber, BondType};
pub use bond_weights::BondDistanceWeights;
pub use extract::{ExtractedBall, InducedSubgraph};
pub use frag_graph::{FragEdge, FragGraph};
pub use fragment::{BeadError, Fragment, Port, PortId, PortKind};
pub use mapping::Mapping;
