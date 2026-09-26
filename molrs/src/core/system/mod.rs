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
pub mod fragment;
pub mod graph_hash;
pub mod link;
pub mod molgraph;
pub mod topology;

pub use bond::{BondNumber, BondType};
pub use bond_weights::BondDistanceWeights;
pub use extract::{ExtractedBall, InducedSubgraph};
pub use fragment::{Fragment, MergeMaps, Port, PortId, PortKind};
pub use link::LinkError;
