//! Molecular system representations: the domain-agnostic
//! [`MolGraph`](molgraph::MolGraph) and its two newtype leaves — all-atom
//! [`Atomistic`](atomistic::Atomistic) and coarse-grained
//! [`CoarseGrain`](coarsegrain::CoarseGrain) — plus ports, the named
//! attachment points any graph may carry ([`port`]), the bond vocabulary, connectivity [`Topology`](topology::Topology),
//! subgraph extraction, graph hashing and element data.

pub mod atomistic;
pub mod bond;
pub mod bond_weights;
pub mod coarsegrain;
pub(crate) mod element;
pub mod entity_table;
pub mod extract;
pub mod graph_hash;
pub mod link;
pub mod molgraph;
pub mod port;
pub mod topology;

pub use bond::{BondNumber, BondType};
pub use bond_weights::BondDistanceWeights;
pub use extract::{ExtractedBall, InducedSubgraph};
pub use link::{LinkError, LinkManyError};
pub use port::{Port, PortId, PortKind};
