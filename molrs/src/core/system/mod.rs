//! Molecular system representations: the domain-agnostic
//! [`MolGraph`] and its two newtype leaves — all-atom
//! [`Atomistic`] and coarse-grained
//! [`CoarseGrain`] — plus ports, the named
//! attachment points any graph may carry ([`Port`]), the bond vocabulary, connectivity [`Topology`],
//! subgraph extraction, graph hashing and element data.

mod atomistic;
mod bond;
mod bond_weights;
mod coarsegrain;
pub(crate) mod element;
pub mod entity_table;
mod extract;
pub(crate) mod graph_hash;
mod link;
pub(crate) mod molgraph;
mod port;
mod topology;

pub use atomistic::{Atomistic, ExtractedAtomistic};
pub use bond::{BondNumber, BondType};
pub use bond_weights::BondDistanceWeights;
pub use coarsegrain::{CoarseGrain, ExtractedCoarseGrain};
pub use element::Element;
pub use extract::{ExtractedBall, InducedSubgraph};
pub use graph_hash::{canonical_order, is_isomorphic, structural_hash};
pub use link::{LinkError, LinkManyError};
pub use molgraph::{
    Atom, FromMolGraph, KindId, MolGraph, NodeId, PropValue, Relation, RelationId, node_from_u64,
    node_to_u64, relation_from_u64, relation_to_u64,
};
pub use port::{Port, PortKind};
pub use topology::{Topology, TopologyError, TopologyRingInfo};
