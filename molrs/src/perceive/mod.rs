//! Chemical perception algorithms operating on molecular graphs:
//! aromaticity, bond-type perception, hydrogen handling, ring detection,
//! stereochemistry, rotatable bonds, SMARTS matching, and subgraph matching
//! of coarse-grained bead graphs.
//!
//! *Perception* means deriving chemical facts that a connectivity graph
//! implies but does not state — which atoms lie on rings, which rings are
//! aromatic, which centres are chiral. SMARTS (SMILES Arbitrary Target
//! Specification) is the substructure query language matched here;
//! [`SubgraphMatcher`] is its coarse-grained counterpart, finding groups of
//! beads (one node per group of atoms) by bead type. [`Coarsener`] maps
//! disjoint node groups onto the sites of a new coarse-grained graph, each at
//! its group's centre of mass and records its axis ([`CoarsenError`] names a
//! refusal).
//!
//! Gasteiger charges used to live here. They are a *charge model*, not a
//! perception, and they now sit with the other charge models in
//! `crate::ff::charge` (feature `ff`) — one implementation, reached through the
//! `ChargeModel` trait there.
//!
//! The layer's public face is the [`Perceive`] builder, which gives every
//! perception one shape — graph in / graph out, non-mutating:
//! `Perceive::new().find_rings(&mol) -> Atomistic`. The free functions it wraps
//! remain available (and re-exported at the crate root) for callers that want
//! the raw side table / map.

/// Executable specification for the Aromatic Bond Representation Standard.
/// Tests only; see the module docs for how to run the red line.
#[cfg(all(test, feature = "smiles"))]
mod aromatic_standard;
pub mod aromaticity;
pub mod bond_order;
pub mod bond_type;
pub mod builder;
pub mod coarsen;
pub mod equivalence;
mod hybridization;
pub mod hydrogens;
pub mod ring_class;
pub mod rings;
pub mod rotatable;
pub mod smarts;
pub mod stereo;
pub mod subgraph;

pub use builder::Perceive;
pub use coarsen::{CoarsenError, Coarsener};
pub use hybridization::{Hybridization, conjugated_atoms, hybridizations};
pub use subgraph::SubgraphMatcher;
