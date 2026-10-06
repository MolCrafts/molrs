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
//! beads (one node per group of atoms) by bead type. Building a new
//! coarse-grained graph from such groups is construction, not perception: it
//! is `crate::builder::Coarsener`.
//!
//! Gasteiger charges used to live here. They are a *charge model*, not a
//! perception, and they now sit with the other charge models in
//! `crate::ff::charge` (feature `ff`) — one implementation, reached through the
//! `ChargeModel` trait there.
//!
//! # One way to perceive
//!
//! Writing perceived facts onto a graph has one public spelling: the
//! [`Perceive`] builder, graph in / graph out and non-mutating —
//! `Perceive::new().find_rings(&mol) -> Atomistic`. The graph-writing
//! functions behind it (hydrogen addition, aromaticity, bond orders, BCC bond
//! types, Kekulé numbers) are crate-private.
//!
//! The modules keep only the *side tables* public — what the builder
//! projects onto props, for callers that need the table itself rather than a
//! graph: [`rings::find_rings`] → [`rings::RingInfo`],
//! [`rotatable::detect_rotatable_bonds`], [`stereo::assign_stereo_from_3d`],
//! [`equivalence::find_equivalence_classes`], [`bond_order::judge_bond_orders`],
//! [`hybridizations`] / [`conjugated_atoms`] (→ [`Hybridization`]).
//! [`hydrogens::remove_hydrogens`] is an edit, not a perception, and stays a
//! function.

/// Executable specification for the Aromatic Bond Representation Standard.
/// Tests only; see the module docs for how to run the red line.
#[cfg(all(test, feature = "smiles"))]
mod aromatic_standard;
pub mod aromaticity;
pub mod bond_order;
pub mod bond_type;
pub mod builder;
pub mod equivalence;
mod hybridization;
pub mod hydrogens;
pub mod ring_class;
pub mod rings;
pub mod rotatable;
// SMARTS matching compiles from the `io::smiles` parser, so it needs `smiles`.
#[cfg(feature = "smiles")]
pub mod smarts;
pub mod stereo;
pub mod subgraph;

pub use builder::Perceive;
pub use hybridization::{Hybridization, conjugated_atoms, hybridizations};
pub use subgraph::SubgraphMatcher;
