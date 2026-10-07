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
//! # Two verbs: `perceive_*` reports, `assign_*` writes
//!
//! Every perception is a free function, in one of two shapes — the split
//! RDKit makes between a query and an `Assign*` (`AssignStereochemistry`,
//! `AssignAtomChiralTagsFromStructure`), and the term the literature uses for
//! the computation itself (ring perception, bond-order perception):
//!
//! - **`perceive_<fact>(mol) -> table`** computes a fact and returns it as a
//!   side table, leaving the graph alone: [`perceive_rings`] → [`RingInfo`],
//!   [`perceive_rotatable_bonds`], [`perceive_chiral_centers`],
//!   [`perceive_tetrahedral_stereo`], [`perceive_bond_stereo`],
//!   [`perceive_equivalence_classes`], [`perceive_bond_orders`],
//!   [`perceive_hybridizations`], [`perceive_conjugated_atoms`],
//!   [`perceive_ring_classes`].
//! - **`assign_<fact>(mol) -> Atomistic`** writes a perceived fact onto a
//!   *clone* of the graph as atom / bond props and returns it — graph in /
//!   graph out, non-mutating, so the steps compose: [`assign_rings`],
//!   [`assign_aromaticity`], [`assign_stereo`], [`assign_rotatable_bonds`],
//!   [`assign_bond_orders`], [`assign_kekule_bond_orders`],
//!   [`assign_bcc_bond_types`], [`assign_bcc_bond_types_from_connectivity`],
//!   [`assign_equivalence_classes`].
//!
//! [`add_hydrogens`] / [`remove_hydrogens`] are edits of the graph, not
//! perceptions, and keep their verbs.
//!
//! | Function | Atom props | Bond props |
//! |---|---|---|
//! | [`assign_rings`] | `is_in_ring` (0/1), `n_rings` | `is_in_ring` (0/1), `n_rings` |
//! | [`assign_aromaticity`] | `is_aromatic` (0/1) | `bond_type` (aromatic), `bond_number` |
//! | [`assign_stereo`] | `stereo` (`"CW"` / `"CCW"`) | `stereo` (`"E"` / `"Z"` / `"either"`) |
//! | [`assign_rotatable_bonds`] | — | `is_rotatable` (0/1) |
//! | [`assign_bond_orders`] | — | `bond_number`, `bond_type` (antechamber's Kekulé structure) |
//! | [`assign_kekule_bond_orders`] | — | `bond_number` of every aromatic bond |
//! | [`assign_bcc_bond_types`] | — | `bcc_bond_type` (1/2/3/6/7/8/9) |
//! | [`assign_bcc_bond_types_from_connectivity`] | — | `bcc_bond_type`, from connectivity-judged orders |
//! | [`assign_equivalence_classes`] | `equiv_class` (0-based class id) | — |
//!
//! The modules are private; every name is re-exported here, so the Rust path
//! `molrs::perceive::<name>` is the Python path `molrs.perceive.<name>`.

/// Executable specification for the Aromatic Bond Representation Standard.
/// Tests only; see the module docs for how to run the red line.
#[cfg(all(test, feature = "smiles"))]
mod aromatic_standard;
mod aromaticity;
mod bcc_bond_class;
mod bond_order;
mod equivalence;
mod hybridization;
mod hydrogens;
mod kekule;
mod ring_class;
mod rings;
mod rotatable;
// SMARTS matching compiles from the `io::smiles` parser, so it needs `smiles`.
#[cfg(feature = "smiles")]
pub mod smarts;
mod stereo;
mod subgraph;

pub use aromaticity::assign_aromaticity;
// The in-place marker, for SMARTS reactions and the conformer pipeline.
#[cfg(feature = "smiles")]
pub(crate) use aromaticity::mark_aromaticity;
pub use bcc_bond_class::{assign_bcc_bond_types, assign_bcc_bond_types_from_connectivity};
pub use bond_order::{assign_bond_orders, perceive_bond_orders};
pub use equivalence::{
    EquivalenceClasses, EquivalenceLevel, EquivalenceOptions, assign_equivalence_classes,
    perceive_equivalence_classes,
};
pub use hybridization::{Hybridization, perceive_conjugated_atoms, perceive_hybridizations};
pub use hydrogens::{add_hydrogens, implicit_h_count, remove_hydrogens};
pub use kekule::assign_kekule_bond_orders;
pub use ring_class::{
    AntechamberRingMembership, AntechamberRingSummary, RingClasses, perceive_ring_classes,
};
pub use rings::{RingInfo, assign_rings, perceive_rings, small_ring_closure};
pub use rotatable::{
    RotatableBond, UnknownBondPolicy, assign_rotatable_bonds, downstream_atoms,
    perceive_rotatable_bonds, perceive_rotatable_bonds_with_downstream,
};
pub use stereo::{
    BondStereo, TetrahedralStereo, assign_stereo, chiral_volume, perceive_bond_stereo,
    perceive_chiral_centers, perceive_tetrahedral_stereo,
};
pub use subgraph::SubgraphMatcher;
