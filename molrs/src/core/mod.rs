//! # molrs
//!
//! A Rust library providing core molecular modeling functionality.
//!
//! ## Module layout
//!
//! - [`store`] — columnar data containers (`Block`, `Frame`, `Trajectory`, keys)
//! - [`system`] — molecular representations (`Atomistic`, `MolGraph`, `Topology`, elements)
//! - [`chem`] — chemical perception (aromaticity, charges, rings, stereo, SMARTS)
//! - [`spatial`] — regions, neighbor lists, geometry
//! - [`math`], [`units`] — numerical and unit-system foundations
//!
//! ## Examples
//!
//! ### Element lookup
//!
//! ```
//! use molrs::Element;
//!
//! // Look up elements by atomic number
//! let hydrogen = Element::by_number(1).unwrap();
//! assert_eq!(hydrogen.symbol(), "H");
//!
//! // Or by symbol (case-insensitive)
//! let h = Element::by_symbol("h").unwrap();
//! assert_eq!(h.name(), "Hydrogen");
//! ```
//!
//! ### Packing
//!
//! Molecular packing (Packmol port) lives in the standalone
//! [`molcrafts-molpack`](https://crates.io/crates/molcrafts-molpack) crate.

#![allow(missing_docs)]
#![warn(rustdoc::missing_crate_level_docs)]

// Embedded data files
pub mod data;

// Domain groups
pub mod chem;
pub mod spatial;
pub mod store;
pub mod system;

// Foundations
pub mod error;
pub mod math;
pub mod types;
pub mod units;

// Public re-exports for common types
pub use chem::aromaticity::perceive_aromaticity;
pub use chem::gasteiger::{GasteigerCharges, compute_gasteiger_charges};
pub use chem::hydrogens::{add_hydrogens, implicit_h_count, remove_hydrogens};
pub use chem::rings::{RingInfo, find_rings};
pub use chem::smarts::{MatchOptions, Reaction, SmartsMatch, SmartsPattern};
pub use chem::stereo::{
    BondStereo, TetrahedralStereo, assign_bond_stereo_from_3d, assign_stereo_from_3d,
    chiral_volume, find_chiral_centers,
};
pub use error::MolRsError;
pub use store::block::Block;
pub use store::frame::Frame;
pub use store::frame_access::FrameAccess;
pub use store::frame_view::FrameView;
pub use store::trajectory::{
    ObservableData, ObservableKind, ObservableRecord, SchemaValue, Trajectory,
};
pub use system::atomistic::{
    AngleId, AtomId, Atomistic, Bond, BondId, DihedralId, ExtractedAtomistic, ImproperId,
};
pub use system::coarsegrain::{CoarseGrain, ExtractedCoarseGrain};
pub use system::element::Element;
pub use system::extract::{ExtractedBall, InducedSubgraph};
pub use system::graph_hash::{canonical_order, is_isomorphic, structural_hash};
pub use system::mapping::{CGMapping, WeightScheme};
pub use system::molgraph::{Atom, Bead, KindId, MolGraph, NodeId, PropValue, Relation};
pub use system::topology::{Topology, TopologyRingInfo};
pub use units::{Dimension, Quantity, Unit, UnitDef, UnitRegistry, UnitsError};
