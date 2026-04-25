//! # molrs
//!
//! A Rust library providing core molecular modeling functionality.
//!
//! ## Features
//!
//! - Element data and lookup by atomic number or symbol
//! - Core data structures, geometry, IO, and neighbor-list algorithms
//! - Type-safe molecular representations
//!
//! ## Examples
//!
//! ### Element lookup
//!
//! ```
//! use molrs_core::Element;
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

// Modules formerly in core/
pub mod atomistic;
pub mod block;
pub mod coarsegrain;
pub mod element;
pub mod field;
pub mod frame;
pub mod frame_access;
pub mod frame_view;
pub mod gasteiger;
pub mod grid;
pub mod hydrogens;
pub mod mapping;
pub mod molgraph;
pub mod molrec;
pub mod region;
pub mod rings;
pub mod rotatable;
pub mod stereo;
pub mod topology;
pub mod types;

// Other top-level modules
pub mod error;
pub mod math;
pub mod neighbors;

// Public re-exports for common types
pub use atomistic::Atomistic;
pub use block::Block;
pub use coarsegrain::CoarseGrain;
pub use element::Element;
pub use error::MolRsError;
pub use field::UniformGridField;
pub use frame::Frame;
pub use frame_access::FrameAccess;
pub use frame_view::FrameView;
pub use gasteiger::{GasteigerCharges, compute_gasteiger_charges};
pub use grid::Grid;
pub use hydrogens::{add_hydrogens, implicit_h_count, remove_hydrogens};
pub use mapping::{CGMapping, WeightScheme};
pub use molgraph::{Atom, AtomId, Bead, Bond, BondId, MolGraph, PropValue};
pub use rings::{RingInfo, find_rings};
pub use stereo::{
    BondStereo, TetrahedralStereo, assign_bond_stereo_from_3d, assign_stereo_from_3d,
    chiral_volume, find_chiral_centers,
};
pub use topology::{Topology, TopologyRingInfo};

pub use molrec::{
    MolRec, ObservableData, ObservableKind, ObservableRecord, SchemaValue, Trajectory,
};
