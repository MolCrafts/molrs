//! # molrs
//!
//! A Rust library providing core molecular modeling functionality.
//!
//! ## Module layout
//!
//! - [`store`] — columnar data containers (`Block`, `Frame`, `Trajectory`, keys)
//! - [`system`] — molecular representations (`Atomistic`, `MolGraph`, `Topology`, elements)
//! - [`spatial`] — regions, neighbor lists, geometry
//! - [`generate`] — structure generators (inverse of compute: parameters → coordinates)
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

// There is no `data` module: molrs embeds no parameter text. Every force-field
// table is typed, compiled Rust under `ff::params` — MMFF94/94s and OPLS-AA
// included, since `chem-perceive-14` — so nothing here `include_str!`s an XML to
// re-parse at runtime.

// Domain groups
pub mod spatial;
pub mod store;
pub mod system;

// Foundations
pub mod error;
pub mod math;
pub mod types;
pub mod units;

#[cfg(all(test, feature = "rayon"))]
pub(crate) mod test_rayon;

// NOTE: chemical perception (rings, aromaticity, hydrogens, stereo, rotatable,
// Gasteiger, SMARTS) used to live here as `core::chem`. It now sits one layer up
// in `crate::perceive` — above `core`, below `ff`. Its crate-root re-exports moved
// with it to `lib.rs`, so `molrs::find_rings`, `molrs::SmartsPattern`, … still
// resolve unchanged.

// Public re-exports for common types
pub use error::MolRsError;
pub use spatial::simbox::{BoxError, BoxKind, Mic, SimBox};
pub use store::block::Block;
pub use store::frame::Frame;
pub use store::frame_access::FrameAccess;
pub use store::frame_view::FrameView;
pub use store::meta::{MetaMap, MetaValue};
pub use store::record::{
    MolRec as Record, Observables, RECORD_FORMAT_NAME, RECORD_SCHEMA_VERSION, RESERVED_META_KEYS,
};
pub use store::trajectory::{
    ObservableData, ObservableKind, ObservableRecord, SchemaValue, Trajectory,
};
pub use system::atomistic::{
    AngleId, AtomId, Atomistic, Bond, BondId, DihedralId, ExtractedAtomistic, ImproperId,
};
pub use system::coarsegrain::{CoarseGrain, ExtractedCoarseGrain};
pub use system::extract::{ExtractedBall, InducedSubgraph};
pub use system::graph_hash::{canonical_order, is_isomorphic, structural_hash};
pub use system::mapping::{CGMapping, WeightScheme};
pub use system::molgraph::{Atom, Bead, KindId, MolGraph, NodeId, PropValue, Relation};
pub use system::topology::{Topology, TopologyRingInfo};
pub use units::{
    Dimension, Quantity, Unit, UnitDef, UnitPreset, UnitPresetRegistry, UnitRegistry, UnitsError,
};
