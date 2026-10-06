//! The core data model and its foundations. Not a public namespace: each
//! domain below is re-exported at the crate root as its own facade.
//!
//! ## Module layout
//!
//! - [`store`] — columnar data containers (`Block`, `Frame`, `Trajectory`, keys)
//! - [`system`] — molecular representations (`Atomistic`, `MolGraph`, `Topology`, elements)
//! - [`spatial`] — regions, neighbor lists, geometry
//! - [`math`], [`units`] — numerical and unit-system foundations
//! - [`error`] — the crate error type
//!
//! Structure builders live in `crate::builder` (feature `builder`), above the
//! core layer.
//!
//! ## Examples
//!
//! ### Element lookup
//!
//! ```
//! use molrs::system::Element;
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

// Domain groups — each re-exported at the crate root as a facade.
pub mod spatial;
pub mod store;
pub mod system;

// Foundations
pub mod error;
pub mod math;
pub mod units;

#[cfg(all(test, feature = "rayon"))]
pub(crate) mod test_rayon;

// Chemical perception (rings, aromaticity, hydrogens, stereo, rotatable, SMARTS)
// sits one layer up in `crate::perceive` — above `core`, below `ff`.
