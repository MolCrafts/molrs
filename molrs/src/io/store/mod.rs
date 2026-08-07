//! Serialization of the store types themselves — as opposed to `io/data`
//! and `io/trajectory`, which read molecular *file formats*.
//!
//! - [`csv`] — `Block` ↔ CSV text. Lived in `core/store/block` until it was
//!   moved here: a format parser has no business inside the container it
//!   parses into.
//! - [`zarr`] — Zarr V3 persistence backend (feature `zarr`).

pub mod csv;

#[cfg(feature = "zarr")]
pub mod zarr;
