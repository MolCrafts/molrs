//! File I/O for molecular data, organized by content kind:
//!
//! - [`data`] — single-structure formats (PDB, XYZ, GRO, mol2, SDF, CIF,
//!   LAMMPS data, XSF, CHGCAR/POSCAR, Cube, AMBER inpcrd / prmtop structure)
//! - [`trajectory`] — multi-frame formats (DCD, LAMMPS dump)
//! - [`mesh`] — surface meshes (STL); reads into a
//!   [`TriMesh`](crate::spatial::TriMesh), not a [`Frame`](crate::store::Frame)
//! - [`mrec`] / [`csv`] — serialization of the store types themselves, as
//!   opposed to [`data`] and [`trajectory`], which read molecular file
//!   formats. [`mrec`] writes and reads a [`crate::store::Frame`] or
//!   [`crate::store::Trajectory`] as a `*.mrec` directory or packed `*.mrec.zip`
//!   (Zarr V3 on disk; Cargo feature `zarr`, adapter crate-private)
//! - [`read_frame`] / [`write_frame`] ([`FrameFormat`]), the one door that picks a
//!   structure format from a file name (or format name) and hands off to it
//! - [`reader`] / [`writer`] / [`streaming`] — shared traits and the
//!   chunk-based frame-indexing infrastructure
//! - [`smiles`] — SMILES/SMARTS and CGsmiles notation parsing (feature
//!   `smiles`)

pub mod csv;
pub mod data;
mod format;
/// Shared LAMMPS primitives (atom_style layouts, box bounds, helpers).
/// Used by both the data-file and dump trajectory readers.
pub(crate) mod lammps;
// Log-file parsers (LAMMPS run output / thermo diagnostics).
pub mod log;
pub mod mesh;
pub mod trajectory;

pub mod reader;
pub mod streaming;
pub mod writer;

#[cfg(feature = "zarr")]
pub mod mrec;
#[cfg(feature = "smiles")]
pub mod smiles;
#[cfg(feature = "zarr")]
pub(crate) mod zarr;

pub use format::{FrameFormat, read_frame, write_frame};

/// The one `InvalidData` error of the io readers and writers: a parse or
/// shape failure carrying `e`'s message.
pub(crate) fn invalid_data<E: std::fmt::Display>(e: E) -> std::io::Error {
    std::io::Error::new(std::io::ErrorKind::InvalidData, e.to_string())
}
