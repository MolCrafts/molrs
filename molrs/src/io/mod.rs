//! File I/O for molecular data, organized by content kind:
//!
//! - [`data`] — single-structure formats (PDB, XYZ, GRO, mol2, SDF, CIF,
//!   LAMMPS data, XSF, CHGCAR/POSCAR, Cube, AMBER inpcrd / prmtop structure)
//! - [`trajectory`] — multi-frame formats (DCD, LAMMPS dump)
//! - [`store`] — persistence backends (Zarr V3, feature `zarr`)
//! - [`reader`] / [`writer`] / [`streaming`] — shared traits and the
//!   chunk-based frame-indexing infrastructure
//! - [`smiles`] — SMILES/SMARTS notation parsing (feature `smiles`)

pub mod data;
/// Shared LAMMPS primitives (atom_style layouts, box bounds, helpers).
/// Used by both the data-file and dump trajectory readers.
pub(crate) mod lammps;
/// Log-file parsers (LAMMPS run output / thermo diagnostics).
pub mod log;
pub mod store;
pub mod trajectory;

pub mod reader;
pub mod streaming;
pub mod writer;

#[cfg(feature = "smiles")]
pub mod smiles;
