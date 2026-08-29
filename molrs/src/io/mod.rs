//! File I/O for molecular data, organized by content kind:
//!
//! - [`data`] — single-structure formats (PDB, XYZ, GRO, mol2, SDF, CIF,
//!   LAMMPS data, XSF, CHGCAR/POSCAR, Cube, AMBER inpcrd / prmtop structure)
//! - [`trajectory`] — multi-frame formats (DCD, LAMMPS dump)
//! - [`zarr`] / [`csv`] — serialization of the store types themselves, as
//!   opposed to [`data`] and [`trajectory`], which read molecular file
//!   formats (Zarr V3 is feature `zarr`)
//! - [`reader`] / [`writer`] / [`streaming`] — shared traits and the
//!   chunk-based frame-indexing infrastructure
//! - [`smiles`] — SMILES/SMARTS notation parsing (feature `smiles`)

pub mod csv;
pub mod data;
/// Shared LAMMPS primitives (atom_style layouts, box bounds, helpers).
/// Used by both the data-file and dump trajectory readers.
pub(crate) mod lammps;
/// Log-file parsers (LAMMPS run output / thermo diagnostics).
pub mod log;
pub mod trajectory;

pub mod reader;
pub mod streaming;
pub mod writer;

#[cfg(feature = "smiles")]
pub mod smiles;
#[cfg(feature = "zarr")]
pub mod zarr;
