//! File I/O and format conversion for the WASM API.
//!
//! Provides readers, writers, and parsers for common molecular file
//! formats:
//!
//! | Module | JS class / function | Formats |
//! |--------|-------------------|---------|
//! | [`reader`] | `XYZReader`, `PDBReader`, `LAMMPSReader`, `LAMMPSTrajReader` | Read XYZ/ExtXYZ, PDB, LAMMPS data/dump files |
//! | [`writer`] | `writeFrame(frame, format)` | Write XYZ, PDB, LAMMPS dump |
//! | [`zarr`] | `MolRecReader` | Read MolRec Zarr V3 archives |
//!
//! All readers consume string content (not file handles) since
//! WASM does not have filesystem access. Use the File API in the
//! browser to read files, then pass the text content to the reader.

pub mod reader;
pub mod writer;
pub mod zarr;

pub use reader::*;
pub use writer::*;
pub use zarr::*;
