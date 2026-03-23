//! File I/O and format conversion for the WASM API.
//!
//! Provides readers, writers, and parsers for common molecular file
//! formats:
//!
//! | Module | JS class / function | Formats |
//! |--------|-------------------|---------|
//! | [`reader`] | `XYZReader`, `PDBReader`, `LAMMPSReader` | Read XYZ/ExtXYZ, PDB, LAMMPS data files |
//! | [`writer`] | `writeFrame(frame, format)` | Write XYZ, PDB |
//! | [`smiles`] | `parseSMILES(str)` -> `SmilesIR` | Parse SMILES notation |
//! | [`zarr`] | `SimulationReader` | Read Zarr V3 simulation archives |
//!
//! All readers consume string content (not file handles) since
//! WASM does not have filesystem access. Use the File API in the
//! browser to read files, then pass the text content to the reader.

pub mod reader;
pub mod smiles;
pub mod writer;
pub mod zarr;

pub use reader::*;
pub use smiles::*;
pub use writer::*;
pub use zarr::*;
