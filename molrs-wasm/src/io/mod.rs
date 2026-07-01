//! File I/O and format conversion for the WASM API.
//!
//! Provides readers, writers, and parsers for common molecular file
//! formats:
//!
//! | Module | JS class / function | Formats |
//! |--------|-------------------|---------|
//! | [`reader`] | `XYZReader`, `PDBReader`, `CIFReader`, `LAMMPSReader`, `LAMMPSTrajReader`, `SDFReader`, `CubeReader`, `CHGCARReader`, `GROReader`, `MOL2Reader`, `POSCARReader`, `DCDReader`, `TRRReader`, `XTCReader` | Read XYZ/ExtXYZ, PDB, CIF, LAMMPS data/dump, SDF, Cube, CHGCAR, GRO, MOL2, POSCAR, DCD, TRR, XTC (GRO/TRR/XTC convert nm→Å on read) |
//! | [`streaming`] | `WasmLammpsDumpStream`, `WasmXyzStream`, `WasmPdbStream`, `WasmLammpsDataStream`, `WasmSdfStream` | Streaming readers driven by a chunk-fed `FrameIndexBuilder` |
//! | [`writer`] | `writeFrame(frame, format)` | Write XYZ, PDB, LAMMPS dump |
//! | [`zarr`] | `MolRecReader` | Read MolRec Zarr V3 archives |
//!
//! All readers consume string content (not file handles) since
//! WASM does not have filesystem access. Use the File API in the
//! browser to read files, then pass the text content to the reader.

pub mod reader;
pub mod streaming;
pub mod writer;
pub mod zarr;

pub use reader::*;
pub use streaming::*;
pub use writer::*;
pub use zarr::*;
