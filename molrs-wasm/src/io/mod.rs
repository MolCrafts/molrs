//! File I/O and format conversion for the WASM API.
//!
//! Provides readers, writers, and parsers for common molecular file
//! formats:
//!
//! | Module | JS class / function | Formats |
//! |--------|-------------------|---------|
//! | [`reader`] | `CIFReader`, `CubeReader`, `CHGCARReader`, `GROReader`, `MOL2Reader`, `POSCARReader`, `XSFReader`, `AmberInpcrdReader`, `AcReader` | Whole-content readers for the formats without a stream: CIF, Cube, CHGCAR, GRO, MOL2, POSCAR, XSF, AMBER inpcrd, AC |
//! | [`streaming`] | `LAMMPSTrajStream`, `XYZStream`, `PDBStream`, `LAMMPSStream`, `SDFStream`, `DCDStream`, `XTCStream`, `TRRStream` | The one reader of XYZ/ExtXYZ, PDB, LAMMPS data/dump, SDF, DCD, XTC, TRR: chunk-fed `FrameIndexBuilder` + per-range parse |
//! | [`writer`] | `writeFrame(frame, format)` | Write XYZ, PDB, LAMMPS dump |
//! | [`log`] | `readLammpsLogThermo`, `isLammpsLog` | LAMMPS log thermo tables |
//! | `smiles` | `parseSMILES` → `SmilesIR` | SMILES strings (`smiles` feature) |
//! | [`zarr`] | `TrajectoryReader` | Read frame-sequence Zarr V3 archives |
//! | [`mesh`] | `readSTL(bytes)` | STL surface meshes (ASCII or binary) — produces a `Mesh`, not a `Frame` |
//!
//! No reader takes a file handle, since WASM has no filesystem access: a
//! whole-content reader takes the file's text (or bytes), a stream takes
//! chunks the host copies into its input buffer. Each format has exactly one
//! of the two.

pub mod log;
pub mod mesh;
pub mod reader;
#[cfg(feature = "smiles")]
pub mod smiles;
pub mod streaming;
pub mod writer;
pub mod zarr;

pub use log::*;
pub use mesh::*;
pub use reader::*;
#[cfg(feature = "smiles")]
pub use smiles::*;
pub use streaming::*;
pub use writer::*;
pub use zarr::*;
