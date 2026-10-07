//! File I/O and format conversion for the WASM API.
//!
//! Provides readers, writers, and parsers for common molecular file
//! formats:
//!
//! | Module | JS class / function | Formats |
//! |--------|-------------------|---------|
//! | [`reader`] | `CifReader`, `CubeReader`, `VaspChgcarReader`, `GroReader`, `Mol2Reader`, `VaspPoscarReader`, `XsfReader`, `AmberInpcrdReader`, `AmberAcReader`, `readMsgpackFrameBytes`, `readJsonFrameStr` | Whole-content readers for the formats without a stream: CIF, Cube, CHGCAR, GRO, MOL2, POSCAR, XSF, AMBER inpcrd, AC; the `molrs::stream` wire encodings |
//! | [`streaming`] | `LammpsDumpStream`, `XyzStream`, `PdbStream`, `LammpsDataStream`, `SdfStream`, `DcdStream`, `XtcStream`, `TrrStream` | The one reader of XYZ/ExtXYZ, PDB, LAMMPS data/dump, SDF, DCD, XTC, TRR: chunk-fed `FrameIndexBuilder` + per-range parse |
//! | [`writer`] | `writeXyzStr`, `writePdbStr`, …, `writeDcdBytes`, …, `writeMsgpackFrameBytes`, `writeJsonFrameStr` | One writer per format — no export picks a format from a string |
//! | [`log`] | `readLammpsLogThermo`, `isLammpsLog` | LAMMPS log thermo tables |
//! | `smiles` | `readSmilesStr`, `SmilesIr.parse` | SMILES strings (`smiles` feature) |
//! | [`mrec`] | `MrecReader`, `readMrecFrame`, `mrecSections` | `*.mrec` scientific records (Zarr V3) |
//! | [`mesh`] | `readStlBytes(bytes)` | STL surface meshes (ASCII or binary) — produces a `Mesh`, not a `Frame` |
//!
//! No reader takes a file handle, since WASM has no filesystem access: a
//! whole-content reader takes the file's text (or bytes), a stream takes
//! chunks the host copies into its input buffer. Each format has exactly one
//! of the two.

pub mod log;
pub mod mesh;
pub mod mrec;
pub mod reader;
#[cfg(feature = "smiles")]
pub mod smiles;
pub mod streaming;
pub mod writer;

pub use log::*;
pub use mesh::*;
pub use mrec::*;
pub use reader::*;
#[cfg(feature = "smiles")]
pub use smiles::*;
pub use streaming::*;
pub use writer::*;
