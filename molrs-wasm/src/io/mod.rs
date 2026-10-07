//! File I/O and format conversion for the WASM API.
//!
//! One module per format, as in `molrs::io`: each holds that format's reader
//! (a whole-content `*Reader` class or a chunk-fed `*Stream` class, never
//! both) and its writer export(s).
//!
//! | Module | JS classes / functions | Format |
//! |--------|------------------------|--------|
//! | [`xyz`] | `XyzStream`, `writeXyzStr` | XYZ / Extended XYZ |
//! | [`pdb`] | `PdbStream`, `writePdbStr` | Protein Data Bank |
//! | [`gro`] | `GroReader`, `writeGroStr` | GROMACS GRO (nm ↔ Å at the molrs boundary) |
//! | [`sdf`] | `SdfStream` | MDL SDF / MOL (read-only: molrs has no SDF writer) |
//! | [`mol2`] | `Mol2Reader`, `writeMol2Str` | Tripos MOL2 |
//! | [`cif`] | `CifReader`, `writeCifStr` | Crystallographic Information File |
//! | [`vasp`] | `VaspPoscarReader`, `VaspChgcarReader`, `writeVaspPoscarStr` | VASP POSCAR / CONTCAR, CHGCAR (read-only) |
//! | [`dcd`] | `DcdStream`, `writeDcdBytes` | DCD trajectory |
//! | [`trr`] | `TrrStream`, `writeTrrBytes` | GROMACS TRR |
//! | [`xtc`] | `XtcStream`, `writeXtcBytes` | GROMACS XTC |
//! | [`lammps`] | `LammpsDataStream`, `LammpsDumpStream`, `writeLammpsDataStr`, `writeLammpsDumpStr`; `readLammpsLogStr`, `isLammpsLog` ([`lammps::log`]) | LAMMPS data, dump, run log |
//! | [`amber`] | `AmberInpcrdReader`, `AmberAcReader` | AMBER inpcrd / restrt, Antechamber AC |
//! | `smiles` | `readSmilesStr`, `SmilesIr.parse` | SMILES strings (`smiles` feature) |
//! | [`mrec`] | `MrecReader`, `readMrecFrame`, `sectionNames` | `*.mrec` scientific records (Zarr V3) |
//! | [`cube`] | `CubeReader`, `writeCubeStr` | Gaussian Cube |
//! | [`xsf`] | `XsfReader`, `writeXsfStr` | XCrySDen XSF |
//! | [`stl`] | `readStlBytes` | STL surface meshes (ASCII or binary) — produces a `TriMesh`, not a `Frame` |
//! | `frame_encoding` | `readMsgpackFrameBytes`, `writeMsgpackFrameBytes`, `readJsonFrameStr`, `writeJsonFrameStr` | `molrs::stream` wire encodings (`stream` feature) |
//! | [`frame_index`] | `FrameOffset` | Shared by every `*Stream`: the chunk-fed `FrameIndexBuilder` protocol |
//!
//! No reader takes a file handle, since WASM has no filesystem access: a
//! whole-content reader takes the file's text (or bytes), a stream takes
//! chunks the host copies into its input buffer. No export picks a format
//! from a string.

/// Write `frame` (a [`Frame`](crate::core::frame::Frame)) through a
/// `molrs::io` writer class into memory, returning the bytes.
macro_rules! write_bytes {
    ($writer:ident, $frame:expr, $what:literal) => {
        $frame.with_frame(|rs_frame| {
            use molrs::io::writer::{FrameWriter, Writer};
            let mut buf: Vec<u8> = Vec::new();
            <$writer<_> as Writer>::new(&mut buf)
                .write(rs_frame)
                .map_err(|e| JsValue::from_str(&format!("{} writing error: {e}", $what)))?;
            Ok(buf)
        })
    };
}

#[macro_use]
pub mod frame_index;

pub mod amber;
pub mod cif;
pub mod cube;
pub mod dcd;
#[cfg(feature = "stream")]
pub mod frame_encoding;
pub mod gro;
pub mod lammps;
pub mod mol2;
pub mod mrec;
pub mod pdb;
pub mod sdf;
#[cfg(feature = "smiles")]
pub mod smiles;
pub mod stl;
pub mod trr;
pub mod vasp;
pub mod xsf;
pub mod xtc;
pub mod xyz;

pub use amber::*;
pub use cif::*;
pub use cube::*;
pub use dcd::*;
#[cfg(feature = "stream")]
pub use frame_encoding::*;
pub use frame_index::*;
pub use gro::*;
pub use lammps::*;
pub use mol2::*;
pub use mrec::*;
pub use pdb::*;
pub use sdf::*;
#[cfg(feature = "smiles")]
pub use smiles::*;
pub use stl::*;
pub use trr::*;
pub use vasp::*;
pub use xsf::*;
pub use xtc::*;
pub use xyz::*;

use wasm_bindgen::prelude::*;

/// The text a reader was built from (it was a JS string, so it is UTF-8).
fn utf8_text(content: &[u8]) -> Result<&str, JsValue> {
    std::str::from_utf8(content).map_err(|e| JsValue::from_str(&format!("UTF-8 error: {e}")))
}

/// Text from a writer's output bytes.
fn utf8_string(bytes: Vec<u8>) -> Result<String, JsValue> {
    String::from_utf8(bytes).map_err(|e| JsValue::from_str(&format!("UTF-8 conversion error: {e}")))
}

#[cfg(test)]
mod test_fixtures {
    use crate::core::frame::Frame;
    use crate::core::nd_array::JsFloatArray;
    use wasm_bindgen::JsCast;
    use wasm_bindgen::prelude::*;

    /// Owned `f64` column `key` of `block`.
    pub(super) fn float_col(block: &crate::core::Block, key: &str) -> js_sys::Float64Array {
        JsValue::from(block.copy(key, None).expect(key)).unchecked_into()
    }

    /// A two-atom (H, O) frame with `x`/`y`/`z` and `element` columns.
    pub(super) fn two_atom_frame() -> Frame {
        let frame = Frame::new();
        let mut atoms = frame.create_block("atoms").expect("atoms block");
        for (key, values) in [("x", [0.0, 1.0]), ("y", [0.0, 0.0]), ("z", [0.0, 0.5])] {
            let col = JsFloatArray::from(&values[..]);
            atoms
                .set(key, JsValue::from(col).unchecked_into(), None)
                .expect(key);
        }
        let elements = js_sys::Array::new();
        elements.push(&JsValue::from_str("H"));
        elements.push(&JsValue::from_str("O"));
        atoms
            .set("element", JsValue::from(elements).unchecked_into(), None)
            .expect("element");
        frame
    }
}
