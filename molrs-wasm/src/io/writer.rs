//! Molecular file writers for the WASM API: one export per format, as
//! `molrs::io` has one door per format — no export picks a format from a
//! string.
//!
//! | JS function | Output | Format |
//! |---|---|---|
//! | `writeXyzStr` | `string` | XYZ / Extended XYZ |
//! | `writePdbStr` | `string` | Protein Data Bank |
//! | `writeCifStr` | `string` | Crystallographic Information File |
//! | `writeCubeStr` | `string` | Gaussian Cube (needs a `"grid"` block) |
//! | `writeGroStr` | `string` | GROMACS GRO (Å → nm on write) |
//! | `writeMol2Str` | `string` | Tripos MOL2 |
//! | `writeVaspPoscarStr` | `string` | VASP POSCAR (needs a `box`) |
//! | `writeXsfStr` | `string` | XCrySDen XSF |
//! | `writeLammpsDataStr` | `string` | LAMMPS data file |
//! | `writeLammpsDumpStr` | `string` | LAMMPS dump (one snapshot) |
//! | `writeDcdBytes` | `Uint8Array` | DCD trajectory |
//! | `writeTrrBytes` | `Uint8Array` | GROMACS TRR (Å → nm on write) |
//! | `writeXtcBytes` | `Uint8Array` | GROMACS XTC (Å → nm on write) |
//! | `writeMsgpackFrameBytes` | `Uint8Array` | `molrs::stream` MessagePack wire encoding (`stream` feature) |
//! | `writeJsonFrameStr` | `string` | `molrs::stream` JSON wire encoding (`stream` feature) |
//!
//! molrs has no SDF or CHGCAR writer, so those remain read-only.

use crate::core::frame::Frame;
use molrs::io::cif::CifWriter;
use molrs::io::dcd::DcdWriter;
use molrs::io::gro::GroWriter;
use molrs::io::lammps::{LammpsDataWriter, LammpsDumpWriter};
use molrs::io::mol2::Mol2Writer;
use molrs::io::pdb::PdbWriter;
use molrs::io::trr::TrrWriter;
use molrs::io::writer::{FrameWriter, Writer};
use molrs::io::xtc::XtcWriter;
use molrs::io::xyz::XyzWriter;
use std::io::Cursor;
use wasm_bindgen::prelude::*;

/// Write `frame` through a `molrs::io` writer class into memory.
macro_rules! write_bytes {
    ($writer:ident, $frame:expr, $what:literal) => {
        $frame.with_frame(|rs_frame| {
            let mut buf: Vec<u8> = Vec::new();
            <$writer<_> as Writer>::new(&mut buf)
                .write(rs_frame)
                .map_err(|e| JsValue::from_str(&format!("{} writing error: {e}", $what)))?;
            Ok(buf)
        })
    };
}

fn utf8(bytes: Vec<u8>) -> Result<String, JsValue> {
    String::from_utf8(bytes).map_err(|e| JsValue::from_str(&format!("UTF-8 conversion error: {e}")))
}

/// Write `frame` as (extended) XYZ text.
#[wasm_bindgen(js_name = writeXyzStr)]
pub fn write_xyz_str(frame: &Frame) -> Result<String, JsValue> {
    utf8(write_bytes!(XyzWriter, frame, "XYZ")?)
}

/// Write `frame` as PDB text.
#[wasm_bindgen(js_name = writePdbStr)]
pub fn write_pdb_str(frame: &Frame) -> Result<String, JsValue> {
    utf8(write_bytes!(PdbWriter, frame, "PDB")?)
}

/// Write `frame` as CIF text.
#[wasm_bindgen(js_name = writeCifStr)]
pub fn write_cif_str(frame: &Frame) -> Result<String, JsValue> {
    utf8(write_bytes!(CifWriter, frame, "CIF")?)
}

/// Write `frame` as GROMACS GRO text (Å → nm).
#[wasm_bindgen(js_name = writeGroStr)]
pub fn write_gro_str(frame: &Frame) -> Result<String, JsValue> {
    utf8(write_bytes!(GroWriter, frame, "GRO")?)
}

/// Write `frame` as Tripos MOL2 text.
#[wasm_bindgen(js_name = writeMol2Str)]
pub fn write_mol2_str(frame: &Frame) -> Result<String, JsValue> {
    utf8(write_bytes!(Mol2Writer, frame, "MOL2")?)
}

/// Write `frame` as a LAMMPS data file.
#[wasm_bindgen(js_name = writeLammpsDataStr)]
pub fn write_lammps_data_str(frame: &Frame) -> Result<String, JsValue> {
    utf8(write_bytes!(LammpsDataWriter, frame, "LAMMPS data")?)
}

/// Write `frame` as one LAMMPS dump snapshot.
#[wasm_bindgen(js_name = writeLammpsDumpStr)]
pub fn write_lammps_dump_str(frame: &Frame) -> Result<String, JsValue> {
    utf8(write_bytes!(LammpsDumpWriter, frame, "LAMMPS dump")?)
}

/// Write `frame` as Gaussian cube text (needs a `"grid"` block).
#[wasm_bindgen(js_name = writeCubeStr)]
pub fn write_cube_str(frame: &Frame) -> Result<String, JsValue> {
    frame.with_frame(|f| {
        molrs::io::write_cube_str(f)
            .map_err(|e| JsValue::from_str(&format!("Cube writing error: {e}")))
    })
}

/// Write `frame` as VASP POSCAR text (needs a box).
#[wasm_bindgen(js_name = writeVaspPoscarStr)]
pub fn write_vasp_poscar_str(frame: &Frame) -> Result<String, JsValue> {
    frame.with_frame(|f| {
        molrs::io::write_vasp_poscar_str(f)
            .map_err(|e| JsValue::from_str(&format!("POSCAR writing error: {e}")))
    })
}

/// Write `frame` as XCrySDen XSF text.
#[wasm_bindgen(js_name = writeXsfStr)]
pub fn write_xsf_str(frame: &Frame) -> Result<String, JsValue> {
    frame.with_frame(|f| {
        molrs::io::write_xsf_str(f)
            .map_err(|e| JsValue::from_str(&format!("XSF writing error: {e}")))
    })
}

/// Write `frame` as a one-frame DCD trajectory.
#[wasm_bindgen(js_name = writeDcdBytes)]
pub fn write_dcd_bytes(frame: &Frame) -> Result<Vec<u8>, JsValue> {
    frame.with_frame(|rs_frame| {
        let mut buf: Vec<u8> = Vec::new();
        // DcdWriter needs Write + Seek (it patches NSET after each frame);
        // Cursor<&mut Vec<u8>> satisfies both and leaves the bytes in `buf`
        // once the writer drops.
        DcdWriter::new(Cursor::new(&mut buf))
            .write(rs_frame)
            .map_err(|e| JsValue::from_str(&format!("DCD writing error: {e}")))?;
        Ok(buf)
    })
}

/// Write `frame` as a one-frame GROMACS TRR trajectory (Å → nm).
#[wasm_bindgen(js_name = writeTrrBytes)]
pub fn write_trr_bytes(frame: &Frame) -> Result<Vec<u8>, JsValue> {
    write_bytes!(TrrWriter, frame, "TRR")
}

/// Write `frame` as a one-frame GROMACS XTC trajectory (Å → nm).
#[wasm_bindgen(js_name = writeXtcBytes)]
pub fn write_xtc_bytes(frame: &Frame) -> Result<Vec<u8>, JsValue> {
    write_bytes!(XtcWriter, frame, "XTC")
}

/// Encode `frame` in the `molrs::stream` MessagePack wire encoding — what a
/// publisher puts on the socket. The inverse of `readMsgpackFrameBytes`.
#[cfg(feature = "stream")]
#[wasm_bindgen(js_name = writeMsgpackFrameBytes)]
pub fn write_msgpack_frame_bytes(frame: &Frame) -> Result<Vec<u8>, JsValue> {
    frame.with_frame(|f| {
        molrs::stream::write_msgpack_frame_bytes(f).map_err(|e| JsValue::from_str(&e.to_string()))
    })
}

/// Encode `frame` in the `molrs::stream` JSON wire encoding. The inverse of
/// `readJsonFrameStr`.
#[cfg(feature = "stream")]
#[wasm_bindgen(js_name = writeJsonFrameStr)]
pub fn write_json_frame_str(frame: &Frame) -> Result<String, JsValue> {
    frame.with_frame(|f| {
        molrs::stream::write_json_frame_str(f).map_err(|e| JsValue::from_str(&e.to_string()))
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::frame::Frame;
    use crate::core::types::JsFloatArray;
    use wasm_bindgen_test::*;

    #[wasm_bindgen_test]
    fn test_write_frame_formats() {
        use js_sys::Array as JsArray;

        let frame = Frame::new();
        let mut atoms = frame.create_block("atoms").expect("atoms block");

        let x = JsFloatArray::from(&[0.0, 1.0][..]);
        let y = JsFloatArray::from(&[0.0, 0.0][..]);
        let z = JsFloatArray::from(&[0.0, 0.5][..]);

        use wasm_bindgen::JsCast;
        atoms
            .set("x", JsValue::from(x).unchecked_into(), None)
            .expect("x");
        atoms
            .set("y", JsValue::from(y).unchecked_into(), None)
            .expect("y");
        atoms
            .set("z", JsValue::from(z).unchecked_into(), None)
            .expect("z");

        let elements = JsArray::new();
        elements.push(&JsValue::from_str("H"));
        elements.push(&JsValue::from_str("O"));
        atoms
            .set("element", JsValue::from(elements).unchecked_into(), None)
            .expect("element");

        let xyz_output = write_xyz_str(&frame).expect("xyz output");
        assert!(xyz_output.lines().next().unwrap_or("").starts_with('2'));

        let pdb_output = write_pdb_str(&frame).expect("pdb output");
        assert!(pdb_output.contains("ATOM"));
    }
}
