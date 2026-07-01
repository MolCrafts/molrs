//! Molecular file writers for the WASM API.
//!
//! Provides a single free function [`writeFrame`](write_frame_export)
//! that serializes a [`Frame`] to a string in a supported format.
//!
//! # Supported output formats
//!
//! Text formats go through [`writeFrame`](write_frame_export) (returns a
//! `String`); binary trajectory formats go through
//! [`writeFrameBytes`](write_frame_bytes_export) (returns a `Uint8Array`).
//! Every format molrs can write is covered (molrs has no SDF or CHGCAR
//! writer, so those remain read-only).
//!
//! | Format string | Kind | Output |
//! |---------------|------|--------|
//! | `"xyz"` | text | XYZ / Extended XYZ |
//! | `"pdb"` | text | Protein Data Bank |
//! | `"cif"` | text | Crystallographic Information File |
//! | `"cube"` | text | Gaussian Cube (needs a `"grid"` block) |
//! | `"gro"` | text | GROMACS GRO (Å → nm on write) |
//! | `"mol2"` | text | Tripos MOL2 |
//! | `"poscar"` | text | VASP POSCAR (needs a `simbox`) |
//! | `"lammps-data"` / `"lammps"` | text | LAMMPS data file |
//! | `"lammps-dump"` / `"lammpstrj"` | text | LAMMPS dump |
//! | `"dcd"` | binary | DCD trajectory |
//! | `"trr"` | binary | GROMACS TRR (Å → nm on write) |
//! | `"xtc"` | binary | GROMACS XTC (Å → nm on write) |

use super::reader::{NM_TO_ANGSTROM, scale_frame_lengths};
use crate::core::frame::Frame;
use molrs::io::data::cif::write_cif_frame;
use molrs::io::data::cube::write_cube_to_writer;
use molrs::io::data::gro::write_gro_frame;
use molrs::io::data::lammps_data::LAMMPSDataWriter;
use molrs::io::data::mol2::write_mol2_frame;
use molrs::io::data::pdb::PDBWriter;
use molrs::io::data::poscar::write_poscar_to_writer;
use molrs::io::data::xyz::XYZFrameWriter;
use molrs::io::trajectory::dcd::DcdWriter;
use molrs::io::trajectory::lammps_dump::LAMMPSDumpWriter;
use molrs::io::trajectory::trr::TrrWriter;
use molrs::io::trajectory::xtc::XtcWriter;
use molrs::io::writer::{FrameWriter, Writer};
use std::io::Cursor;
use wasm_bindgen::prelude::*;

/// Serialize a [`Frame`] to a string in the specified format.
///
/// The frame must have an `"atoms"` block with at least an element/name
/// string column and `x`, `y`, `z` float columns (coordinates in
/// angstrom).
///
/// # Arguments
///
/// * `frame` - The [`Frame`] to write
/// * `format` - Output format string: `"xyz"` or `"pdb"`
///   (case-insensitive)
///
/// # Returns
///
/// The formatted file content as a string.
///
/// # Errors
///
/// Throws a `JsValue` string if:
/// - The format is not recognized
/// - The frame is missing required columns
/// - The writer encounters an error
///
/// # Example (JavaScript)
///
/// ```js
/// const xyzStr = writeFrame(frame, "xyz");
/// console.log(xyzStr);
/// // 2
/// //
/// // H  0.000000  0.000000  0.000000
/// // O  1.000000  0.000000  0.500000
///
/// const pdbStr = writeFrame(frame, "pdb");
/// // download or display the PDB string
/// ```
#[wasm_bindgen(js_name = writeFrame)]
pub fn write_frame_export(frame: &Frame, format: &str) -> Result<String, JsValue> {
    let buffer = frame.with_frame(|rs_frame| {
        let mut buf: Vec<u8> = Vec::new();
        match format.to_lowercase().as_str() {
            "xyz" => {
                let mut writer = <XYZFrameWriter<_> as Writer>::new(&mut buf);
                writer
                    .write_frame(rs_frame)
                    .map_err(|e| JsValue::from_str(&format!("XYZ writing error: {}", e)))?;
            }
            "pdb" => {
                let mut writer = <PDBWriter<_> as Writer>::new(&mut buf);
                writer
                    .write_frame(rs_frame)
                    .map_err(|e| JsValue::from_str(&format!("PDB writing error: {}", e)))?;
            }
            "lammps-data" | "lammps" => {
                let mut writer = <LAMMPSDataWriter<_> as Writer>::new(&mut buf);
                writer
                    .write_frame(rs_frame)
                    .map_err(|e| JsValue::from_str(&format!("LAMMPS data writing error: {}", e)))?;
            }
            "lammps-dump" | "lammpstrj" => {
                let mut writer = <LAMMPSDumpWriter<_> as Writer>::new(&mut buf);
                writer
                    .write_frame(rs_frame)
                    .map_err(|e| JsValue::from_str(&format!("LAMMPS dump writing error: {}", e)))?;
            }
            "cif" => {
                write_cif_frame(&mut buf, rs_frame)
                    .map_err(|e| JsValue::from_str(&format!("CIF writing error: {}", e)))?;
            }
            "cube" => {
                write_cube_to_writer(&mut buf, rs_frame)
                    .map_err(|e| JsValue::from_str(&format!("Cube writing error: {:?}", e)))?;
            }
            "gro" => {
                // GRO is GROMACS-native nm; scale Å → nm on a throwaway clone.
                let mut nm = rs_frame.clone();
                scale_frame_lengths(&mut nm, 1.0 / NM_TO_ANGSTROM)?;
                write_gro_frame(&mut buf, &nm)
                    .map_err(|e| JsValue::from_str(&format!("GRO writing error: {}", e)))?;
            }
            "mol2" => {
                write_mol2_frame(&mut buf, rs_frame)
                    .map_err(|e| JsValue::from_str(&format!("MOL2 writing error: {}", e)))?;
            }
            "poscar" => {
                write_poscar_to_writer(&mut buf, rs_frame)
                    .map_err(|e| JsValue::from_str(&format!("POSCAR writing error: {}", e)))?;
            }
            _ => {
                return Err(JsValue::from_str(&format!(
                    "unsupported format: {}",
                    format
                )));
            }
        }
        Ok(buf)
    })?;

    String::from_utf8(buffer)
        .map_err(|e| JsValue::from_str(&format!("UTF-8 conversion error: {}", e)))
}

/// Serialize a [`Frame`] to bytes in a binary trajectory format.
///
/// Mirrors [`write_frame_export`] for the formats whose output is not valid
/// UTF-8: `"dcd"`, `"trr"`, `"xtc"` (case-insensitive). Returns a
/// `Uint8Array` to JavaScript. The GROMACS formats (`trr`/`xtc`) are written
/// in nm — coordinates and box are scaled Å → nm on a throwaway clone first.
///
/// # Errors
///
/// Throws a `JsValue` string if the format is not a known binary format, the
/// frame is missing required columns, or the writer encounters an error.
#[wasm_bindgen(js_name = writeFrameBytes)]
pub fn write_frame_bytes_export(frame: &Frame, format: &str) -> Result<Vec<u8>, JsValue> {
    frame.with_frame(|rs_frame| {
        let mut buf: Vec<u8> = Vec::new();
        match format.to_lowercase().as_str() {
            "dcd" => {
                // DcdWriter needs Write + Seek (it patches NSET after each
                // frame); Cursor<&mut Vec<u8>> satisfies both and leaves the
                // bytes in `buf` once the writer drops.
                let mut writer = DcdWriter::new(Cursor::new(&mut buf));
                writer
                    .write_frame(rs_frame)
                    .map_err(|e| JsValue::from_str(&format!("DCD writing error: {}", e)))?;
            }
            "trr" => {
                let mut nm = rs_frame.clone();
                scale_frame_lengths(&mut nm, 1.0 / NM_TO_ANGSTROM)?;
                let mut writer = TrrWriter::new(&mut buf);
                writer
                    .write_frame(&nm)
                    .map_err(|e| JsValue::from_str(&format!("TRR writing error: {}", e)))?;
            }
            "xtc" => {
                let mut nm = rs_frame.clone();
                scale_frame_lengths(&mut nm, 1.0 / NM_TO_ANGSTROM)?;
                let mut writer = XtcWriter::new(&mut buf);
                writer
                    .write_frame(&nm)
                    .map_err(|e| JsValue::from_str(&format!("XTC writing error: {}", e)))?;
            }
            _ => {
                return Err(JsValue::from_str(&format!(
                    "unsupported binary format: {}",
                    format
                )));
            }
        }
        Ok(buf)
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

        atoms.set_col_f("x", &x, None).expect("x");
        atoms.set_col_f("y", &y, None).expect("y");
        atoms.set_col_f("z", &z, None).expect("z");

        let elements = JsArray::new();
        elements.push(&JsValue::from_str("H"));
        elements.push(&JsValue::from_str("O"));
        atoms.set_col_str("element", elements).expect("element");

        let xyz_output = write_frame_export(&frame, "xyz").expect("xyz output");
        assert!(xyz_output.lines().next().unwrap_or("").starts_with('2'));

        let pdb_output = write_frame_export(&frame, "pdb").expect("pdb output");
        assert!(pdb_output.contains("ATOM"));
    }
}
