//! Molecular file writers for the WASM API.
//!
//! Provides a single free function [`writeFrame`](write_frame_export)
//! that serializes a [`Frame`] to a string in a supported format.
//!
//! # Supported output formats
//!
//! | Format string | Description | Required columns in `"atoms"` block |
//! |---------------|-------------|-------------------------------------|
//! | `"xyz"` | XYZ / Extended XYZ | `element` (string), `x`, `y`, `z` (F) |
//! | `"pdb"` | Protein Data Bank | `element` or `name` (string), `x`, `y`, `z` (F) |
//! | `"lammps-data"` / `"lammps"` | LAMMPS data file | `id`, `type` (i32), `x`, `y`, `z` (F) |
//! | `"lammps-dump"` / `"lammpstrj"` | LAMMPS dump | columns from atoms block |

use crate::core::frame::Frame;
use molrs::io::lammps_data::LAMMPSDataWriter;
use molrs::io::lammps_dump::LAMMPSDumpWriter;
use molrs::io::pdb::PDBWriter;
use molrs::io::writer::{FrameWriter, Writer};
use molrs::io::xyz::XYZFrameWriter;
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
    // Get the core frame data by cloning all blocks
    let rs_frame = frame_to_core(frame)?;

    let mut buffer: Vec<u8> = Vec::new();
    match format.to_lowercase().as_str() {
        "xyz" => {
            let mut writer = <XYZFrameWriter<_> as Writer>::new(&mut buffer);
            writer
                .write_frame(&rs_frame)
                .map_err(|e| JsValue::from_str(&format!("XYZ writing error: {}", e)))?;
        }
        "pdb" => {
            let mut writer = <PDBWriter<_> as Writer>::new(&mut buffer);
            writer
                .write_frame(&rs_frame)
                .map_err(|e| JsValue::from_str(&format!("PDB writing error: {}", e)))?;
        }
        "lammps-data" | "lammps" => {
            let mut writer = <LAMMPSDataWriter<_> as Writer>::new(&mut buffer);
            writer
                .write_frame(&rs_frame)
                .map_err(|e| JsValue::from_str(&format!("LAMMPS data writing error: {}", e)))?;
        }
        "lammps-dump" | "lammpstrj" => {
            let mut writer = <LAMMPSDumpWriter<_> as Writer>::new(&mut buffer);
            writer
                .write_frame(&rs_frame)
                .map_err(|e| JsValue::from_str(&format!("LAMMPS dump writing error: {}", e)))?;
        }
        _ => {
            return Err(JsValue::from_str(&format!(
                "unsupported format: {}",
                format
            )));
        }
    }

    String::from_utf8(buffer)
        .map_err(|e| JsValue::from_str(&format!("UTF-8 conversion error: {}", e)))
}

/// Helper to convert FFI Frame to core Frame
fn frame_to_core(frame: &Frame) -> Result<molrs::frame::Frame, JsValue> {
    frame.clone_core_frame()
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
