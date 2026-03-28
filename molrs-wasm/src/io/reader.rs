//! Molecular file readers for the WASM API.
//!
//! Each reader takes a file's text content as a string (since WASM has
//! no filesystem access) and produces [`Frame`] objects. All readers
//! implement a uniform two-method interface:
//!
//! - `read(step)` -- read a specific frame by index
//! - `len()` -- return the number of available frames
//!
//! # Supported formats
//!
//! | JS class | Format | Multi-frame? | Produces |
//! |----------|--------|-------------|----------|
//! | `XYZReader` | XYZ / ExtXYZ | Yes | `"atoms"` block with `element`, `x`, `y`, `z` |
//! | `PDBReader` | Protein Data Bank | No (step=0 only) | `"atoms"` block with `name`, `resname`, `x`, `y`, `z`, etc. |
//! | `LAMMPSReader` | LAMMPS data file | No (step=0 only) | `"atoms"` block + `"bonds"` block + simbox |
//! | `LAMMPSDumpReader` | LAMMPS dump trajectory | Yes | `"atoms"` block with columns from dump header |

use crate::core::frame::Frame;
use molrs::io::lammps_data::LAMMPSDataReader;
use molrs::io::lammps_dump::LAMMPSDumpReader;
use molrs::io::pdb::PDBReader;
use molrs::io::reader::{FrameReader, TrajReader};
use molrs::io::xyz::XYZReader;
use std::io::Cursor;
use wasm_bindgen::prelude::*;

/// XYZ / Extended XYZ file reader.
///
/// Supports multi-frame trajectory files. Each frame produces a
/// [`Frame`] with an `"atoms"` block containing `element` (string)
/// and `x`, `y`, `z` (F, coordinates in angstrom) columns.
///
/// # Example (JavaScript)
///
/// ```js
/// const content = await file.text(); // read file in browser
/// const reader = new XYZReader(content);
/// console.log(reader.len()); // number of frames
///
/// const frame = reader.read(0); // first frame
/// const atoms = frame.getBlock("atoms");
/// const x = atoms.copyColF("x");
/// ```
#[wasm_bindgen(js_name = XYZReader)]
pub struct XyzReader {
    inner: XYZReader<Cursor<Vec<u8>>>,
}

#[wasm_bindgen(js_class = XYZReader)]
impl XyzReader {
    /// Create a new XYZ reader from a string containing the file content.
    ///
    /// # Arguments
    ///
    /// * `content` - The full text content of an XYZ or Extended XYZ file
    ///
    /// # Example (JavaScript)
    ///
    /// ```js
    /// const reader = new XYZReader(fileContent);
    /// ```
    #[wasm_bindgen(constructor)]
    pub fn new(content: &str) -> XyzReader {
        let bytes = content.as_bytes().to_vec();
        XyzReader {
            inner: XYZReader::new(Cursor::new(bytes)),
        }
    }

    /// Read a frame at the given step index.
    ///
    /// # Arguments
    ///
    /// * `step` - Zero-based frame index
    ///
    /// # Returns
    ///
    /// A [`Frame`] if the step exists, or `undefined` if `step` is
    /// out of range.
    ///
    /// # Errors
    ///
    /// Throws a `JsValue` string on parse errors.
    ///
    /// # Example (JavaScript)
    ///
    /// ```js
    /// const frame = reader.read(0);
    /// if (frame) {
    ///   const atoms = frame.getBlock("atoms");
    /// }
    /// ```
    #[wasm_bindgen]
    pub fn read(&mut self, step: usize) -> Result<Option<Frame>, JsValue> {
        let rs_frame = self
            .inner
            .read_step(step)
            .map_err(|e| JsValue::from_str(&format!("XYZ read error: {}", e)))?;

        match rs_frame {
            Some(frame_data) => Ok(Some(Frame::from_rs_frame(frame_data)?)),
            None => Ok(None),
        }
    }

    /// Return the number of frames in the file.
    ///
    /// # Returns
    ///
    /// The total frame count.
    ///
    /// # Errors
    ///
    /// Throws a `JsValue` string if the file cannot be scanned.
    ///
    /// # Example (JavaScript)
    ///
    /// ```js
    /// console.log(reader.len()); // e.g., 100
    /// ```
    #[wasm_bindgen]
    pub fn len(&mut self) -> Result<usize, JsValue> {
        self.inner
            .len()
            .map_err(|e| JsValue::from_str(&format!("XYZ len error: {}", e)))
    }

    /// Check whether the file contains no frames.
    ///
    /// # Errors
    ///
    /// Throws a `JsValue` string if the file cannot be scanned.
    #[wasm_bindgen(js_name = isEmpty)]
    pub fn is_empty(&mut self) -> Result<bool, JsValue> {
        Ok(self.len()? == 0)
    }
}

/// Protein Data Bank (PDB) file reader.
///
/// PDB files contain a single molecular structure. The reader produces
/// a [`Frame`] with an `"atoms"` block containing columns such as
/// `name` (string), `resname` (string), `x`, `y`, `z` (F, angstrom),
/// and optionally `occupancy` and `bfactor` (F).
///
/// # Example (JavaScript)
///
/// ```js
/// const reader = new PDBReader(pdbContent);
/// const frame = reader.read(0);
/// const atoms = frame.getBlock("atoms");
/// const names = atoms.copyColStr("name"); // ["CA", "CB", ...]
/// const x = atoms.copyColF("x");
/// ```
#[wasm_bindgen(js_name = PDBReader)]
pub struct PdbReader {
    content: Vec<u8>,
    cached_len: Option<usize>,
}

#[wasm_bindgen(js_class = PDBReader)]
impl PdbReader {
    /// Create a new PDB reader from a string containing the file content.
    ///
    /// # Arguments
    ///
    /// * `content` - The full text content of a PDB file
    ///
    /// # Example (JavaScript)
    ///
    /// ```js
    /// const reader = new PDBReader(pdbString);
    /// ```
    #[wasm_bindgen(constructor)]
    pub fn new(content: &str) -> PdbReader {
        PdbReader {
            content: content.as_bytes().to_vec(),
            cached_len: None,
        }
    }

    /// Read the frame at the given step index.
    ///
    /// PDB files contain a single structure, so only `step = 0` is
    /// valid. Passing any other step returns `undefined`.
    ///
    /// # Arguments
    ///
    /// * `step` - Frame index (must be `0` for PDB files)
    ///
    /// # Returns
    ///
    /// A [`Frame`] if `step == 0` and the file is valid, or `undefined`.
    ///
    /// # Errors
    ///
    /// Throws a `JsValue` string on parse errors.
    ///
    /// # Example (JavaScript)
    ///
    /// ```js
    /// const frame = reader.read(0);
    /// ```
    #[wasm_bindgen]
    pub fn read(&mut self, step: usize) -> Result<Option<Frame>, JsValue> {
        if step > 0 {
            return Ok(None);
        }

        let mut reader = PDBReader::new(Cursor::new(self.content.as_slice()));
        let rs_frame = reader
            .read_frame()
            .map_err(|e| JsValue::from_str(&format!("PDB read error: {}", e)))?;

        match rs_frame {
            Some(frame_data) => Ok(Some(Frame::from_rs_frame(frame_data)?)),
            None => Ok(None),
        }
    }

    /// Return the number of frames (always 0 or 1 for PDB files).
    ///
    /// # Errors
    ///
    /// Throws a `JsValue` string on parse errors.
    #[wasm_bindgen]
    pub fn len(&mut self) -> Result<usize, JsValue> {
        if let Some(n) = self.cached_len {
            return Ok(n);
        }
        let has_frame = self.read(0)?.is_some();
        let n = if has_frame { 1 } else { 0 };
        self.cached_len = Some(n);
        Ok(n)
    }

    /// Check whether the file contains no valid frames.
    ///
    /// # Errors
    ///
    /// Throws a `JsValue` string on parse errors.
    #[wasm_bindgen(js_name = isEmpty)]
    pub fn is_empty(&mut self) -> Result<bool, JsValue> {
        Ok(self.len()? == 0)
    }
}

/// LAMMPS data file reader.
///
/// Reads LAMMPS data files (the format written by `write_data`). The
/// reader produces a [`Frame`] containing:
///
/// - `"atoms"` block: `type` (i32), `x`, `y`, `z` (F, angstrom),
///   and optionally `charge` (F)
/// - `"bonds"` block (if present): `i`, `j` (u32), `type` (i32)
/// - Simulation box (`simbox`) with PBC
///
/// Only a single frame is supported (`step = 0`).
///
/// # Example (JavaScript)
///
/// ```js
/// const reader = new LAMMPSReader(dataFileContent);
/// const frame = reader.read(0);
/// const atoms = frame.getBlock("atoms");
/// const bonds = frame.getBlock("bonds");
/// const box   = frame.simbox;
/// ```
#[wasm_bindgen(js_name = LAMMPSReader)]
pub struct LammpsReader {
    inner: LAMMPSDataReader<Cursor<Vec<u8>>>,
    cached_len: Option<usize>,
}

#[wasm_bindgen(js_class = LAMMPSReader)]
impl LammpsReader {
    /// Create a new LAMMPS data file reader from string content.
    ///
    /// # Arguments
    ///
    /// * `content` - The full text content of a LAMMPS data file
    ///
    /// # Example (JavaScript)
    ///
    /// ```js
    /// const reader = new LAMMPSReader(dataFileString);
    /// ```
    #[wasm_bindgen(constructor)]
    pub fn new(content: &str) -> LammpsReader {
        let bytes = content.as_bytes().to_vec();
        LammpsReader {
            inner: LAMMPSDataReader::new(Cursor::new(bytes)),
            cached_len: None,
        }
    }

    /// Read the frame at the given step index.
    ///
    /// LAMMPS data files contain a single configuration, so only
    /// `step = 0` is valid. Passing any other step returns `undefined`.
    ///
    /// # Arguments
    ///
    /// * `step` - Frame index (must be `0`)
    ///
    /// # Returns
    ///
    /// A [`Frame`] with atoms, optional bonds, and simbox, or `undefined`.
    ///
    /// # Errors
    ///
    /// Throws a `JsValue` string on parse errors.
    ///
    /// # Example (JavaScript)
    ///
    /// ```js
    /// const frame = reader.read(0);
    /// ```
    #[wasm_bindgen]
    pub fn read(&mut self, step: usize) -> Result<Option<Frame>, JsValue> {
        if step > 0 {
            return Ok(None);
        }

        let rs_frame = self
            .inner
            .read_frame()
            .map_err(|e| JsValue::from_str(&format!("LAMMPS read error: {}", e)))?;

        match rs_frame {
            Some(frame_data) => Ok(Some(Frame::from_rs_frame(frame_data)?)),
            None => Ok(None),
        }
    }

    /// Return the number of frames (always 0 or 1 for LAMMPS data files).
    ///
    /// # Errors
    ///
    /// Throws a `JsValue` string on parse errors.
    #[wasm_bindgen]
    pub fn len(&mut self) -> Result<usize, JsValue> {
        if let Some(n) = self.cached_len {
            return Ok(n);
        }
        let has_frame = self.read(0)?.is_some();
        let n = if has_frame { 1 } else { 0 };
        self.cached_len = Some(n);
        Ok(n)
    }

    /// Check whether the file contains no valid frames.
    ///
    /// # Errors
    ///
    /// Throws a `JsValue` string on parse errors.
    #[wasm_bindgen(js_name = isEmpty)]
    pub fn is_empty(&mut self) -> Result<bool, JsValue> {
        Ok(self.len()? == 0)
    }
}

/// LAMMPS dump trajectory file reader.
///
/// Reads multi-frame LAMMPS dump files (the format produced by the
/// `dump` command). Each frame produces a [`Frame`] containing an
/// `"atoms"` block with columns matching the dump header (e.g.
/// `id`, `type`, `x`, `y`, `z`, `vx`, `vy`, `vz`).
///
/// # Example (JavaScript)
///
/// ```js
/// const reader = new LAMMPSDumpReader(dumpContent);
/// console.log(reader.len()); // number of timesteps
/// const frame = reader.read(0);
/// const atoms = frame.getBlock("atoms");
/// ```
#[wasm_bindgen(js_name = LAMMPSDumpReader)]
pub struct LammpsDumpReader {
    inner: LAMMPSDumpReader<Cursor<Vec<u8>>>,
}

#[wasm_bindgen(js_class = LAMMPSDumpReader)]
impl LammpsDumpReader {
    /// Create a new LAMMPS dump reader from string content.
    #[wasm_bindgen(constructor)]
    pub fn new(content: &str) -> LammpsDumpReader {
        let bytes = content.as_bytes().to_vec();
        LammpsDumpReader {
            inner: LAMMPSDumpReader::new(Cursor::new(bytes)),
        }
    }

    /// Read a frame at the given step index.
    #[wasm_bindgen]
    pub fn read(&mut self, step: usize) -> Result<Option<Frame>, JsValue> {
        let rs_frame = self
            .inner
            .read_step(step)
            .map_err(|e| JsValue::from_str(&format!("LAMMPS dump read error: {}", e)))?;

        match rs_frame {
            Some(frame_data) => Ok(Some(Frame::from_rs_frame(frame_data)?)),
            None => Ok(None),
        }
    }

    /// Return the number of frames in the dump file.
    #[wasm_bindgen]
    pub fn len(&mut self) -> Result<usize, JsValue> {
        self.inner
            .len()
            .map_err(|e| JsValue::from_str(&format!("LAMMPS dump len error: {}", e)))
    }

    /// Check whether the file contains no frames.
    #[wasm_bindgen(js_name = isEmpty)]
    pub fn is_empty(&mut self) -> Result<bool, JsValue> {
        Ok(self.len()? == 0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use wasm_bindgen_test::*;

    #[wasm_bindgen_test]
    fn test_pdb_reader() {
        let pdb_content = r#"ATOM      1  C   MOL     1       1.000   2.000   3.000  1.00  0.00           C
ATOM      2  N   MOL     1       4.000   5.000   6.000  1.00  0.00           N
END"#;

        let mut reader = PdbReader::new(pdb_content);

        let frame = reader.read(0).expect("read failed").expect("no frame");

        let block = frame.get_block("atoms").expect("no atoms block");

        let x = block.copy_col_f("x").expect("no x column");
        assert_eq!(x.length(), 2);
        assert_eq!(x.get_index(0), 1.0);
        assert_eq!(x.get_index(1), 4.0);
    }
}
