//! Crystallographic Information File for the WASM API — the face of `molrs::io::cif`.
//!
//! | JS | molrs |
//! |----|-------|
//! | `CifReader` | `CifReader` (whole-content; the one CIF reader) |
//! | `writeCifStr` | `CifWriter` |

use molrs::io::cif::{CifReader as RsCifReader, CifWriter};
use molrs::io::reader::Reader;
use std::io::Cursor;
use wasm_bindgen::prelude::*;

use crate::core::frame::Frame;

/// Crystallographic Information File (CIF / mmCIF) reader.
///
/// Each `data_*` block in the file becomes one [`Frame`]. Most CIF files
/// contain a single block (one structure), but multi-block files (e.g.
/// polymorphs of the same compound) are also supported and are exposed
/// as a multi-frame sequence. The unit cell parameters
/// (`_cell_length_a`, `_b`, `_c`, `_cell_angle_alpha`, `_beta`,
/// `_gamma`) are converted to a 3x3 h-matrix on the Rust side and
/// surface on the JS side as `frame.simbox`.
///
/// Produces a [`Frame`] with an `"atoms"` block containing
/// `element` (string), `x`, `y`, `z` (F, angstrom in Cartesian
/// coordinates) and (when present in the file) `label`, `occupancy`,
/// `bfactor` columns.
///
/// CIF parsing reads the entire file on each `read(step)` call --
/// random access is therefore O(file_size), but typical CIF files are
/// small (< 1 MB) and the molvis lazy trajectory caches frames at the
/// JS level, so this is rarely a bottleneck.
///
/// # Example (JavaScript)
///
/// ```js
/// const content = await file.text();
/// const reader = new CifReader(content);
/// const frame  = reader.read(0);
/// const atoms  = frame.get("atoms");
/// const box    = frame.simbox;        // populated from the unit cell
/// ```
#[wasm_bindgen(js_name = CifReader)]
pub struct CifReader {
    content: Vec<u8>,
    cached_len: Option<usize>,
}

#[wasm_bindgen(js_class = CifReader)]
impl CifReader {
    /// Create a new CIF reader from a string containing the file content.
    ///
    /// # Arguments
    ///
    /// * `content` - The full text content of a CIF file
    ///
    /// # Example (JavaScript)
    ///
    /// ```js
    /// const reader = new CifReader(cifString);
    /// ```
    #[wasm_bindgen(constructor)]
    pub fn new(content: &str) -> CifReader {
        CifReader {
            content: content.as_bytes().to_vec(),
            cached_len: None,
        }
    }

    /// Read the frame at the given block index.
    ///
    /// # Arguments
    ///
    /// * `step` - 0-based index of the `data_*` block to return
    ///
    /// # Returns
    ///
    /// A [`Frame`] for the requested block, or `undefined` when
    /// `step >= len()`.
    ///
    /// # Errors
    ///
    /// Throws a `JsValue` string on parse errors.
    #[wasm_bindgen]
    pub fn read(&mut self, step: usize) -> Result<Option<Frame>, JsValue> {
        let mut reader = RsCifReader::new(Cursor::new(self.content.as_slice()));
        let frames = molrs::io::reader::collect_frames(&mut reader)
            .map_err(|e| JsValue::from_str(&format!("CIF read error: {}", e)))?;
        if step >= frames.len() {
            return Ok(None);
        }
        let rs_frame = frames.into_iter().nth(step).expect("bounds checked");
        Ok(Some(Frame::from_rs(rs_frame)?))
    }

    /// Return the number of `data_*` blocks in the file.
    ///
    /// # Errors
    ///
    /// Throws a `JsValue` string on parse errors.
    #[wasm_bindgen]
    pub fn len(&mut self) -> Result<usize, JsValue> {
        if let Some(n) = self.cached_len {
            return Ok(n);
        }
        let mut reader = RsCifReader::new(Cursor::new(self.content.as_slice()));
        let n = molrs::io::reader::collect_frames(&mut reader)
            .map_err(|e| JsValue::from_str(&format!("CIF len error: {}", e)))?
            .len();
        self.cached_len = Some(n);
        Ok(n)
    }

    /// Check whether the file contains no valid blocks.
    ///
    /// # Errors
    ///
    /// Throws a `JsValue` string on parse errors.
    #[wasm_bindgen(js_name = isEmpty)]
    pub fn is_empty(&mut self) -> Result<bool, JsValue> {
        Ok(self.len()? == 0)
    }
}

/// Write `frame` as CIF text.
#[wasm_bindgen(js_name = writeCifStr)]
pub fn write_cif_str(frame: &Frame) -> Result<String, JsValue> {
    super::utf8_string(write_bytes!(CifWriter, frame, "CIF")?)
}
