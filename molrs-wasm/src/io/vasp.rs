//! VASP for the WASM API — the face of `molrs::io::vasp`.
//!
//! | JS | molrs |
//! |----|-------|
//! | `VaspPoscarReader` | `read_vasp_poscar_str` (POSCAR / CONTCAR) |
//! | `VaspChgcarReader` | `read_vasp_chgcar_str` (CHGCAR / CHGDIF) |
//! | `writeVaspPoscarStr` | `write_vasp_poscar_str` (needs a `box`) |
//!
//! molrs has no CHGCAR writer, so CHGCAR is read-only.

use molrs::io::{read_vasp_chgcar_str, read_vasp_poscar_str};
use wasm_bindgen::prelude::*;

use crate::core::frame::Frame;

/// VASP POSCAR / CONTCAR structure reader.
///
/// POSCAR describes a single crystalline cell. Coordinates are returned as
/// Cartesian angstrom (`Direct` files are converted on read by molrs).
/// Produces an `"atoms"` block (`x`/`y`/`z`, optional `symbol`,
/// selective-dynamics flags, velocities) and a periodic `box`.
/// Single-frame: any `step != 0` returns `undefined`.
#[wasm_bindgen(js_name = VaspPoscarReader)]
pub struct VaspPoscarReader {
    content: Vec<u8>,
    cached_len: Option<usize>,
}

#[wasm_bindgen(js_class = VaspPoscarReader)]
impl VaspPoscarReader {
    /// Create a new POSCAR reader from a string containing the file content.
    #[wasm_bindgen(constructor)]
    pub fn new(content: &str) -> VaspPoscarReader {
        VaspPoscarReader {
            content: content.as_bytes().to_vec(),
            cached_len: None,
        }
    }

    /// Read the frame at `step`. POSCAR is single-frame, so any `step != 0`
    /// returns `undefined`.
    #[wasm_bindgen]
    pub fn read(&mut self, step: usize) -> Result<Option<Frame>, JsValue> {
        if step > 0 {
            return Ok(None);
        }
        let rs_frame = read_vasp_poscar_str(super::utf8_text(&self.content)?)
            .map_err(|e| JsValue::from_str(&format!("POSCAR read error: {}", e)))?;
        Ok(Some(Frame::from_rs(rs_frame)?))
    }

    /// Return the number of frames (always 0 or 1 for POSCAR files).
    #[wasm_bindgen]
    pub fn len(&mut self) -> Result<usize, JsValue> {
        if let Some(n) = self.cached_len {
            return Ok(n);
        }
        let n = if self.read(0)?.is_some() { 1 } else { 0 };
        self.cached_len = Some(n);
        Ok(n)
    }

    /// Check whether the file contains no valid frame.
    #[wasm_bindgen(js_name = isEmpty)]
    pub fn is_empty(&mut self) -> Result<bool, JsValue> {
        Ok(self.len()? == 0)
    }
}

/// VASP CHGCAR / CHGDIF volumetric data reader.
///
/// Reads VASP-format charge density files (extension-less canonical name
/// `CHGCAR` or `CHGCAR_*`). Produces a [`Frame`] with:
/// - `"atoms"` block: `element` (string), `x`/`y`/`z` (F, Cartesian Å).
/// - `"grid"` block: structural shape `[nx, ny, nz]`, columns
///   `total` (always) and `diff` (when ISPIN=2).
/// - `box`: triclinic POSCAR lattice in Å, fully periodic.
///
/// CHGCAR is single-frame; only `step = 0` is valid.
///
/// # Example (JavaScript)
///
/// ```js
/// const content = await file.text();
/// const reader  = new VaspChgcarReader(content);
/// const frame   = reader.read(0);
/// const grid    = frame.get("grid");
/// const total   = grid.copy("total");
/// ```
#[wasm_bindgen(js_name = VaspChgcarReader)]
pub struct VaspChgcarReader {
    content: Vec<u8>,
    cached_len: Option<usize>,
}

#[wasm_bindgen(js_class = VaspChgcarReader)]
impl VaspChgcarReader {
    /// Create a new CHGCAR reader from the file's text content.
    #[wasm_bindgen(constructor)]
    pub fn new(content: &str) -> VaspChgcarReader {
        VaspChgcarReader {
            content: content.as_bytes().to_vec(),
            cached_len: None,
        }
    }

    /// Read the frame at `step`. CHGCAR is single-frame.
    ///
    /// # Errors
    ///
    /// Throws a `JsValue` string on parse errors.
    #[wasm_bindgen]
    pub fn read(&mut self, step: usize) -> Result<Option<Frame>, JsValue> {
        if step > 0 {
            return Ok(None);
        }
        let rs_frame = read_vasp_chgcar_str(super::utf8_text(&self.content)?)
            .map_err(|e| JsValue::from_str(&format!("CHGCAR read error: {}", e)))?;
        Ok(Some(Frame::from_rs(rs_frame)?))
    }

    #[wasm_bindgen]
    pub fn len(&mut self) -> Result<usize, JsValue> {
        if let Some(n) = self.cached_len {
            return Ok(n);
        }
        let n = if self.read(0)?.is_some() { 1 } else { 0 };
        self.cached_len = Some(n);
        Ok(n)
    }

    #[wasm_bindgen(js_name = isEmpty)]
    pub fn is_empty(&mut self) -> Result<bool, JsValue> {
        Ok(self.len()? == 0)
    }
}

/// Write `frame` as VASP POSCAR text (needs a box).
#[wasm_bindgen(js_name = writeVaspPoscarStr)]
pub fn write_vasp_poscar_str(frame: &Frame) -> Result<String, JsValue> {
    frame.with_frame(|f| {
        molrs::io::write_vasp_poscar_str(f)
            .map_err(|e| JsValue::from_str(&format!("POSCAR writing error: {e}")))
    })
}
