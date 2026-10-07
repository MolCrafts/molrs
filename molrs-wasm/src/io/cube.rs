//! Gaussian Cube for the WASM API — the face of `molrs::io::cube`.
//!
//! | JS | molrs |
//! |----|-------|
//! | `CubeReader` | `read_cube_str` |
//! | `writeCubeStr` | `write_cube_str` (needs a `"grid"` block) |

use molrs::io::read_cube_str;
use wasm_bindgen::prelude::*;

use crate::core::frame::Frame;

/// Gaussian Cube file reader.
///
/// Cube files describe a single voxel grid with embedded atom geometry.
/// The reader produces a [`Frame`] with:
/// - `"atoms"` block: `element` (string), `atomic_number` (i32),
///   `charge` (F), `x`/`y`/`z` (F, **always Å** — Bohr files are converted
///   on read).
/// - `"grid"` block: structural shape `[nx, ny, nz]` and one f64 column
///   per scalar field — `density` for single-density files,
///   `mo_<idx>` for negative-natoms multi-orbital files.
/// - `box`: voxel cell × dims in Å.
///
/// Cube is inherently single-frame (only `step = 0` is valid).
///
/// # Example (JavaScript)
///
/// ```js
/// const content = await file.text();
/// const reader  = new CubeReader(content);
/// const frame   = reader.read(0);
/// const grid    = frame.get("grid");   // shape [nx, ny, nz]
/// const density = grid.copy("density"); // owned Float64Array
/// ```
#[wasm_bindgen(js_name = CubeReader)]
pub struct CubeReader {
    content: Vec<u8>,
    cached_len: Option<usize>,
}

#[wasm_bindgen(js_class = CubeReader)]
impl CubeReader {
    /// Create a new Cube reader from the file's text content.
    #[wasm_bindgen(constructor)]
    pub fn new(content: &str) -> CubeReader {
        CubeReader {
            content: content.as_bytes().to_vec(),
            cached_len: None,
        }
    }

    /// Read the frame at `step`. Cube files are single-frame, so any
    /// `step != 0` returns `undefined`.
    ///
    /// # Errors
    ///
    /// Throws a `JsValue` string on parse errors.
    #[wasm_bindgen]
    pub fn read(&mut self, step: usize) -> Result<Option<Frame>, JsValue> {
        if step > 0 {
            return Ok(None);
        }
        let rs_frame = read_cube_str(super::utf8_text(&self.content)?)
            .map_err(|e| JsValue::from_str(&format!("Cube read error: {}", e)))?;
        Ok(Some(Frame::from_rs(rs_frame)?))
    }

    /// Return the number of frames (always 0 or 1).
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

/// Write `frame` as Gaussian cube text (needs a `"grid"` block).
#[wasm_bindgen(js_name = writeCubeStr)]
pub fn write_cube_str(frame: &Frame) -> Result<String, JsValue> {
    frame.with_frame(|f| {
        molrs::io::write_cube_str(f)
            .map_err(|e| JsValue::from_str(&format!("Cube writing error: {e}")))
    })
}
