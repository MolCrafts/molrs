//! GROMACS GRO for the WASM API — the face of `molrs::io::gro`.
//!
//! | JS | molrs |
//! |----|-------|
//! | `GroReader` | `GroReader` (whole-content; the one GRO reader) |
//! | `writeGroStr` | `GroWriter` (Å → nm on write) |

use molrs::io::gro::{GroReader as RsGroReader, GroWriter};
use molrs::io::reader::FrameReader;
use std::io::Cursor;
use wasm_bindgen::prelude::*;

use crate::core::frame::Frame;

/// GROMACS GRO structure / trajectory reader.
///
/// GRO is a fixed-column text format for GROMACS structures and
/// single-precision trajectories. Multi-frame files expose each frame via
/// `read(step)`. Coordinates and box are GROMACS-native nm in the file and
/// arrive in angstrom — the molrs GRO reader normalises at its own boundary,
/// so this binder scales nothing. Each frame produces an `"atoms"` block
/// (`res_id`, `res_name`, `name`, `element`, `id`, `x`/`y`/`z`, optional
/// `vx`/`vy`/`vz`) and a `box` from the box-vector line.
#[wasm_bindgen(js_name = GroReader)]
pub struct GroReader {
    content: Vec<u8>,
    cached_len: Option<usize>,
}

#[wasm_bindgen(js_class = GroReader)]
impl GroReader {
    /// Create a new GRO reader from a string containing the file content.
    #[wasm_bindgen(constructor)]
    pub fn new(content: &str) -> GroReader {
        GroReader {
            content: content.as_bytes().to_vec(),
            cached_len: None,
        }
    }

    /// Read the frame at the given step index (0-based). Coordinates arrive
    /// in angstrom (the molrs reader converts from the file's nm).
    #[wasm_bindgen]
    pub fn read(&mut self, step: usize) -> Result<Option<Frame>, JsValue> {
        let mut reader = RsGroReader::new(Cursor::new(self.content.as_slice()));
        for current in 0..=step {
            let rs_frame = reader
                .read()
                .map_err(|e| JsValue::from_str(&format!("GRO read error: {}", e)))?;
            match rs_frame {
                Some(frame) if current == step => {
                    return Ok(Some(Frame::from_rs(frame)?));
                }
                Some(_) => continue,
                None => return Ok(None),
            }
        }
        Ok(None)
    }

    /// Return the number of frames in the GRO file.
    #[wasm_bindgen]
    pub fn len(&mut self) -> Result<usize, JsValue> {
        if let Some(n) = self.cached_len {
            return Ok(n);
        }
        let mut reader = RsGroReader::new(Cursor::new(self.content.as_slice()));
        let mut count = 0usize;
        while reader
            .read()
            .map_err(|e| JsValue::from_str(&format!("GRO len error: {}", e)))?
            .is_some()
        {
            count += 1;
        }
        self.cached_len = Some(count);
        Ok(count)
    }

    /// Check whether the file contains no frames.
    #[wasm_bindgen(js_name = isEmpty)]
    pub fn is_empty(&mut self) -> Result<bool, JsValue> {
        Ok(self.len()? == 0)
    }
}

/// Write `frame` as GROMACS GRO text (Å → nm).
#[wasm_bindgen(js_name = writeGroStr)]
pub fn write_gro_str(frame: &Frame) -> Result<String, JsValue> {
    super::utf8_string(write_bytes!(GroWriter, frame, "GRO")?)
}
