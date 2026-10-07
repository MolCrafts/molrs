//! Tripos MOL2 for the WASM API — the face of `molrs::io::mol2`.
//!
//! | JS | molrs |
//! |----|-------|
//! | `Mol2Reader` | `Mol2Reader` (whole-content; the one MOL2 reader) |
//! | `writeMol2Str` | `Mol2Writer` |

use molrs::io::mol2::{Mol2Reader as RsMol2Reader, Mol2Writer};
use molrs::io::reader::{FrameReader, Reader};
use std::io::Cursor;
use wasm_bindgen::prelude::*;

use crate::core::frame::Frame;

/// Tripos MOL2 reader.
///
/// MOL2 is a section-delimited (`@<TRIPOS>...`) text format. Multi-molecule
/// files expose each `MOLECULE` record as a frame via `read(step)`.
/// Coordinates are already in angstrom. Produces an `"atoms"` block (`id`,
/// `name`, `x`/`y`/`z`, `atom_type`, optional `subst_id`/`subst_name`/
/// `charge`) and, when present, a `"bonds"` block (`atomi`/`atomj` 0-based,
/// `bond_type`).
#[wasm_bindgen(js_name = Mol2Reader)]
pub struct Mol2Reader {
    content: Vec<u8>,
    cached_len: Option<usize>,
}

#[wasm_bindgen(js_class = Mol2Reader)]
impl Mol2Reader {
    /// Create a new MOL2 reader from a string containing the file content.
    #[wasm_bindgen(constructor)]
    pub fn new(content: &str) -> Mol2Reader {
        Mol2Reader {
            content: content.as_bytes().to_vec(),
            cached_len: None,
        }
    }

    /// Read the molecule record at the given step index (0-based).
    #[wasm_bindgen]
    pub fn read(&mut self, step: usize) -> Result<Option<Frame>, JsValue> {
        let mut reader = RsMol2Reader::new(Cursor::new(self.content.as_slice()));
        for current in 0..=step {
            let rs_frame = reader
                .read()
                .map_err(|e| JsValue::from_str(&format!("MOL2 read error: {}", e)))?;
            match rs_frame {
                Some(frame) if current == step => return Ok(Some(Frame::from_rs(frame)?)),
                Some(_) => continue,
                None => return Ok(None),
            }
        }
        Ok(None)
    }

    /// Return the number of molecule records in the MOL2 file.
    #[wasm_bindgen]
    pub fn len(&mut self) -> Result<usize, JsValue> {
        if let Some(n) = self.cached_len {
            return Ok(n);
        }
        let mut reader = RsMol2Reader::new(Cursor::new(self.content.as_slice()));
        let mut count = 0usize;
        while reader
            .read()
            .map_err(|e| JsValue::from_str(&format!("MOL2 len error: {}", e)))?
            .is_some()
        {
            count += 1;
        }
        self.cached_len = Some(count);
        Ok(count)
    }

    /// Check whether the file contains no records.
    #[wasm_bindgen(js_name = isEmpty)]
    pub fn is_empty(&mut self) -> Result<bool, JsValue> {
        Ok(self.len()? == 0)
    }
}

/// Write `frame` as Tripos MOL2 text.
#[wasm_bindgen(js_name = writeMol2Str)]
pub fn write_mol2_str(frame: &Frame) -> Result<String, JsValue> {
    super::utf8_string(write_bytes!(Mol2Writer, frame, "MOL2")?)
}
