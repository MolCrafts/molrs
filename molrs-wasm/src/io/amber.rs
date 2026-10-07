//! AMBER for the WASM API — the face of `molrs::io::amber`.
//!
//! | JS | molrs |
//! |----|-------|
//! | `AmberInpcrdReader` | `read_amber_inpcrd_str` (ASCII inpcrd / restrt) |
//! | `AmberAcReader` | `read_amber_ac_str` (Antechamber `.ac`) |

use molrs::io::{read_amber_ac_str, read_amber_inpcrd_str};
use wasm_bindgen::prelude::*;

use crate::core::frame::Frame;

/// AMBER ASCII inpcrd / restrt coordinate reader.
#[wasm_bindgen(js_name = AmberInpcrdReader)]
pub struct AmberInpcrdReader {
    content: Vec<u8>,
    cached_len: Option<usize>,
}

#[wasm_bindgen(js_class = AmberInpcrdReader)]
impl AmberInpcrdReader {
    #[wasm_bindgen(constructor)]
    pub fn new(content: &str) -> AmberInpcrdReader {
        AmberInpcrdReader {
            content: content.as_bytes().to_vec(),
            cached_len: None,
        }
    }

    #[wasm_bindgen]
    pub fn read(&mut self, step: usize) -> Result<Option<Frame>, JsValue> {
        if step > 0 {
            return Ok(None);
        }
        let rs_frame = read_amber_inpcrd_str(super::utf8_text(&self.content)?)
            .map_err(|e| JsValue::from_str(&format!("AMBER inpcrd read error: {}", e)))?;
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

/// Antechamber `.ac` structure reader.
#[wasm_bindgen(js_name = AmberAcReader)]
pub struct AmberAcReader {
    content: String,
    cached_len: Option<usize>,
}

#[wasm_bindgen(js_class = AmberAcReader)]
impl AmberAcReader {
    #[wasm_bindgen(constructor)]
    pub fn new(content: &str) -> AmberAcReader {
        AmberAcReader {
            content: content.to_string(),
            cached_len: None,
        }
    }

    #[wasm_bindgen]
    pub fn read(&mut self, step: usize) -> Result<Option<Frame>, JsValue> {
        if step > 0 {
            return Ok(None);
        }
        let rs_frame = read_amber_ac_str(&self.content)
            .map_err(|e| JsValue::from_str(&format!("AC read error: {}", e)))?;
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
