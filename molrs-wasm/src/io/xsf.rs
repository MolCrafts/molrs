//! XCrySDen XSF for the WASM API — the face of `molrs::io::xsf`.
//!
//! | JS | molrs |
//! |----|-------|
//! | `XsfReader` | `read_xsf_str` |
//! | `writeXsfStr` | `write_xsf_str` |

use molrs::io::read_xsf_str;
use wasm_bindgen::prelude::*;

use crate::core::frame::Frame;

/// XCrySDen XSF structure reader.
#[wasm_bindgen(js_name = XsfReader)]
pub struct XsfReader {
    content: Vec<u8>,
    cached_len: Option<usize>,
}

#[wasm_bindgen(js_class = XsfReader)]
impl XsfReader {
    #[wasm_bindgen(constructor)]
    pub fn new(content: &str) -> XsfReader {
        XsfReader {
            content: content.as_bytes().to_vec(),
            cached_len: None,
        }
    }

    #[wasm_bindgen]
    pub fn read(&mut self, step: usize) -> Result<Option<Frame>, JsValue> {
        if step > 0 {
            return Ok(None);
        }
        let rs_frame = read_xsf_str(super::utf8_text(&self.content)?)
            .map_err(|e| JsValue::from_str(&format!("XSF read error: {}", e)))?;
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

/// Write `frame` as XCrySDen XSF text.
#[wasm_bindgen(js_name = writeXsfStr)]
pub fn write_xsf_str(frame: &Frame) -> Result<String, JsValue> {
    frame.with_frame(|f| {
        molrs::io::write_xsf_str(f)
            .map_err(|e| JsValue::from_str(&format!("XSF writing error: {e}")))
    })
}
