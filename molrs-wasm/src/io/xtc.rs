//! GROMACS XTC trajectories for the WASM API — the face of `molrs::io::xtc`.
//!
//! | JS | molrs |
//! |----|-------|
//! | `XtcStream` | `XtcIndexBuilder` + `read_xtc_bytes` (the one XTC reader) |
//! | `writeXtcBytes` | `XtcWriter` (Å → nm on write) |

use molrs::io::read_xtc_bytes;
use molrs::io::xtc::{XtcIndexBuilder, XtcWriter};
use wasm_bindgen::prelude::*;

use crate::core::frame::Frame;

impl_wasm_traj_stream! {
    name    = XtcStream,
    indexer = XtcIndexBuilder::new(),
    parse   = |bytes, _ctx| read_xtc_bytes(bytes),
}

/// Write `frame` as a one-frame GROMACS XTC trajectory (Å → nm).
#[wasm_bindgen(js_name = writeXtcBytes)]
pub fn write_xtc_bytes(frame: &Frame) -> Result<Vec<u8>, JsValue> {
    write_bytes!(XtcWriter, frame, "XTC")
}
