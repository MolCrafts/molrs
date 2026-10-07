//! DCD trajectories for the WASM API — the face of `molrs::io::dcd`.
//!
//! | JS | molrs |
//! |----|-------|
//! | `DcdStream` | `DcdIndexBuilder` + `read_dcd_bytes` (the one DCD reader; the only stream that uses `decoderState`) |
//! | `writeDcdBytes` | `DcdWriter` |

use molrs::io::dcd::{DcdIndexBuilder, DcdWriter};
use molrs::io::read_dcd_bytes;
use molrs::io::writer::FrameWriter;
use std::io::Cursor;
use wasm_bindgen::prelude::*;

use crate::core::frame::Frame;

impl_wasm_traj_stream! {
    name    = DcdStream,
    indexer = DcdIndexBuilder::new(),
    parse   = read_dcd_bytes,
}

/// Write `frame` as a one-frame DCD trajectory.
#[wasm_bindgen(js_name = writeDcdBytes)]
pub fn write_dcd_bytes(frame: &Frame) -> Result<Vec<u8>, JsValue> {
    frame.with_frame(|rs_frame| {
        let mut buf: Vec<u8> = Vec::new();
        // DcdWriter needs Write + Seek (it patches NSET after each frame);
        // Cursor<&mut Vec<u8>> satisfies both and leaves the bytes in `buf`
        // once the writer drops.
        DcdWriter::new(Cursor::new(&mut buf))
            .write(rs_frame)
            .map_err(|e| JsValue::from_str(&format!("DCD writing error: {e}")))?;
        Ok(buf)
    })
}
