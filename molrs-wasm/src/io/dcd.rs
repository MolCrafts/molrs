//! DCD trajectories for the WASM API — the face of `molrs::io::dcd`.
//!
//! | JS | molrs |
//! |----|-------|
//! | `DcdStream` | `DcdIndexBuilder` + `read_dcd_bytes` (the chunk-fed DCD reader; the only stream that uses `decoderContext`) |
//! | `readDcdBytes`, `writeDcdBytes` | `read_dcd_bytes`, `write_dcd_bytes` |

use molrs::io::dcd::DcdIndexBuilder;
use wasm_bindgen::prelude::*;

use crate::core::frame::Frame;

impl_wasm_traj_stream! {
    name    = DcdStream,
    indexer = DcdIndexBuilder::new(),
    parse   = molrs::io::read_dcd_bytes,
}

/// Read one DCD frame from bytes: a whole one-frame file (what
/// `writeDcdBytes` returns), or a frame body with the decoder `context` it
/// is decoded with.
#[wasm_bindgen(js_name = readDcdBytes)]
pub fn read_dcd_bytes(bytes: &[u8], context: Option<Vec<u8>>) -> Result<Frame, JsValue> {
    let frame = molrs::io::read_dcd_bytes(bytes, context.as_deref())
        .map_err(|e| JsValue::from_str(&format!("DCD read error: {e}")))?;
    Frame::from_rs(frame)
}

write_door!(
    /// Write `frame` as a one-frame DCD trajectory.
    writeDcdBytes => write_dcd_bytes -> Vec<u8>, "DCD"
);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::test_support::{float_col, two_atom_frame};
    use wasm_bindgen_test::*;

    /// `readDcdBytes` reads back the one-frame file `writeDcdBytes` wrote.
    #[wasm_bindgen_test]
    fn read_dcd_bytes_reads_back_write_dcd_bytes() {
        let bytes = write_dcd_bytes(&two_atom_frame()).expect("dcd bytes");
        let frame = read_dcd_bytes(&bytes, None).expect("read back");
        let z = float_col(&frame.get("atoms").expect("atoms"), "z");
        assert_eq!(z.to_vec(), vec![0.0, 0.5]);
    }
}
