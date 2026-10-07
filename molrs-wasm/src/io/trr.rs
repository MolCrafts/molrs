//! GROMACS TRR trajectories for the WASM API — the face of `molrs::io::trr`.
//!
//! | JS | molrs |
//! |----|-------|
//! | `TrrStream` | `TrrIndexBuilder` + `read_trr_bytes` (the one TRR reader) |
//! | `writeTrrBytes` | `TrrWriter` (Å → nm on write) |

use molrs::io::read_trr_bytes;
use molrs::io::trr::{TrrIndexBuilder, TrrWriter};
use wasm_bindgen::prelude::*;

use crate::core::frame::Frame;

impl_wasm_traj_stream! {
    name    = TrrStream,
    indexer = TrrIndexBuilder::new(),
    parse   = |bytes, _ctx| read_trr_bytes(bytes),
}

/// Write `frame` as a one-frame GROMACS TRR trajectory (Å → nm).
#[wasm_bindgen(js_name = writeTrrBytes)]
pub fn write_trr_bytes(frame: &Frame) -> Result<Vec<u8>, JsValue> {
    write_bytes!(TrrWriter, frame, "TRR")
}
