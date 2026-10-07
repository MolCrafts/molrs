//! XYZ / Extended XYZ for the WASM API — the face of `molrs::io::xyz`.
//!
//! | JS | molrs |
//! |----|-------|
//! | `XyzStream` | `XyzIndexBuilder` + `read_xyz_bytes` (the one XYZ reader) |
//! | `writeXyzStr` | `XyzWriter` |

use molrs::io::read_xyz_bytes;
use molrs::io::xyz::{XyzIndexBuilder, XyzWriter};
use wasm_bindgen::prelude::*;

use crate::core::frame::Frame;

impl_wasm_traj_stream! {
    name    = XyzStream,
    indexer = XyzIndexBuilder::new(),
    parse   = |bytes, _ctx| read_xyz_bytes(bytes),
}

/// Write `frame` as (extended) XYZ text.
#[wasm_bindgen(js_name = writeXyzStr)]
pub fn write_xyz_str(frame: &Frame) -> Result<String, JsValue> {
    super::utf8_string(write_bytes!(XyzWriter, frame, "XYZ")?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::test_support::two_atom_frame;
    use wasm_bindgen_test::*;

    #[wasm_bindgen_test]
    fn write_xyz_str_starts_with_the_atom_count() {
        let out = write_xyz_str(&two_atom_frame()).expect("xyz output");
        assert!(out.lines().next().unwrap_or("").starts_with('2'));
    }
}
