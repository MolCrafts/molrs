//! MDL SDF / MOL for the WASM API — the face of `molrs::io::sdf`.
//!
//! | JS | molrs |
//! |----|-------|
//! | `SdfStream` | `SdfIndexBuilder` + `read_sdf_bytes` (the one SDF reader) |
//!
//! molrs has no SDF writer, so SDF is read-only.

use molrs::io::read_sdf_bytes;
use molrs::io::sdf::SdfIndexBuilder;
use wasm_bindgen::prelude::*;

impl_wasm_traj_stream! {
    name    = SdfStream,
    indexer = SdfIndexBuilder::new(),
    parse   = |bytes, _ctx| read_sdf_bytes(bytes),
}
