//! MDL SDF / MOL for the WASM API — the face of `molrs::io::sdf`.
//!
//! | JS | molrs |
//! |----|-------|
//! | `SdfStream` | `SdfIndexBuilder` + `read_sdf_bytes` (the chunk-fed SDF reader) |
//! | `readSdfStr`, `readSdfBytes` | `read_sdf_str`, `read_sdf_bytes` |
//!
//! molrs has no SDF writer, so SDF is read-only.

use molrs::io::sdf::SdfIndexBuilder;
use wasm_bindgen::prelude::*;

impl_wasm_traj_stream! {
    name    = SdfStream,
    indexer = SdfIndexBuilder::new(),
    parse   = |bytes, _ctx| molrs::io::read_sdf_bytes(bytes),
}

read_door!(
    /// Read the first record of SDF / MDL molfile text.
    readSdfStr => read_sdf_str(text: &str), "SDF"
);
read_door!(
    /// Read one SDF record from bytes.
    readSdfBytes => read_sdf_bytes(bytes: &[u8]), "SDF"
);
