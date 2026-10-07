//! XYZ / Extended XYZ for the WASM API — the face of `molrs::io::xyz`.
//!
//! | JS | molrs |
//! |----|-------|
//! | `XyzStream` | `XyzIndexBuilder` + `read_xyz_bytes` (the chunk-fed XYZ reader) |
//! | `readXyzStr`, `readXyzBytes`, `writeXyzStr` | `read_xyz_str`, `read_xyz_bytes`, `write_xyz_str` |

use molrs::io::xyz::XyzIndexBuilder;
use wasm_bindgen::prelude::*;

impl_wasm_traj_stream! {
    name    = XyzStream,
    indexer = XyzIndexBuilder::new(),
    parse   = |bytes, _ctx| molrs::io::read_xyz_bytes(bytes),
}

read_door!(
    /// Read the first frame of (extended) XYZ text.
    readXyzStr => read_xyz_str(text: &str), "XYZ"
);
read_door!(
    /// Read one (extended) XYZ frame from bytes.
    readXyzBytes => read_xyz_bytes(bytes: &[u8]), "XYZ"
);
write_door!(
    /// Write `frame` as (extended) XYZ text.
    writeXyzStr => write_xyz_str -> String, "XYZ"
);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::test_fixtures::{float_col, two_atom_frame};
    use wasm_bindgen_test::*;

    #[wasm_bindgen_test]
    fn write_xyz_str_starts_with_the_atom_count() {
        let out = write_xyz_str(&two_atom_frame()).expect("xyz output");
        assert!(out.lines().next().unwrap_or("").starts_with('2'));
    }

    /// `readXyzStr` and `readXyzBytes` read what `writeXyzStr` wrote.
    #[wasm_bindgen_test]
    fn read_xyz_str_and_bytes_read_back_the_written_text() {
        let out = write_xyz_str(&two_atom_frame()).expect("xyz output");
        for frame in [
            read_xyz_str(&out).expect("str"),
            read_xyz_bytes(out.as_bytes()).expect("bytes"),
        ] {
            let x = float_col(&frame.get("atoms").expect("atoms"), "x");
            assert_eq!(x.to_vec(), vec![0.0, 1.0]);
        }
    }
}
