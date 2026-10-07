//! Protein Data Bank for the WASM API — the face of `molrs::io::pdb`.
//!
//! | JS | molrs |
//! |----|-------|
//! | `PdbStream` | `PdbIndexBuilder` + `read_pdb_bytes` (the one PDB reader) |
//! | `writePdbStr` | `PdbWriter` |

use molrs::io::pdb::{PdbIndexBuilder, PdbWriter};
use molrs::io::read_pdb_bytes;
use wasm_bindgen::prelude::*;

use crate::core::frame::Frame;

impl_wasm_traj_stream! {
    name    = PdbStream,
    indexer = PdbIndexBuilder::new(),
    parse   = |bytes, _ctx| read_pdb_bytes(bytes),
}

/// Write `frame` as PDB text.
#[wasm_bindgen(js_name = writePdbStr)]
pub fn write_pdb_str(frame: &Frame) -> Result<String, JsValue> {
    super::utf8_string(write_bytes!(PdbWriter, frame, "PDB")?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::test_fixtures::{float_col, two_atom_frame};
    use wasm_bindgen_test::*;

    /// A PDB file is read through `PdbStream`, its one reader.
    #[wasm_bindgen_test]
    fn pdb_stream_reads_the_atoms() {
        let pdb = "ATOM      1  C   MOL     1       1.000   2.000   3.000  1.00  0.00           C\n\
ATOM      2  N   MOL     1       4.000   5.000   6.000  1.00  0.00           N\n\
END\n";
        let mut stream = PdbStream::new();
        let entries = index_whole!(stream, pdb.as_bytes());
        assert_eq!(entries.len(), 1);
        let e = &entries[0];
        let frame = stream
            .parse_range_in_input(e.byte_offset() as usize, e.byte_len() as usize)
            .expect("parse");
        let x = float_col(&frame.get("atoms").expect("atoms"), "x");
        assert_eq!(x.to_vec(), vec![1.0, 4.0]);
    }

    #[wasm_bindgen_test]
    fn write_pdb_str_emits_atom_records() {
        let out = write_pdb_str(&two_atom_frame()).expect("pdb output");
        assert!(out.contains("ATOM"));
    }
}
