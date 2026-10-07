//! Protein Data Bank for the WASM API — the face of `molrs::io::pdb`.
//!
//! | JS | molrs |
//! |----|-------|
//! | `PdbStream` | `PdbIndexBuilder` + `read_pdb_bytes` (the chunk-fed PDB reader) |
//! | `readPdbStr`, `readPdbBytes`, `writePdbStr` | `read_pdb_str`, `read_pdb_bytes`, `write_pdb_str` |

use molrs::io::pdb::PdbIndexBuilder;
use wasm_bindgen::prelude::*;

impl_wasm_traj_stream! {
    name    = PdbStream,
    indexer = PdbIndexBuilder::new(),
    parse   = |bytes, _ctx| molrs::io::read_pdb_bytes(bytes),
}

read_door!(
    /// Read the first frame of PDB text.
    readPdbStr => read_pdb_str(text: &str), "PDB"
);
read_door!(
    /// Read one PDB frame (a `MODEL` block or a whole file) from bytes.
    readPdbBytes => read_pdb_bytes(bytes: &[u8]), "PDB"
);
write_door!(
    /// Write `frame` as PDB text.
    writePdbStr => write_pdb_str -> String, "PDB"
);

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
