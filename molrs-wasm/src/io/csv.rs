//! CSV tables for the WASM API — the face of `molrs::io`'s CSV doors.
//!
//! | JS | molrs |
//! |----|-------|
//! | `readCsvBlockStr` | `read_csv_block_str` |
//! | `writeCsvBlockStr` | `write_csv_block_str` |

use wasm_bindgen::prelude::*;

use crate::core::block::Block;

/// Parse CSV `text` into a [`Block`]; each column's dtype is inferred
/// int → float → str. With `header`, the text is headerless and those are
/// the column names; without, the first non-empty line names them.
/// `delimiter` defaults to `","`.
#[wasm_bindgen(js_name = readCsvBlockStr)]
pub fn read_csv_block_str(
    text: &str,
    delimiter: Option<char>,
    header: Option<Vec<String>>,
) -> Result<Block, JsValue> {
    let block = molrs::io::read_csv_block_str(text, delimiter.unwrap_or(','), header.as_deref())
        .map_err(|e| JsValue::from_str(&format!("CSV read error: {e}")))?;
    Block::from_rs(block)
}

/// Write `block` as CSV text — the inverse of `readCsvBlockStr`.
/// `delimiter` defaults to `","`, `header` (write the names line) to `true`.
#[wasm_bindgen(js_name = writeCsvBlockStr)]
pub fn write_csv_block_str(
    block: &Block,
    delimiter: Option<char>,
    header: Option<bool>,
) -> Result<String, JsValue> {
    block.with_rs(|b| {
        molrs::io::write_csv_block_str(b, delimiter.unwrap_or(','), header.unwrap_or(true))
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use wasm_bindgen_test::*;

    #[wasm_bindgen_test]
    fn write_csv_block_str_reads_back() {
        let block = read_csv_block_str("a,b\n1,x\n2,y\n", None, None).expect("csv");
        assert_eq!(block.n_rows().expect("rows"), 2);
        let text = write_csv_block_str(&block, None, None).expect("text");
        assert!(text.starts_with("a,b\n"), "{text}");
        let back = read_csv_block_str(&text, None, None).expect("read back");
        assert_eq!(back.keys().expect("keys"), ["a", "b"]);
        assert_eq!(back.n_rows().expect("rows"), 2);
    }
}
