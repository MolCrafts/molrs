//! Hand-written CSV (de)serialization for [`Block`] — no external crate.
//!
//! `block_from_csv` parses CSV text into a `Block`, inferring each column's
//! dtype as int → float → str (the first that parses every cell wins).
//! `block_to_csv` serializes a `Block` back to CSV text. Fields are split /
//! joined on a single-character delimiter; values are trimmed on read. Quoting
//! and escaping are intentionally not handled (simple numeric/label tables).

use ndarray::Array1;

use crate::core::store::block::Block;
use crate::core::store::block::Column;
use crate::types::{F, I};

/// Parse CSV `text` into a [`Block`].
///
/// If `header` is `Some`, the text is treated as headerless and those names are
/// used; otherwise the first non-empty line provides the column names. Blank
/// lines are skipped. Per-column dtype is inferred int → float → str.
pub fn block_from_csv(
    text: &str,
    delimiter: char,
    header: Option<&[String]>,
) -> Result<Block, String> {
    let mut lines = text.lines().filter(|l| !l.trim().is_empty());

    let headers: Vec<String> = match header {
        Some(h) => h.to_vec(),
        None => lines
            .next()
            .ok_or_else(|| "CSV is empty".to_string())?
            .split(delimiter)
            .map(|s| s.trim().to_string())
            .collect(),
    };
    let ncol = headers.len();
    let mut cols: Vec<Vec<String>> = vec![Vec::new(); ncol];
    for line in lines {
        let fields: Vec<&str> = line.split(delimiter).collect();
        // Fewer fields than columns can't be filled → error. Extra trailing
        // fields are ignored (the header count defines how many columns to read).
        if fields.len() < ncol {
            return Err(format!(
                "CSV row has {} field(s) but {} column name(s) were given",
                fields.len(),
                ncol
            ));
        }
        for (i, col) in cols.iter_mut().enumerate() {
            col.push(fields[i].trim().to_string());
        }
    }

    let mut block = Block::new();
    for (name, raw) in headers.into_iter().zip(cols) {
        insert_inferred(&mut block, name, raw)?;
    }
    Ok(block)
}

/// Insert `raw` string cells under `name`, inferring int → float → str.
fn insert_inferred(block: &mut Block, name: String, raw: Vec<String>) -> Result<(), String> {
    let nonempty = !raw.is_empty();

    // A canonical key's dtype is declared, not inferred. Content-driven
    // inference would type `x` as Int whenever a file happens to hold whole
    // numbers, and the column would then reject the first fractional
    // coordinate written to it — the dtype-fixed-on-first-write trap, arrived
    // at from a CSV instead of from a bad writer.
    if let Some(spec) = crate::store::schema::column(&name) {
        return insert_as(block, name, raw, spec.dtype);
    }

    if nonempty && raw.iter().all(|s| s.parse::<I>().is_ok()) {
        let v: Vec<I> = raw.iter().map(|s| s.parse().unwrap()).collect();
        block
            .insert(name, Array1::from(v).into_dyn())
            .map_err(|e| e.to_string())
    } else if nonempty && raw.iter().all(|s| s.parse::<F>().is_ok()) {
        let v: Vec<F> = raw.iter().map(|s| s.parse().unwrap()).collect();
        block
            .insert(name, Array1::from(v).into_dyn())
            .map_err(|e| e.to_string())
    } else {
        block
            .insert(name, Array1::from(raw).into_dyn())
            .map_err(|e| e.to_string())
    }
}

/// Serialize `block` to CSV text (inverse of [`block_from_csv`]).
pub fn block_to_csv(block: &Block, delimiter: char, header: bool) -> String {
    let names: Vec<&str> = block.keys().collect();
    let nrows = block.nrows().unwrap_or(0);
    let delim = delimiter.to_string();

    let mut out = String::new();
    if header {
        out.push_str(&names.join(&delim));
        out.push('\n');
    }
    for row in 0..nrows {
        let cells: Vec<String> = names
            .iter()
            .map(|name| match block.get(name) {
                Some(col) => cell_to_string(col, row),
                None => String::new(),
            })
            .collect();
        out.push_str(&cells.join(&delim));
        out.push('\n');
    }
    out
}

/// Format one cell of `col` at `row` as a string, dispatching on its dtype.
fn cell_to_string(col: &Column, row: usize) -> String {
    if let Some(a) = col.as_float() {
        return a[[row]].to_string();
    }
    if let Some(a) = col.as_int() {
        return a[[row]].to_string();
    }
    if let Some(a) = col.as_uint() {
        return a[[row]].to_string();
    }
    if let Some(a) = col.as_bool() {
        return a[[row]].to_string();
    }
    if let Some(a) = col.as_u8() {
        return a[[row]].to_string();
    }
    if let Some(a) = col.as_string() {
        return a[[row]].clone();
    }
    String::new()
}

/// Parse a column at a dtype the schema declares, rather than inferring one.
fn insert_as(
    block: &mut Block,
    name: String,
    raw: Vec<String>,
    dtype: crate::store::block::DType,
) -> Result<(), String> {
    use crate::store::block::DType;
    let parse_err = |e: std::num::ParseIntError| format!("column '{name}': {e}");
    match dtype {
        DType::Float => {
            let v: Vec<F> = raw
                .iter()
                .map(|s| s.parse::<F>().map_err(|e| format!("column '{name}': {e}")))
                .collect::<Result<_, _>>()?;
            block
                .insert(name, Array1::from(v).into_dyn())
                .map_err(|e| e.to_string())
        }
        DType::Int => {
            let v: Vec<I> = raw
                .iter()
                .map(|s| s.parse::<I>().map_err(parse_err))
                .collect::<Result<_, _>>()?;
            block
                .insert(name, Array1::from(v).into_dyn())
                .map_err(|e| e.to_string())
        }
        DType::UInt => {
            let v: Vec<crate::types::Idx> = raw
                .iter()
                .map(|s| s.parse::<crate::types::Idx>().map_err(parse_err))
                .collect::<Result<_, _>>()?;
            block
                .insert(name, Array1::from(v).into_dyn())
                .map_err(|e| e.to_string())
        }
        DType::Bool => {
            let v: Vec<bool> = raw
                .iter()
                .map(|s| {
                    s.parse::<bool>()
                        .map_err(|e| format!("column '{name}': {e}"))
                })
                .collect::<Result<_, _>>()?;
            block
                .insert(name, Array1::from(v).into_dyn())
                .map_err(|e| e.to_string())
        }
        DType::U8 => {
            let v: Vec<u8> = raw
                .iter()
                .map(|s| s.parse::<u8>().map_err(parse_err))
                .collect::<Result<_, _>>()?;
            block
                .insert(name, Array1::from(v).into_dyn())
                .map_err(|e| e.to_string())
        }
        DType::String => block
            .insert(name, Array1::from(raw).into_dyn())
            .map_err(|e| e.to_string()),
        other => Err(format!(
            "CSV cannot encode {other} columns; only float/int/uint/u8/bool/string"
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_headered() {
        let text = "x,y,name\n0,1.5,a\n3,4.5,b\n";
        let block = block_from_csv(text, ',', None).expect("parse");
        assert_eq!(block.nrows(), Some(2));
        // `x` parses as Float even though the file holds whole numbers: the
        // schema declares the dtype, so inference does not get to make a
        // coordinate column Int and reject the first fractional value later.
        assert!(block.get("x").unwrap().as_float().is_some());
        assert!(block.get("y").unwrap().as_float().is_some());
        assert!(block.get("name").unwrap().as_string().is_some());
        let out = block_to_csv(&block, ',', true);
        let rt = block_from_csv(&out, ',', None).expect("reparse");
        assert_eq!(rt.get("x").unwrap().as_float().unwrap()[[1]], 3.0);
        assert_eq!(rt.get("name").unwrap().as_string().unwrap()[[0]], "a");
    }

    #[test]
    fn headerless_with_names() {
        let names = vec!["a".to_string(), "b".to_string()];
        let block = block_from_csv("1,2\n3,4\n", ',', Some(&names)).expect("parse");
        assert_eq!(block.nrows(), Some(2));
        // Block keys are unordered; assert membership, not order.
        let keys: std::collections::HashSet<&str> = block.keys().collect();
        assert_eq!(keys, ["a", "b"].into_iter().collect());
    }

    #[test]
    fn empty_text_errors() {
        assert!(block_from_csv("", ',', None).is_err());
    }
}
