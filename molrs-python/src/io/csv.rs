//! `molrs.io.read_csv_block[_str]` / `write_csv_block[_str]` — `molrs::io`'s
//! CSV doors of the same names: a `Block` to and from CSV text.

use std::path::PathBuf;

use crate::core::block::PyBlock;
use crate::path::path_str;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

// These read and write a store container rather than a molecular file
// format, but they are IO and they live here for the same reason `read_pdb`
// does: turning text into a container is a reader's job. `Block` carries no
// `from_csv` constructor — a container that parses its own formats grows one
// entry point per format and duplicates this module. (The streaming wire
// encoding of a `Frame` is `molrs.stream`'s.)

/// `delimiter` as the one character the Rust doors take.
fn delimiter_char(delimiter: &str) -> PyResult<char> {
    let mut chars = delimiter.chars();
    match (chars.next(), chars.next()) {
        (Some(c), None) => Ok(c),
        _ => Err(PyValueError::new_err(format!(
            "a CSV delimiter is one character, got {delimiter:?}"
        ))),
    }
}

/// Read CSV ``text`` into a :class:`~molrs.core.Block`.
///
/// Each column's dtype is inferred int → float → str. When ``header`` is
/// given the text is treated as headerless and those names are used;
/// otherwise the first non-empty line names the columns. Blank lines are
/// skipped; fields are split on ``delimiter`` and trimmed, with no quoting.
///
/// Raises
/// ------
/// ValueError
///     The text is empty, a row is ragged, or ``delimiter`` is not one
///     character.
#[pyfunction]
#[pyo3(signature = (text, delimiter = ",", header = None))]
pub fn read_csv_block_str(
    text: &str,
    delimiter: &str,
    header: Option<Vec<String>>,
) -> PyResult<PyBlock> {
    let block = molrs::io::read_csv_block_str(text, delimiter_char(delimiter)?, header.as_deref())
        .map_err(PyValueError::new_err)?;
    PyBlock::from_core_block(block)
}

/// Read the CSV file at ``path`` into a :class:`~molrs.core.Block` —
/// :func:`read_csv_block_str` over the file's UTF-8 text.
///
/// Raises
/// ------
/// ValueError
///     The file cannot be read, or :func:`read_csv_block_str` refuses it.
#[pyfunction]
#[pyo3(signature = (path, delimiter = ",", header = None))]
pub fn read_csv_block(
    path: PathBuf,
    delimiter: &str,
    header: Option<Vec<String>>,
) -> PyResult<PyBlock> {
    let block = molrs::io::read_csv_block(
        path_str(&path)?,
        delimiter_char(delimiter)?,
        header.as_deref(),
    )
    .map_err(PyValueError::new_err)?;
    PyBlock::from_core_block(block)
}

/// Write a :class:`~molrs.core.Block` as CSV text — the inverse of
/// :func:`read_csv_block_str`.
///
/// Raises
/// ------
/// ValueError
///     ``delimiter`` is not one character, or a column has a dtype CSV
///     cannot encode.
#[pyfunction]
#[pyo3(signature = (block, delimiter = ",", header = true))]
pub fn write_csv_block_str(block: &PyBlock, delimiter: &str, header: bool) -> PyResult<String> {
    let delimiter = delimiter_char(delimiter)?;
    PyBlock::with_block(block, |b| {
        molrs::io::write_csv_block_str(b, delimiter, header)
    })
}

/// Write a :class:`~molrs.core.Block` as a CSV file at ``path`` — the
/// inverse of :func:`read_csv_block`.
///
/// Raises
/// ------
/// ValueError
///     The file cannot be written, or :func:`write_csv_block_str` refuses
///     the block.
#[pyfunction]
#[pyo3(signature = (path, block, delimiter = ",", header = true))]
pub fn write_csv_block(
    path: PathBuf,
    block: &PyBlock,
    delimiter: &str,
    header: bool,
) -> PyResult<()> {
    let delimiter = delimiter_char(delimiter)?;
    let path = path_str(&path)?;
    PyBlock::with_block(block, |b| {
        molrs::io::write_csv_block(path, b, delimiter, header)
    })?
    .map_err(PyValueError::new_err)
}

/// Register this module's functions.
pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    for door in [
        wrap_pyfunction!(read_csv_block, m)?,
        wrap_pyfunction!(read_csv_block_str, m)?,
        wrap_pyfunction!(write_csv_block, m)?,
        wrap_pyfunction!(write_csv_block_str, m)?,
    ] {
        crate::add_function(m, "molrs.io", door)?;
    }
    Ok(())
}
