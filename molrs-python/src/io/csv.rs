//! The native half of `molrs.io.read_csv_block[_str]` /
//! `write_csv_block[_str]` (`molrs::io`): a `Block` to and from CSV text. The
//! public doors are Python (`molrs/io/_csv_block.py`), which add the file,
//! encoding and empty-field handling around these two private natives, named
//! after the Rust doors they bind.

use crate::core::block::PyBlock;
use pyo3::prelude::*;

// These read and write a store container rather than a molecular file
// format, but they are still IO and they live here for the same reason
// `read_pdb` does: turning text into a container is a reader's job. `Block`
// deliberately carries no `from_csv` constructor — a container that parses
// its own formats grows one entry point per format and duplicates this
// module. (The streaming wire encoding of a `Frame` is `molrs.stream`'s.)

/// Parse CSV ``text`` into a :class:`Block`.
///
/// Each column's dtype is inferred int → float → str. When ``header`` is given
/// the text is treated as headerless and those names are used; otherwise the
/// first non-empty line provides the column names.
#[pyfunction]
#[pyo3(name = "read_csv_block_str", signature = (text, delimiter = ',', header = None))]
pub fn read_csv_block_str_py(
    text: &str,
    delimiter: char,
    header: Option<Vec<String>>,
) -> PyResult<PyBlock> {
    let block = molrs::io::read_csv_block_str(text, delimiter, header.as_deref())
        .map_err(pyo3::exceptions::PyValueError::new_err)?;
    PyBlock::from_core_block(block)
}

/// Serialize a :class:`Block` to CSV text. The inverse of
/// :func:`read_csv_block_str`.
#[pyfunction]
#[pyo3(name = "write_csv_block_str", signature = (block, delimiter = ',', header = true))]
pub fn write_csv_block_str_py(block: &PyBlock, delimiter: char, header: bool) -> PyResult<String> {
    PyBlock::with_block(block, |b| {
        molrs::io::write_csv_block_str(b, delimiter, header)
    })
}

/// Register this module's classes and functions.
pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(read_csv_block_str_py, m)?)?;
    m.add_function(wrap_pyfunction!(write_csv_block_str_py, m)?)?;
    Ok(())
}
