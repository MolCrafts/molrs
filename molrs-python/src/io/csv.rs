//! Block CSV (`molrs::io::csv`): a `Block` to and from CSV text.

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
#[pyo3(signature = (text, delimiter = ',', header = None))]
pub fn read_block_csv(
    text: &str,
    delimiter: char,
    header: Option<Vec<String>>,
) -> PyResult<PyBlock> {
    let block = molrs::io::csv::block_from_csv(text, delimiter, header.as_deref())
        .map_err(pyo3::exceptions::PyValueError::new_err)?;
    PyBlock::from_core_block(block)
}

/// Serialize a :class:`Block` to CSV text. The inverse of :func:`read_block_csv`.
#[pyfunction]
#[pyo3(signature = (block, delimiter = ',', header = true))]
pub fn write_block_csv(block: &PyBlock, delimiter: char, header: bool) -> PyResult<String> {
    PyBlock::with_block(block, |b| {
        molrs::io::csv::block_to_csv(b, delimiter, header)
    })
}

/// Register this module's classes and functions.
pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(read_block_csv, m)?)?;
    m.add_function(wrap_pyfunction!(write_block_csv, m)?)?;
    Ok(())
}
