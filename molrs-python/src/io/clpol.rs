//! `molrs.io.read_clpol_alpha[_str]` — `molrs::io`'s doors of the same names:
//! a CL&Pol `alpha.ff` polarisation table read into rows.

use std::path::PathBuf;

use molrs::io::clpol::ClpolAlphaRow;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::path::path_str;

/// One row's fields under the names :func:`molrs.ff.params.clpol_polarizability`
/// uses, and its ``type_name``.
fn row_dict(py: Python<'_>, row: ClpolAlphaRow) -> PyResult<Bound<'_, PyDict>> {
    let dict = PyDict::new(py);
    dict.set_item("type_name", row.type_name)?;
    dict.set_item("m_D", row.m_d)?;
    dict.set_item("q_D_sign", row.q_d_sign)?;
    dict.set_item("k_D", row.k_d)?;
    dict.set_item("alpha", row.alpha)?;
    dict.set_item("a_thole", row.a_thole)?;
    Ok(dict)
}

fn row_dicts(py: Python<'_>, rows: Vec<ClpolAlphaRow>) -> PyResult<Vec<Bound<'_, PyDict>>> {
    rows.into_iter().map(|row| row_dict(py, row)).collect()
}

/// Read ``alpha.ff`` text into its rows, in file order.
///
/// The text is whitespace-separated rows ``type m_D q_D k_D alpha a_thole``;
/// ``#`` starts a comment and a line with fewer than six fields is not a
/// row. A type given twice keeps both rows; the later one is the file's last
/// word. The table molrs ships is :func:`molrs.ff.params.clpol_polarizability`.
///
/// Returns
/// -------
/// list[dict]
///     One dict per row: ``type_name``, ``m_D`` (u), ``q_D_sign``, ``k_D``
///     (kJ/mol/Å², ``k/2 r²`` form), ``alpha`` (Å³) and ``a_thole``.
///
/// Raises
/// ------
/// ValueError
///     A row's numeric fields do not parse.
#[pyfunction]
pub fn read_clpol_alpha_str<'py>(py: Python<'py>, text: &str) -> PyResult<Vec<Bound<'py, PyDict>>> {
    let rows = molrs::io::read_clpol_alpha_str(text).map_err(PyValueError::new_err)?;
    row_dicts(py, rows)
}

/// Read the ``alpha.ff`` file at ``path`` — :func:`read_clpol_alpha_str` over
/// its text.
///
/// Raises
/// ------
/// ValueError
///     The file cannot be read, or a row's numeric fields do not parse.
#[pyfunction]
pub fn read_clpol_alpha(py: Python<'_>, path: PathBuf) -> PyResult<Vec<Bound<'_, PyDict>>> {
    let rows = molrs::io::read_clpol_alpha(path_str(&path)?).map_err(PyValueError::new_err)?;
    row_dicts(py, rows)
}

/// Register this module's functions.
pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    for door in [
        wrap_pyfunction!(read_clpol_alpha, m)?,
        wrap_pyfunction!(read_clpol_alpha_str, m)?,
    ] {
        crate::add_function(m, "molrs.io", door)?;
    }
    Ok(())
}
