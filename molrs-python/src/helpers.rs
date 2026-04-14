//! Shared helper functions and type aliases for PyO3 bindings.
//!
//! This module provides error conversion functions that map Rust error types
//! to appropriate Python exceptions, and the [`NpF`] type alias that matches
//! the crate's float precision setting.

use molrs::region::simbox::BoxError;
use molrs::types::F;
use ndarray::{Array1, array};
use numpy::PyReadonlyArray1;
use pyo3::exceptions::{PyIOError, PyRuntimeError, PyValueError};
use pyo3::prelude::*;

/// Numpy float type matching the `F` alias — always `f64`.
pub type NpF = f64;

/// Parse an optional origin array, defaulting to `[0, 0, 0]`.
///
/// # Errors
///
/// Returns `PyValueError` if the array does not have exactly 3 elements.
pub fn parse_origin(origin: Option<PyReadonlyArray1<'_, NpF>>) -> PyResult<Array1<F>> {
    match origin {
        Some(o) => {
            let s = o.as_slice()?;
            if s.len() != 3 {
                return Err(PyValueError::new_err("origin must have length 3"));
            }
            Ok(array![s[0], s[1], s[2]])
        }
        None => Ok(array![0.0 as F, 0.0 as F, 0.0 as F]),
    }
}

/// Parse an optional PBC flag array, defaulting to `[true, true, true]`.
///
/// # Errors
///
/// Returns `PyValueError` if the array does not have exactly 3 elements.
pub fn parse_pbc(pbc: Option<PyReadonlyArray1<'_, bool>>) -> PyResult<[bool; 3]> {
    match pbc {
        Some(p) => {
            let s = p.as_slice()?;
            if s.len() != 3 {
                return Err(PyValueError::new_err("pbc must have 3 elements"));
            }
            Ok([s[0], s[1], s[2]])
        }
        None => Ok([true, true, true]),
    }
}

/// Convert a [`BoxError`] to a Python `ValueError`.
pub fn box_error_to_pyerr(e: BoxError) -> PyErr {
    PyValueError::new_err(format!("{:?}", e))
}

/// Convert a [`std::io::Error`] to a Python `IOError`.
pub fn io_error_to_pyerr(e: std::io::Error) -> PyErr {
    PyIOError::new_err(e.to_string())
}

/// Convert a [`molrs_pack::PackError`] to a Python `RuntimeError`.
pub fn pack_error_to_pyerr(e: molrs_pack::PackError) -> PyErr {
    PyRuntimeError::new_err(e.to_string())
}

/// Convert a [`molrs::MolRsError`] to a Python `ValueError`.
pub fn molrs_error_to_pyerr(e: molrs::MolRsError) -> PyErr {
    PyValueError::new_err(e.to_string())
}

/// Convert a [`molrs_smiles::SmilesError`] to a Python `ValueError`.
pub fn smiles_error_to_pyerr(e: molrs_smiles::SmilesError) -> PyErr {
    PyValueError::new_err(e.to_string())
}
