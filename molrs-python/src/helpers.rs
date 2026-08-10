//! Shared helper functions and type aliases for PyO3 bindings.
//!
//! This module provides error conversion functions that map Rust error types
//! to appropriate Python exceptions, and the [`NpF`] type alias that matches
//! the crate's float precision setting.

use molrs::spatial::simbox::BoxError;
use molrs::types::F;
use ndarray::{Array1, array};
use numpy::PyReadonlyArray1;
use pyo3::exceptions::{PyIOError, PyValueError};
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

/// Convert a [`molrs::MolRsError`] to a Python `ValueError`.
pub fn molrs_error_to_pyerr(e: molrs::MolRsError) -> PyErr {
    PyValueError::new_err(e.to_string())
}

/// Convert a [`molrs::io::smiles::SmilesError`] to a Python `ValueError`.
pub fn smiles_error_to_pyerr(e: molrs::io::smiles::SmilesError) -> PyErr {
    PyValueError::new_err(e.to_string())
}

/// Convert any `Display` error (typically a `molrs-compute` / `molrs-signal`
/// analysis error) to a Python `ValueError`. Shared by the analysis bindings.
pub fn py_value_err<E: std::fmt::Display>(e: E) -> PyErr {
    PyValueError::new_err(e.to_string())
}

/// Resolve a wire-encoding name onto [`MessageFormat`].
///
/// The two spellings are the only ones the Rust side can produce, so an
/// unknown name is an error rather than a silent fall back to MessagePack —
/// a caller who writes `"messagepack"` must find out, not stream bytes the
/// peer will read as JSON.
pub(crate) fn message_format(name: &str) -> PyResult<molrs::stream::MessageFormat> {
    match name {
        "msgpack" => Ok(molrs::stream::MessageFormat::MessagePack),
        "json" => Ok(molrs::stream::MessageFormat::Json),
        other => Err(PyValueError::new_err(format!(
            "unknown wire format {other:?}; expected 'msgpack' or 'json'"
        ))),
    }
}

/// Collect owned core [`Frame`]s from a single `Frame` or a list of them.
/// Used by every batch-`compute` binding to accept both shapes.
///
/// [`Frame`]: molrs::store::frame::Frame
pub(crate) fn collect_frames(
    frames: &Bound<'_, PyAny>,
) -> PyResult<Vec<molrs::store::frame::Frame>> {
    use crate::core::store::frame::PyFrame;
    if let Ok(single) = frames.extract::<PyRef<'_, PyFrame>>() {
        return Ok(vec![single.clone_core_frame()?]);
    }
    let list: Vec<PyRef<'_, PyFrame>> = frames.extract()?;
    list.iter().map(|f| f.clone_core_frame()).collect()
}

/// Collect owned [`Neighbors`] tables from a single wrapper or a list of them.
///
/// The analyses take one materialized table per frame, so this is where the
/// binder accepts either shape. The engine that produced a table
/// (`NeighborList`) is deliberately not accepted: it would have to guess a
/// column policy, and a guess that drops `disp` is exactly the silent failure
/// this chain removed.
///
/// [`Neighbors`]: molrs::spatial::neighbors::Neighbors
pub(crate) fn collect_neighbors(
    arg: &Bound<'_, PyAny>,
) -> PyResult<Vec<molrs::spatial::neighbors::Neighbors>> {
    use crate::core::spatial::neighborlist::PyNeighbors;
    if let Ok(single) = arg.extract::<PyRef<'_, PyNeighbors>>() {
        return Ok(vec![single.inner.clone()]);
    }
    let list: Vec<PyRef<'_, PyNeighbors>> = arg.extract()?;
    Ok(list.iter().map(|n| n.inner.clone()).collect())
}
