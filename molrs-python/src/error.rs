//! Public molrs exception types and helpers.
//!
//! `BlockDtypeError` is the single error a column write raises when the value
//! is not numpy-representable by the Rust Store (object dtype, None-bearing, or
//! ragged/mixed). It subclasses Python `TypeError` so downstream code can
//! `except molrs.BlockDtypeError` precisely while still being caught by broad
//! `except TypeError` handlers.

use pyo3::create_exception;
use pyo3::prelude::*;

create_exception!(
    molrs,
    BlockDtypeError,
    pyo3::exceptions::PyTypeError,
    "Raised when a Block column value is not a numpy-representable dtype \
     (object, None-bearing, or ragged/mixed). The Rust Store holds only \
     float / int / bool / str columns."
);

create_exception!(
    molrs,
    UnitsError,
    pyo3::exceptions::PyValueError,
    "Raised when unit parsing, definition, arithmetic, or conversion fails."
);

/// Preserve the native units error message at the Python boundary.
pub fn units_error(error: ::molrs::units::UnitsError) -> PyErr {
    UnitsError::new_err(error.to_string())
}

/// Build a column-named `BlockDtypeError` for a rejected array.
///
/// Best-effort introspects the offending array to name the detected dtype and,
/// for object arrays, whether it is None-bearing (vs ragged/mixed) so the
/// message guides the caller to coerce or drop the column.
pub fn dtype_reject(key: &str, array: &Bound<'_, PyAny>) -> PyErr {
    let dtype = array
        .getattr("dtype")
        .and_then(|d| d.str())
        .map(|s| s.to_string())
        .unwrap_or_else(|_| "<unknown>".to_string());

    let kind = array
        .getattr("dtype")
        .and_then(|d| d.getattr("kind"))
        .and_then(|k| k.extract::<String>())
        .unwrap_or_default();

    let mut reason = format!("column '{key}' has unsupported dtype '{dtype}'");
    if kind == "O" {
        if object_array_is_none_bearing(array) {
            reason.push_str("; the object array is None-bearing");
        } else {
            reason.push_str("; the object array is ragged/mixed");
        }
    }
    reason.push_str(
        ". Coerce to a numpy float/int/bool/str column or drop the column \
         (the Rust Store has no object-column overflow).",
    );
    BlockDtypeError::new_err(reason)
}

/// True if any element of a flattened object array is Python `None`.
fn object_array_is_none_bearing(array: &Bound<'_, PyAny>) -> bool {
    let Ok(flat) = array.call_method0("ravel") else {
        return false;
    };
    let Ok(iter) = flat.try_iter() else {
        return false;
    };
    iter.flatten().any(|item| item.is_none())
}
