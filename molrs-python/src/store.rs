//! Error conversion and re-exports for the shared FFI store.
//!
//! All [`PyFrame`](crate::frame::PyFrame) and
//! [`PyBlock`](crate::block::PyBlock) instances hold the canonical
//! [`molrs_ffi::SharedStore`] (an `Rc<RefCell<Store>>`) via
//! [`FrameRef`](molrs_ffi::FrameRef) / [`BlockRef`](molrs_ffi::BlockRef).
//! This module only re-exports the alias and maps FFI errors onto
//! Python exceptions.

pub(crate) use molrs_ffi::SharedStore;

use molrs_ffi::FfiError;
use pyo3::exceptions::{PyKeyError, PyRuntimeError, PyTypeError, PyValueError};
use pyo3::prelude::*;

/// Convert an [`FfiError`] to the most appropriate Python exception.
///
/// | Variant               | Python exception   |
/// |-----------------------|--------------------|
/// | `InvalidFrameId`      | `RuntimeError`     |
/// | `InvalidBlockHandle`  | `RuntimeError`     |
/// | `KeyNotFound`         | `KeyError`         |
/// | `NonContiguous`       | `ValueError`       |
/// | `DTypeMismatch`       | `TypeError`        |
pub(crate) fn ffi_error_to_pyerr(err: FfiError) -> PyErr {
    match err {
        FfiError::InvalidFrameId => PyRuntimeError::new_err("invalid frame handle"),
        FfiError::InvalidBlockHandle => PyRuntimeError::new_err("invalid block handle"),
        FfiError::KeyNotFound { key } => PyKeyError::new_err(key),
        FfiError::NonContiguous { key } => {
            PyValueError::new_err(format!("column '{key}' is not contiguous in memory"))
        }
        FfiError::DTypeMismatch {
            key,
            expected,
            actual,
        } => PyTypeError::new_err(format!(
            "column '{key}' has dtype {actual} but expected {expected}"
        )),
    }
}
