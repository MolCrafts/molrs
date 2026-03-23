//! Shared FFI store type alias and error conversion.
//!
//! All [`PyFrame`](crate::frame::PyFrame) and
//! [`PyBlock`](crate::block::PyBlock) instances hold an `Rc<RefCell<Store>>`
//! so that multiple Python wrappers can share the same backing store and
//! benefit from version-tracked invalidation.

use std::cell::RefCell;
use std::rc::Rc;

use molrs_ffi::{FfiError, Store as FfiStore};
use pyo3::exceptions::{PyKeyError, PyRuntimeError, PyValueError};
use pyo3::prelude::*;

/// Reference-counted, interiorly mutable handle to the FFI `Store`.
pub(crate) type SharedStore = Rc<RefCell<FfiStore>>;

/// Convert an [`FfiError`] to the most appropriate Python exception.
///
/// | Variant               | Python exception   |
/// |-----------------------|--------------------|
/// | `InvalidFrameId`      | `RuntimeError`     |
/// | `InvalidBlockHandle`  | `RuntimeError`     |
/// | `KeyNotFound`         | `KeyError`         |
/// | `NonContiguous`       | `ValueError`       |
pub(crate) fn ffi_error_to_pyerr(err: FfiError) -> PyErr {
    match err {
        FfiError::InvalidFrameId => PyRuntimeError::new_err("invalid frame handle"),
        FfiError::InvalidBlockHandle => PyRuntimeError::new_err("invalid block handle"),
        FfiError::KeyNotFound { key } => PyKeyError::new_err(key),
        FfiError::NonContiguous { key } => {
            PyValueError::new_err(format!("column '{key}' is not contiguous in memory"))
        }
    }
}
