//! Error types and thread-local error buffer for the C API.
//!
//! Every `extern "C"` function returns a [`MolrsStatus`] code.  On
//! failure, a human-readable message is stored in a thread-local buffer
//! and can be retrieved via [`crate::molrs_last_error`].
//!
//! Column data types are exposed to C as [`MolrsDType`] discriminants
//! that map one-to-one to the internal [`molrs::block::DType`] enum.

use std::cell::RefCell;
use std::ffi::c_char;

use molrs::block::DType;
use molrs_ffi::FfiError;

/// Status codes returned by every `extern "C"` function.
///
/// A value of `Ok` (0) indicates success.  All other values are errors.
/// Call [`molrs_last_error`](crate::molrs_last_error) to obtain a
/// human-readable description of the most recent error.
///
/// # C mapping
///
/// In the generated header this is `enum MolrsStatus` with the same
/// integer discriminants listed below.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MolrsStatus {
    /// Operation completed successfully.
    Ok = 0,
    /// The supplied `MolrsFrameHandle` does not refer to a live frame.
    InvalidFrameHandle = 1,
    /// The supplied `MolrsBlockHandle` does not refer to a live block,
    /// or its version has been invalidated.
    InvalidBlockHandle = 2,
    /// The supplied `MolrsSimBoxHandle` does not refer to a live SimBox.
    InvalidSimBoxHandle = 3,
    /// The supplied `MolrsForceFieldHandle` does not refer to a live
    /// force field.
    InvalidForceFieldHandle = 4,
    /// A requested key (block name, column name, metadata key) was not
    /// found.
    KeyNotFound = 5,
    /// A column's internal storage is not contiguous in memory, so a
    /// zero-copy pointer cannot be returned.
    NonContiguous = 6,
    /// A function argument has an invalid value (e.g. zero-length shape,
    /// out-of-range index, buffer too small).
    InvalidArgument = 7,
    /// A required pointer argument was `NULL`.
    NullPointer = 8,
    /// The column's data type does not match the requested accessor
    /// (e.g. asking for `float` on an `int` column).
    TypeMismatch = 9,
    /// An unexpected internal error (e.g. a caught panic).
    InternalError = 10,
    /// A C string argument was not valid UTF-8.
    Utf8Error = 11,
    /// The provided 3x3 cell matrix is singular (determinant is zero)
    /// and cannot form a valid simulation box.
    SingularCell = 12,
    /// A parse error occurred (e.g. invalid SMILES or JSON string).
    ParseError = 13,
}

/// Data type discriminants for Block columns.
///
/// Each column in a [`Block`](molrs::block::Block) stores a
/// homogeneously-typed ndarray.  This enum tells C callers which
/// accessor family to use (`molrs_block_get_F`, `molrs_block_get_I`,
/// `molrs_block_get_U`, etc.).
///
/// # C mapping
///
/// | Discriminant | C type (`default` features)        |
/// |--------------|------------------------------------|
/// | `Float` (0)  | `molrs_float_t` (`float` / `double`)|
/// | `Int` (1)    | `molrs_int_t` (`int32_t` / `int64_t`)|
/// | `Bool` (2)   | `bool`                             |
/// | `UInt` (3)   | `molrs_uint_t` (`uint32_t` / `uint64_t`)|
/// | `String` (4) | (not directly accessible via pointer; use dedicated string APIs) |
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MolrsDType {
    /// Floating-point column (`molrs_float_t`).
    Float = 0,
    /// Signed integer column (`molrs_int_t`).
    Int = 1,
    /// Boolean column.
    Bool = 2,
    /// Unsigned integer column (`molrs_uint_t`).
    UInt = 3,
    /// String column (no zero-copy pointer access).
    String = 4,
}

impl From<DType> for MolrsDType {
    fn from(dt: DType) -> Self {
        match dt {
            DType::Float => Self::Float,
            DType::Int => Self::Int,
            DType::Bool => Self::Bool,
            DType::UInt => Self::UInt,
            DType::U8 => unreachable!("U8 columns are not exposed via the C API"),
            DType::String => Self::String,
        }
    }
}

// Thread-local error buffer: null-terminated bytes.
thread_local! {
    static LAST_ERROR: RefCell<Vec<u8>> = const { RefCell::new(Vec::new()) };
}

/// Store an error message in the thread-local buffer.
pub(crate) fn set_last_error(msg: impl Into<String>) {
    LAST_ERROR.with(|e| {
        let s = msg.into();
        let mut buf = e.borrow_mut();
        buf.clear();
        buf.extend_from_slice(s.as_bytes());
        buf.push(0);
    });
}

/// Return a pointer to the last error message (null-terminated).
///
/// Valid until the next error is set on this thread.
pub(crate) fn last_error_ptr() -> *const c_char {
    LAST_ERROR.with(|e| {
        let buf = e.borrow();
        if buf.is_empty() {
            c"".as_ptr()
        } else {
            buf.as_ptr() as *const c_char
        }
    })
}

/// Convert an `FfiError` to a `MolrsStatus`, storing the message.
pub(crate) fn ffi_err_to_status(err: &FfiError) -> MolrsStatus {
    set_last_error(err.to_string());
    match err {
        FfiError::InvalidFrameId => MolrsStatus::InvalidFrameHandle,
        FfiError::InvalidBlockHandle => MolrsStatus::InvalidBlockHandle,
        FfiError::KeyNotFound { .. } => MolrsStatus::KeyNotFound,
        FfiError::NonContiguous { .. } => MolrsStatus::NonContiguous,
        FfiError::DTypeMismatch { .. } => MolrsStatus::TypeMismatch,
    }
}
