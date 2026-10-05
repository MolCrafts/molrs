//! Error types and thread-local error buffer for the C API.
//!
//! Every `extern "C"` function returns a [`MolrsStatus`] code.  On
//! failure, a human-readable message is stored in a thread-local buffer
//! and can be retrieved via [`crate::molrs_last_error`].
//!
//! Column data types are exposed to C as [`MolrsDType`] discriminants
//! that map one-to-one to the internal [`molrs::store::block::DType`] enum.

use std::cell::RefCell;
use std::ffi::c_char;

use molrs::store::block::DType;
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
    /// The supplied `MolrsBoxHandle` does not refer to a live SimBox.
    InvalidBoxHandle = 3,
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
    /// The supplied `MolrsRegionHandle` does not refer to a live region.
    InvalidRegionHandle = 14,
}

/// Data type discriminants for Block columns.
///
/// Each column in a [`Block`](molrs::store::block::Block) stores a
/// homogeneously-typed ndarray.  This enum is the stored variant, not a
/// width bucket: an `i64` column is [`Int64`](Self::Int64), not [`Int`](Self::Int).
/// Discriminants 0–4 stay where they were; later variants are appended.
///
/// A string column has no flat scalar buffer. [`molrs_block_get`](crate::molrs_block_get)
/// reports that as `TypeMismatch`, not as a missing key.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MolrsDType {
    /// `f64` column.
    Float = 0,
    /// `i32` column.
    Int = 1,
    /// Boolean column, one byte per element.
    Bool = 2,
    /// `u64` column.
    UInt = 3,
    /// String column. No flat scalar buffer.
    String = 4,
    /// `i8` column.
    Int8 = 5,
    /// `i16` column.
    Int16 = 6,
    /// `i64` column.
    Int64 = 7,
    /// `u8` column.
    U8 = 8,
    /// `u16` column.
    UInt16 = 9,
    /// `u32` column.
    UInt32 = 10,
    /// `complex64` column, a pair of `f32` per element.
    Complex64 = 11,
    /// `complex128` column, a pair of `f64` per element.
    Complex128 = 12,
}

impl From<DType> for MolrsDType {
    fn from(dt: DType) -> Self {
        match dt {
            DType::Float => Self::Float,
            DType::Int => Self::Int,
            DType::Bool => Self::Bool,
            DType::UInt => Self::UInt,
            DType::String => Self::String,
            DType::Int8 => Self::Int8,
            DType::Int16 => Self::Int16,
            DType::Int64 => Self::Int64,
            DType::U8 => Self::U8,
            DType::UInt16 => Self::UInt16,
            DType::UInt32 => Self::UInt32,
            DType::Complex64 => Self::Complex64,
            DType::Complex128 => Self::Complex128,
            // `DType` is `non_exhaustive`. Every variant that exists today is
            // named above; a future one must not be reported as `String`.
            other => unreachable!("no C dtype for {other:?}"),
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dtype_reports_the_stored_variant() {
        assert_eq!(MolrsDType::from(DType::Float) as u8, 0);
        assert_eq!(MolrsDType::from(DType::Int) as u8, 1);
        assert_eq!(MolrsDType::from(DType::Bool) as u8, 2);
        assert_eq!(MolrsDType::from(DType::UInt) as u8, 3);
        assert_eq!(MolrsDType::from(DType::String) as u8, 4);
        assert_eq!(MolrsDType::from(DType::Int8), MolrsDType::Int8);
        assert_eq!(MolrsDType::from(DType::Int16), MolrsDType::Int16);
        assert_eq!(MolrsDType::from(DType::Int64), MolrsDType::Int64);
        assert_eq!(MolrsDType::from(DType::U8), MolrsDType::U8);
        assert_eq!(MolrsDType::from(DType::UInt16), MolrsDType::UInt16);
        assert_eq!(MolrsDType::from(DType::UInt32), MolrsDType::UInt32);
        assert_eq!(MolrsDType::from(DType::Complex64), MolrsDType::Complex64);
        assert_eq!(MolrsDType::from(DType::Complex128), MolrsDType::Complex128);
        assert_ne!(MolrsDType::from(DType::Int64), MolrsDType::Int);
        assert_ne!(MolrsDType::from(DType::U8), MolrsDType::UInt);
        assert_ne!(MolrsDType::from(DType::Complex64), MolrsDType::Float);
        assert_ne!(MolrsDType::from(DType::Complex128), MolrsDType::Float);
    }
}
