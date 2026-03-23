//! Error types for the FFI layer.

use std::fmt;

/// Errors that can occur in FFI operations.
#[derive(Debug, Clone)]
pub enum FfiError {
    /// Frame ID is invalid (frame was dropped or never existed)
    InvalidFrameId,

    /// Block handle is invalid (block was removed/replaced or frame gone)
    InvalidBlockHandle,

    /// Key not found in frame or block
    KeyNotFound { key: String },

    /// Array is not contiguous (zero-copy view not possible)
    NonContiguous { key: String },
}

impl fmt::Display for FfiError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            FfiError::InvalidFrameId => write!(f, "Invalid frame ID"),
            FfiError::InvalidBlockHandle => write!(f, "Invalid block handle"),
            FfiError::KeyNotFound { key } => write!(f, "Key '{}' not found", key),
            FfiError::NonContiguous { key } => {
                write!(f, "Column '{}' is not contiguous in memory", key)
            }
        }
    }
}

impl std::error::Error for FfiError {}
