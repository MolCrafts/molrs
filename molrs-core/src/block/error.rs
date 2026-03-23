//! Error types for Block operations.

use std::fmt;

/// Errors that can occur when manipulating a [`Block`](super::Block).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BlockError {
    /// Inserted array has rank 0 (no axis-0 length can be defined)
    RankZero {
        /// The key for which the error occurred
        key: String,
    },
    /// Inserted array's axis-0 length does not match the Block's `nrows`
    RaggedAxis0 {
        /// The key for which the mismatch was detected
        key: String,
        /// The axis-0 length the block expects
        expected: usize,
        /// The axis-0 length provided by the inserted array
        got: usize,
    },
    /// General validation error
    Validation {
        /// Error message
        message: String,
    },
}

impl fmt::Display for BlockError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BlockError::RankZero { key } => {
                write!(
                    f,
                    "array for key '{}' has rank 0; expected at least 1D",
                    key
                )
            }
            BlockError::RaggedAxis0 { key, expected, got } => write!(
                f,
                "array for key '{}' has axis-0 length {} but block expects {}",
                key, got, expected
            ),
            BlockError::Validation { message } => write!(f, "{}", message),
        }
    }
}

impl std::error::Error for BlockError {}

impl BlockError {
    /// Creates a validation error with the given message.
    pub fn validation(message: impl Into<String>) -> Self {
        BlockError::Validation {
            message: message.into(),
        }
    }
}
