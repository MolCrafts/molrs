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
    /// A canonical column key was written at a dtype the vocabulary does not
    /// allow.
    ///
    /// The vocabulary binds a key wherever it appears, so this fires without
    /// the block knowing its own name. A column's dtype is fixed by its first
    /// write and molrs refuses to coerce, so accepting the wrong dtype here
    /// means the *next* correct write silently fails to land.
    SchemaDtype {
        /// The canonical key.
        key: String,
        /// Dtype the vocabulary declares.
        expected: crate::store::block::DType,
        /// Dtype the caller supplied.
        got: crate::store::block::DType,
    },
    /// A canonical column key was written at a shape the vocabulary does not
    /// allow.
    SchemaShape {
        /// The canonical key.
        key: String,
        /// Shape the vocabulary declares.
        expected: crate::store::schema::ColShape,
        /// Shape the caller supplied.
        got: Vec<usize>,
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
            BlockError::SchemaDtype { key, expected, got } => write!(
                f,
                "column '{key}' is declared '{expected}' by the Frame schema, got '{got}'"
            ),
            BlockError::SchemaShape { key, expected, got } => write!(
                f,
                "column '{key}' is declared {expected} by the Frame schema, got {got:?}"
            ),
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
