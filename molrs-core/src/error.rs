//! Unified error types for the molrs library.

use std::fmt;
use std::io;

use crate::block::BlockError;

/// Main error type for the molrs library.
#[derive(Debug)]
pub enum MolRsError {
    /// Error from Block operations
    Block(BlockError),

    /// IO error (file reading/writing)
    Io(io::Error),

    /// Parse error with context
    Parse {
        /// Line number where error occurred (if applicable)
        line: Option<usize>,
        /// Error message
        message: String,
    },

    /// Validation error
    Validation {
        /// Error message
        message: String,
    },

    /// Zarr I/O error (only with "zarr" feature)
    #[cfg(feature = "zarr")]
    Zarr {
        /// Error message
        message: String,
    },

    /// Entity not found (atom, bond, angle, dihedral)
    NotFound {
        /// Kind of entity ("atom", "bond", "angle", "dihedral")
        entity: &'static str,
        /// Human-readable message
        message: String,
    },

    /// Generic error with message
    Other(String),
}

impl fmt::Display for MolRsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MolRsError::Block(e) => write!(f, "Block error: {}", e),
            MolRsError::Io(e) => write!(f, "IO error: {}", e),
            MolRsError::Parse {
                line: Some(line),
                message,
            } => {
                write!(f, "Parse error at line {}: {}", line, message)
            }
            MolRsError::Parse {
                line: None,
                message,
            } => {
                write!(f, "Parse error: {}", message)
            }
            MolRsError::Validation { message } => {
                write!(f, "Validation error: {}", message)
            }
            MolRsError::NotFound { entity, message } => {
                write!(f, "{} not found: {}", entity, message)
            }
            #[cfg(feature = "zarr")]
            MolRsError::Zarr { message } => {
                write!(f, "Zarr error: {}", message)
            }
            MolRsError::Other(msg) => write!(f, "{}", msg),
        }
    }
}

impl std::error::Error for MolRsError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            MolRsError::Block(e) => Some(e),
            MolRsError::Io(e) => Some(e),
            _ => None,
        }
    }
}

// Automatic conversions
impl From<BlockError> for MolRsError {
    fn from(err: BlockError) -> Self {
        MolRsError::Block(err)
    }
}

impl From<io::Error> for MolRsError {
    fn from(err: io::Error) -> Self {
        MolRsError::Io(err)
    }
}

impl From<String> for MolRsError {
    fn from(msg: String) -> Self {
        MolRsError::Other(msg)
    }
}

impl From<&str> for MolRsError {
    fn from(msg: &str) -> Self {
        MolRsError::Other(msg.to_string())
    }
}

// Helper constructors
impl MolRsError {
    /// Create a parse error with line number
    pub fn parse_error(line: usize, message: impl Into<String>) -> Self {
        MolRsError::Parse {
            line: Some(line),
            message: message.into(),
        }
    }

    /// Create a parse error without line number
    pub fn parse(message: impl Into<String>) -> Self {
        MolRsError::Parse {
            line: None,
            message: message.into(),
        }
    }

    /// Create a validation error
    pub fn validation(message: impl Into<String>) -> Self {
        MolRsError::Validation {
            message: message.into(),
        }
    }

    /// Create a not-found error
    pub fn not_found(entity: &'static str, message: impl Into<String>) -> Self {
        MolRsError::NotFound {
            entity,
            message: message.into(),
        }
    }

    /// Create a Zarr I/O error
    #[cfg(feature = "zarr")]
    pub fn zarr(message: impl Into<String>) -> Self {
        MolRsError::Zarr {
            message: message.into(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::block::BlockError;

    #[test]
    fn test_error_display() {
        let err = MolRsError::parse_error(42, "unexpected token");
        assert_eq!(
            format!("{}", err),
            "Parse error at line 42: unexpected token"
        );

        let err = MolRsError::parse("invalid format");
        assert_eq!(format!("{}", err), "Parse error: invalid format");

        let err = MolRsError::validation("inconsistent dimensions");
        assert_eq!(
            format!("{}", err),
            "Validation error: inconsistent dimensions"
        );
    }

    #[test]
    fn test_from_block_error() {
        let block_err = BlockError::RankZero {
            key: "test".to_string(),
        };
        let err: MolRsError = block_err.into();
        assert!(matches!(err, MolRsError::Block(_)));
    }

    #[test]
    fn test_from_string() {
        let err: MolRsError = "test error".into();
        assert!(matches!(err, MolRsError::Other(_)));
        assert_eq!(format!("{}", err), "test error");
    }
}
