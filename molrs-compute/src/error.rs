use std::fmt;

use molrs::MolRsError;

/// Error type for compute operations.
#[derive(Debug)]
pub enum ComputeError {
    /// Required block not found in Frame.
    MissingBlock { name: &'static str },

    /// Required column not found in a block.
    MissingColumn {
        block: &'static str,
        col: &'static str,
    },

    /// Frame has no SimBox but the compute requires one.
    MissingSimBox,

    /// Array dimensions do not match expectations.
    DimensionMismatch { expected: usize, got: usize },

    /// Reducer queried before any frames were fed.
    NoFrames,

    /// Forwarded from molrs-core.
    MolRs(MolRsError),
}

impl fmt::Display for ComputeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MissingBlock { name } => {
                write!(f, "missing block '{name}' in Frame")
            }
            Self::MissingColumn { block, col } => {
                write!(f, "missing column '{col}' in block '{block}'")
            }
            Self::MissingSimBox => write!(f, "Frame has no SimBox"),
            Self::DimensionMismatch { expected, got } => {
                write!(f, "dimension mismatch: expected {expected}, got {got}")
            }
            Self::NoFrames => write!(f, "no frames have been fed to the reducer"),
            Self::MolRs(e) => write!(f, "{e}"),
        }
    }
}

impl std::error::Error for ComputeError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::MolRs(e) => Some(e),
            _ => None,
        }
    }
}

impl From<MolRsError> for ComputeError {
    fn from(err: MolRsError) -> Self {
        Self::MolRs(err)
    }
}
