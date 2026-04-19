use std::fmt;

use molrs::MolRsError;

use crate::graph::NodeId;

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
    DimensionMismatch {
        expected: usize,
        got: usize,
        what: &'static str,
    },

    /// A composite value has the wrong shape (e.g. ragged matrix).
    BadShape { expected: String, got: String },

    /// Non-finite value (NaN / ±Inf) at the given flat index of the named field.
    NonFinite { where_: &'static str, index: usize },

    /// Scalar input out of its valid range.
    OutOfRange { field: &'static str, value: String },

    /// `frames` slice is empty but the compute needs at least one frame.
    EmptyInput,

    /// Graph DAG contains a cycle involving these nodes.
    CyclicDependency { nodes: Vec<NodeId> },

    /// `Inputs` did not bind a value for this input slot.
    MissingInput { slot: NodeId },

    /// Value stored at a slot has a different type than the accessor expects.
    TypeMismatch {
        slot: NodeId,
        expected: &'static str,
        got: &'static str,
    },

    /// A compute node failed; `source` carries the original error, `node_id`
    /// identifies which node in the Graph raised it.
    Node {
        node_id: NodeId,
        source: Box<ComputeError>,
    },

    /// Forwarded from molrs-core.
    MolRs(MolRsError),
}

impl fmt::Display for ComputeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MissingBlock { name } => write!(f, "missing block '{name}' in Frame"),
            Self::MissingColumn { block, col } => {
                write!(f, "missing column '{col}' in block '{block}'")
            }
            Self::MissingSimBox => write!(f, "Frame has no SimBox"),
            Self::DimensionMismatch {
                expected,
                got,
                what,
            } => write!(
                f,
                "{what} dimension mismatch: expected {expected}, got {got}"
            ),
            Self::BadShape { expected, got } => {
                write!(f, "bad shape: expected {expected}, got {got}")
            }
            Self::NonFinite { where_, index } => {
                write!(f, "non-finite value in {where_} at index {index}")
            }
            Self::OutOfRange { field, value } => {
                write!(f, "{field} out of range: {value}")
            }
            Self::EmptyInput => write!(f, "empty frames slice"),
            Self::CyclicDependency { nodes } => {
                write!(f, "cyclic dependency among nodes {nodes:?}")
            }
            Self::MissingInput { slot } => {
                write!(f, "input slot {slot:?} not bound in Inputs")
            }
            Self::TypeMismatch {
                slot,
                expected,
                got,
            } => write!(
                f,
                "type mismatch for slot {slot:?}: expected {expected}, got {got}"
            ),
            Self::Node { node_id, source } => {
                write!(f, "node {node_id:?} failed: {source}")
            }
            Self::MolRs(e) => write!(f, "{e}"),
        }
    }
}

impl std::error::Error for ComputeError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::MolRs(e) => Some(e),
            Self::Node { source, .. } => Some(source.as_ref()),
            _ => None,
        }
    }
}

impl From<MolRsError> for ComputeError {
    fn from(err: MolRsError) -> Self {
        Self::MolRs(err)
    }
}
