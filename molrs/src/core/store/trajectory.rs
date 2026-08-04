//! Backend-agnostic trajectory + observable logical model.
//!
//! `Trajectory` is a plain frame-sequence carrier (frames plus optional
//! step/time index arrays); `ObservableRecord` is a named, typed observable
//! payload. Neither is a record aggregate — the canonical entity is `Frame`.

use serde::{Deserialize, Serialize};
use serde_json::{Map as JsonMap, Value as JsonValue};

use crate::MolRsError;
use crate::store::block::Column;
use crate::store::frame::Frame;
use crate::types::F;

/// Hierarchical schema node used by observable/provenance metadata.
pub type SchemaValue = JsonValue;

/// Trajectory-like list of frame states plus shared indexing arrays.
#[derive(Debug, Clone, Default)]
pub struct Trajectory {
    /// Ordered frame-like states.
    pub frames: Vec<Frame>,
    /// Optional discrete step indices.
    pub step: Option<Vec<i64>>,
    /// Optional physical time values.
    pub time: Option<Vec<F>>,
}

impl Trajectory {
    /// Create an empty trajectory.
    pub fn new() -> Self {
        Self::default()
    }

    /// Build a trajectory from frame states.
    pub fn from_frames(frames: Vec<Frame>) -> Self {
        Self {
            frames,
            step: None,
            time: None,
        }
    }

    /// Number of states.
    pub fn len(&self) -> usize {
        self.frames.len()
    }

    /// Returns true when no states are stored.
    pub fn is_empty(&self) -> bool {
        self.frames.is_empty()
    }

    /// Validate shared axis lengths.
    pub fn validate(&self) -> Result<(), MolRsError> {
        let n = self.frames.len();
        if let Some(step) = &self.step
            && step.len() != n
        {
            return Err(MolRsError::validation(format!(
                "trajectory.step length mismatch: expected {}, got {}",
                n,
                step.len()
            )));
        }
        if let Some(time) = &self.time
            && time.len() != n
        {
            return Err(MolRsError::validation(format!(
                "trajectory.time length mismatch: expected {}, got {}",
                n,
                time.len()
            )));
        }
        Ok(())
    }
}

/// Observable kind aligned with the observable metadata contract.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ObservableKind {
    Scalar,
    Vector,
}

impl ObservableKind {
    /// Contract spelling of the kind, as written to `observables/meta/<name>/kind`.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Scalar => "scalar",
            Self::Vector => "vector",
        }
    }

    /// Parse the contract spelling; `None` for a kind this build does not define.
    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "scalar" => Some(Self::Scalar),
            "vector" => Some(Self::Vector),
            _ => None,
        }
    }
}

/// Raw observable payload.
#[derive(Debug, Clone)]
pub enum ObservableData {
    /// Typed ndarray-style data.
    Column(Column),
}

/// Named observable with semantic metadata.
#[derive(Debug, Clone)]
pub struct ObservableRecord {
    pub name: String,
    pub kind: ObservableKind,
    pub description: String,
    pub time_dependent: bool,
    pub unit: Option<String>,
    pub axes: Vec<String>,
    pub sampling: Option<String>,
    pub domain: Option<String>,
    pub target: Option<String>,
    pub extra: JsonMap<String, JsonValue>,
    pub data: ObservableData,
}

impl ObservableRecord {
    /// Build a scalar observable.
    pub fn scalar(name: impl Into<String>, data: Column) -> Self {
        Self {
            name: name.into(),
            kind: ObservableKind::Scalar,
            description: String::new(),
            time_dependent: false,
            unit: None,
            axes: Vec::new(),
            sampling: None,
            domain: None,
            target: None,
            extra: JsonMap::new(),
            data: ObservableData::Column(data),
        }
    }

    /// Build a vector observable.
    pub fn vector(name: impl Into<String>, data: Column) -> Self {
        Self {
            name: name.into(),
            kind: ObservableKind::Vector,
            description: String::new(),
            time_dependent: false,
            unit: None,
            axes: Vec::new(),
            sampling: None,
            domain: None,
            target: None,
            extra: JsonMap::new(),
            data: ObservableData::Column(data),
        }
    }

    /// Validate the observable payload against the declared kind.
    pub fn validate(&self) -> Result<(), MolRsError> {
        match (&self.kind, &self.data) {
            (ObservableKind::Scalar | ObservableKind::Vector, ObservableData::Column(_)) => Ok(()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_trajectory_has_no_frames() {
        let traj = Trajectory::new();
        assert_eq!(traj.len(), 0);
        assert!(traj.is_empty());
    }

    #[test]
    fn from_frames_counts_states() {
        let traj = Trajectory::from_frames(vec![Frame::new(), Frame::new()]);
        assert_eq!(traj.len(), 2);
        assert!(!traj.is_empty());
        traj.validate().unwrap();
    }

    #[test]
    fn mismatched_step_axis_fails_validation() {
        let mut traj = Trajectory::from_frames(vec![Frame::new(), Frame::new()]);
        traj.step = Some(vec![0]); // length 1 != 2 frames
        assert!(traj.validate().is_err());
    }
}
