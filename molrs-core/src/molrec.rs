//! Backend-agnostic MolRec logical model.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use serde_json::{Map as JsonMap, Value as JsonValue};

use crate::MolRsError;
use crate::block::Column;
use crate::frame::Frame;
use crate::grid::Grid;
use crate::types::F;

/// Hierarchical schema node used by `meta`, `method`, and `parameters`.
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

/// Observable kind aligned with the MolRec metadata contract.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ObservableKind {
    Scalar,
    Vector,
    Grid,
}

/// Raw observable payload.
#[derive(Debug, Clone)]
pub enum ObservableData {
    /// Typed ndarray-style data.
    Column(Column),
    /// Volumetric grid data.
    Grid(Grid),
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

    /// Build a grid observable.
    pub fn grid(name: impl Into<String>, grid: Grid) -> Self {
        Self {
            name: name.into(),
            kind: ObservableKind::Grid,
            description: String::new(),
            time_dependent: false,
            unit: None,
            axes: Vec::new(),
            sampling: None,
            domain: None,
            target: None,
            extra: JsonMap::new(),
            data: ObservableData::Grid(grid),
        }
    }

    /// Validate the observable payload against the declared kind.
    pub fn validate(&self) -> Result<(), MolRsError> {
        match (&self.kind, &self.data) {
            (ObservableKind::Scalar | ObservableKind::Vector, ObservableData::Column(_)) => Ok(()),
            (ObservableKind::Grid, ObservableData::Grid(_)) => Ok(()),
            _ => Err(MolRsError::validation("observable kind/data mismatch")),
        }
    }
}

/// Single logical MolRec object.
#[derive(Debug, Clone)]
pub struct MolRec {
    /// Record-level metadata.
    pub meta: SchemaValue,
    /// Canonical frame.
    pub frame: Frame,
    /// Optional trajectory states.
    pub trajectory: Option<Trajectory>,
    /// Named observables.
    pub observables: BTreeMap<String, ObservableRecord>,
    /// Method-level metadata.
    pub method: SchemaValue,
    /// Parameter metadata.
    pub parameters: SchemaValue,
}

impl Default for MolRec {
    fn default() -> Self {
        Self::new(Frame::new())
    }
}

impl MolRec {
    /// Create a MolRec around one canonical frame.
    pub fn new(frame: Frame) -> Self {
        Self {
            meta: empty_object(),
            frame,
            trajectory: None,
            observables: BTreeMap::new(),
            method: empty_object(),
            parameters: empty_object(),
        }
    }

    /// Build a MolRec from one canonical frame plus explicit trajectory states.
    pub fn from_frames(frame: Frame, frames: Vec<Frame>) -> Self {
        let mut rec = Self::new(frame);
        let trajectory = Trajectory::from_frames(frames);
        rec.trajectory = Some(trajectory);
        rec
    }

    /// Build a MolRec from a dedicated trajectory object.
    ///
    /// The canonical frame defaults to the first trajectory frame.
    pub fn from_trajectory(trajectory: Trajectory) -> Result<Self, MolRsError> {
        trajectory.validate()?;
        let Some(frame) = trajectory.frames.first().cloned() else {
            return Err(MolRsError::validation(
                "cannot build MolRec from an empty trajectory",
            ));
        };
        let mut rec = Self::new(frame);
        rec.trajectory = Some(trajectory);
        Ok(rec)
    }

    /// Total number of accessible frames.
    pub fn count_frames(&self) -> usize {
        match &self.trajectory {
            Some(traj) if !traj.frames.is_empty() => traj.frames.len(),
            _ => 1,
        }
    }

    /// Return one accessible frame.
    pub fn frame_at(&self, index: usize) -> Option<Frame> {
        match &self.trajectory {
            Some(traj) if !traj.frames.is_empty() => traj.frames.get(index).cloned(),
            _ if index == 0 => Some(self.frame.clone()),
            _ => None,
        }
    }

    /// Replace the canonical frame.
    pub fn set_frame(&mut self, frame: Frame) {
        self.frame = frame;
    }

    /// Append one frame to the trajectory, creating it if needed.
    pub fn add_frame(&mut self, frame: Frame) {
        match &mut self.trajectory {
            Some(traj) => traj.frames.push(frame),
            None => {
                self.trajectory = Some(Trajectory::from_frames(vec![frame]));
            }
        }
    }

    /// Replace the trajectory.
    pub fn set_trajectory(&mut self, trajectory: Option<Trajectory>) {
        self.trajectory = trajectory;
    }

    /// Insert or replace an observable by name.
    pub fn add_observable(&mut self, observable: ObservableRecord) -> Option<ObservableRecord> {
        self.observables.insert(observable.name.clone(), observable)
    }

    /// Borrow an observable by name.
    pub fn get_observable(&self, name: &str) -> Option<&ObservableRecord> {
        self.observables.get(name)
    }

    /// Remove an observable by name.
    pub fn remove_observable(&mut self, name: &str) -> Option<ObservableRecord> {
        self.observables.remove(name)
    }
}

fn empty_object() -> JsonValue {
    JsonValue::Object(JsonMap::new())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn static_molrec_counts_one_frame() {
        let rec = MolRec::new(Frame::new());
        assert_eq!(rec.count_frames(), 1);
        assert!(rec.frame_at(0).is_some());
        assert!(rec.frame_at(1).is_none());
    }

    #[test]
    fn from_trajectory_uses_first_frame_as_canonical() {
        let mut traj = Trajectory::new();
        traj.frames.push(Frame::new());
        traj.frames.push(Frame::new());
        let rec = MolRec::from_trajectory(traj).unwrap();
        assert_eq!(rec.count_frames(), 2);
    }
}
