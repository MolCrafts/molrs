//! Backend-agnostic MolRec logical model.

use std::collections::BTreeMap;

use ndarray::{Array1, Array2, array};
use serde::{Deserialize, Serialize};
use serde_json::{Map as JsonMap, Value as JsonValue, json};

use crate::MolRsError;
use crate::block::Column;
use crate::forcefield::ForceField;
use crate::frame::Frame;
use crate::region::simbox::SimBox;
use crate::types::F;

/// Hierarchical schema node used by `meta`, `method`, and `parameters`.
pub type SchemaValue = JsonValue;

/// One simulation box snapshot in record space.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CellBox {
    /// Cell vectors in Cartesian coordinates.
    pub vectors: [[F; 3]; 3],
    /// Cell origin.
    pub origin: [F; 3],
    /// Boundary semantics for x, y, z.
    pub boundary: [String; 3],
}

impl CellBox {
    /// Convert a molrs `SimBox` into a record box.
    pub fn from_simbox(simbox: &SimBox) -> Self {
        let h = simbox.h_view();
        let origin = simbox.origin_view();
        let pbc = simbox.pbc_view();
        Self {
            vectors: [
                [h[[0, 0]], h[[0, 1]], h[[0, 2]]],
                [h[[1, 0]], h[[1, 1]], h[[1, 2]]],
                [h[[2, 0]], h[[2, 1]], h[[2, 2]]],
            ],
            origin: [origin[0], origin[1], origin[2]],
            boundary: [
                boundary_label(pbc[0]),
                boundary_label(pbc[1]),
                boundary_label(pbc[2]),
            ],
        }
    }

    /// Convert a record box into a molrs `SimBox`.
    pub fn to_simbox(&self) -> Result<SimBox, MolRsError> {
        let h: Array2<F> = array![
            [self.vectors[0][0], self.vectors[0][1], self.vectors[0][2]],
            [self.vectors[1][0], self.vectors[1][1], self.vectors[1][2]],
            [self.vectors[2][0], self.vectors[2][1], self.vectors[2][2]],
        ];
        let origin: Array1<F> = array![self.origin[0], self.origin[1], self.origin[2]];
        let pbc = [
            is_periodic(&self.boundary[0]),
            is_periodic(&self.boundary[1]),
            is_periodic(&self.boundary[2]),
        ];
        SimBox::new(h, origin, pbc)
            .map_err(|e| MolRsError::validation(format!("invalid MolRec box: {:?}", e)))
    }
}

/// Root-level box semantics for a record.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "mode", rename_all = "snake_case")]
pub enum RecordBox {
    /// One static box shared by all frames.
    Static { cell: CellBox },
    /// One box per trajectory state.
    Dynamic { cells: Vec<CellBox> },
}

impl RecordBox {
    /// Return the box appropriate for one accessible frame.
    pub fn cell_at(&self, index: usize) -> Option<&CellBox> {
        match self {
            Self::Static { cell } => Some(cell),
            Self::Dynamic { cells } => cells.get(index),
        }
    }
}

/// Trajectory-like list of frame states plus shared indexing arrays.
#[derive(Debug, Clone, Default)]
pub struct Trajectory {
    /// Ordered frame-like states.
    pub frames: Vec<Frame>,
    /// Optional discrete step indices.
    pub step: Option<Vec<i64>>,
    /// Optional physical time values.
    pub time: Option<Vec<F>>,
    /// Optional box information attached before the trajectory is wrapped in a record.
    pub box_data: Option<RecordBox>,
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
            box_data: None,
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

    /// Return a copy without embedded box data.
    pub fn without_box(mut self) -> Self {
        self.box_data = None;
        self
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
        if let Some(RecordBox::Dynamic { cells }) = &self.box_data
            && cells.len() != n
        {
            return Err(MolRsError::validation(format!(
                "trajectory box length mismatch: expected {}, got {}",
                n,
                cells.len()
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

/// Single logical MolRec object.
#[derive(Debug, Clone)]
pub struct MolRec {
    /// Record-level metadata.
    pub meta: SchemaValue,
    /// Canonical frame.
    pub frame: Frame,
    /// Root-level box information.
    pub box_data: Option<RecordBox>,
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
        let box_data = frame.simbox.as_ref().map(|sb| RecordBox::Static {
            cell: CellBox::from_simbox(sb),
        });
        Self {
            meta: empty_object(),
            frame,
            box_data,
            trajectory: None,
            observables: BTreeMap::new(),
            method: empty_object(),
            parameters: empty_object(),
        }
    }

    /// Build a MolRec from one canonical frame plus explicit trajectory states.
    pub fn from_frames(frame: Frame, frames: Vec<Frame>) -> Self {
        let mut rec = Self::new(frame.clone());
        let mut trajectory = Trajectory::from_frames(frames);
        rec.box_data = rec
            .box_data
            .clone()
            .or_else(|| infer_box_from_frames(&trajectory.frames));
        trajectory.box_data = None;
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
        rec.box_data = trajectory
            .box_data
            .clone()
            .or_else(|| infer_box_from_frames(&trajectory.frames));
        rec.trajectory = Some(trajectory.without_box());
        Ok(rec)
    }

    /// Build a MolRec whose method metadata comes from a force field definition.
    pub fn from_forcefield(frame: Frame, forcefield: &ForceField) -> Self {
        let mut rec = Self::new(frame);
        let styles: Vec<JsonValue> = forcefield
            .styles()
            .iter()
            .map(|style| {
                json!({
                    "category": style.category(),
                    "name": style.name,
                })
            })
            .collect();
        rec.method = json!({
            "type": "classical",
            "description": "Force-field-derived molecular record",
            "classical": {
                "force_field": {
                    "name": forcefield.name,
                    "styles": styles,
                }
            }
        });
        rec
    }

    /// Total number of accessible frames.
    pub fn count_frames(&self) -> usize {
        match &self.trajectory {
            Some(traj) if !traj.frames.is_empty() => traj.frames.len(),
            _ => 1,
        }
    }

    /// Return one accessible frame with record-level box projected onto it.
    pub fn frame_at(&self, index: usize) -> Option<Frame> {
        let mut frame = match &self.trajectory {
            Some(traj) if !traj.frames.is_empty() => traj.frames.get(index)?.clone(),
            _ if index == 0 => self.frame.clone(),
            _ => return None,
        };

        if let Some(cell) = self.box_data.as_ref().and_then(|b| b.cell_at(index))
            && let Ok(simbox) = cell.to_simbox()
        {
            frame.simbox = Some(simbox);
        }

        Some(frame)
    }

    /// Replace the canonical frame.
    ///
    /// If no box has been set yet, the frame's simbox (if any) is promoted to
    /// a static `RecordBox`.
    pub fn set_frame(&mut self, frame: Frame) {
        if self.box_data.is_none() {
            self.box_data = frame.simbox.as_ref().map(|sb| RecordBox::Static {
                cell: CellBox::from_simbox(sb),
            });
        }
        self.frame = frame;
    }

    /// Replace the root-level box.
    pub fn set_box(&mut self, box_data: Option<RecordBox>) {
        self.box_data = box_data;
    }

    /// Append one frame to the trajectory, creating it if needed.
    ///
    /// Box is (re-)inferred from all trajectory frames after each append.
    pub fn add_frame(&mut self, frame: Frame) {
        match &mut self.trajectory {
            Some(traj) => traj.frames.push(frame),
            None => {
                self.trajectory = Some(Trajectory::from_frames(vec![frame]));
            }
        }
        if let Some(traj) = &self.trajectory {
            if let Some(inferred) = infer_box_from_frames(&traj.frames) {
                self.box_data = Some(inferred);
            }
        }
    }

    /// Populate method metadata from a force-field definition.
    pub fn set_forcefield(&mut self, ff: &ForceField) {
        let styles: Vec<JsonValue> = ff
            .styles()
            .iter()
            .map(|style| {
                json!({
                    "category": style.category(),
                    "name": style.name,
                })
            })
            .collect();
        self.method = json!({
            "type": "classical",
            "description": "Force-field-derived molecular record",
            "classical": {
                "force_field": {
                    "name": ff.name,
                    "styles": styles,
                }
            }
        });
    }

    /// Replace the trajectory.
    pub fn set_trajectory(&mut self, mut trajectory: Option<Trajectory>) {
        if let Some(ref mut traj) = trajectory {
            if self.box_data.is_none() {
                self.box_data = traj
                    .box_data
                    .clone()
                    .or_else(|| infer_box_from_frames(&traj.frames));
            }
            traj.box_data = None;
        }
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

#[cfg(feature = "zarr")]
impl MolRec {
    /// Read a MolRec from a Zarr v3 store.
    pub fn read_zarr_store(
        store: zarrs::storage::ReadableWritableListableStorage,
    ) -> Result<Self, crate::error::MolRsError> {
        crate::io::zarr::read_molrec_store(store)
    }

    /// Count addressable frames in a MolRec Zarr v3 store.
    pub fn count_zarr_frames(
        store: zarrs::storage::ReadableWritableListableStorage,
    ) -> Result<u64, crate::error::MolRsError> {
        crate::io::zarr::count_molrec_frames_in_store(store)
    }
}

#[cfg(all(feature = "zarr", feature = "filesystem"))]
impl MolRec {
    /// Read a MolRec from a Zarr v3 directory.
    pub fn read_zarr(path: impl AsRef<std::path::Path>) -> Result<Self, crate::error::MolRsError> {
        crate::io::zarr::read_molrec_file(path)
    }

    /// Write a MolRec into a Zarr v3 directory.
    pub fn write_zarr(
        &self,
        path: impl AsRef<std::path::Path>,
    ) -> Result<(), crate::error::MolRsError> {
        crate::io::zarr::write_molrec_file(path, self)
    }
}

fn empty_object() -> JsonValue {
    JsonValue::Object(JsonMap::new())
}

fn boundary_label(periodic: bool) -> String {
    if periodic {
        "periodic".into()
    } else {
        "fixed".into()
    }
}

fn is_periodic(boundary: &str) -> bool {
    matches!(boundary, "periodic" | "wrap" | "pbc" | "true")
}

fn infer_box_from_frames(frames: &[Frame]) -> Option<RecordBox> {
    let cells: Vec<CellBox> = frames
        .iter()
        .filter_map(|frame| frame.simbox.as_ref().map(CellBox::from_simbox))
        .collect();
    if cells.is_empty() {
        return None;
    }
    if cells.len() != frames.len() {
        return None;
    }
    if cells.iter().all(|cell| cell == &cells[0]) {
        Some(RecordBox::Static {
            cell: cells[0].clone(),
        })
    } else {
        Some(RecordBox::Dynamic { cells })
    }
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
