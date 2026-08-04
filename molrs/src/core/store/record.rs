//! MolRec record aggregate — L2 of the MolRec contract.
//!
//! A [`MolRec`] is one openable root carrying `meta` plus at least one of
//! `frame`, `system`, or `status`. It is backend-neutral: the reference Zarr V3
//! binding lives in `crate::io::store::zarr`, and nothing here depends on it.
//!
//! Contract: <https://github.com/MolCrafts/molrec> (`docs/spec/record.md`).
//! `meta.record_schema_version` is the **sole** version key of a record; there is
//! no parallel per-frame schema version.

use std::collections::BTreeMap;

use serde_json::{Map as JsonMap, Value as JsonValue};

use crate::MolRsError;
use crate::store::frame::Frame;
use crate::store::trajectory::{ObservableRecord, Trajectory};

/// Sole schema version of a MolRec record (root layout + L1 encoding).
pub const RECORD_SCHEMA_VERSION: u64 = 1;

/// Binding identifier written to `meta/format_name` by the Zarr writer.
pub const RECORD_FORMAT_NAME: &str = "molrec";

/// Reserved `meta` keys owned by the contract rather than by the producer.
pub const RESERVED_META_KEYS: [&str; 2] = ["record_schema_version", "format_name"];

/// Named observables of a record, keyed by observable name.
///
/// Data and semantic metadata are one unit: the contract forbids standalone
/// observable arrays, so a record can only carry a fully described
/// [`ObservableRecord`].
#[derive(Debug, Clone, Default)]
pub struct Observables {
    records: BTreeMap<String, ObservableRecord>,
}

impl Observables {
    /// Create an empty collection.
    pub fn new() -> Self {
        Self::default()
    }

    /// Number of observables.
    pub fn len(&self) -> usize {
        self.records.len()
    }

    /// Returns true when no observable is stored.
    pub fn is_empty(&self) -> bool {
        self.records.is_empty()
    }

    /// Returns true when `name` is present.
    pub fn contains(&self, name: &str) -> bool {
        self.records.contains_key(name)
    }

    /// Borrow one observable.
    pub fn get(&self, name: &str) -> Option<&ObservableRecord> {
        self.records.get(name)
    }

    /// Insert an observable, replacing any earlier one of the same name.
    pub fn insert(&mut self, record: ObservableRecord) -> Result<(), MolRsError> {
        record.validate()?;
        self.records.insert(record.name.clone(), record);
        Ok(())
    }

    /// Remove one observable.
    pub fn remove(&mut self, name: &str) -> Option<ObservableRecord> {
        self.records.remove(name)
    }

    /// Observable names, sorted.
    pub fn names(&self) -> impl Iterator<Item = &String> {
        self.records.keys()
    }

    /// Iterate over `(name, record)` pairs, sorted by name.
    pub fn iter(&self) -> impl Iterator<Item = (&String, &ObservableRecord)> {
        self.records.iter()
    }
}

/// One self-describing record: the unit of interchange between MolCrafts tools.
///
/// Sections map one-to-one onto the contract's root layout. `meta` is always
/// written; the remaining sections are optional, and a reader preserves sections
/// it does not interpret in [`extra_sections`](Self::extra_sections).
#[derive(Debug, Clone, Default)]
pub struct MolRec {
    /// Record-level metadata. The writer adds the reserved contract keys
    /// (see [`RESERVED_META_KEYS`]); everything else is the producer's.
    pub meta: JsonMap<String, JsonValue>,
    /// How the record was produced (run surface).
    pub method: JsonMap<String, JsonValue>,
    /// Lifecycle / progress (run surface).
    pub status: JsonMap<String, JsonValue>,
    /// Append-only run measurements (run surface).
    pub metrics: JsonMap<String, JsonValue>,
    /// System definition — topology and types, without instantaneous state.
    pub system: Option<Frame>,
    /// Instantaneous snapshot.
    pub frame: Option<Frame>,
    /// Ordered frame sequence.
    pub trajectory: Option<Trajectory>,
    /// Named scientific results.
    pub observables: Observables,
    /// Sections this build does not interpret, kept verbatim so a round-trip
    /// through an older reader does not delete a newer producer's data.
    pub extra_sections: BTreeMap<String, Frame>,
}

impl MolRec {
    /// Create an empty record.
    pub fn new() -> Self {
        Self::default()
    }

    /// Append a frame to the trajectory section, creating it when absent.
    pub fn add_frame(&mut self, frame: Frame) {
        self.trajectory
            .get_or_insert_with(Trajectory::new)
            .frames
            .push(frame);
    }

    /// Number of frames the record carries: the trajectory length when present,
    /// otherwise one for a bare snapshot.
    pub fn count_frames(&self) -> usize {
        match &self.trajectory {
            Some(traj) => traj.len(),
            None => usize::from(self.frame.is_some()),
        }
    }

    /// Check the contract's minimum record shape.
    ///
    /// A record must carry at least one of `frame`, `system`, or `status`; a
    /// Run-shaped record (`meta` + `status`) needs no frame.
    pub fn validate(&self) -> Result<(), MolRsError> {
        if self.frame.is_none() && self.system.is_none() && self.status.is_empty() {
            return Err(MolRsError::validation(
                "record must carry at least one of 'frame', 'system', or 'status'",
            ));
        }
        if let Some(traj) = &self.trajectory {
            traj.validate()?;
        }
        for record in self.observables.records.values() {
            record.validate()?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::block::Column;
    use ndarray::ArrayD;

    fn scalar_column(values: &[f64]) -> Column {
        Column::from_float(ArrayD::from_shape_vec(vec![values.len()], values.to_vec()).unwrap())
    }

    #[test]
    fn empty_record_fails_minimum_shape() {
        assert!(MolRec::new().validate().is_err());
    }

    #[test]
    fn frame_only_record_is_valid() {
        let mut rec = MolRec::new();
        rec.frame = Some(Frame::new());
        rec.validate().unwrap();
    }

    #[test]
    fn system_only_record_is_valid() {
        let mut rec = MolRec::new();
        rec.system = Some(Frame::new());
        rec.validate().unwrap();
    }

    #[test]
    fn run_shaped_record_needs_no_frame() {
        let mut rec = MolRec::new();
        rec.status.insert("state".into(), "running".into());
        rec.validate().unwrap();
    }

    #[test]
    fn bare_snapshot_counts_one_frame() {
        let mut rec = MolRec::new();
        rec.frame = Some(Frame::new());
        assert_eq!(rec.count_frames(), 1);
    }

    #[test]
    fn trajectory_length_overrides_snapshot_count() {
        let mut rec = MolRec::new();
        rec.frame = Some(Frame::new());
        rec.add_frame(Frame::new());
        rec.add_frame(Frame::new());
        assert_eq!(rec.count_frames(), 2);
    }

    #[test]
    fn empty_record_counts_no_frames() {
        assert_eq!(MolRec::new().count_frames(), 0);
    }

    #[test]
    fn mismatched_trajectory_axis_fails_record_validation() {
        let mut rec = MolRec::new();
        rec.frame = Some(Frame::new());
        rec.add_frame(Frame::new());
        rec.trajectory.as_mut().unwrap().step = Some(vec![0, 1]); // 2 steps, 1 frame
        assert!(rec.validate().is_err());
    }

    #[test]
    fn observables_round_trip_by_name() {
        let mut obs = Observables::new();
        obs.insert(ObservableRecord::scalar(
            "total_energy",
            scalar_column(&[1.0, 2.0]),
        ))
        .unwrap();
        assert!(obs.contains("total_energy"));
        assert_eq!(obs.len(), 1);
        assert_eq!(obs.get("total_energy").unwrap().name, "total_energy");
    }

    #[test]
    fn inserting_same_observable_name_replaces() {
        let mut obs = Observables::new();
        obs.insert(ObservableRecord::scalar("e", scalar_column(&[1.0])))
            .unwrap();
        obs.insert(ObservableRecord::vector("e", scalar_column(&[2.0])))
            .unwrap();
        assert_eq!(obs.len(), 1);
    }
}
