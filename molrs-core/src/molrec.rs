//! Backend-agnostic MolRec logical model.

use std::collections::BTreeMap;

use crate::frame::Frame;

/// Single logical MolRec object.
///
/// MolRec is backend-agnostic. Concrete persistence is provided by backend
/// adapters such as the Zarr v3 implementation.
#[derive(Debug, Clone, Default)]
pub struct MolRec {
    /// Record-level metadata.
    pub meta: BTreeMap<String, String>,
    /// Canonical atomistic frame.
    pub frame: Frame,
    /// Optional ordered state realizations.
    pub trajectory: Vec<Frame>,
    /// Method-level metadata.
    pub method: BTreeMap<String, String>,
}

impl MolRec {
    /// Create a MolRec around one canonical frame.
    pub fn new(frame: Frame) -> Self {
        Self {
            meta: BTreeMap::new(),
            frame,
            trajectory: Vec::new(),
            method: BTreeMap::new(),
        }
    }

    /// Total number of accessible frames.
    ///
    /// Returns `1` when no trajectory is present because the canonical frame
    /// remains the single accessible state.
    pub fn count_frames(&self) -> usize {
        if self.trajectory.is_empty() {
            1
        } else {
            self.trajectory.len()
        }
    }

    /// Return one accessible frame, projecting record-level fields onto it.
    pub fn frame_at(&self, index: usize) -> Option<Frame> {
        if self.trajectory.is_empty() {
            if index == 0 {
                return Some(self.frame.clone());
            }
            return None;
        }

        let mut frame = self.trajectory.get(index)?.clone();
        for (name, field) in self.frame.fields() {
            if !frame.has_field(name) {
                frame.add_field(name.to_string(), field.clone());
            }
        }
        Some(frame)
    }

    /// Replace the canonical frame.
    pub fn set_frame(&mut self, frame: Frame) {
        self.frame = frame;
    }

    /// Append one trajectory state.
    pub fn push_frame(&mut self, frame: Frame) {
        self.trajectory.push(frame);
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

#[cfg(test)]
mod tests {
    use crate::field::{FieldObservable, UniformGridField};

    use super::*;

    #[test]
    fn static_molrec_counts_one_frame() {
        let rec = MolRec::new(Frame::new());
        assert_eq!(rec.count_frames(), 1);
        assert!(rec.frame_at(0).is_some());
        assert!(rec.frame_at(1).is_none());
    }

    #[test]
    fn canonical_fields_are_projected_onto_trajectory_frames() {
        let mut base = Frame::new();
        base.add_field(
            "electron_density",
            FieldObservable::uniform_grid(
                "electron_density",
                "electron_density",
                "e/Angstrom^3",
                UniformGridField {
                    shape: [1, 1, 1],
                    origin: [0.0, 0.0, 0.0],
                    cell: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                    pbc: [false, false, false],
                    values: vec![0.2],
                },
            ),
        );

        let mut rec = MolRec::new(base);
        rec.push_frame(Frame::new());

        let frame = rec.frame_at(0).unwrap();
        assert!(frame.has_field("electron_density"));
    }
}
