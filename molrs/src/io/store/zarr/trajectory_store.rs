//! Zarr v3 storage for a `Trajectory` (frame sequence).
//!
//! Persists a plain frame sequence — each frame via the shared
//! `frame_io::write_system` — plus optional `step`/`time` index arrays, under a
//! group tagged `molrs_format="frames"`. There is no record aggregate: the
//! stored payload is `Frame`s, matching the external molrec frame spec
//! (github.com/MolCrafts/molrec) without depending on any runtime `MolRec`.

#[cfg(feature = "filesystem")]
use std::path::Path;
#[cfg(feature = "filesystem")]
use std::sync::Arc;

#[cfg(feature = "filesystem")]
use zarrs::array::ArrayBuilder;
#[cfg(feature = "filesystem")]
use zarrs::array::data_type;
use zarrs::array::data_type::{Float32DataType, Float64DataType, Int64DataType};
use zarrs::array::{Array, ArraySubset};
#[cfg(feature = "filesystem")]
use zarrs::filesystem::FilesystemStore;
#[cfg(feature = "filesystem")]
use zarrs::group::GroupBuilder;
use zarrs::node::{Node, NodeMetadata};
use zarrs::storage::ReadableWritableListableStorage;

#[cfg(not(feature = "filesystem"))]
use crate::io::store::zarr::frame_io::{join_path, read_system};
#[cfg(feature = "filesystem")]
use crate::io::store::zarr::frame_io::{join_path, read_system, write_system};
use molrs::MolRsError;
use molrs::store::frame::Frame;
use molrs::store::trajectory::Trajectory;
use molrs::types::F;

/// Internal Zarr v3 backend for a stored frame sequence.
pub(crate) struct TrajectoryZarrBackend {
    store: ReadableWritableListableStorage,
    prefix: String,
}

impl TrajectoryZarrBackend {
    #[cfg(feature = "filesystem")]
    pub fn create_file(
        path: impl AsRef<Path>,
        trajectory: &Trajectory,
    ) -> Result<Self, MolRsError> {
        let store: ReadableWritableListableStorage =
            Arc::new(FilesystemStore::new(path.as_ref()).map_err(zerr)?);
        Self::create_in_store(store, "/", trajectory)
    }

    #[cfg(feature = "filesystem")]
    pub fn open_file(path: impl AsRef<Path>) -> Result<Self, MolRsError> {
        let store: ReadableWritableListableStorage =
            Arc::new(FilesystemStore::new(path.as_ref()).map_err(zerr)?);
        Self::open_in_store(store, "/")
    }

    pub fn open_store(store: ReadableWritableListableStorage) -> Result<Self, MolRsError> {
        Self::open_in_store(store, "/")
    }

    #[cfg(feature = "filesystem")]
    pub(crate) fn create_in_store(
        store: ReadableWritableListableStorage,
        prefix: &str,
        trajectory: &Trajectory,
    ) -> Result<Self, MolRsError> {
        trajectory.validate()?;

        let mut attrs = serde_json::Map::new();
        attrs.insert("molrs_format".into(), "frames".into());
        attrs.insert("version".into(), 2.into());
        attrs.insert(
            "frame_count".into(),
            (trajectory.frames.len() as u64).into(),
        );
        GroupBuilder::new()
            .attributes(attrs)
            .build(store.clone(), prefix)?
            .store_metadata()?;

        if let Some(step) = &trajectory.step {
            write_i64_array(
                &store,
                &join_path(prefix, "step"),
                &[step.len() as u64],
                step,
            )?;
        }
        if let Some(time) = &trajectory.time {
            let data: Vec<f32> = time.iter().map(|&v| v as f32).collect();
            write_f32_array(
                &store,
                &join_path(prefix, "time"),
                &[time.len() as u64],
                &data,
            )?;
        }

        let frames_path = join_path(prefix, "frames");
        GroupBuilder::new()
            .build(store.clone(), &frames_path)?
            .store_metadata()?;
        for (index, frame) in trajectory.frames.iter().enumerate() {
            write_system(&store, &join_path(&frames_path, &index.to_string()), frame)?;
        }

        Ok(Self {
            store,
            prefix: prefix.to_string(),
        })
    }

    pub(crate) fn open_in_store(
        store: ReadableWritableListableStorage,
        prefix: &str,
    ) -> Result<Self, MolRsError> {
        let root = zarrs::group::Group::open(store.clone(), prefix)?;
        let format = root
            .attributes()
            .get("molrs_format")
            .and_then(|v| v.as_str())
            .unwrap_or("");
        if format != "frames" {
            return Err(MolRsError::zarr(format!(
                "expected frames zarr v3 store, found '{}'",
                format
            )));
        }
        Ok(Self {
            store,
            prefix: prefix.to_string(),
        })
    }

    pub fn read(&self) -> Result<Trajectory, MolRsError> {
        let step_path = join_path(&self.prefix, "step");
        let time_path = join_path(&self.prefix, "time");
        let step = if let Ok(arr) = Array::open(self.store.clone(), &step_path) {
            Some(read_i64_values(arr)?)
        } else {
            None
        };
        let time = if let Ok(arr) = Array::open(self.store.clone(), &time_path) {
            Some(read_float_values(arr)?)
        } else {
            None
        };

        let mut frames = Vec::new();
        let frames_path = join_path(&self.prefix, "frames");
        if let Ok(node) = Node::open(&self.store, &frames_path) {
            let mut children: Vec<_> = node
                .children()
                .iter()
                .filter(|child| matches!(child.metadata(), NodeMetadata::Group(_)))
                .collect();
            children.sort_by_key(|child| {
                child
                    .path()
                    .as_str()
                    .rsplit('/')
                    .next()
                    .and_then(|s| s.parse::<usize>().ok())
                    .unwrap_or(usize::MAX)
            });
            for child in children {
                frames.push(read_system(&self.store, child.path().as_str())?);
            }
        }

        Ok(Trajectory { frames, step, time })
    }

    pub fn count_frames(&self) -> Result<u64, MolRsError> {
        let root = zarrs::group::Group::open(self.store.clone(), &self.prefix)?;
        Ok(root
            .attributes()
            .get("frame_count")
            .and_then(|v| v.as_u64())
            .unwrap_or(0))
    }

    pub fn read_frame(&self, index: usize) -> Result<Option<Frame>, MolRsError> {
        let traj = self.read()?;
        Ok(traj.frames.into_iter().nth(index))
    }
}

#[cfg(feature = "filesystem")]
pub fn write_trajectory_file(
    path: impl AsRef<Path>,
    trajectory: &Trajectory,
) -> Result<(), MolRsError> {
    let _ = TrajectoryZarrBackend::create_file(path, trajectory)?;
    Ok(())
}

#[cfg(feature = "filesystem")]
pub fn read_trajectory_file(path: impl AsRef<Path>) -> Result<Trajectory, MolRsError> {
    TrajectoryZarrBackend::open_file(path)?.read()
}

pub fn read_trajectory_store(
    store: ReadableWritableListableStorage,
) -> Result<Trajectory, MolRsError> {
    TrajectoryZarrBackend::open_store(store)?.read()
}

pub fn read_frame_from_store(
    store: ReadableWritableListableStorage,
    index: usize,
) -> Result<Option<Frame>, MolRsError> {
    TrajectoryZarrBackend::open_store(store)?.read_frame(index)
}

pub fn count_frames_in_store(store: ReadableWritableListableStorage) -> Result<u64, MolRsError> {
    TrajectoryZarrBackend::open_store(store)?.count_frames()
}

#[cfg(feature = "filesystem")]
fn write_f32_array(
    store: &ReadableWritableListableStorage,
    path: &str,
    shape: &[u64],
    data: &[f32],
) -> Result<(), MolRsError> {
    let arr = ArrayBuilder::new(shape.to_vec(), shape.to_vec(), data_type::float32(), 0.0f32)
        .build(store.clone(), path)?;
    arr.store_metadata()?;
    arr.store_array_subset(&ArraySubset::new_with_shape(shape.to_vec()), data)?;
    Ok(())
}

#[cfg(feature = "filesystem")]
fn write_i64_array(
    store: &ReadableWritableListableStorage,
    path: &str,
    shape: &[u64],
    data: &[i64],
) -> Result<(), MolRsError> {
    let arr = ArrayBuilder::new(shape.to_vec(), shape.to_vec(), data_type::int64(), 0i64)
        .build(store.clone(), path)?;
    arr.store_metadata()?;
    arr.store_array_subset(&ArraySubset::new_with_shape(shape.to_vec()), data)?;
    Ok(())
}

fn read_float_values<
    TStorage: ?Sized + zarrs::storage::ReadableWritableListableStorageTraits + 'static,
>(
    arr: Array<TStorage>,
) -> Result<Vec<F>, MolRsError> {
    let shape = arr.shape().to_vec();
    let subset = ArraySubset::new_with_shape(shape);
    let dt = arr.data_type();
    if dt.is::<Float32DataType>() {
        let data: Vec<f32> = arr.retrieve_array_subset(&subset).map_err(zerr)?;
        Ok(data.into_iter().map(|v| v as F).collect())
    } else if dt.is::<Float64DataType>() {
        let data: Vec<f64> = arr.retrieve_array_subset(&subset).map_err(zerr)?;
        Ok(data.into_iter().map(|v| v as F).collect())
    } else {
        Err(MolRsError::zarr(format!(
            "expected float array, got {:?}",
            dt
        )))
    }
}

fn read_i64_values<
    TStorage: ?Sized + zarrs::storage::ReadableWritableListableStorageTraits + 'static,
>(
    arr: Array<TStorage>,
) -> Result<Vec<i64>, MolRsError> {
    let subset = ArraySubset::new_with_shape(arr.shape().to_vec());
    let dt = arr.data_type();
    if dt.is::<Int64DataType>() {
        arr.retrieve_array_subset(&subset).map_err(zerr)
    } else {
        Err(MolRsError::zarr(format!(
            "expected int64 array, got {:?}",
            dt
        )))
    }
}

fn zerr(e: impl std::fmt::Display) -> MolRsError {
    MolRsError::zarr(e.to_string())
}

#[cfg(test)]
mod tests {
    use tempfile::tempdir;

    use super::*;

    #[test]
    fn trajectory_store_roundtrip_preserves_frames() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("record.zarr");

        let mut frame = Frame::new();
        frame.meta.insert("key".into(), "value".into());

        let traj = Trajectory::from_frames(vec![frame]);

        write_trajectory_file(&path, &traj).unwrap();
        let loaded = read_trajectory_file(&path).unwrap();
        assert_eq!(loaded.frames.len(), 1);
        assert_eq!(loaded.frames[0].meta.get("key").unwrap(), "value");
    }
}
