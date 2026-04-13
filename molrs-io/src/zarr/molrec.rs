//! MolRec Zarr v3 backend.

#[cfg(feature = "filesystem")]
use std::path::Path;
#[cfg(feature = "filesystem")]
use std::sync::Arc;

use serde_json::Value as JsonValue;
#[cfg(feature = "filesystem")]
use zarrs::array::ArrayBuilder;
#[cfg(feature = "filesystem")]
use zarrs::array::data_type;
use zarrs::array::data_type::{Float32DataType, Float64DataType, Int64DataType, StringDataType};
use zarrs::array::{Array, ArraySubset};
#[cfg(feature = "filesystem")]
use zarrs::filesystem::FilesystemStore;
#[cfg(feature = "filesystem")]
use zarrs::group::GroupBuilder;
use zarrs::node::{Node, NodeMetadata};
use zarrs::storage::ReadableWritableListableStorage;

use molrs::MolRsError;
use molrs::frame::Frame;
#[cfg(not(feature = "filesystem"))]
use crate::zarr::frame_io::{join_path, read_column, read_grid, read_system};
#[cfg(feature = "filesystem")]
use crate::zarr::frame_io::{
    join_path, read_column, read_grid, read_system, write_column, write_grid, write_system,
};
use molrs::molrec::{MolRec, ObservableData, ObservableKind, ObservableRecord, Trajectory};
use molrs::types::F;

/// Internal Zarr v3 backend state for MolRec.
pub(crate) struct MolRecZarrBackend {
    store: ReadableWritableListableStorage,
    prefix: String,
}

impl MolRecZarrBackend {
    #[cfg(feature = "filesystem")]
    pub fn create_file(path: impl AsRef<Path>, molrec: &MolRec) -> Result<Self, MolRsError> {
        let store: ReadableWritableListableStorage =
            Arc::new(FilesystemStore::new(path.as_ref()).map_err(zerr)?);
        Self::create_in_store(store, "/", molrec)
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
        molrec: &MolRec,
    ) -> Result<Self, MolRsError> {
        let mut attrs = serde_json::Map::new();
        attrs.insert("molrs_format".into(), "molrec".into());
        attrs.insert("version".into(), 2.into());
        attrs.insert("frame_count".into(), (molrec.count_frames() as u64).into());
        GroupBuilder::new()
            .attributes(attrs)
            .build(store.clone(), prefix)?
            .store_metadata()?;

        write_json_group(&store, &join_path(prefix, "meta"), &molrec.meta)?;
        write_json_group(&store, &join_path(prefix, "method"), &molrec.method)?;
        write_json_group(&store, &join_path(prefix, "parameters"), &molrec.parameters)?;

        write_system(&store, &join_path(prefix, "frame"), &molrec.frame)?;

        if !molrec.observables.is_empty() {
            let observable_prefix = join_path(prefix, "observables");
            GroupBuilder::new()
                .build(store.clone(), &observable_prefix)?
                .store_metadata()?;
            for (name, observable) in &molrec.observables {
                write_observable(&store, &join_path(&observable_prefix, name), observable)?;
            }
        }

        if let Some(trajectory) = &molrec.trajectory
            && !trajectory.frames.is_empty()
        {
            write_trajectory(&store, &join_path(prefix, "trajectory"), trajectory)?;
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
        if format != "molrec" {
            return Err(MolRsError::zarr(format!(
                "expected MolRec zarr v3 store, found '{}'",
                format
            )));
        }
        Ok(Self {
            store,
            prefix: prefix.to_string(),
        })
    }

    pub fn read(&self) -> Result<MolRec, MolRsError> {
        let frame = read_system(&self.store, &join_path(&self.prefix, "frame"))?;
        let mut rec = MolRec::new(frame);
        rec.meta = read_json_group(&self.store, &join_path(&self.prefix, "meta"))?;
        rec.method = read_json_group(&self.store, &join_path(&self.prefix, "method"))?;
        rec.parameters = read_json_group(&self.store, &join_path(&self.prefix, "parameters"))?;

        let observable_prefix = join_path(&self.prefix, "observables");
        if let Ok(node) = Node::open(&self.store, &observable_prefix) {
            for child in node.children() {
                if !matches!(child.metadata(), NodeMetadata::Group(_)) {
                    continue;
                }
                let name = child.path().as_str().rsplit('/').next().unwrap_or("");
                if name.is_empty() {
                    continue;
                }
                let observable = read_observable(&self.store, child.path().as_str())?;
                rec.observables.insert(name.to_string(), observable);
            }
        }

        let traj_path = join_path(&self.prefix, "trajectory");
        if Node::open(&self.store, &traj_path).is_ok() {
            rec.trajectory = Some(read_trajectory(&self.store, &traj_path)?);
        } else {
            rec.trajectory = None;
        }

        Ok(rec)
    }

    pub fn count_frames(&self) -> Result<u64, MolRsError> {
        let root = zarrs::group::Group::open(self.store.clone(), &self.prefix)?;
        Ok(root
            .attributes()
            .get("frame_count")
            .and_then(|v| v.as_u64())
            .unwrap_or(1))
    }

    pub fn read_frame(&self, index: usize) -> Result<Option<Frame>, MolRsError> {
        let rec = self.read()?;
        Ok(rec.frame_at(index))
    }
}

#[cfg(feature = "filesystem")]
pub fn write_molrec_file(path: impl AsRef<Path>, molrec: &MolRec) -> Result<(), MolRsError> {
    let _ = MolRecZarrBackend::create_file(path, molrec)?;
    Ok(())
}

#[cfg(feature = "filesystem")]
pub fn read_molrec_file(path: impl AsRef<Path>) -> Result<MolRec, MolRsError> {
    MolRecZarrBackend::open_file(path)?.read()
}

pub fn read_molrec_store(store: ReadableWritableListableStorage) -> Result<MolRec, MolRsError> {
    MolRecZarrBackend::open_store(store)?.read()
}

pub fn read_molrec_frame_from_store(
    store: ReadableWritableListableStorage,
    index: usize,
) -> Result<Option<Frame>, MolRsError> {
    MolRecZarrBackend::open_store(store)?.read_frame(index)
}

pub fn count_molrec_frames_in_store(
    store: ReadableWritableListableStorage,
) -> Result<u64, MolRsError> {
    MolRecZarrBackend::open_store(store)?.count_frames()
}

#[cfg(feature = "filesystem")]
fn write_json_group(
    store: &ReadableWritableListableStorage,
    path: &str,
    value: &JsonValue,
) -> Result<(), MolRsError> {
    let mut attrs = serde_json::Map::new();
    attrs.insert(
        "json".into(),
        JsonValue::String(serde_json::to_string(value).map_err(json_err)?),
    );
    GroupBuilder::new()
        .attributes(attrs)
        .build(store.clone(), path)?
        .store_metadata()?;
    Ok(())
}

fn read_json_group(
    store: &ReadableWritableListableStorage,
    path: &str,
) -> Result<JsonValue, MolRsError> {
    let group = match zarrs::group::Group::open(store.clone(), path) {
        Ok(group) => group,
        Err(_) => return Ok(JsonValue::Object(Default::default())),
    };
    let json_text = group
        .attributes()
        .get("json")
        .and_then(|v| v.as_str())
        .unwrap_or("{}");
    serde_json::from_str(json_text).map_err(json_err)
}

#[cfg(feature = "filesystem")]
fn write_trajectory(
    store: &ReadableWritableListableStorage,
    path: &str,
    trajectory: &Trajectory,
) -> Result<(), MolRsError> {
    trajectory.validate()?;
    GroupBuilder::new()
        .build(store.clone(), path)?
        .store_metadata()?;

    if let Some(step) = &trajectory.step {
        write_i64_array(store, &format!("{}/step", path), &[step.len() as u64], step)?;
    }
    if let Some(time) = &trajectory.time {
        let data: Vec<f32> = time.iter().map(|&v| v as f32).collect();
        write_f32_array(
            store,
            &format!("{}/time", path),
            &[time.len() as u64],
            &data,
        )?;
    }

    let frames_path = format!("{}/frames", path);
    GroupBuilder::new()
        .build(store.clone(), &frames_path)?
        .store_metadata()?;
    for (index, frame) in trajectory.frames.iter().enumerate() {
        write_system(store, &format!("{}/{}", frames_path, index), frame)?;
    }
    Ok(())
}

fn read_trajectory(
    store: &ReadableWritableListableStorage,
    path: &str,
) -> Result<Trajectory, MolRsError> {
    let step_path = format!("{}/step", path);
    let time_path = format!("{}/time", path);
    let step = if let Ok(arr) = Array::open(store.clone(), &step_path) {
        Some(read_i64_values(arr)?)
    } else {
        None
    };
    let time = if let Ok(arr) = Array::open(store.clone(), &time_path) {
        Some(read_float_values(arr)?)
    } else {
        None
    };

    let mut frames = Vec::new();
    let frames_path = format!("{}/frames", path);
    if let Ok(node) = Node::open(store, &frames_path) {
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
            frames.push(read_system(store, child.path().as_str())?);
        }
    }

    Ok(Trajectory {
        frames,
        step,
        time,
    })
}

#[cfg(feature = "filesystem")]
fn write_observable(
    store: &ReadableWritableListableStorage,
    path: &str,
    observable: &ObservableRecord,
) -> Result<(), MolRsError> {
    observable.validate()?;
    let mut attrs = serde_json::Map::new();
    attrs.insert("name".into(), JsonValue::String(observable.name.clone()));
    attrs.insert(
        "kind".into(),
        JsonValue::String(match observable.kind {
            ObservableKind::Scalar => "scalar".into(),
            ObservableKind::Vector => "vector".into(),
            ObservableKind::Grid => "grid".into(),
        }),
    );
    attrs.insert(
        "description".into(),
        JsonValue::String(observable.description.clone()),
    );
    attrs.insert(
        "time_dependent".into(),
        JsonValue::Bool(observable.time_dependent),
    );
    attrs.insert(
        "axes".into(),
        JsonValue::String(serde_json::to_string(&observable.axes).map_err(json_err)?),
    );
    attrs.insert(
        "extra".into(),
        JsonValue::String(serde_json::to_string(&observable.extra).map_err(json_err)?),
    );
    if let Some(unit) = &observable.unit {
        attrs.insert("unit".into(), JsonValue::String(unit.clone()));
    }
    if let Some(sampling) = &observable.sampling {
        attrs.insert("sampling".into(), JsonValue::String(sampling.clone()));
    }
    if let Some(domain) = &observable.domain {
        attrs.insert("domain".into(), JsonValue::String(domain.clone()));
    }
    if let Some(target) = &observable.target {
        attrs.insert("target".into(), JsonValue::String(target.clone()));
    }

    GroupBuilder::new()
        .attributes(attrs)
        .build(store.clone(), path)?
        .store_metadata()?;

    match &observable.data {
        ObservableData::Column(column) => write_column(store, &format!("{}/data", path), column)?,
        ObservableData::Grid(grid) => write_grid(store, &format!("{}/data", path), grid)?,
    }

    Ok(())
}

fn read_observable(
    store: &ReadableWritableListableStorage,
    path: &str,
) -> Result<ObservableRecord, MolRsError> {
    let group = zarrs::group::Group::open(store.clone(), path)?;
    let attrs = group.attributes();
    let kind = match attrs
        .get("kind")
        .and_then(|v| v.as_str())
        .unwrap_or("scalar")
    {
        "scalar" => ObservableKind::Scalar,
        "vector" => ObservableKind::Vector,
        "grid" => ObservableKind::Grid,
        other => {
            return Err(MolRsError::zarr(format!(
                "unsupported observable kind '{}'",
                other
            )));
        }
    };
    let description = attrs
        .get("description")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();
    let time_dependent = attrs
        .get("time_dependent")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    let axes = attrs
        .get("axes")
        .and_then(|v| v.as_str())
        .and_then(|s| serde_json::from_str::<Vec<String>>(s).ok())
        .unwrap_or_default();
    let extra = attrs
        .get("extra")
        .and_then(|v| v.as_str())
        .and_then(|s| serde_json::from_str(s).ok())
        .unwrap_or_default();

    let data_path = format!("{}/data", path);
    let data = if kind == ObservableKind::Grid {
        ObservableData::Grid(read_grid(store, &data_path)?)
    } else {
        ObservableData::Column(read_column(store, &data_path)?)
    };

    Ok(ObservableRecord {
        name: attrs
            .get("name")
            .and_then(|v| v.as_str())
            .unwrap_or_else(|| path.rsplit('/').next().unwrap_or("observable"))
            .to_string(),
        kind,
        description,
        time_dependent,
        unit: attrs
            .get("unit")
            .and_then(|v| v.as_str())
            .map(str::to_string),
        axes,
        sampling: attrs
            .get("sampling")
            .and_then(|v| v.as_str())
            .map(str::to_string),
        domain: attrs
            .get("domain")
            .and_then(|v| v.as_str())
            .map(str::to_string),
        target: attrs
            .get("target")
            .and_then(|v| v.as_str())
            .map(str::to_string),
        extra,
        data,
    })
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

fn json_err(e: impl std::fmt::Display) -> MolRsError {
    MolRsError::zarr(format!("json error: {}", e))
}

#[cfg(test)]
mod tests {
    use tempfile::tempdir;

    use super::*;
    use molrs::grid::Grid;

    #[test]
    fn molrec_store_roundtrip_preserves_grid_in_frame() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("record.zarr");

        let mut frame = Frame::new();
        let mut grid = Grid::new(
            [2, 2, 2],
            [0.0; 3],
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            [false; 3],
        );
        grid.insert("density", vec![0.1; 8]).unwrap();
        frame.insert_grid("electron_density", grid);

        let mut rec = MolRec::new(frame);
        rec.meta = serde_json::json!({"version": [0, 2]});
        rec.method = serde_json::json!({"type": "workflow"});

        write_molrec_file(&path, &rec).unwrap();
        let loaded = read_molrec_file(&path).unwrap();
        assert!(loaded.frame.has_grid("electron_density"));
        assert_eq!(loaded.method["type"], "workflow");
    }
}
