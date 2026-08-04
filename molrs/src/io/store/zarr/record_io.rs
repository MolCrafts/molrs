//! Zarr V3 binding for [`MolRec`] — the reference L4 binding of the MolRec
//! contract (<https://github.com/MolCrafts/molrec>) for **array sections**.
//!
//! One record is one openable root:
//!
//! ```text
//! <root>/
//! ├── meta/          record_schema_version = 1, format_name = "molrec", + producer keys
//! ├── system/        frame-shaped group (topology / types)
//! ├── frame/         frame-shaped group (snapshot)
//! ├── trajectory/    step, time, frames/0..n-1/
//! ├── observables/   meta/<name> (semantics) + <name> (data)
//! ├── method/        JSON attributes
//! ├── status/        JSON attributes
//! └── metrics/       JSON attributes (closed snapshot / summary only)
//! ```
//!
//! ## Metrics are not a Zarr append stream
//!
//! Live training / monitor curves use the **JSONL** reference binding under
//! `metrics/metrics.jsonl` (molrec `docs/spec/metrics.md`). Do **not** use
//! this module's Zarr group attributes as the live append path — Zarr is a
//! poor fit for per-step scalar append. Higher layers (molnex provisional
//! writer, later molpy) own the JSONL stream.
//!
//! When `write_record_*` materialises a `metrics` map into a Zarr group, that
//! is a **closed summary** for pure-Zarr aggregates, not a substitute for
//! `metrics.jsonl`. A hybrid filesystem record root may keep JSONL beside
//! Zarr array sections; readers that need the full curve must open the
//! JSONL file when it exists.
//!
//! Sections the reader does not interpret are preserved verbatim into
//! [`MolRec::extra_sections`] rather than dropped.
//!
//! Trajectories are stored as an ordered list of frame groups. The contract also
//! describes a *packed* convention (per-block arrays with a leading time axis);
//! that is a storage optimisation over the same logical model and is not
//! implemented here.

#[cfg(feature = "filesystem")]
use std::path::Path;
#[cfg(feature = "filesystem")]
use std::sync::Arc;

use serde_json::{Map as JsonMap, Value as JsonValue};
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

use crate::io::store::zarr::frame_io::{join_path, read_column, read_frame_group};
#[cfg(feature = "filesystem")]
use crate::io::store::zarr::frame_io::{write_column, write_frame_group};
use molrs::MolRsError;
use molrs::store::record::{MolRec, RECORD_FORMAT_NAME, RECORD_SCHEMA_VERSION};
use molrs::store::trajectory::{ObservableData, ObservableKind, ObservableRecord, Trajectory};
use molrs::types::F;

// ---------------------------------------------------------------------------
// Write
// ---------------------------------------------------------------------------

/// Write a record to a filesystem path.
#[cfg(feature = "filesystem")]
pub fn write_record_file(path: impl AsRef<Path>, record: &MolRec) -> Result<(), MolRsError> {
    let store: ReadableWritableListableStorage =
        Arc::new(FilesystemStore::new(path.as_ref()).map_err(zerr)?);
    write_record_store(store, record)
}

/// Write a record into an open store, rooted at `/`.
#[cfg(feature = "filesystem")]
pub fn write_record_store(
    store: ReadableWritableListableStorage,
    record: &MolRec,
) -> Result<(), MolRsError> {
    record.validate()?;
    let prefix = "/";

    GroupBuilder::new()
        .build(store.clone(), prefix)?
        .store_metadata()?;

    write_meta(&store, &join_path(prefix, "meta"), &record.meta)?;

    if let Some(system) = &record.system {
        write_frame_group(&store, &join_path(prefix, "system"), system)?;
    }
    if let Some(frame) = &record.frame {
        write_frame_group(&store, &join_path(prefix, "frame"), frame)?;
    }
    if let Some(trajectory) = &record.trajectory {
        write_trajectory_section(&store, &join_path(prefix, "trajectory"), trajectory)?;
    }
    if !record.observables.is_empty() {
        write_observables(&store, &join_path(prefix, "observables"), record)?;
    }
    for (name, section) in [
        ("method", &record.method),
        ("status", &record.status),
        ("metrics", &record.metrics),
    ] {
        if !section.is_empty() {
            write_json_group(&store, &join_path(prefix, name), section)?;
        }
    }
    for (name, frame) in &record.extra_sections {
        write_frame_group(&store, &join_path(prefix, name), frame)?;
    }

    Ok(())
}

/// Write `meta`, stamping the contract-owned keys over any producer copy.
#[cfg(feature = "filesystem")]
fn write_meta(
    store: &ReadableWritableListableStorage,
    path: &str,
    meta: &JsonMap<String, JsonValue>,
) -> Result<(), MolRsError> {
    let mut attrs = meta.clone();
    attrs.insert("record_schema_version".into(), RECORD_SCHEMA_VERSION.into());
    attrs.insert("format_name".into(), RECORD_FORMAT_NAME.into());
    write_json_group(store, path, &attrs)
}

#[cfg(feature = "filesystem")]
fn write_json_group(
    store: &ReadableWritableListableStorage,
    path: &str,
    attrs: &JsonMap<String, JsonValue>,
) -> Result<(), MolRsError> {
    GroupBuilder::new()
        .attributes(attrs.clone())
        .build(store.clone(), path)?
        .store_metadata()?;
    Ok(())
}

#[cfg(feature = "filesystem")]
fn write_trajectory_section(
    store: &ReadableWritableListableStorage,
    prefix: &str,
    trajectory: &Trajectory,
) -> Result<(), MolRsError> {
    trajectory.validate()?;

    GroupBuilder::new()
        .build(store.clone(), prefix)?
        .store_metadata()?;

    if let Some(step) = &trajectory.step {
        write_i64_array(
            store,
            &join_path(prefix, "step"),
            &[step.len() as u64],
            step,
        )?;
    }
    if let Some(time) = &trajectory.time {
        write_f64_array(
            store,
            &join_path(prefix, "time"),
            &[time.len() as u64],
            time,
        )?;
    }

    let frames_path = join_path(prefix, "frames");
    GroupBuilder::new()
        .build(store.clone(), &frames_path)?
        .store_metadata()?;
    for (index, frame) in trajectory.frames.iter().enumerate() {
        write_frame_group(store, &join_path(&frames_path, &index.to_string()), frame)?;
    }
    Ok(())
}

#[cfg(feature = "filesystem")]
fn write_observables(
    store: &ReadableWritableListableStorage,
    prefix: &str,
    record: &MolRec,
) -> Result<(), MolRsError> {
    GroupBuilder::new()
        .build(store.clone(), prefix)?
        .store_metadata()?;
    let meta_path = join_path(prefix, "meta");
    GroupBuilder::new()
        .build(store.clone(), &meta_path)?
        .store_metadata()?;

    for (name, obs) in record.observables.iter() {
        let mut attrs = obs.extra.clone();
        attrs.insert("kind".into(), obs.kind.as_str().into());
        attrs.insert("description".into(), obs.description.clone().into());
        attrs.insert("time_dependent".into(), obs.time_dependent.into());
        if let Some(unit) = &obs.unit {
            attrs.insert("unit".into(), unit.clone().into());
        }
        if !obs.axes.is_empty() {
            attrs.insert("axes".into(), obs.axes.clone().into());
        }
        if let Some(sampling) = &obs.sampling {
            attrs.insert("sampling".into(), sampling.clone().into());
        }
        if let Some(domain) = &obs.domain {
            attrs.insert("domain".into(), domain.clone().into());
        }
        if let Some(target) = &obs.target {
            attrs.insert("target".into(), target.clone().into());
        }
        write_json_group(store, &join_path(&meta_path, name), &attrs)?;

        let ObservableData::Column(column) = &obs.data;
        write_column(store, &join_path(prefix, name), column)?;
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Read
// ---------------------------------------------------------------------------

/// Read a record from a filesystem path.
#[cfg(feature = "filesystem")]
pub fn read_record_file(path: impl AsRef<Path>) -> Result<MolRec, MolRsError> {
    let store: ReadableWritableListableStorage =
        Arc::new(FilesystemStore::new(path.as_ref()).map_err(zerr)?);
    read_record_store(store)
}

/// Read a record from an open store, rooted at `/`.
pub fn read_record_store(store: ReadableWritableListableStorage) -> Result<MolRec, MolRsError> {
    let prefix = "/";
    let mut record = MolRec::new();

    record.meta = read_meta(&store, &join_path(prefix, "meta"))?;

    let root = Node::open(&store, prefix)?;
    for child in root.children() {
        if !matches!(child.metadata(), NodeMetadata::Group(_)) {
            continue;
        }
        let path = child.path().as_str().to_string();
        let name = path.rsplit('/').next().unwrap_or("").to_string();
        if name.is_empty() {
            continue;
        }
        match name.as_str() {
            "meta" => {}
            "system" => record.system = Some(read_frame_group(&store, &path)?),
            "frame" => record.frame = Some(read_frame_group(&store, &path)?),
            "trajectory" => record.trajectory = Some(read_trajectory_section(&store, &path)?),
            "observables" => read_observables(&store, &path, &mut record)?,
            "method" => record.method = read_json_group(&store, &path)?,
            "status" => record.status = read_json_group(&store, &path)?,
            "metrics" => record.metrics = read_json_group(&store, &path)?,
            _ => {
                // Preserve the unknown: keep foreign sections rather than
                // silently dropping a newer producer's data on round-trip.
                record
                    .extra_sections
                    .insert(name, read_frame_group(&store, &path)?);
            }
        }
    }

    Ok(record)
}

/// Read and validate the mandatory `meta` section.
fn read_meta(
    store: &ReadableWritableListableStorage,
    path: &str,
) -> Result<JsonMap<String, JsonValue>, MolRsError> {
    let group = zarrs::group::Group::open(store.clone(), path)
        .map_err(|_| MolRsError::zarr("not a MolRec record: missing required 'meta' section"))?;
    let attrs = group.attributes().clone();

    let version = attrs
        .get("record_schema_version")
        .and_then(JsonValue::as_u64)
        .ok_or_else(|| MolRsError::zarr("meta is missing 'record_schema_version'"))?;
    if version != RECORD_SCHEMA_VERSION {
        return Err(MolRsError::zarr(format!(
            "unsupported record_schema_version {version}; expected {RECORD_SCHEMA_VERSION}"
        )));
    }
    match attrs.get("format_name").and_then(JsonValue::as_str) {
        Some(RECORD_FORMAT_NAME) => {}
        Some(other) => {
            return Err(MolRsError::zarr(format!(
                "unsupported format_name '{other}'; expected '{RECORD_FORMAT_NAME}'"
            )));
        }
        None => return Err(MolRsError::zarr("meta is missing 'format_name'")),
    }
    Ok(attrs)
}

fn read_json_group(
    store: &ReadableWritableListableStorage,
    path: &str,
) -> Result<JsonMap<String, JsonValue>, MolRsError> {
    Ok(zarrs::group::Group::open(store.clone(), path)?
        .attributes()
        .clone())
}

fn read_trajectory_section(
    store: &ReadableWritableListableStorage,
    prefix: &str,
) -> Result<Trajectory, MolRsError> {
    let step = match Array::open(store.clone(), &join_path(prefix, "step")) {
        Ok(arr) => Some(read_i64_values(arr)?),
        Err(_) => None,
    };
    let time = match Array::open(store.clone(), &join_path(prefix, "time")) {
        Ok(arr) => Some(read_float_values(arr)?),
        Err(_) => None,
    };

    let mut frames = Vec::new();
    let frames_path = join_path(prefix, "frames");
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
            frames.push(read_frame_group(store, child.path().as_str())?);
        }
    }

    let trajectory = Trajectory { frames, step, time };
    trajectory.validate()?;
    Ok(trajectory)
}

fn read_observables(
    store: &ReadableWritableListableStorage,
    prefix: &str,
    record: &mut MolRec,
) -> Result<(), MolRsError> {
    let meta_path = join_path(prefix, "meta");
    let node = Node::open(store, prefix)?;

    for child in node.children() {
        if !matches!(child.metadata(), NodeMetadata::Array(_)) {
            continue;
        }
        let path = child.path().as_str();
        let name = path.rsplit('/').next().unwrap_or("");
        if name.is_empty() {
            continue;
        }

        let attrs = zarrs::group::Group::open(store.clone(), &join_path(&meta_path, name))
            .map_err(|_| {
                MolRsError::zarr(format!(
                    "observable '{name}' has data but no 'observables/meta/{name}' entry"
                ))
            })?
            .attributes()
            .clone();

        let kind_str = attrs
            .get("kind")
            .and_then(JsonValue::as_str)
            .ok_or_else(|| MolRsError::zarr(format!("observable '{name}' is missing 'kind'")))?;
        let kind = ObservableKind::parse(kind_str)
            .ok_or_else(|| MolRsError::zarr(format!("unknown observable kind '{kind_str}'")))?;

        let mut extra = attrs.clone();
        for key in [
            "kind",
            "description",
            "time_dependent",
            "unit",
            "axes",
            "sampling",
            "domain",
            "target",
        ] {
            extra.remove(key);
        }

        let obs = ObservableRecord {
            name: name.to_string(),
            kind,
            description: attrs
                .get("description")
                .and_then(JsonValue::as_str)
                .unwrap_or("")
                .to_string(),
            time_dependent: attrs
                .get("time_dependent")
                .and_then(JsonValue::as_bool)
                .unwrap_or(false),
            unit: attrs
                .get("unit")
                .and_then(JsonValue::as_str)
                .map(str::to_string),
            axes: attrs
                .get("axes")
                .and_then(JsonValue::as_array)
                .map(|items| {
                    items
                        .iter()
                        .filter_map(|v| v.as_str().map(str::to_string))
                        .collect()
                })
                .unwrap_or_default(),
            sampling: attrs
                .get("sampling")
                .and_then(JsonValue::as_str)
                .map(str::to_string),
            domain: attrs
                .get("domain")
                .and_then(JsonValue::as_str)
                .map(str::to_string),
            target: attrs
                .get("target")
                .and_then(JsonValue::as_str)
                .map(str::to_string),
            extra,
            data: ObservableData::Column(read_column(store, path)?),
        };
        record.observables.insert(obs)?;
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Trajectory-only doors (narrow entry points onto the same record layout)
// ---------------------------------------------------------------------------

/// Write a trajectory as a record whose only state section is `trajectory`.
#[cfg(feature = "filesystem")]
pub fn write_trajectory_file(
    path: impl AsRef<Path>,
    trajectory: &Trajectory,
) -> Result<(), MolRsError> {
    let mut record = MolRec::new();
    record.frame = trajectory.frames.first().cloned();
    record.trajectory = Some(trajectory.clone());
    write_record_file(path, &record)
}

/// Read the `trajectory` section of a record.
#[cfg(feature = "filesystem")]
pub fn read_trajectory_file(path: impl AsRef<Path>) -> Result<Trajectory, MolRsError> {
    Ok(read_record_file(path)?.trajectory.unwrap_or_default())
}

/// Read a single frame of a record's trajectory by index.
pub fn read_frame_from_store(
    store: ReadableWritableListableStorage,
    index: usize,
) -> Result<Option<molrs::store::frame::Frame>, MolRsError> {
    Ok(read_record_store(store)?
        .trajectory
        .and_then(|traj| traj.frames.into_iter().nth(index)))
}

/// Count the frames a stored record carries.
pub fn count_frames_in_store(store: ReadableWritableListableStorage) -> Result<u64, MolRsError> {
    Ok(read_record_store(store)?.count_frames() as u64)
}

// ---------------------------------------------------------------------------
// Primitive array helpers
// ---------------------------------------------------------------------------

#[cfg(feature = "filesystem")]
fn write_f64_array(
    store: &ReadableWritableListableStorage,
    path: &str,
    shape: &[u64],
    data: &[F],
) -> Result<(), MolRsError> {
    let arr = ArrayBuilder::new(shape.to_vec(), shape.to_vec(), data_type::float64(), 0.0f64)
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
    let subset = ArraySubset::new_with_shape(arr.shape().to_vec());
    let dt = arr.data_type();
    if dt.is::<Float32DataType>() {
        let data: Vec<f32> = arr.retrieve_array_subset(&subset).map_err(zerr)?;
        Ok(data.into_iter().map(|v| v as F).collect())
    } else if dt.is::<Float64DataType>() {
        arr.retrieve_array_subset(&subset).map_err(zerr)
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
    if arr.data_type().is::<Int64DataType>() {
        arr.retrieve_array_subset(&subset).map_err(zerr)
    } else {
        Err(MolRsError::zarr(format!(
            "expected int64 array, got {:?}",
            arr.data_type()
        )))
    }
}

fn zerr(e: impl std::fmt::Display) -> MolRsError {
    MolRsError::zarr(e.to_string())
}

#[cfg(all(test, feature = "filesystem"))]
mod tests {
    use super::*;
    use molrs::store::block::{Block, Column};
    use molrs::store::frame::Frame;
    use molrs::store::record::RESERVED_META_KEYS;
    use ndarray::ArrayD;
    use tempfile::tempdir;

    fn float_column(values: &[F]) -> Column {
        Column::from_float(ArrayD::from_shape_vec(vec![values.len()], values.to_vec()).unwrap())
    }

    fn frame_with_atoms(n: usize) -> Frame {
        let mut block = Block::new();
        block
            .insert("x", ArrayD::from_shape_vec(vec![n], vec![1.0; n]).unwrap())
            .unwrap();
        let mut frame = Frame::new();
        frame.insert("atoms", block);
        frame
    }

    fn write_then_read(record: &MolRec) -> MolRec {
        let dir = tempdir().unwrap();
        let path = dir.path().join("record.zarr");
        write_record_file(&path, record).unwrap();
        read_record_file(&path).unwrap()
    }

    #[test]
    fn meta_carries_the_contract_keys() {
        let mut rec = MolRec::new();
        rec.frame = Some(Frame::new());
        let loaded = write_then_read(&rec);
        assert_eq!(
            loaded.meta.get("record_schema_version").unwrap().as_u64(),
            Some(RECORD_SCHEMA_VERSION)
        );
        assert_eq!(
            loaded.meta.get("format_name").unwrap().as_str(),
            Some(RECORD_FORMAT_NAME)
        );
    }

    #[test]
    fn no_section_carries_a_frame_schema_version() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("record.zarr");
        let mut rec = MolRec::new();
        rec.frame = Some(frame_with_atoms(3));
        rec.add_frame(frame_with_atoms(3));
        write_record_file(&path, &rec).unwrap();

        // The contract forbids a parallel per-frame schema version anywhere.
        for entry in walk_json(&path) {
            let text = std::fs::read_to_string(&entry).unwrap();
            assert!(
                !text.contains("frame_schema_version"),
                "{} still emits frame_schema_version",
                entry.display()
            );
        }
    }

    fn walk_json(root: &Path) -> Vec<std::path::PathBuf> {
        let mut out = Vec::new();
        let mut stack = vec![root.to_path_buf()];
        while let Some(dir) = stack.pop() {
            for entry in std::fs::read_dir(&dir).unwrap().flatten() {
                let path = entry.path();
                if path.is_dir() {
                    stack.push(path);
                } else if path.extension().is_some_and(|e| e == "json") {
                    out.push(path);
                }
            }
        }
        out
    }

    #[test]
    fn producer_meta_and_method_round_trip() {
        let mut rec = MolRec::new();
        rec.frame = Some(frame_with_atoms(2));
        rec.meta
            .insert("creator".into(), serde_json::json!({"name": "unit-test"}));
        rec.method.insert("type".into(), "static_structure".into());

        let loaded = write_then_read(&rec);
        assert_eq!(loaded.meta["creator"]["name"], "unit-test");
        assert_eq!(loaded.method["type"], "static_structure");
    }

    #[test]
    fn frame_blocks_round_trip() {
        let mut rec = MolRec::new();
        rec.frame = Some(frame_with_atoms(5));
        let loaded = write_then_read(&rec);
        assert_eq!(loaded.frame.unwrap().get("atoms").unwrap().nrows(), Some(5));
    }

    #[test]
    fn trajectory_round_trips_with_index_arrays() {
        let mut rec = MolRec::new();
        rec.frame = Some(frame_with_atoms(4));
        rec.add_frame(frame_with_atoms(4));
        rec.add_frame(frame_with_atoms(4));
        let traj = rec.trajectory.as_mut().unwrap();
        traj.step = Some(vec![0, 1]);
        traj.time = Some(vec![0.0, 0.5]);

        let loaded = write_then_read(&rec);
        assert_eq!(loaded.count_frames(), 2);
        let traj = loaded.trajectory.as_ref().unwrap();
        assert_eq!(traj.frames.len(), 2);
        assert_eq!(traj.step, Some(vec![0, 1]));
        assert_eq!(traj.time, Some(vec![0.0, 0.5]));
    }

    #[test]
    fn observables_round_trip_with_semantics() {
        let mut rec = MolRec::new();
        rec.frame = Some(Frame::new());
        let mut obs = ObservableRecord::scalar("total_energy", float_column(&[1.0, 1.5, 2.0]));
        obs.description = "Total energy by step".into();
        obs.unit = Some("eV".into());
        obs.axes = vec!["timestep".into()];
        obs.time_dependent = true;
        obs.domain = Some("trajectory".into());
        rec.observables.insert(obs).unwrap();

        let loaded = write_then_read(&rec);
        let got = loaded.observables.get("total_energy").unwrap();
        assert_eq!(got.kind, ObservableKind::Scalar);
        assert_eq!(got.unit.as_deref(), Some("eV"));
        assert_eq!(got.axes, vec!["timestep".to_string()]);
        assert!(got.time_dependent);
        assert_eq!(got.domain.as_deref(), Some("trajectory"));
    }

    #[test]
    fn unknown_sections_survive_a_round_trip() {
        let mut rec = MolRec::new();
        rec.frame = Some(Frame::new());
        rec.extra_sections
            .insert("future_section".into(), frame_with_atoms(3));

        let loaded = write_then_read(&rec);
        assert_eq!(
            loaded
                .extra_sections
                .get("future_section")
                .unwrap()
                .get("atoms")
                .unwrap()
                .nrows(),
            Some(3)
        );
    }

    #[test]
    fn record_without_meta_is_rejected() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("record.zarr");
        let mut rec = MolRec::new();
        rec.frame = Some(Frame::new());
        write_record_file(&path, &rec).unwrap();
        std::fs::remove_dir_all(path.join("meta")).unwrap();
        assert!(read_record_file(&path).is_err());
    }

    #[test]
    fn every_noncurrent_record_schema_version_is_rejected() {
        for version in [None, Some(0_u64), Some(2), Some(99)] {
            let dir = tempdir().unwrap();
            let path = dir.path().join("record.zarr");
            let mut rec = MolRec::new();
            rec.frame = Some(Frame::new());
            write_record_file(&path, &rec).unwrap();

            let metadata_path = path.join("meta/zarr.json");
            let mut metadata: JsonValue =
                serde_json::from_slice(&std::fs::read(&metadata_path).unwrap()).unwrap();
            match version {
                Some(v) => metadata["attributes"]["record_schema_version"] = v.into(),
                None => {
                    metadata["attributes"]
                        .as_object_mut()
                        .unwrap()
                        .remove("record_schema_version");
                }
            }
            std::fs::write(&metadata_path, serde_json::to_vec(&metadata).unwrap()).unwrap();
            assert!(
                read_record_file(&path).is_err(),
                "accepted record_schema_version {version:?}"
            );
        }
    }

    #[test]
    fn foreign_format_name_is_rejected() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("record.zarr");
        let mut rec = MolRec::new();
        rec.frame = Some(Frame::new());
        write_record_file(&path, &rec).unwrap();

        let metadata_path = path.join("meta/zarr.json");
        let mut metadata: JsonValue =
            serde_json::from_slice(&std::fs::read(&metadata_path).unwrap()).unwrap();
        metadata["attributes"]["format_name"] = "molpy-zarr".into();
        std::fs::write(&metadata_path, serde_json::to_vec(&metadata).unwrap()).unwrap();
        assert!(read_record_file(&path).is_err());
    }

    #[test]
    fn a_record_with_no_state_section_is_refused_at_write() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("record.zarr");
        assert!(write_record_file(&path, &MolRec::new()).is_err());
    }

    #[test]
    fn trajectory_door_round_trips_through_the_record_layout() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("record.zarr");
        let mut frame = frame_with_atoms(3);
        frame.meta.insert("key", "value");
        let traj = Trajectory::from_frames(vec![frame]);

        write_trajectory_file(&path, &traj).unwrap();
        let loaded = read_trajectory_file(&path).unwrap();
        assert_eq!(loaded.frames.len(), 1);
        assert_eq!(
            loaded.frames[0].meta.get("key").unwrap().as_str(),
            Some("value")
        );

        // The narrow door writes a conforming record, not a private layout.
        let record = read_record_file(&path).unwrap();
        assert_eq!(
            record.meta.get("format_name").unwrap().as_str(),
            Some(RECORD_FORMAT_NAME)
        );
    }

    #[test]
    fn reserved_meta_keys_are_owned_by_the_writer() {
        let mut rec = MolRec::new();
        rec.frame = Some(Frame::new());
        // A producer trying to claim the contract keys must not win.
        rec.meta
            .insert("record_schema_version".into(), 99u64.into());
        rec.meta.insert("format_name".into(), "not-molrec".into());

        let loaded = write_then_read(&rec);
        for key in RESERVED_META_KEYS {
            assert!(loaded.meta.contains_key(key));
        }
        assert_eq!(
            loaded.meta["record_schema_version"].as_u64(),
            Some(RECORD_SCHEMA_VERSION)
        );
        assert_eq!(loaded.meta["format_name"], RECORD_FORMAT_NAME);
    }

    #[test]
    fn simbox_geometry_roundtrips_as_f64() {
        use molrs::spatial::simbox::SimBox;
        use ndarray::{Array2, array};

        let dir = tempdir().unwrap();
        let path = dir.path().join("simbox_f64.zarr");
        let h_val = 123.456789012345;
        let h = Array2::from_shape_vec(
            (3, 3),
            vec![h_val, 0.0, 0.0, 0.0, 50.0, 0.0, 0.0, 0.0, 40.0],
        )
        .unwrap();
        let origin = array![0.1, 0.2, 0.3];
        let mut frame = Frame::new();
        frame.simbox = Some(SimBox::new(h, origin, [true, true, true]).unwrap());
        let mut rec = MolRec::new();
        rec.frame = Some(frame);
        write_record_file(&path, &rec).unwrap();
        let back = read_record_file(&path).unwrap();
        let sb = back.frame.as_ref().unwrap().simbox.as_ref().unwrap();
        let h_back = sb.h_view()[[0, 0]];
        assert!(
            (h_back - h_val).abs() < 1e-15,
            "f64 simbox lost precision: {h_back} vs {h_val}"
        );
        assert!((sb.origin_view()[0] - 0.1).abs() < 1e-15);
    }
}
