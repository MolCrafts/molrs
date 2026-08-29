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
//! ├── trajectory/    the frame sequence — see [`crate::io::zarr::sequence`]
//! ├── observables/   meta/<name> (semantics) + <name> (data)
//! ├── method/        JSON attributes
//! ├── status/        JSON attributes
//! └── metrics/       dense series arrays + catalog attrs; optional JSONL WAL
//! ```
//!
//! ## Metrics: dense Zarr SoT, JSONL WAL for live append
//!
//! Closed training / monitor curves densify to **Zarr series arrays** under
//! `metrics/` (molrec `docs/spec/metrics.md`). Live append uses
//! `metrics/metrics.jsonl` — do **not** use per-step Zarr chunk append.
//! Higher layers (molexp / molnex) own the WAL → densify path.
//!
//! When `write_record_*` materialises a `metrics` map into a Zarr group, that
//! is a **closed catalog / summary**. Readers that need the full curve MUST
//! open dense series arrays when present, else fall back to the JSONL WAL.
//!
//! Sections the reader does not interpret are preserved verbatim into
//! [`MolRec::extra_sections`] rather than dropped.
//!
//! ## The `trajectory/` section has one owner
//!
//! This module owns the **record-level** sections — `meta`, `frame`, `system`,
//! `observables`, the JSON groups — and the two path-taking doors. It does
//! **not** own the `trajectory/` layout: that layout has exactly one encoder
//! and one decoder, both in [`crate::io::zarr::sequence`], and the record
//! writer and reader drive them rather than restating them. Nothing here
//! encodes or decodes a row pointer, a step index, or a frame group.

#[cfg(feature = "filesystem")]
use std::path::Path;
#[cfg(feature = "filesystem")]
use std::sync::Arc;

use serde_json::{Map as JsonMap, Value as JsonValue};
use zarrs::array::{Array, ArraySubset};
#[cfg(feature = "filesystem")]
use zarrs::filesystem::FilesystemStore;
#[cfg(feature = "zarr")]
use zarrs::group::GroupBuilder;
use zarrs::node::{Node, NodeMetadata};
use zarrs::storage::ReadableWritableListableStorage;
#[cfg(feature = "zarr")]
use zarrs::storage::WritableStorageTraits;

use crate::io::zarr::frame_io::{join_path, read_column, read_frame_group};
#[cfg(feature = "zarr")]
use crate::io::zarr::frame_io::{node_prefix, write_column, write_frame_group};
use crate::io::zarr::sequence::FrameSequence;
#[cfg(feature = "zarr")]
use crate::io::zarr::sequence::{FrameSequenceWriter, SequenceSchema};
#[cfg(feature = "filesystem")]
use crate::io::zarr::store::PositionalWriteStore;
use molrs::MolRsError;
use molrs::store::record::{MolRec, RECORD_FORMAT_NAME, RECORD_SCHEMA_VERSION};
#[cfg(feature = "filesystem")]
use molrs::store::trajectory::Trajectory;
use molrs::store::trajectory::{ObservableData, ObservableKind, ObservableRecord};

// ---------------------------------------------------------------------------
// Write
// ---------------------------------------------------------------------------

/// Write a record to a filesystem path.
///
/// The store is a `PositionalWriteStore` (private to [`crate::io::zarr`]), not
/// a bare `FilesystemStore`: on
/// the stock store every partial write is a whole-value read-modify-write, so
/// a shard-sized value pays O(file) per append (see [`crate::io::zarr`]).
/// [`write_trajectory_file`] reaches the same door through here.
#[cfg(feature = "filesystem")]
pub fn write_record_file(path: impl AsRef<Path>, record: &MolRec) -> Result<(), MolRsError> {
    let store: ReadableWritableListableStorage =
        Arc::new(PositionalWriteStore::new(path.as_ref())?);
    write_record_store(store, record)
}

/// Write a record into an open store, rooted at `/`.
#[cfg(feature = "zarr")]
pub fn write_record_store(
    store: ReadableWritableListableStorage,
    record: &MolRec,
) -> Result<(), MolRsError> {
    record.validate()?;
    let prefix = "/";

    // Erase before writing: this record is the whole content of the store root,
    // so a rewrite must not inherit the previous record's sections, blocks,
    // columns or frames. Every writer clears its own target node this way.
    store.erase_prefix(&node_prefix(prefix)?)?;

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
        // The `trajectory/` layout has exactly one encoder and it is not here:
        // this door mints the schema from the frames themselves and drives
        // `FrameSequenceWriter`. The root erase above has already emptied the
        // node, which is what `create` insists on before it will mint.
        let mut writer = FrameSequenceWriter::create(
            store.clone(),
            SequenceSchema::from_frames(&trajectory.frames)?,
        )?;
        for (index, frame) in trajectory.frames.iter().enumerate() {
            // `record.validate()` above pinned `step` and `time` to
            // `frames.len()`, so both indexings are in range.
            let time = trajectory.time.as_ref().map(|times| times[index]);
            match (&trajectory.step, time) {
                (Some(steps), _) => writer.append_at(frame, steps[index], time)?,
                // No step numbers of its own and no time to carry: the auto
                // door owns the numbering, so nothing restates it.
                (None, None) => writer.append(frame)?,
                // A time but no step number. The auto door takes no time, so
                // its numbering (0, 1, 2, …) has to be spelled out — dropping
                // the time instead would be silent data loss.
                (None, Some(_)) => writer.append_at(frame, index as i64, time)?,
            }
        }
        writer.close()?;
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
#[cfg(feature = "zarr")]
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

#[cfg(feature = "zarr")]
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

#[cfg(feature = "zarr")]
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
            "trajectory" => {
                // One decoder, and it is not here. `FrameSequence::open`
                // resolves the same `/trajectory` node this child is, and a
                // store still carrying the pre-0.14 `trajectory/frames/` tree
                // errors out of it rather than reading back as empty.
                // Demoted on the way in: the decoder is a read door, so it is
                // handed the store's read-only view rather than this one.
                let mut sequence = FrameSequence::open(store.clone().readable_listable())?;
                record.trajectory = Some(sequence.to_trajectory()?);
            }
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
            // An observable's array is its whole column.
            data: ObservableData::Column(read_column(
                store,
                path,
                &ArraySubset::new_with_shape(Array::open(store.clone(), path)?.shape().to_vec()),
            )?),
        };
        record.observables.insert(obs)?;
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Trajectory-only doors (narrow entry points onto the same record layout)
// ---------------------------------------------------------------------------

/// Write a trajectory as a record whose only state section is `trajectory`.
///
/// The encoding is [`FrameSequenceWriter`]'s, reached through
/// [`write_record_store`]; this door only shapes the record around it.
#[cfg(feature = "filesystem")]
pub fn write_trajectory_file(
    path: impl AsRef<Path>,
    trajectory: &Trajectory,
) -> Result<(), MolRsError> {
    let mut record = MolRec::new();
    record.trajectory = Some(trajectory.clone());
    write_record_file(path, &record)
}

/// Read the `trajectory` section of a record.
///
/// The decoding is [`FrameSequence`]'s, reached through [`read_record_store`];
/// a store still carrying the pre-0.14 `trajectory/frames/` tree therefore
/// fails here too, naming the legacy layout.
#[cfg(feature = "filesystem")]
pub fn read_trajectory_file(path: impl AsRef<Path>) -> Result<Trajectory, MolRsError> {
    Ok(read_record_file(path)?.trajectory.unwrap_or_default())
}

pub(in crate::io::zarr) fn zerr(e: impl std::fmt::Display) -> MolRsError {
    MolRsError::zarr(e.to_string())
}

#[cfg(all(test, feature = "filesystem"))]
mod tests {
    use super::*;
    use molrs::store::block::{Block, Column};
    use molrs::store::frame::Frame;
    use molrs::store::record::RESERVED_META_KEYS;
    use molrs::types::F;
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

    /// One `atoms` block with one `f64` column, at the caller's values.
    ///
    /// [`frame_with_atoms`] fills with 1.0, which cannot tell a rewritten
    /// store from a stale one; these values can.
    fn frame_with_x(values: &[F]) -> Frame {
        let mut block = Block::new();
        block.insert_column("x", float_column(values)).unwrap();
        let mut frame = Frame::new();
        frame.insert("atoms", block);
        frame
    }

    /// Column `x` of the `atoms` block, as it came back from the store.
    fn atoms_x(frame: &Frame) -> Vec<F> {
        frame
            .get("atoms")
            .expect("the frame carries an atoms block")
            .get_float("x")
            .expect("column x arrived as f64")
            .iter()
            .copied()
            .collect()
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

    /// Every node in the store, as a path relative to the record root (the
    /// root group itself is the empty string). One `zarr.json` is one node.
    fn node_paths(root: &Path) -> Vec<String> {
        let mut paths: Vec<String> = walk_json(root)
            .iter()
            .map(|file| {
                file.parent()
                    .unwrap()
                    .strip_prefix(root)
                    .unwrap()
                    .to_string_lossy()
                    .into_owned()
            })
            .collect();
        paths.sort();
        paths
    }

    #[test]
    fn rewrite_leaves_no_stale_node() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("record.zarr");

        // A wide record first: a second block, a second column, and a
        // trajectory/frames/ subtree.
        let mut atoms = Block::new();
        atoms
            .insert("x", ArrayD::from_shape_vec(vec![5], vec![1.0; 5]).unwrap())
            .unwrap();
        atoms
            .insert("y", ArrayD::from_shape_vec(vec![5], vec![2.0; 5]).unwrap())
            .unwrap();
        let mut bonds = Block::new();
        bonds
            .insert(
                "atomi",
                ArrayD::from_shape_vec(vec![2], vec![0u64, 1u64]).unwrap(),
            )
            .unwrap();
        let mut frame = Frame::new();
        frame.insert("atoms", atoms);
        frame.insert("bonds", bonds);
        let mut wide = MolRec::new();
        wide.frame = Some(frame);
        wide.add_frame(frame_with_atoms(5));
        write_record_file(&path, &wide).unwrap();

        // Then a strictly smaller one into the same path.
        let mut narrow = MolRec::new();
        narrow.frame = Some(frame_with_atoms(3));
        write_record_file(&path, &narrow).unwrap();

        assert_eq!(
            node_paths(&path),
            vec!["", "frame", "frame/atoms", "frame/atoms/x", "meta"]
        );
    }

    /// The trajectory door writes the frames **once**.
    ///
    /// Today it copies `frames[0]` into the record's `frame` section to get
    /// past `MolRec::validate`, so frame 0 lands in the store twice — once as
    /// `frame/` and once as row 0 of the sequence. That duplicate is a second
    /// copy of real data (bytes, and a wrong `frame` section a reader will
    /// believe), so the pin is on the store's own node list: a trajectory-only
    /// record has a `trajectory` section and no `frame` one at all.
    #[test]
    fn a_written_trajectory_store_holds_no_duplicate_frame_zero() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("record.zarr");

        let traj = Trajectory {
            frames: vec![
                frame_with_x(&[1.0, 2.0, 3.0]),
                frame_with_x(&[4.0, 5.0, 6.0]),
            ],
            step: Some(vec![0, 1]),
            time: None,
        };
        write_trajectory_file(&path, &traj).unwrap();

        let nodes = node_paths(&path);
        let frame_nodes: Vec<&String> = nodes
            .iter()
            .filter(|node| node.as_str() == "frame" || node.starts_with("frame/"))
            .collect();
        assert!(
            frame_nodes.is_empty(),
            "a trajectory-only record must not duplicate frame 0 into a \
             'frame' section; store holds {frame_nodes:?} among {nodes:?}"
        );
        assert!(
            nodes.iter().any(|node| node == "trajectory"),
            "the frames belong to the trajectory section: {nodes:?}"
        );
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

    /// A second trajectory record written over the same path replaces the
    /// first, and the read-back is the second one.
    ///
    /// This is the seam between two rules that only look compatible:
    /// [`write_record_store`] erases the root before writing, and
    /// `FrameSequenceWriter::create` **refuses** a path that still holds a
    /// `trajectory/step` array rather than silently overwriting somebody's
    /// run. Get the order or the prefix wrong and the second write is a hard
    /// `Err`, not a wrong answer — so the pin is that it succeeds *and* that
    /// no frame, step or time of the first record survives it.
    #[test]
    fn rewriting_a_trajectory_record_over_the_same_path_succeeds() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("record.zarr");

        let first = Trajectory {
            frames: vec![
                frame_with_x(&[1.0, 2.0, 3.0]),
                frame_with_x(&[4.0, 5.0, 6.0]),
            ],
            step: Some(vec![0, 1]),
            time: None,
        };
        write_trajectory_file(&path, &first).unwrap();

        // Different in every axis the layout carries: fewer rows per frame,
        // more frames, other step numbers, and times where there were none.
        let second = Trajectory {
            frames: vec![
                frame_with_x(&[-1.5, -2.5]),
                frame_with_x(&[-3.5, -4.5]),
                frame_with_x(&[-5.5, -6.5]),
            ],
            step: Some(vec![7, 8, 9]),
            time: Some(vec![0.25, 0.5, 0.75]),
        };
        write_trajectory_file(&path, &second).unwrap();

        let loaded = read_trajectory_file(&path).unwrap();
        assert_eq!(
            loaded.frames.iter().map(atoms_x).collect::<Vec<Vec<F>>>(),
            second.frames.iter().map(atoms_x).collect::<Vec<Vec<F>>>(),
            "the read-back must be the second record's frames, bit for bit"
        );
        assert_eq!(loaded.step, second.step, "and its step numbers");
        assert_eq!(loaded.time, second.time, "and its times");
    }

    /// The eager door refuses a legacy store in the same words
    /// `FrameSequence::open` uses (ac-019's second door, decision 10).
    ///
    /// Both doors lead to the one decoder, so this is not a second
    /// implementation of the check — it is the pin that the record reader
    /// does not swallow it. The fixture is a real 0.14 record with the
    /// pre-0.14 `trajectory/frames/` tree grafted back on: the old writer is
    /// gone, and that group is what identifies the layout.
    #[test]
    fn read_trajectory_file_refuses_a_legacy_store() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("record.zarr");
        write_trajectory_file(
            &path,
            &Trajectory::from_frames(vec![frame_with_x(&[1.0, 2.0])]),
        )
        .unwrap();

        let store: ReadableWritableListableStorage = Arc::new(FilesystemStore::new(&path).unwrap());
        for group in ["/trajectory/frames", "/trajectory/frames/0"] {
            GroupBuilder::new()
                .build(store.clone(), group)
                .unwrap()
                .store_metadata()
                .unwrap();
        }

        let message = read_trajectory_file(&path)
            .expect_err("the old layout must be refused, not read as an empty trajectory")
            .to_string();
        // The comparison glyph is the implementer's; everything around it is
        // the pinned message, shared with the `FrameSequence::open` door.
        assert!(
            message.contains("legacy layout (written by molrs"),
            "must name the layout: {message}"
        );
        assert!(
            message.contains("0.13); re-write with 0.13"),
            "must say which writer produced it and how to migrate: {message}"
        );
    }
}
