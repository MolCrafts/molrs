//! Zarr V3 binding for [`MolRec`] — the reference L4 binding of the MolRec
//! contract (<https://github.com/MolCrafts/molrec>) for **array sections**.
//!
//! One record is one openable root:
//!
//! ```text
//! <root>/
//! ├── meta/          molrec_version = 1, + producer keys
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
use crate::io::zarr::schema;
use crate::io::zarr::sequence::FrameSequence;
#[cfg(feature = "zarr")]
use crate::io::zarr::sequence::{FrameSequenceWriter, SequenceSchema};
#[cfg(feature = "filesystem")]
use crate::io::zarr::store::PositionalWriteStore;
use molrs::MolRsError;
use molrs::store::block::Column;
// Not `filesystem`-gated: the store-taking section door below names it in
// every configuration, wasm included.
use molrs::store::frame::Frame;
use molrs::store::record::MolRec;
#[cfg(feature = "filesystem")]
use molrs::store::trajectory::Trajectory;
use molrs::store::trajectory::{ObservableData, ObservableKind, ObservableRecord};

// ---------------------------------------------------------------------------
// Write
// ---------------------------------------------------------------------------

/// Write a [`crate::Record`] to a filesystem path as a `*.mrec` directory.
///
/// The conventional suffix is `.mrec` (for example `water.mrec/`). Paths whose
/// file name ends in `.zarr` or `.zarr.zip` are refused; those were the
/// previous scientific suffixes and are not migrated. Other names are
/// accepted. A second write to the same path replaces the previous record
/// entirely — leftover sections from a wider record do not survive.
///
/// The writer writes the reserved `meta` key over any producer copy:
/// [`crate::MOLREC_VERSION`] (`molrec_version = 1`). A trajectory section is encoded by
/// [`crate::io::mrec::FrameSequenceWriter`]. [`write_trajectory_file`] is the
/// same write, with the record shaped to carry only a trajectory.
///
/// # Errors
///
/// A [`MolRsError::Zarr`] when `path` uses a retired `.zarr` suffix, when
/// `path` cannot be created as a directory store, or when a section fails to
/// encode. A [`MolRsError::Validation`] when [`crate::Record::validate`]
/// rejects the record (no state section, or a `step`/`time` length that does
/// not match the frame count).
///
/// # Examples
///
/// ```
/// # fn main() -> Result<(), molrs::MolRsError> {
/// use molrs::io::mrec::{read_record_file, write_record_file};
///
/// let dir = tempfile::tempdir().unwrap();
/// let path = dir.path().join("water.mrec");
///
/// let mut record = molrs::Record::new();
/// record.frame = Some(molrs::Frame::new());
/// write_record_file(&path, &record)?;
///
/// let loaded = read_record_file(&path)?;
/// assert!(loaded.frame.is_some());
/// # Ok(())
/// # }
/// ```
#[cfg(feature = "filesystem")]
pub fn write_record_file(path: impl AsRef<Path>, record: &MolRec) -> Result<(), MolRsError> {
    let path = path.as_ref();
    schema::validate_path(path)?;
    let store: ReadableWritableListableStorage = Arc::new(PositionalWriteStore::new(path)?);
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
    for (name, section) in [("method", &record.method), ("status", &record.status)] {
        if !section.is_empty() {
            write_json_group(&store, &join_path(prefix, name), section)?;
        }
    }
    if !record.metrics.is_empty() || !record.metrics_series.is_empty() {
        write_metrics(&store, &join_path(prefix, "metrics"), record)?;
    }
    for (name, frame) in &record.extra_sections {
        write_frame_group(&store, &join_path(prefix, name), frame)?;
    }

    Ok(())
}

/// Write `meta` as the producer handed it.
///
/// No version key is stamped: during development `molrec_version` is optional
/// and its absence means no version validation. A producer that wants one
/// puts it in `meta` itself; [`schema::validate_meta`] checks it on the way
/// back in.
#[cfg(feature = "zarr")]
fn write_meta(
    store: &ReadableWritableListableStorage,
    path: &str,
    meta: &JsonMap<String, JsonValue>,
) -> Result<(), MolRsError> {
    schema::validate_meta(meta)?;
    write_json_group(store, path, meta)
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

/// Percent-encode a metrics series name into a legal array name
/// (`train/loss` → `train%2Floss`): every byte outside `[A-Za-z0-9._-]`
/// becomes `%XX` with uppercase hex. Mirrors molrec's `safe_name` — the two
/// implementations must mangle identically or produce stores neither can
/// read back.
fn safe_series_name(name: &str) -> String {
    let mut out = String::with_capacity(name.len());
    for byte in name.bytes() {
        match byte {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'.' | b'_' | b'-' => out.push(byte as char),
            _ => {
                use std::fmt::Write as _;
                let _ = write!(out, "%{byte:02X}");
            }
        }
    }
    out
}

/// The inverse of [`safe_series_name`].
fn original_series_name(encoded: &str) -> Result<String, MolRsError> {
    let bytes = encoded.as_bytes();
    let mut raw = Vec::with_capacity(bytes.len());
    let mut index = 0;
    while index < bytes.len() {
        if bytes[index] == b'%' {
            let hex = encoded.get(index + 1..index + 3).ok_or_else(|| {
                MolRsError::zarr(format!(
                    "metrics series name '{encoded}': truncated %-escape"
                ))
            })?;
            raw.push(u8::from_str_radix(hex, 16).map_err(|_| {
                MolRsError::zarr(format!(
                    "metrics series name '{encoded}': bad %-escape '%{hex}'"
                ))
            })?);
            index += 3;
        } else {
            raw.push(bytes[index]);
            index += 1;
        }
    }
    String::from_utf8(raw)
        .map_err(|_| MolRsError::zarr(format!("metrics series name '{encoded}' is not UTF-8")))
}

/// Write the `metrics/` section: the catalog document as group attributes,
/// and each closed series as one float64 array at
/// `metrics/series/<safe_name>`.
///
/// The live JSONL WAL (`metrics/metrics.jsonl`) is host-owned and never
/// written here.
#[cfg(feature = "zarr")]
fn write_metrics(
    store: &ReadableWritableListableStorage,
    prefix: &str,
    record: &MolRec,
) -> Result<(), MolRsError> {
    write_json_group(store, prefix, &record.metrics)?;
    if record.metrics_series.is_empty() {
        return Ok(());
    }
    let series_path = join_path(prefix, "series");
    GroupBuilder::new()
        .build(store.clone(), &series_path)?
        .store_metadata()?;
    for (name, values) in &record.metrics_series {
        let column = Column::from_float(
            ndarray::ArrayD::from_shape_vec(vec![values.len()], values.clone())
                .map_err(|e| MolRsError::zarr(format!("metrics series '{name}': {e}")))?,
        );
        write_column(
            store,
            &join_path(&series_path, &safe_series_name(name)),
            &column,
        )?;
    }
    Ok(())
}

/// Read `metrics/series/<name>` float64 arrays back into
/// [`MolRec::metrics_series`]. A `metrics/` group without a `series/` child
/// is a document-only section; a non-float64 series is an error rather than
/// a silent narrowing.
fn read_metrics_series(
    store: &ReadableWritableListableStorage,
    metrics_path: &str,
    record: &mut MolRec,
) -> Result<(), MolRsError> {
    let node = Node::open(store, metrics_path)?;
    let has_series = node.children().iter().any(|child| {
        matches!(child.metadata(), NodeMetadata::Group(_))
            && child.path().as_str().ends_with("/series")
    });
    if !has_series {
        return Ok(());
    }
    let series_path = join_path(metrics_path, "series");
    let series_node = Node::open(store, &series_path)?;
    for child in series_node.children() {
        if !matches!(child.metadata(), NodeMetadata::Array(_)) {
            continue;
        }
        let path = child.path().as_str();
        let name = path.rsplit('/').next().unwrap_or("");
        if name.is_empty() {
            continue;
        }
        let column = read_column(
            store,
            path,
            &ArraySubset::new_with_shape(Array::open(store.clone(), path)?.shape().to_vec()),
        )?;
        let values = column.as_float().ok_or_else(|| {
            MolRsError::zarr(format!(
                "metrics series '{name}' must be a float64 array, got {}",
                column.dtype().name()
            ))
        })?;
        record.metrics_series.insert(
            original_series_name(name)?,
            values.iter().copied().collect(),
        );
    }
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

/// Read a [`crate::Record`] from a `*.mrec` directory.
///
/// Paths whose file name ends in `.zarr` or `.zarr.zip` are refused. The
/// `meta` section must carry `molrec_version` in `1..=`[`crate::MOLREC_VERSION`];
/// a missing key or an unsupported value is an error. Sections this build does
/// not interpret are kept in [`crate::Record::extra_sections`] rather than dropped.
///
/// A store still carrying the pre-0.14 `trajectory/frames/` tree is refused
/// by name; it is not migrated and is not read back as empty.
///
/// # Errors
///
/// A [`MolRsError::Zarr`] when `path` uses a retired `.zarr` suffix, when
/// `path` is not a readable record store, when `meta` is missing or does not
/// carry a supported `molrec_version`, or when a section fails to
/// decode — including a legacy `trajectory/frames/` layout.
#[cfg(feature = "filesystem")]
pub fn read_record_file(path: impl AsRef<Path>) -> Result<MolRec, MolRsError> {
    let path = path.as_ref();
    schema::validate_path(path)?;
    let store: ReadableWritableListableStorage =
        Arc::new(FilesystemStore::new(path).map_err(zerr)?);
    read_record_store(store)
}

/// Read **one** `Frame`-shaped section of a record from an open store.
///
/// Sections are independent: a record may carry a `frame`, a `system`, a
/// `trajectory`, or several at once, and a caller that wants one of them has
/// no business paying for the rest. [`read_record_store`] decodes everything
/// it finds — including a `trajectory` of any size — so it is the wrong door
/// for "give me the topology out of this run".
///
/// `section` is a top-level group name (`"frame"`, `"system"`, or a producer's
/// own). `Ok(None)` when the record has no such section; the store is listed,
/// not decoded, to find that out.
///
/// # Errors
///
/// The same store errors as [`read_record_store`].
pub fn read_frame_section_store(
    store: ReadableWritableListableStorage,
    section: &str,
) -> Result<Option<Frame>, MolRsError> {
    let root = Node::open(&store, "/")?;
    for child in root.children() {
        if !matches!(child.metadata(), NodeMetadata::Group(_)) {
            continue;
        }
        let path = child.path().as_str().to_string();
        if path.rsplit('/').next().unwrap_or("") == section {
            return Ok(Some(read_frame_group(&store, &path)?));
        }
    }
    Ok(None)
}

/// The record's top-level section names, without decoding any of them.
///
/// The store-taking twin of [`section_names`], which needs a filesystem path.
///
/// # Errors
///
/// The same store errors as [`read_record_store`].
pub fn section_names_store(
    store: ReadableWritableListableStorage,
) -> Result<Vec<String>, MolRsError> {
    let root = Node::open(&store, "/")?;
    let mut names = Vec::new();
    for child in root.children() {
        if !matches!(child.metadata(), NodeMetadata::Group(_)) {
            continue;
        }
        let name = child
            .path()
            .as_str()
            .rsplit('/')
            .next()
            .unwrap_or("")
            .to_string();
        if !name.is_empty() {
            names.push(name);
        }
    }
    names.sort();
    Ok(names)
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
                let sequence = FrameSequence::open(store.clone().readable_listable())?;
                record.trajectory = Some(sequence.to_trajectory()?);
            }
            "observables" => read_observables(&store, &path, &mut record)?,
            "method" => record.method = read_json_group(&store, &path)?,
            "status" => record.status = read_json_group(&store, &path)?,
            "metrics" => {
                record.metrics = read_json_group(&store, &path)?;
                read_metrics_series(&store, &path, &mut record)?;
            }
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

/// Read and validate the `meta` section.
///
/// Every writer creates the group, but a reader tolerates its absence — an
/// empty document — so a store a foreign tool assembled without one still
/// opens. A present `molrec_version` is validated.
fn read_meta(
    store: &ReadableWritableListableStorage,
    path: &str,
) -> Result<JsonMap<String, JsonValue>, MolRsError> {
    let attrs = match zarrs::group::Group::open(store.clone(), path) {
        Ok(group) => group.attributes().clone(),
        Err(zarrs::group::GroupCreateError::MissingMetadata) => JsonMap::new(),
        Err(e) => return Err(e.into()),
    };
    schema::validate_meta(&attrs)?;
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

/// Write a [`crate::Trajectory`] as a record whose only state section is
/// `trajectory`.
///
/// Same path rules as [`write_record_file`]: conventional suffix `.mrec`,
/// retired `.zarr` / `.zarr.zip` refused, a second write replaces the first.
/// The frames are encoded by [`crate::io::mrec::FrameSequenceWriter`]; no
/// duplicate `frame/` snapshot is written beside them.
///
/// # Errors
///
/// The same errors as [`write_record_file`], including validation of `step`
/// and `time` lengths against the frame count.
///
/// # Examples
///
/// ```
/// # fn main() -> Result<(), molrs::MolRsError> {
/// use molrs::Trajectory;
/// use molrs::io::mrec::{read_trajectory_file, write_trajectory_file};
///
/// let dir = tempfile::tempdir().unwrap();
/// let path = dir.path().join("run.mrec");
///
/// let traj = Trajectory::from_frames(vec![molrs::Frame::new()]);
/// write_trajectory_file(&path, &traj)?;
///
/// let loaded = read_trajectory_file(&path)?;
/// assert_eq!(loaded.len(), 1);
/// # Ok(())
/// # }
/// ```
#[cfg(feature = "filesystem")]
pub fn write_trajectory_file(
    path: impl AsRef<Path>,
    trajectory: &Trajectory,
) -> Result<(), MolRsError> {
    let mut record = MolRec::new();
    record.trajectory = Some(trajectory.clone());
    write_record_file(path, &record)
}

/// Write a [`Frame`] as a record whose only state section is `frame`.
///
/// Same path rules as [`write_record_file`]. This is the Structure shape
/// (`meta` + `frame/`). Pass `system` to also persist a `system/` section
/// beside the snapshot; the two remain separate groups.
///
/// # Errors
///
/// The same errors as [`write_record_file`].
#[cfg(feature = "filesystem")]
pub fn write_frame_file(
    path: impl AsRef<Path>,
    frame: &Frame,
    system: Option<&Frame>,
    meta: Option<&JsonMap<String, JsonValue>>,
) -> Result<(), MolRsError> {
    let mut record = MolRec::new();
    record.frame = Some(frame.clone());
    record.system = system.cloned();
    if let Some(meta) = meta {
        record.meta = meta.clone();
    }
    write_record_file(path, &record)
}

/// Write a [`Frame`] as a record whose only state section is `system`.
///
/// Same path rules as [`write_record_file`]. This is the System-def shape
/// (`meta` + `system/`).
///
/// # Errors
///
/// The same errors as [`write_record_file`].
#[cfg(feature = "filesystem")]
pub fn write_system_file(
    path: impl AsRef<Path>,
    system: &Frame,
    meta: Option<&JsonMap<String, JsonValue>>,
) -> Result<(), MolRsError> {
    let mut record = MolRec::new();
    record.system = Some(system.clone());
    if let Some(meta) = meta {
        record.meta = meta.clone();
    }
    write_record_file(path, &record)
}

/// Read the `frame` section of a record at `path`.
///
/// # Errors
///
/// The same errors as [`read_record_file`], plus a missing `frame` section.
#[cfg(feature = "filesystem")]
pub fn read_frame_file(path: impl AsRef<Path>) -> Result<Frame, MolRsError> {
    read_record_file(path)?
        .frame
        .ok_or_else(|| MolRsError::zarr("record has no 'frame' section"))
}

/// Read the `system` section of a record at `path`.
///
/// # Errors
///
/// The same errors as [`read_record_file`], plus a missing `system` section.
#[cfg(feature = "filesystem")]
pub fn read_system_file(path: impl AsRef<Path>) -> Result<Frame, MolRsError> {
    read_record_file(path)?
        .system
        .ok_or_else(|| MolRsError::zarr("record has no 'system' section"))
}

/// Read the mandatory `meta` document of a record at `path`.
///
/// # Errors
///
/// The same path and brand errors as [`read_record_file`].
#[cfg(feature = "filesystem")]
pub fn read_meta_file(path: impl AsRef<Path>) -> Result<JsonMap<String, JsonValue>, MolRsError> {
    let path = path.as_ref();
    schema::validate_path(path)?;
    let store: ReadableWritableListableStorage =
        Arc::new(FilesystemStore::new(path).map_err(zerr)?);
    read_meta(&store, &join_path("/", "meta"))
}

/// Child group names at the record root (`meta`, `frame`, `system`, …).
///
/// This is the inspect door: callers ask which sections are present instead
/// of probing `read_frame` / `read_system` and catching a missing-section
/// error.
///
/// # Errors
///
/// The same path errors as [`read_record_file`].
#[cfg(feature = "filesystem")]
pub fn section_names(path: impl AsRef<Path>) -> Result<Vec<String>, MolRsError> {
    let path = path.as_ref();
    schema::validate_path(path)?;
    let store: ReadableWritableListableStorage =
        Arc::new(FilesystemStore::new(path).map_err(zerr)?);
    let root = Node::open(&store, "/")?;
    let mut names = Vec::new();
    for child in root.children() {
        if !matches!(child.metadata(), NodeMetadata::Group(_)) {
            continue;
        }
        let name = child
            .path()
            .as_str()
            .rsplit('/')
            .next()
            .unwrap_or("")
            .to_string();
        if !name.is_empty() {
            names.push(name);
        }
    }
    names.sort();
    Ok(names)
}

/// Read the `trajectory` section of a record at `path`.
///
/// Same path rules as [`read_record_file`]. A store with no `trajectory`
/// section returns an empty [`crate::Trajectory`], not an error. A store still
/// carrying the pre-0.14 `trajectory/frames/` tree is refused by name — the
/// same failure [`crate::io::mrec::FrameSequence::open`] reports.
///
/// # Errors
///
/// The same errors as [`read_record_file`].
#[cfg(feature = "filesystem")]
pub fn read_trajectory_file(path: impl AsRef<Path>) -> Result<Trajectory, MolRsError> {
    Ok(read_record_file(path)?.trajectory.unwrap_or_default())
}

/// Open a lazy [`FrameSequence`] cursor on a filesystem path.
///
/// Same path rules as [`read_record_file`]: conventional suffix `.mrec`,
/// retired `.zarr` / `.zarr.zip` refused. The cursor is index-only at open;
/// each [`FrameSequence::frame`] call decodes one committed frame.
///
/// This is the filesystem door for a caller that does not already hold a
/// store — Python in particular, so the binder does not take a `zarrs`
/// dependency of its own. [`FrameSequence::open`] remains the store-taking
/// door (in-memory stores, packed zip adapters).
///
/// # Errors
///
/// The same path errors as [`read_record_file`], plus
/// [`FrameSequence::open`]'s store errors (legacy `trajectory/frames/`
/// layout, schema mismatch, missing index).
///
/// # Examples
///
/// ```
/// # fn main() -> Result<(), molrs::MolRsError> {
/// use molrs::Trajectory;
/// use molrs::io::mrec::{open_trajectory_sequence, write_trajectory_file};
///
/// let dir = tempfile::tempdir().unwrap();
/// let path = dir.path().join("run.mrec");
///
/// let traj = Trajectory::from_frames(vec![molrs::Frame::new()]);
/// write_trajectory_file(&path, &traj)?;
///
/// let mut seq = open_trajectory_sequence(&path)?;
/// assert!(seq.frame(0)?.is_some());
/// # Ok(())
/// # }
/// ```
#[cfg(feature = "filesystem")]
pub fn open_trajectory_sequence(path: impl AsRef<Path>) -> Result<FrameSequence, MolRsError> {
    let path = path.as_ref();
    schema::validate_path(path)?;
    let store = Arc::new(FilesystemStore::new(path).map_err(zerr)?);
    FrameSequence::open(store)
}

pub(in crate::io::zarr) fn zerr(e: impl std::fmt::Display) -> MolRsError {
    MolRsError::zarr(e.to_string())
}

/// Refuse the retired scientific path brand `.zarr` / `.zarr.zip`.
#[cfg(feature = "filesystem")]
pub(in crate::io::zarr) fn reject_retired_zarr_path(path: &Path) -> Result<(), MolRsError> {
    schema::validate_path(path)
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
        let path = dir.path().join("record.mrec");
        write_record_file(&path, record).unwrap();
        read_record_file(&path).unwrap()
    }

    /// Development contract: the writer stamps no version key. `meta` comes
    /// back exactly as the producer handed it — empty here.
    /// A reader tolerates a missing `meta/` group as an empty document; every
    /// molrs writer creates the group, but a foreign store may not.
    #[test]
    fn a_record_without_a_meta_group_reads_as_an_empty_document() {
        let mut rec = MolRec::new();
        rec.frame = Some(Frame::new());
        let loaded = write_then_read(&rec);
        assert!(loaded.meta.is_empty(), "{:?}", loaded.meta);
    }

    /// A producer that does write `molrec_version` keeps it, and a supported
    /// value round-trips untouched.
    #[test]
    fn a_producer_molrec_version_round_trips() {
        let mut rec = MolRec::new();
        rec.frame = Some(Frame::new());
        rec.meta
            .insert("molrec_version".into(), schema::MOLREC_VERSION.into());
        let loaded = write_then_read(&rec);
        assert_eq!(
            loaded.meta["molrec_version"].as_u64(),
            Some(schema::MOLREC_VERSION)
        );
    }

    #[test]
    fn no_section_carries_a_frame_schema_version() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("record.mrec");
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
        let path = dir.path().join("record.mrec");
        let mut rec = MolRec::new();
        rec.frame = Some(Frame::new());
        write_record_file(&path, &rec).unwrap();
        std::fs::remove_dir_all(path.join("meta")).unwrap();
        let loaded = read_record_file(&path).unwrap();
        assert!(loaded.meta.is_empty());
    }

    /// An absent `molrec_version` is not validated (development contract); a
    /// present one outside `1..=MOLREC_VERSION` is refused.
    #[test]
    fn a_present_molrec_version_outside_the_supported_range_is_rejected() {
        for version in [None, Some(0_u64), Some(2), Some(99)] {
            let dir = tempdir().unwrap();
            let path = dir.path().join("record.mrec");
            let mut rec = MolRec::new();
            rec.frame = Some(Frame::new());
            write_record_file(&path, &rec).unwrap();

            let metadata_path = path.join("meta/zarr.json");
            let mut metadata: JsonValue =
                serde_json::from_slice(&std::fs::read(&metadata_path).unwrap()).unwrap();
            // The writer stamps no version, so `None` is the store as written;
            // the other three overwrite the (absent) key.
            if let Some(v) = version {
                metadata["attributes"]["molrec_version"] = v.into();
            }
            std::fs::write(&metadata_path, serde_json::to_vec(&metadata).unwrap()).unwrap();
            let result = read_record_file(&path);
            match version {
                None => assert!(result.is_ok(), "refused a store without molrec_version"),
                Some(_) => assert!(result.is_err(), "accepted molrec_version {version:?}"),
            }
        }
    }

    #[test]
    fn a_record_with_no_state_section_is_refused_at_write() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("record.mrec");
        assert!(write_record_file(&path, &MolRec::new()).is_err());
    }

    #[test]
    fn trajectory_door_round_trips_through_the_record_layout() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("record.mrec");
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

        // The narrow door writes a conforming record, not a private layout:
        // a root, a `meta/` group, and the trajectory section.
        assert!(section_names(&path).unwrap().contains(&"meta".to_string()));
        let record = read_record_file(&path).unwrap();
        assert!(record.trajectory.is_some());
    }

    #[test]
    fn frame_door_writes_the_frame_section() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("snapshot.mrec");
        let frame = frame_with_atoms(3);
        write_frame_file(&path, &frame, None, None).unwrap();

        let loaded = read_frame_file(&path).unwrap();
        assert_eq!(loaded.get("atoms").unwrap().nrows(), Some(3));
        assert!(read_system_file(&path).is_err());
        assert!(section_names(&path).unwrap().contains(&"meta".to_string()));
        let record = read_record_file(&path).unwrap();
        assert!(record.trajectory.is_none());
    }

    #[test]
    fn system_door_writes_the_system_section() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("system.mrec");
        write_system_file(&path, &frame_with_atoms(2), None).unwrap();

        let loaded = read_system_file(&path).unwrap();
        assert_eq!(loaded.get("atoms").unwrap().nrows(), Some(2));
        assert!(read_frame_file(&path).is_err());
    }

    #[test]
    fn frame_door_can_carry_a_system_section_beside_the_snapshot() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("both.mrec");
        let frame = frame_with_atoms(3);
        let system = frame_with_atoms(3);
        write_frame_file(&path, &frame, Some(&system), None).unwrap();

        assert_eq!(
            section_names(&path).unwrap(),
            vec!["frame", "meta", "system"]
        );
        // The identity document is present and, with nothing handed in, empty.
        assert!(read_meta_file(&path).unwrap().is_empty());

        assert_eq!(
            read_frame_file(&path)
                .unwrap()
                .get("atoms")
                .unwrap()
                .nrows(),
            Some(3)
        );
        assert_eq!(
            read_system_file(&path)
                .unwrap()
                .get("atoms")
                .unwrap()
                .nrows(),
            Some(3)
        );
    }

    #[test]
    fn metrics_series_round_trip_as_float64_arrays() {
        let mut rec = MolRec::new();
        rec.frame = Some(Frame::new());
        rec.metrics.insert("catalog".into(), "summary".into());
        rec.metrics_series
            .insert("train/loss".into(), vec![1.0, 0.5, 0.25]);
        rec.metrics_series.insert("lr".into(), vec![1e-3, 1e-4]);

        let loaded = write_then_read(&rec);
        assert_eq!(loaded.metrics["catalog"], "summary");
        assert_eq!(loaded.metrics_series.len(), 2);
        assert_eq!(loaded.metrics_series["train/loss"], vec![1.0, 0.5, 0.25]);
        assert_eq!(loaded.metrics_series["lr"], vec![1e-3, 1e-4]);

        // A second write of what was read must not shed the series — this is
        // the preserve-the-unknown guarantee for densified curves.
        let again = write_then_read(&loaded);
        assert_eq!(again.metrics_series, rec.metrics_series);
    }

    /// A live host WAL (`metrics/metrics.jsonl`) is a stray text file inside
    /// the store; reading the record must tolerate it rather than error.
    #[test]
    fn a_stray_metrics_wal_file_is_tolerated_on_read() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("record.mrec");
        let mut rec = MolRec::new();
        rec.frame = Some(Frame::new());
        rec.metrics.insert("live".into(), true.into());
        write_record_file(&path, &rec).unwrap();

        std::fs::write(
            path.join("metrics/metrics.jsonl"),
            "{\"t\":\"scalar\",\"k\":\"loss\",\"v\":0.5}\n",
        )
        .unwrap();

        let loaded = read_record_file(&path).unwrap();
        assert_eq!(loaded.metrics["live"], true);
        assert!(loaded.metrics_series.is_empty());
    }

    #[test]
    fn a_non_float64_metrics_series_is_rejected_loud() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("record.mrec");
        let mut rec = MolRec::new();
        rec.frame = Some(Frame::new());
        rec.metrics.insert("catalog".into(), "summary".into());
        write_record_file(&path, &rec).unwrap();

        // Plant a u64 array where a float64 series belongs.
        let store: ReadableWritableListableStorage =
            std::sync::Arc::new(super::PositionalWriteStore::new(&path).unwrap());
        GroupBuilder::new()
            .build(store.clone(), "/metrics/series")
            .unwrap()
            .store_metadata()
            .unwrap();
        write_column(
            &store,
            "/metrics/series/steps",
            &Column::from_uint(ndarray::ArrayD::from_shape_vec(vec![2], vec![1u64, 2]).unwrap()),
        )
        .unwrap();

        let err = read_record_file(&path).unwrap_err().to_string();
        assert!(err.contains("steps"), "{err}");
        assert!(err.contains("float64"), "{err}");
    }

    /// The contract key is validated on the way out as well as in: a producer
    /// cannot write a version this build does not support.
    #[test]
    fn a_producer_molrec_version_out_of_range_is_refused_at_write() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("record.mrec");
        let mut rec = MolRec::new();
        rec.frame = Some(Frame::new());
        rec.meta.insert("molrec_version".into(), 99u64.into());
        let err = write_record_file(&path, &rec).unwrap_err().to_string();
        for key in RESERVED_META_KEYS {
            assert!(err.contains(key), "{err}");
        }
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
        let path = dir.path().join("record.mrec");

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
        let path = dir.path().join("record.mrec");

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
        let path = dir.path().join("simbox_f64.mrec");
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
        let path = dir.path().join("record.mrec");

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
        let path = dir.path().join("record.mrec");
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
