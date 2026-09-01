//! The `trajectory/` frame sequence: one on-disk layout, one encoder, one
//! decoder.
//!
//! # One object, three access forms
//!
//! A frame sequence is a single thing. Three Rust names reach it, and they are
//! named after the **access form**, never after the content:
//!
//! - [`Trajectory`] — the **eager in-memory carrier**. Every frame is
//!   materialized in a `Vec<Frame>`; it does not know a store exists.
//! - [`FrameSequence`] — the **lazy store cursor**. It opens index-only (each
//!   section's `step_index` and `offset`, plus the schema) and reads **one**
//!   frame per call. [`FrameSequence::to_trajectory`] is the named lazy → eager
//!   conversion, not a second representation.
//! - [`FrameSequenceWriter`] — the sequence's **streaming producer**. One frame
//!   per [`append`](FrameSequenceWriter::append),
//!   [`flush`](FrameSequenceWriter::flush) commits.
//!
//! The `trajectory/` group on disk is the sequence's serialized form; none of
//! the three names is a second name for it.
//!
//! # Error vocabulary
//!
//! Every [`FrameSequence`] and [`FrameSequenceWriter`] door yields
//! [`MolRsError`]. The three [`TrajectoryReader`] methods keep `io::Result` and
//! convert `MolRsError` in at the boundary — **lossy, and deliberately so**:
//! the trait is the shape every backend shares and must not be captured by one
//! backend's error type.
//!
//! # The layout
//!
//! ```text
//! trajectory/                    group; attributes carry the pinned schema
//!   step     i64  [nstep]        the commit marker, extended last
//!   time     f64  [nstep]        only when the run supplies times
//!   meta/<key>    typed [nstep] (or [nstep][3|6|9]); attr molrs_meta_dtype
//!   box/
//!     step_index u64  [n_updates]
//!     vectors    f64  [n_updates][3][3]
//!     origin     f64  [n_updates][3]
//!     boundary   bool [n_updates][3]
//!   <block>/                     attr structural_shape when one is declared
//!     step_index u64  [n_updates]
//!     offset     u64  [n_updates+1]   CSR row pointer, offset[0] = 0
//!     <column>        [total_rows][...trailing]
//! ```
//!
//! `offset` is a **CSR** (compressed sparse row) row pointer, the standard way
//! to store a ragged sequence of row groups in one flat array: entry `j` holds
//! the index of the first row of update `j`, so update `j` owns the half-open
//! row range `offset[j]..offset[j+1]`, and the array is therefore one entry
//! longer than `step_index`.
//!
//! Resolving block `B` at frame `i`: binary-search `B/step_index` for the
//! largest entry `<= i`, giving update `j`; the frame's rows are
//! `offset[j]..offset[j+1]`. **No entry `<= i` means the block does not exist
//! at that step** — absence, not an empty block. Row *counts* are never
//! stored; they are `diff(offset)`.
//!
//! Two consequences of that encoding are worth stating rather than
//! discovering:
//!
//! - A block that was present and then goes away is landed as a **zero-row
//!   update** at the step it disappears, and any update of zero rows reads
//!   back as absent. A block that is genuinely present with zero rows is
//!   therefore indistinguishable from an absent one.
//! - The `box/` section has no `offset`, so it has no absence marker: once a
//!   run writes a cell, every later step resolves to the most recent one. A
//!   frame that drops its cell mid-run keeps the previous cell on read.
//!
//! # Chunks, shards, and branch A
//!
//! Three words from the Zarr storage model carry the rest of this module, so
//! they are defined here once rather than assumed.
//!
//! An **inner chunk** is the smallest unit the codec compresses and rewrites: a
//! fixed number of rows, frozen when the array is created and not negotiable
//! afterwards. A **shard** is one file on disk holding a fixed number of
//! consecutive inner chunks plus a small index of their offsets within the
//! file; choosing how many chunks a shard spans is the file-count lever, and it
//! is what keeps a long run to a handful of files rather than one per frame.
//!
//! **Branch A** is the append strategy this module implements. The name comes
//! from the spike verdict recorded in [`crate::io::zarr`]'s module doc: a
//! partially filled *trailing* inner chunk can be rewritten in place as a
//! tail-only write, so a commit may land any number of rows and never has to
//! wait for a chunk boundary. Its price is that the superseded copy of a
//! rewritten chunk stays in the shard file as dead bytes until that shard is
//! re-encoded — which [`FrameSequenceWriter::flush`] does for a shard the run
//! has just completed, and [`FrameSequenceWriter::close`] does for the final,
//! still-partial one.
//!
//! [`Trajectory`]: molrs::store::trajectory::Trajectory
//! [`TrajectoryReader`]: crate::io::reader::TrajectoryReader

use std::collections::BTreeMap;
use std::sync::Arc;

use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};
use zarrs::array::codec::GzipCodec;
use zarrs::array::codec::array_to_bytes::sharding::{ShardingCodecOptions, SubchunkWriteOrder};
use zarrs::array::{
    Array, ArrayBuilder, ArrayBytes, ArraySubset, CodecOptions, CodecSpecificOptions,
};
use zarrs::config::global_config;
use zarrs::group::{Group, GroupBuilder};
use zarrs::node::{Node, NodeMetadata, NodePath, get_child_nodes};
use zarrs::storage::{
    ListableStorageTraits, ReadableListableStorage, ReadableStorageTraits,
    ReadableWritableListableStorage, ReadableWritableListableStorageTraits, StorageHandle,
    StorePrefix, WritableStorageTraits,
};

use molrs::MolRsError;
use molrs::spatial::simbox::SimBox;
use molrs::store::block::{Block, Column, DType};
use molrs::store::frame::Frame;
use molrs::store::meta::MetaValue;
use molrs::store::trajectory::Trajectory;
use molrs::types::F;

use crate::io::reader::TrajectoryReader;

use super::chunking::TARGET_CHUNK_BYTES;
use super::frame_io::{
    BOX_GROUP, GZIP_LEVEL, insert_column_into_block, join_path, node_prefix, read_column,
    zarr_dtype,
};
use super::record_io::zerr;

// ---------------------------------------------------------------------------
// Layout vocabulary
// ---------------------------------------------------------------------------

/// The sequence root. Absolute, because every array path is built from it.
const TRAJECTORY_GROUP: &str = "/trajectory";
/// The commit marker.
const STEP_ARRAY: &str = "step";
/// Physical time, present only when the run supplies it.
const TIME_ARRAY: &str = "time";
/// Parent group of the per-step typed metadata arrays.
const META_GROUP: &str = "meta";
/// CSR row pointer of a block section.
const OFFSET_ARRAY: &str = "offset";
/// Frame ordinals at which a section was updated.
const STEP_INDEX_ARRAY: &str = "step_index";
/// Cell matrices of the `box/` section.
const VECTORS_ARRAY: &str = "vectors";
/// Cell origins of the `box/` section.
const ORIGIN_ARRAY: &str = "origin";
/// Per-axis periodicity of the `box/` section.
const BOUNDARY_ARRAY: &str = "boundary";
/// The per-frame group tree molrs <= 0.13 wrote under `trajectory/`.
const LEGACY_FRAMES_GROUP: &str = "frames";
/// Group attribute of `trajectory/` holding the pinned schema.
const SCHEMA_ATTRIBUTE: &str = "molrs_sequence_schema";
/// Array attribute naming a per-step meta array's [`MetaValue::dtype`] tag.
const META_DTYPE_ATTRIBUTE: &str = "molrs_meta_dtype";
/// Optional `box/` group attribute; absent means a defined cell, matching
/// `read_simbox`'s rule for the frame section.
const CELL_DEFINED_ATTRIBUTE: &str = "cell_defined";
/// Optional block-section group attribute mirroring the block's structural
/// shape, in the same spelling `write_frame_group` uses for the frame section.
const STRUCTURAL_SHAPE_ATTRIBUTE: &str = "structural_shape";

/// `trajectory/`'s own children: no block may take one of these names.
const RESERVED_BLOCK_NAMES: [&str; 4] = [STEP_ARRAY, TIME_ARRAY, META_GROUP, BOX_GROUP];
/// A block section's own children: no column may take one of these names.
const RESERVED_COLUMN_NAMES: [&str; 2] = [OFFSET_ARRAY, STEP_INDEX_ARRAY];

/// Bytes one shard file aims for, and therefore the file-count lever: a store
/// costs `total_bytes / SHARD_TARGET_BYTES + O(arrays)` files rather than one
/// per frame.
///
/// Read where `k` is derived, in [`derive_extents`]. It is only a *target*:
/// `k` falls out of it and the chunk size, and
/// [`FrameSequenceWriter::with_chunks_per_shard`] overrides it outright.
const SHARD_TARGET_BYTES: u64 = 256 * 1024 * 1024;

/// Bytes assumed for one element of a variable-width ([`DType::String`])
/// column when sizing a growth array.
///
/// [`DType::itemsize`] reports `None` there and molrec's `plan` declines to
/// size such an array at all — but a growth array's extents are frozen at
/// creation, so "decline" is not an option and a number has to be chosen. It
/// moves a chunk boundary and nothing else: a string chunk that misses the
/// byte target costs throughput, never correctness.
const ASSUMED_STRING_ITEMSIZE: u64 = 16;

/// The options every write of this module carries.
///
/// Built explicitly and threaded into the `_opt` methods on purpose: the
/// non-`_opt` methods construct their own defaults, where
/// `experimental_partial_encoding` is `false` and every flush silently becomes
/// an O(shard) rewrite. `global_config_mut()` is never touched — it is a
/// process-wide `RwLock` belonging to the library, not to this writer.
fn partial_encoding_options() -> CodecOptions {
    global_config()
        .codec_options()
        .with_experimental_partial_encoding(true)
}

/// The options a shard re-encode carries.
///
/// Partial encoding is deliberately **off**: a seal is a whole-value write of
/// one shard, and the dead bytes partial encoding leaves behind are exactly
/// what it exists to undo.
fn compaction_options() -> CodecOptions {
    global_config().codec_options()
}

/// Codec options that pin a shard's subchunk layout to C (row-major) order.
///
/// A sealed shard's bytes must not depend on thread scheduling, and by default
/// they do: zarrs 0.23.13 lays newly written subchunks out in the iteration
/// order of a `HashMap` (`sharding_partial_encoder.rs:363`) and, in the
/// whole-shard encoder, in completion order
/// (`SubchunkWriteOrder::Unordered`, the default). Measured, not feared — the
/// same 24-frame run sealed its two inner chunks in either order from run to
/// run. `SubchunkWriteOrder::C` makes the layout the shard index's own order,
/// which is what lets "a closed store's bytes do not remember how often it was
/// flushed" be a claim about bytes and not only about values.
fn deterministic_shard_layout() -> CodecSpecificOptions {
    CodecSpecificOptions::default().with_option(
        ShardingCodecOptions::default().with_subchunk_write_order(SubchunkWriteOrder::C),
    )
}

/// The tag a column's storage width is written as in the schema attributes.
///
/// The three domain aliases (`Float`/`Int`/`UInt`) are tagged with molrec's
/// concrete-width names (`f64`/`i32`/`u64`) so a written schema validates
/// against molrec's published dtype enum and matches the meta-side tags
/// ([`meta_layout`]); every other width is [`DType::name`] verbatim. The match
/// carries no wildcard, so it stays exhaustive over `DType` inside this crate:
/// a width added to the enum stops the build here until [`dtype_from_tag`] can
/// read it back.
fn dtype_tag(dtype: DType) -> &'static str {
    match dtype {
        DType::Float => "f64",
        DType::Int => "i32",
        DType::UInt => "u64",
        DType::Float16
        | DType::Float32
        | DType::Int8
        | DType::Int16
        | DType::Int64
        | DType::Bool
        | DType::U8
        | DType::UInt16
        | DType::UInt32
        | DType::String
        | DType::Complex64
        | DType::Complex128 => dtype.name(),
    }
}

/// Every storage width a schema column can declare — the search space
/// [`dtype_tag`] is read backwards over.
const SCHEMA_WIDTHS: [DType; 15] = [
    DType::Float16,
    DType::Float32,
    DType::Float,
    DType::Int8,
    DType::Int16,
    DType::Int,
    DType::Int64,
    DType::Bool,
    DType::UInt,
    DType::U8,
    DType::UInt16,
    DType::UInt32,
    DType::String,
    DType::Complex64,
    DType::Complex128,
];

/// [`dtype_tag`] read backwards.
fn dtype_from_tag(tag: &str) -> Result<DType, MolRsError> {
    // Stores written by molrs < 0.14 tagged the domain aliases `float`/`int`/
    // `uint`; those stay readable forever. New stores use the molrec-conformant
    // `f64`/`i32`/`u64` spelling that `dtype_tag` now emits.
    match tag {
        "float" => return Ok(DType::Float),
        "int" => return Ok(DType::Int),
        "uint" => return Ok(DType::UInt),
        _ => {}
    }
    SCHEMA_WIDTHS
        .into_iter()
        .find(|dtype| dtype_tag(*dtype) == tag)
        .ok_or_else(|| MolRsError::zarr(format!("unknown column dtype {tag:?} in sequence schema")))
}

/// The [`DType`] a stored array carries, in this crate's dtype vocabulary.
///
/// A search over [`zarr_dtype`] rather than a second table: `open` has to
/// report what it *found* on disk in the same words the schema declares it in.
fn dtype_of_stored(stored: &zarrs::array::DataType) -> Option<DType> {
    SCHEMA_WIDTHS
        .into_iter()
        .find(|dtype| zarr_dtype(*dtype).0 == *stored)
}

/// The stored width and trailing shape of a per-step meta array, from the
/// [`MetaValue::dtype`] tag it declares.
///
/// Both vocabularies are the existing ones: `MetaValue`'s 20 stable tags name
/// the variant, and [`DType`] names the width it is stored at.
fn meta_layout(tag: &str) -> Option<(DType, Vec<u64>)> {
    let (dtype, width) = match tag {
        "bool" => (DType::Bool, 1),
        "i32" => (DType::Int, 1),
        "i64" => (DType::Int64, 1),
        "u32" => (DType::UInt32, 1),
        "u64" => (DType::UInt, 1),
        "f32" => (DType::Float32, 1),
        "f64" => (DType::Float, 1),
        // A JSON document is stored as its text: the tag says how to read it
        // back, so the width is the only thing the array has to carry.
        "string" | "json" => (DType::String, 1),
        "bool3" => (DType::Bool, 3),
        "i32x3" => (DType::Int, 3),
        "i64x3" => (DType::Int64, 3),
        "u32x3" => (DType::UInt32, 3),
        "u64x3" => (DType::UInt, 3),
        "f32x3" => (DType::Float32, 3),
        "f64x3" => (DType::Float, 3),
        "f32x6" => (DType::Float32, 6),
        "f64x6" => (DType::Float, 6),
        "f32x9" => (DType::Float32, 9),
        "f64x9" => (DType::Float, 9),
        _ => return None,
    };
    Some((dtype, if width == 1 { Vec::new() } else { vec![width] }))
}

/// Bytes one row of an array of `dtype` with `trailing` axes occupies.
fn row_bytes(dtype: DType, trailing: &[u64]) -> u64 {
    let width = dtype
        .itemsize()
        .map_or(ASSUMED_STRING_ITEMSIZE, |width| width as u64);
    trailing
        .iter()
        .fold(width, |acc, &axis| acc.saturating_mul(axis.max(1)))
        .max(1)
}

/// `(rows per inner chunk, inner chunks per shard)` for a growth array whose
/// row is `row` bytes wide.
///
/// Derived, not configured: `R` is how many rows fit in the 512 KiB chunk
/// target and `k` how many such chunks fit in the 256 MiB shard target, each
/// floored at one. The two knobs override the two halves independently, which
/// is the only way a unit test reaches a chunk or shard boundary in a handful
/// of frames.
fn derive_extents(
    rows_per_chunk: Option<u64>,
    chunks_per_shard: Option<u64>,
    row: u64,
) -> (u64, u64) {
    let rows = rows_per_chunk.unwrap_or_else(|| (TARGET_CHUNK_BYTES / row).max(1));
    let chunk = rows.saturating_mul(row).max(1);
    let chunks = chunks_per_shard.unwrap_or_else(|| (SHARD_TARGET_BYTES / chunk).max(1));
    (rows.max(1), chunks.max(1))
}

/// The subset covering `rows` rows from `start` of an array whose trailing
/// axes are `trailing`.
fn rows_subset(start: u64, rows: u64, trailing: &[u64]) -> Result<ArraySubset, MolRsError> {
    let mut origin = Vec::with_capacity(trailing.len() + 1);
    origin.push(start);
    origin.extend(std::iter::repeat_n(0u64, trailing.len()));
    let mut shape = Vec::with_capacity(trailing.len() + 1);
    shape.push(rows);
    shape.extend_from_slice(trailing);
    ArraySubset::new_with_start_shape(origin, shape).map_err(zerr)
}

/// Bitwise column equality — the predicate behind a section's `step_index`.
///
/// Bitwise rather than numeric, via [`Column::raw_bytes`]: a NaN coordinate
/// has to count as *unchanged*, or a section that never moves would earn an
/// entry at every step.
fn same_column(left: &Column, right: &Column) -> bool {
    if left.dtype() != right.dtype() || left.shape() != right.shape() {
        return false;
    }
    match (left.raw_bytes(), right.raw_bytes()) {
        (Some(left), Some(right)) => left == right,
        // The one variable-width column has no byte image; compare elements.
        (None, None) => left.as_string() == right.as_string(),
        _ => false,
    }
}

/// Whether two blocks carry the same content, column for column.
fn same_block(left: &Block, right: &Block) -> bool {
    left.len() == right.len()
        && left.nrows() == right.nrows()
        && left.structural_shape() == right.structural_shape()
        && left.iter().all(|(name, column)| {
            right
                .get(name)
                .is_some_and(|other| same_column(column, other))
        })
}

/// Whether two cells are the same cell, bit for bit.
fn same_simbox(left: &SimBox, right: &SimBox) -> bool {
    left.is_cell_defined() == right.is_cell_defined()
        && left.pbc_view() == right.pbc_view()
        && left
            .h_view()
            .iter()
            .map(|value| value.to_bits())
            .eq(right.h_view().iter().map(|value| value.to_bits()))
        && left
            .origin_view()
            .iter()
            .map(|value| value.to_bits())
            .eq(right.origin_view().iter().map(|value| value.to_bits()))
}

/// Origin is optional on disk: all zeros is the default and is not written.
fn origin_is_default(cell: &SimBox) -> bool {
    cell.origin_view().iter().all(|value| *value == 0.0)
}

/// Boundary is optional on disk: all-periodic is the default and is not written.
fn boundary_is_default(cell: &SimBox) -> bool {
    cell.pbc() == [true, true, true]
}

// ---------------------------------------------------------------------------
// Schema
// ---------------------------------------------------------------------------

/// One declared column: the width and trailing shape it arrived with.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
struct ColumnSchema {
    /// [`dtype_tag`] of the column's storage width.
    dtype: String,
    /// Axes after the leading row axis, e.g. `[3]` for xyz.
    #[serde(default)]
    trailing: Vec<u64>,
}

/// One declared block: its columns, and the structural shape it declares.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
struct BlockSchema {
    columns: BTreeMap<String, ColumnSchema>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    structural_shape: Option<Vec<usize>>,
}

/// One declared per-step metadata key.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
struct MetaSchema {
    /// The [`MetaValue::dtype`] tag every step must carry.
    dtype: String,
    /// Value written for a step that omits the key, as
    /// [`MetaValue::to_json_value`]'s envelope. `None` makes the omission an
    /// error — there is no implicit fill.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    fill: Option<serde_json::Value>,
}

/// The blocks, columns, dtypes and trailing shapes a sequence is pinned to.
///
/// **Derived only.** The two mints read the frames' own columns; there is no
/// hand-written dtype entry point, because a dtype written by hand is a dtype
/// that can disagree with the data. A later frame may present a *subset* of
/// the union — that is how the layout expresses sparsity — but never anything
/// outside it: a run that decides halfway through to record forces needs a new
/// store.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SequenceSchema {
    blocks: BTreeMap<String, BlockSchema>,
    #[serde(default)]
    meta: BTreeMap<String, MetaSchema>,
}

impl SequenceSchema {
    /// Mint from one frame's own columns.
    ///
    /// # Errors
    ///
    /// The single-frame case of [`from_frames`](Self::from_frames), and it
    /// yields the same [`MolRsError::Zarr`] for the same reasons: a block or
    /// column that takes one of the layout's reserved names. A one-frame union
    /// cannot conflict with itself, so the conflict errors listed there cannot
    /// arise here.
    pub fn from_frame(frame: &Frame) -> Result<Self, MolRsError> {
        Self::from_frames(std::slice::from_ref(frame))
    }

    /// Mint from the **union** of `frames`.
    ///
    /// This is how a heterogeneous run is expressed: blocks and columns are
    /// unioned across the frames, and each frame later presents whichever
    /// subset it has. A column appearing twice with a conflicting dtype or
    /// trailing shape is an error *here*, at mint, rather than at the append
    /// that would have discovered it.
    ///
    /// # Errors
    ///
    /// Every case is a [`MolRsError::Zarr`] naming the offending block, column
    /// or metadata key:
    ///
    /// - a block named `step`, `time`, `meta` or `box` — those four are
    ///   `trajectory/`'s own children, so no block may take one;
    /// - a column named `offset` or `step_index` — a block section's own two
    ///   index arrays, for the same reason;
    /// - the same block declaring two different structural shapes across the
    ///   frames;
    /// - the same column declaring two different dtypes, or two different
    ///   trailing shapes, across the frames;
    /// - the same `meta` key carrying two different `MetaValue` dtypes across
    ///   the frames.
    pub fn from_frames(frames: &[Frame]) -> Result<Self, MolRsError> {
        let mut blocks: BTreeMap<String, BlockSchema> = BTreeMap::new();
        let mut shapes: BTreeMap<String, Option<Vec<usize>>> = BTreeMap::new();
        let mut meta: BTreeMap<String, MetaSchema> = BTreeMap::new();

        for frame in frames {
            for (name, block) in frame.iter() {
                if RESERVED_BLOCK_NAMES.contains(&name) {
                    return Err(MolRsError::zarr(format!(
                        "{name:?} is a reserved child of the trajectory group; a block cannot take it"
                    )));
                }
                let shape = block.structural_shape().map(<[usize]>::to_vec);
                match shapes.get(name) {
                    None => {
                        shapes.insert(name.to_string(), shape);
                    }
                    Some(previous) if *previous == shape => {}
                    Some(previous) => {
                        return Err(MolRsError::zarr(format!(
                            "sequence schema conflict: block {name:?} declares structural shape \
                             {previous:?} in one frame and {shape:?} in another"
                        )));
                    }
                }

                let entry = blocks
                    .entry(name.to_string())
                    .or_insert_with(|| BlockSchema {
                        columns: BTreeMap::new(),
                        structural_shape: None,
                    });
                for (column, values) in block.iter() {
                    if RESERVED_COLUMN_NAMES.contains(&column) {
                        return Err(MolRsError::zarr(format!(
                            "{column:?} is a reserved child of a block section; a column cannot \
                             take it"
                        )));
                    }
                    let declared = ColumnSchema {
                        dtype: dtype_tag(values.dtype()).to_string(),
                        trailing: values.shape().iter().skip(1).map(|&n| n as u64).collect(),
                    };
                    match entry.columns.get(column) {
                        None => {
                            entry.columns.insert(column.to_string(), declared);
                        }
                        Some(existing) if existing.dtype != declared.dtype => {
                            return Err(MolRsError::zarr(format!(
                                "sequence schema conflict: column {column:?} of block {name:?} is \
                                 {} in one frame and {} in another",
                                existing.dtype, declared.dtype
                            )));
                        }
                        Some(existing) if existing.trailing != declared.trailing => {
                            return Err(MolRsError::zarr(format!(
                                "sequence schema conflict: column {column:?} of block {name:?} has \
                                 trailing shape {:?} in one frame and {:?} in another",
                                existing.trailing, declared.trailing
                            )));
                        }
                        Some(_) => {}
                    }
                }
            }

            for (key, value) in frame.meta.iter() {
                // `step` / `time` are the sequence's own arrays, not per-step
                // meta. A producer that stashed the commit marker on the frame
                // must not mint a duplicate `trajectory/meta/step`.
                if key == STEP_ARRAY || key == TIME_ARRAY {
                    continue;
                }
                let tag = value.dtype();
                match meta.get(key) {
                    None => {
                        meta.insert(
                            key.clone(),
                            MetaSchema {
                                dtype: tag.to_string(),
                                fill: None,
                            },
                        );
                    }
                    Some(existing) if existing.dtype == tag => {}
                    Some(existing) => {
                        return Err(MolRsError::zarr(format!(
                            "sequence schema conflict: meta key {key:?} is {} in one frame and \
                             {tag} in another",
                            existing.dtype
                        )));
                    }
                }
            }
        }

        for (name, shape) in shapes {
            if let Some(entry) = blocks.get_mut(&name) {
                entry.structural_shape = shape;
            }
        }
        Ok(Self { blocks, meta })
    }

    /// Declare `key`'s value for the steps that omit it.
    ///
    /// The declared dtype is the fill's own — the only dtype source this type
    /// admits. Without a fill an omitted key is an error at append: there is
    /// deliberately no implicit NaN (not-a-number, the IEEE-754 floating-point
    /// value used elsewhere to stand for "missing"), because a NaN nobody asked
    /// for is a measurement nobody made.
    ///
    /// # Errors
    ///
    /// A [`MolRsError::Zarr`] naming `key` when `fill`'s dtype is one no
    /// per-step array can store. Every `MetaValue` variant this build defines
    /// can, so that arm is unreachable today and stands as a guard against a
    /// variant added later without a per-step layout. The twenty tags it
    /// accepts are the scalar forms `bool`, `i32`, `i64`, `u32`, `u64`, `f32`,
    /// `f64`, `string` and `json`, the three-component forms `bool3`, `i32x3`,
    /// `i64x3`, `u32x3`, `u64x3`, `f32x3` and `f64x3`, and the six- and
    /// nine-component float forms `f32x6`, `f64x6`, `f32x9` and `f64x9`.
    ///
    /// Also a [`MolRsError::Zarr`] when `key` is already declared with a
    /// different dtype: a declaration is a pin, and one key cannot mean two
    /// widths in one sequence.
    pub fn declare_meta(&mut self, key: &str, fill: MetaValue) -> Result<(), MolRsError> {
        let tag = fill.dtype();
        if meta_layout(tag).is_none() {
            return Err(MolRsError::zarr(format!(
                "meta key {key:?} cannot be stored per step: unknown dtype {tag}"
            )));
        }
        if let Some(existing) = self.meta.get(key)
            && existing.dtype != tag
        {
            return Err(MolRsError::zarr(format!(
                "meta key {key:?} is already declared {} and the fill value is {tag}",
                existing.dtype
            )));
        }
        self.meta.insert(
            key.to_string(),
            MetaSchema {
                dtype: tag.to_string(),
                fill: Some(fill.to_json_value()),
            },
        );
        Ok(())
    }
}

/// Read the schema back out of the `trajectory/` group attributes, if one is
/// pinned there.
///
/// The attributes are authoritative when present: they exist from `create`,
/// before a single array does, which is exactly what lets `open` notice that
/// the arrays no longer agree with them.
///
/// A missing pin is `Ok(None)`, not an error, because it is the one fact the
/// two doors answer differently: [`schema_of`] refuses it, and
/// [`FrameSequence::open`] derives from the store instead.
///
/// Generic over the store, and asking only for reads, so the writer's
/// read-write store and [`FrameSequence`]'s read-only one reach the same
/// decoder rather than two copies of it.
fn pinned_schema<S>(store: &Arc<S>) -> Result<Option<SequenceSchema>, MolRsError>
where
    S: ?Sized + ReadableStorageTraits + 'static,
{
    let group = Group::open(store.clone(), TRAJECTORY_GROUP)?;
    let Some(attribute) = group.attributes().get(SCHEMA_ATTRIBUTE) else {
        return Ok(None);
    };
    serde_json::from_value(attribute.clone())
        .map(Some)
        .map_err(zerr)
}

/// [`pinned_schema`], requiring the pin.
///
/// The strict half of the seam, and the writer's: appending needs the
/// *declared* schema — the union a later frame is checked against, and the
/// fills for the meta keys a step omits — and neither is recoverable from data
/// that was never written.
fn schema_of<S>(store: &Arc<S>) -> Result<SequenceSchema, MolRsError>
where
    S: ?Sized + ReadableStorageTraits + 'static,
{
    pinned_schema(store)?.ok_or_else(|| {
        MolRsError::zarr(format!(
            "{TRAJECTORY_GROUP} carries no {SCHEMA_ATTRIBUTE} attribute; it is not a molrs frame \
             sequence"
        ))
    })
}

/// The direct child nodes of the group at `path` — one level deep, metadata
/// only.
///
/// A path holding no children at all yields an empty vector rather than an
/// error, which is how a sequence that declared no per-step metadata (and so
/// has no `meta/` group) reads as "no keys".
fn children<S>(store: &Arc<S>, path: &str) -> Result<Vec<Node>, MolRsError>
where
    S: ?Sized + ReadableStorageTraits + ListableStorageTraits + 'static,
{
    let path: NodePath = path.try_into().map_err(zerr)?;
    Ok(get_child_nodes(store, &path, false)?)
}

/// The schema of a store that pins none: derived from the layout itself.
///
/// Everything the read path needs is already on disk. A block is a child
/// *group* of `trajectory/` that is not one of [`RESERVED_BLOCK_NAMES`]; its
/// columns are that group's arrays that are not one of
/// [`RESERVED_COLUMN_NAMES`]; each column's width and trailing shape are its
/// own array metadata, read through [`dtype_of_stored`] — the mapping the
/// pinned path already validates against, not a second table. Per-step
/// metadata keys are `trajectory/meta/`'s arrays, each carrying its own
/// [`META_DTYPE_ATTRIBUTE`] tag.
///
/// Every derived [`MetaSchema::fill`] is `None`, and that is the honest
/// reading rather than a gap: a fill is what a *writer* chose to land for a
/// step that omitted the key, and no store records that choice. `None` already
/// means "an omitted key is an error", which is exactly what a reader that was
/// never told can say.
fn schema_from_store<S>(store: &Arc<S>) -> Result<SequenceSchema, MolRsError>
where
    S: ?Sized + ReadableStorageTraits + ListableStorageTraits + 'static,
{
    let mut blocks = BTreeMap::new();
    for section in children(store, TRAJECTORY_GROUP)? {
        // A block section is a group. `step` and `time` are arrays and the two
        // reserved groups are named, so anything else at this level is not a
        // block section and minting one from it would only fail deeper in.
        if !matches!(section.metadata(), NodeMetadata::Group(_)) {
            continue;
        }
        let name = section.name().as_str().to_string();
        if RESERVED_BLOCK_NAMES.contains(&name.as_str()) {
            continue;
        }
        let path = join_path(TRAJECTORY_GROUP, &name);

        let mut columns = BTreeMap::new();
        for child in children(store, &path)? {
            if !matches!(child.metadata(), NodeMetadata::Array(_)) {
                continue;
            }
            let column = child.name().as_str().to_string();
            if RESERVED_COLUMN_NAMES.contains(&column.as_str()) {
                continue;
            }
            let column_path = join_path(&path, &column);
            let array = Array::open(store.clone(), &column_path)?;
            let dtype = dtype_of_stored(array.data_type()).ok_or_else(|| {
                MolRsError::zarr(format!(
                    "{column_path} is stored as {:?}, which is no column width molrs reads",
                    array.data_type()
                ))
            })?;
            columns.insert(
                column,
                ColumnSchema {
                    dtype: dtype_tag(dtype).to_string(),
                    trailing: array.shape().iter().skip(1).copied().collect(),
                },
            );
        }

        // The section mirrors its structural shape for exactly this reader;
        // `read_block_rows` applies it only when it matches the row count, so
        // a shape that does not is inert rather than fatal.
        let structural_shape = Group::open(store.clone(), &path)?
            .attributes()
            .get(STRUCTURAL_SHAPE_ATTRIBUTE)
            .and_then(serde_json::Value::as_array)
            .map(|axes| {
                axes.iter()
                    .filter_map(|axis| axis.as_u64().map(|axis| axis as usize))
                    .collect::<Vec<usize>>()
            })
            .filter(|shape| !shape.is_empty());

        blocks.insert(
            name,
            BlockSchema {
                columns,
                structural_shape,
            },
        );
    }

    let meta_path = join_path(TRAJECTORY_GROUP, META_GROUP);
    let mut meta = BTreeMap::new();
    for child in children(store, &meta_path)? {
        if !matches!(child.metadata(), NodeMetadata::Array(_)) {
            continue;
        }
        let key = child.name().as_str().to_string();
        let key_path = join_path(&meta_path, &key);
        let array = Array::open(store.clone(), &key_path)?;
        let tag = array
            .attributes()
            .get(META_DTYPE_ATTRIBUTE)
            .and_then(serde_json::Value::as_str)
            .ok_or_else(|| {
                MolRsError::zarr(format!(
                    "{key_path} carries no {META_DTYPE_ATTRIBUTE} attribute; its per-step values \
                     cannot be read back"
                ))
            })?;
        meta.insert(
            key,
            MetaSchema {
                dtype: tag.to_string(),
                fill: None,
            },
        );
    }

    Ok(SequenceSchema { blocks, meta })
}

/// Refuse a store written by molrs <= 0.13, whose `trajectory/frames/<i>/`
/// groups this layout replaced.
///
/// Loud rather than empty: the one quadrant of the compatibility matrix that
/// could be bought for free.
fn ensure_not_legacy<S>(store: &Arc<S>) -> Result<(), MolRsError>
where
    S: ?Sized + ListableStorageTraits,
{
    let prefix = StorePrefix::new(format!(
        "{}/{LEGACY_FRAMES_GROUP}/",
        TRAJECTORY_GROUP.trim_start_matches('/')
    ))
    .map_err(zerr)?;
    if store.list_prefix(&prefix)?.is_empty() {
        return Ok(());
    }
    Err(MolRsError::zarr(
        "legacy layout (written by molrs <= 0.13); re-write with 0.13",
    ))
}

/// Whether an array exists at `path`.
fn array_exists<S>(store: &Arc<S>, path: &str) -> Result<bool, MolRsError>
where
    S: ?Sized + ReadableStorageTraits + 'static,
{
    match Array::open(store.clone(), path) {
        Ok(_) => Ok(true),
        Err(zarrs::array::ArrayCreateError::MissingMetadata) => Ok(false),
        Err(e) => Err(e.into()),
    }
}

/// Every element of the array at `path`.
fn read_whole<T, S>(store: &Arc<S>, path: &str) -> Result<Vec<T>, MolRsError>
where
    T: zarrs::array::ElementOwned,
    S: ?Sized + ReadableStorageTraits + 'static,
{
    let array = Array::open(store.clone(), path)?;
    let subset = ArraySubset::new_with_shape(array.shape().to_vec());
    array.retrieve_array_subset(&subset).map_err(zerr)
}

// ---------------------------------------------------------------------------
// Growth arrays
// ---------------------------------------------------------------------------

/// A `zarrs` array whose leading axis grows and whose extents never do.
///
/// `zarrs::Array::set_shape` grows the leading axis and rebuilds the chunk grid
/// from *frozen* chunk metadata, so the inner-chunk and shard extents are
/// decided once, at creation, and are not negotiable afterwards. Everything
/// this type offers follows from that: [`extend`](Self::extend) opens a row
/// range, [`compact`](Self::compact) re-encodes one shard as a clean full
/// write.
struct GrowthArray {
    array: Array<dyn ReadableWritableListableStorageTraits>,
    /// Rows one shard file spans — the seal unit.
    shard_rows: u64,
    /// Rows landed so far.
    rows: u64,
}

impl GrowthArray {
    /// Create the array at `path`, zero rows long, extents frozen.
    fn create(
        store: &ReadableWritableListableStorage,
        path: &str,
        dtype: DType,
        trailing: &[u64],
        extents: (u64, u64),
        attributes: serde_json::Map<String, serde_json::Value>,
        sharded: bool,
    ) -> Result<Self, MolRsError> {
        let (rows_per_chunk, chunks_per_shard) = extents;
        let (data_type, fill) = zarr_dtype(dtype);
        let mut shape = Vec::with_capacity(trailing.len() + 1);
        shape.push(0u64);
        shape.extend_from_slice(trailing);
        // A chunk extent must be non-zero on every axis, including a trailing
        // axis that happens to be empty.
        let mut inner: Vec<u64> = shape.iter().map(|&axis| axis.max(1)).collect();
        inner[0] = rows_per_chunk;
        // Data columns (CSR coordinates) shard so file count stays O(arrays).
        // Index / optional arrays (step, offset, box origin…) stay unsharded:
        // they rarely fill even one inner chunk, and the shard wrapper is
        // several kilobytes around a few hundred bytes of payload.
        let chunk_extent = if sharded {
            let mut shard = inner.clone();
            shard[0] = rows_per_chunk.saturating_mul(chunks_per_shard);
            shard
        } else {
            inner.clone()
        };

        let mut builder = ArrayBuilder::new(shape, chunk_extent.clone(), data_type, fill);
        if sharded {
            // Sharded data columns keep lossless gzip — the one compressor
            // every reader of this store (wasm32 included) can decode. A
            // precision study admits no lossy codec here.
            builder.bytes_to_bytes_codecs(vec![Arc::new(
                GzipCodec::new(GZIP_LEVEL)
                    .map_err(|e| MolRsError::zarr(format!("gzip level {GZIP_LEVEL}: {e}")))?,
            )]);
            builder.subchunk_shape(inner);
        }
        // Unsharded index/optional arrays (step, offset, step_index, box) carry
        // a few hundred bytes and are appended on every flush. gzip has
        // `partial_encode: false`, so a compressor there forces a whole-chunk
        // read-decode-modify-encode-write of a fill-padded chunk per append —
        // ~5 ms of pure codec for no ratio on already-tiny data. Left as the
        // raw `bytes` codec, each append is a real positional write instead.
        builder.attributes(attributes);
        let array = builder
            .build(store.clone(), path)?
            .with_codec_specific_options(&deterministic_shard_layout());
        array.store_metadata()?;
        Ok(Self {
            array,
            shard_rows: chunk_extent[0].max(1),
            rows: 0,
        })
    }

    /// Reopen an existing array, adopting the extents it was created with.
    ///
    /// Adopting rather than re-deriving is not a shortcut: a re-planned grid
    /// cannot be laid over live data, so what is on disk *is* the plan.
    fn open(store: &ReadableWritableListableStorage, path: &str) -> Result<Self, MolRsError> {
        let array = Array::open(store.clone(), path)?
            .with_codec_specific_options(&deterministic_shard_layout());
        let metadata = serde_json::to_value(array.metadata()).map_err(zerr)?;
        let shard_rows = metadata["chunk_grid"]["configuration"]["chunk_shape"]
            .as_array()
            .and_then(|shape| shape.first())
            .and_then(serde_json::Value::as_u64)
            .ok_or_else(|| {
                MolRsError::zarr(format!(
                    "array {path:?} has no regular chunk grid to append to"
                ))
            })?;
        let rows = array.shape().first().copied().unwrap_or(0);
        Ok(Self {
            array,
            shard_rows: shard_rows.max(1),
            rows,
        })
    }

    /// The array's axes after the leading row axis.
    fn trailing(&self) -> Vec<u64> {
        self.array.shape().iter().skip(1).copied().collect()
    }

    /// Grow the leading axis by `count` rows and return the subset they land
    /// in. The caller writes the data.
    fn extend(&mut self, count: u64) -> Result<ArraySubset, MolRsError> {
        let start = self.rows;
        self.rows += count;
        let mut shape = self.array.shape().to_vec();
        shape[0] = self.rows;
        self.array.set_shape(shape)?;
        self.array.store_metadata()?;
        rows_subset(start, count, &self.trailing())
    }

    /// Re-encode shard `shard` as one clean full write.
    ///
    /// Branch A's price is paid here. Every rewrite of a partially filled
    /// trailing chunk leaves the superseded copy behind as dead bytes in the
    /// shard file; reading the shard's live rows, erasing it and writing them
    /// back once drops all of them at the cost of one sequential write.
    fn compact(&mut self, shard: u64) -> Result<(), MolRsError> {
        if shard.saturating_mul(self.shard_rows) >= self.rows {
            return Ok(());
        }
        let options = compaction_options();
        let mut indices = vec![0u64; self.array.shape().len()];
        indices[0] = shard;
        // Decode the shard whole and store it whole: one `set` of the shard
        // key, which truncates, so no superseded copy can survive it.
        let live: ArrayBytes<'static> = self.array.retrieve_chunk_opt(&indices, &options)?;
        self.array.store_chunk_opt(&indices, live, &options)?;
        Ok(())
    }

    /// Seal every shard this write completed.
    fn seal(&mut self, before: u64) -> Result<(), MolRsError> {
        let first = before / self.shard_rows;
        let completed = self.rows / self.shard_rows;
        for shard in first..completed {
            self.compact(shard)?;
        }
        Ok(())
    }

    /// Seal the final, still-partial shard — `close`'s half of the bargain.
    fn compact_tail(&mut self) -> Result<(), MolRsError> {
        if self.rows == 0 || self.rows.is_multiple_of(self.shard_rows) {
            // Nothing to seal, or the boundary crossing already sealed it.
            return Ok(());
        }
        self.compact(self.rows / self.shard_rows)
    }

    /// Land `columns` — one per update carrying this block — as one
    /// contiguous row range.
    fn store_columns(
        &self,
        subset: &ArraySubset,
        dtype: DType,
        columns: &[&Column],
        options: &CodecOptions,
    ) -> Result<(), MolRsError> {
        macro_rules! landed {
            ($variant:ident) => {{
                let mut data = Vec::new();
                for column in columns {
                    let Column::$variant(holder) = column else {
                        return Err(MolRsError::zarr(format!(
                            "column changed width mid-buffer: expected {dtype}, found {}",
                            column.dtype()
                        )));
                    };
                    data.extend(holder.as_standard_layout().iter().cloned());
                }
                self.array.store_array_subset_opt(subset, data, options)?;
            }};
        }
        match dtype {
            DType::Float16 => landed!(Float16),
            DType::Float32 => landed!(Float32),
            DType::Float => landed!(Float),
            DType::Int8 => landed!(Int8),
            DType::Int16 => landed!(Int16),
            DType::Int => landed!(Int),
            DType::Int64 => landed!(Int64),
            DType::Bool => landed!(Bool),
            DType::UInt => landed!(UInt),
            DType::U8 => landed!(U8),
            DType::UInt16 => landed!(UInt16),
            DType::UInt32 => landed!(UInt32),
            DType::String => landed!(String),
            DType::Complex64 => landed!(Complex64),
            DType::Complex128 => landed!(Complex128),
        }
        Ok(())
    }

    /// Land one typed value per pending frame into a per-step meta array.
    fn store_meta(
        &self,
        subset: &ArraySubset,
        key: &str,
        tag: &str,
        values: &[&MetaValue],
        options: &CodecOptions,
    ) -> Result<(), MolRsError> {
        let mismatch = |value: &MetaValue| {
            MolRsError::zarr(format!(
                "meta key {key:?} is declared {tag} but a frame carries {}",
                value.dtype()
            ))
        };
        macro_rules! scalar {
            ($variant:ident, $ty:ty) => {{
                let mut data: Vec<$ty> = Vec::with_capacity(values.len());
                for value in values.iter().copied() {
                    let MetaValue::$variant(inner) = value else {
                        return Err(mismatch(value));
                    };
                    data.push(inner.clone());
                }
                self.array.store_array_subset_opt(subset, data, options)?;
            }};
        }
        macro_rules! vector {
            ($variant:ident, $ty:ty) => {{
                let mut data: Vec<$ty> = Vec::new();
                for value in values.iter().copied() {
                    let MetaValue::$variant(inner) = value else {
                        return Err(mismatch(value));
                    };
                    data.extend_from_slice(inner);
                }
                self.array.store_array_subset_opt(subset, data, options)?;
            }};
        }
        match tag {
            "bool" => scalar!(Bool, bool),
            "i32" => scalar!(I32, i32),
            "i64" => scalar!(I64, i64),
            "u32" => scalar!(U32, u32),
            "u64" => scalar!(U64, u64),
            "f32" => scalar!(F32, f32),
            "f64" => scalar!(F64, f64),
            "string" => scalar!(String, String),
            "json" => {
                let mut data: Vec<String> = Vec::with_capacity(values.len());
                for value in values.iter().copied() {
                    let MetaValue::Json(inner) = value else {
                        return Err(mismatch(value));
                    };
                    data.push(serde_json::to_string(inner).map_err(zerr)?);
                }
                self.array.store_array_subset_opt(subset, data, options)?;
            }
            "bool3" => vector!(Bool3, bool),
            "i32x3" => vector!(I32x3, i32),
            "i64x3" => vector!(I64x3, i64),
            "u32x3" => vector!(U32x3, u32),
            "u64x3" => vector!(U64x3, u64),
            "f32x3" => vector!(F32x3, f32),
            "f64x3" => vector!(F64x3, f64),
            "f32x6" => vector!(F32x6, f32),
            "f64x6" => vector!(F64x6, f64),
            "f32x9" => vector!(F32x9, f32),
            "f64x9" => vector!(F64x9, f64),
            other => {
                return Err(MolRsError::zarr(format!(
                    "meta key {key:?} declares unknown dtype {other:?}"
                )));
            }
        }
        Ok(())
    }
}

/// One value of a per-step meta array, at step `index`.
fn read_meta_value(
    store: &ReadableListableStorage,
    path: &str,
    key: &str,
    tag: &str,
    index: u64,
) -> Result<MetaValue, MolRsError> {
    let (_, trailing) = meta_layout(tag).ok_or_else(|| {
        MolRsError::zarr(format!("meta key {key:?} declares unknown dtype {tag:?}"))
    })?;
    let array = Array::open(store.clone(), path)?;
    let subset = rows_subset(index, 1, &trailing)?;
    let short = |count: usize| {
        MolRsError::zarr(format!(
            "meta key {key:?} is {tag} but step {index} holds {count} value(s)"
        ))
    };
    macro_rules! scalar {
        ($variant:ident, $ty:ty) => {{
            let values: Vec<$ty> = array.retrieve_array_subset(&subset)?;
            let count = values.len();
            MetaValue::$variant(values.into_iter().next().ok_or_else(|| short(count))?)
        }};
    }
    macro_rules! vector {
        ($variant:ident, $ty:ty, $len:expr) => {{
            let values: Vec<$ty> = array.retrieve_array_subset(&subset)?;
            let values: [$ty; $len] = values
                .try_into()
                .map_err(|values: Vec<$ty>| short(values.len()))?;
            MetaValue::$variant(values)
        }};
    }
    Ok(match tag {
        "bool" => scalar!(Bool, bool),
        "i32" => scalar!(I32, i32),
        "i64" => scalar!(I64, i64),
        "u32" => scalar!(U32, u32),
        "u64" => scalar!(U64, u64),
        "f32" => scalar!(F32, f32),
        "f64" => scalar!(F64, f64),
        "string" => scalar!(String, String),
        "json" => {
            let values: Vec<String> = array.retrieve_array_subset(&subset)?;
            let count = values.len();
            let text = values.into_iter().next().ok_or_else(|| short(count))?;
            MetaValue::Json(serde_json::from_str(&text).map_err(zerr)?)
        }
        "bool3" => vector!(Bool3, bool, 3),
        "i32x3" => vector!(I32x3, i32, 3),
        "i64x3" => vector!(I64x3, i64, 3),
        "u32x3" => vector!(U32x3, u32, 3),
        "u64x3" => vector!(U64x3, u64, 3),
        "f32x3" => vector!(F32x3, f32, 3),
        "f64x3" => vector!(F64x3, f64, 3),
        "f32x6" => vector!(F32x6, f32, 6),
        "f64x6" => vector!(F64x6, f64, 6),
        "f32x9" => vector!(F32x9, f32, 9),
        "f64x9" => vector!(F64x9, f64, 9),
        other => {
            return Err(MolRsError::zarr(format!(
                "meta key {key:?} declares unknown dtype {other:?}"
            )));
        }
    })
}

/// Read update `j` of a block section back as a [`Block`].
fn read_block_rows<S>(
    store: &Arc<S>,
    path: &str,
    schema: &BlockSchema,
    start: u64,
    rows: u64,
) -> Result<Block, MolRsError>
where
    S: ?Sized + ReadableStorageTraits + 'static,
{
    let mut block = Block::new();
    for (column, declared) in &schema.columns {
        let subset = rows_subset(start, rows, &declared.trailing)?;
        let values = read_column(store, &join_path(path, column), &subset)?;
        insert_column_into_block(&mut block, column, values)?;
    }
    if schema.columns.is_empty() {
        // A declared block with no columns is still a block with a row count.
        block.resize(rows as usize)?;
    }
    if let Some(shape) = &schema.structural_shape
        && shape.iter().product::<usize>() == rows as usize
    {
        block
            .set_shape(shape)
            .map_err(|e| MolRsError::zarr(format!("block {path:?} shape {shape:?}: {e}")))?;
    }
    Ok(block)
}

/// Read cell update `j` of the `box/` section.
fn read_box_row<S>(store: &Arc<S>, index: u64, cell_defined: bool) -> Result<SimBox, MolRsError>
where
    S: ?Sized + ReadableStorageTraits + 'static,
{
    let prefix = join_path(TRAJECTORY_GROUP, BOX_GROUP);
    let cell: Vec<F> = Array::open(store.clone(), &join_path(&prefix, VECTORS_ARRAY))?
        .retrieve_array_subset(&rows_subset(index, 1, &[3, 3])?)?;
    let origin_path = join_path(&prefix, ORIGIN_ARRAY);
    let origin: Vec<F> = if array_exists(store, &origin_path)? {
        Array::open(store.clone(), &origin_path)?.retrieve_array_subset(&rows_subset(
            index,
            1,
            &[3],
        )?)?
    } else {
        vec![0.0, 0.0, 0.0]
    };
    let boundary_path = join_path(&prefix, BOUNDARY_ARRAY);
    let boundary: Vec<bool> = if array_exists(store, &boundary_path)? {
        Array::open(store.clone(), &boundary_path)?.retrieve_array_subset(&rows_subset(
            index,
            1,
            &[3],
        )?)?
    } else {
        vec![true, true, true]
    };
    if cell.len() != 9 || origin.len() != 3 || boundary.len() != 3 {
        return Err(MolRsError::zarr(format!(
            "box update {index} is malformed: {} cell values, {} origin values, {} boundary flags",
            cell.len(),
            origin.len(),
            boundary.len()
        )));
    }
    SimBox::new_cell(
        Array2::from_shape_vec((3, 3), cell).map_err(zerr)?,
        Array1::from(origin),
        [boundary[0], boundary[1], boundary[2]],
        cell_defined,
    )
    .map_err(|e| MolRsError::zarr(format!("box update {index} is not a valid cell: {e:?}")))
}

// ---------------------------------------------------------------------------
// Writer
// ---------------------------------------------------------------------------

/// The arrays of one block section.
struct BlockArrays {
    columns: BTreeMap<String, GrowthArray>,
    offset: GrowthArray,
    step_index: GrowthArray,
    /// Rows landed across every update.
    total_rows: u64,
    /// Updates landed.
    updates: u64,
}

/// The arrays of the `box/` section.
///
/// `vectors` is the only required array. Origin (default zeros), boundary
/// (default all-periodic) and `step_index` (implied `[0]` for a single
/// update at ordinal 0) are created only when they carry information.
struct BoxArrays {
    step_index: Option<GrowthArray>,
    vectors: GrowthArray,
    origin: Option<GrowthArray>,
    boundary: Option<GrowthArray>,
}

impl BoxArrays {
    fn create_optional(
        store: &ReadableWritableListableStorage,
        name: &str,
        dtype: DType,
        trailing: &[u64],
        rows_per_chunk: Option<u64>,
        chunks_per_shard: Option<u64>,
    ) -> Result<GrowthArray, MolRsError> {
        let prefix = join_path(TRAJECTORY_GROUP, BOX_GROUP);
        GrowthArray::create(
            store,
            &join_path(&prefix, name),
            dtype,
            trailing,
            derive_extents(rows_per_chunk, chunks_per_shard, row_bytes(dtype, trailing)),
            serde_json::Map::new(),
            false,
        )
    }

    fn ensure_origin(
        &mut self,
        store: &ReadableWritableListableStorage,
        rows_per_chunk: Option<u64>,
        chunks_per_shard: Option<u64>,
        previous: u64,
        options: &CodecOptions,
    ) -> Result<&mut GrowthArray, MolRsError> {
        if self.origin.is_none() {
            let mut origin = Self::create_optional(
                store,
                ORIGIN_ARRAY,
                DType::Float,
                &[3],
                rows_per_chunk,
                chunks_per_shard,
            )?;
            if previous > 0 {
                let fill = vec![0.0 as F; (previous * 3) as usize];
                let subset = origin.extend(previous)?;
                origin
                    .array
                    .store_array_subset_opt(&subset, fill, options)?;
                origin.seal(0)?;
            }
            self.origin = Some(origin);
        }
        Ok(self.origin.as_mut().expect("origin was just created"))
    }

    fn ensure_boundary(
        &mut self,
        store: &ReadableWritableListableStorage,
        rows_per_chunk: Option<u64>,
        chunks_per_shard: Option<u64>,
        previous: u64,
        options: &CodecOptions,
    ) -> Result<&mut GrowthArray, MolRsError> {
        if self.boundary.is_none() {
            let mut boundary = Self::create_optional(
                store,
                BOUNDARY_ARRAY,
                DType::Bool,
                &[3],
                rows_per_chunk,
                chunks_per_shard,
            )?;
            if previous > 0 {
                let fill = vec![true; (previous * 3) as usize];
                let subset = boundary.extend(previous)?;
                boundary
                    .array
                    .store_array_subset_opt(&subset, fill, options)?;
                boundary.seal(0)?;
            }
            self.boundary = Some(boundary);
        }
        Ok(self.boundary.as_mut().expect("boundary was just created"))
    }

    fn ensure_step_index(
        &mut self,
        store: &ReadableWritableListableStorage,
        rows_per_chunk: Option<u64>,
        chunks_per_shard: Option<u64>,
        previous: u64,
        options: &CodecOptions,
    ) -> Result<&mut GrowthArray, MolRsError> {
        if self.step_index.is_none() {
            let mut step_index = Self::create_optional(
                store,
                STEP_INDEX_ARRAY,
                DType::UInt,
                &[],
                rows_per_chunk,
                chunks_per_shard,
            )?;
            if previous > 0 {
                // The omitted form is one update at ordinal 0.
                let fill = vec![0u64; previous as usize];
                let subset = step_index.extend(previous)?;
                step_index
                    .array
                    .store_array_subset_opt(&subset, fill, options)?;
                step_index.seal(0)?;
            }
            self.step_index = Some(step_index);
        }
        Ok(self
            .step_index
            .as_mut()
            .expect("step_index was just created"))
    }
}

/// Every array of one sequence, created at the first append and never
/// re-planned afterwards.
struct SequenceArrays {
    step: GrowthArray,
    time: Option<GrowthArray>,
    meta: BTreeMap<String, GrowthArray>,
    blocks: BTreeMap<String, BlockArrays>,
    cell: Option<BoxArrays>,
}

impl SequenceArrays {
    /// Create every zero-length array the schema declares.
    fn create(
        store: &ReadableWritableListableStorage,
        schema: &SequenceSchema,
        rows_per_chunk: Option<u64>,
        chunks_per_shard: Option<u64>,
        with_time: bool,
    ) -> Result<Self, MolRsError> {
        let dense = |dtype: DType, trailing: &[u64]| {
            derive_extents(rows_per_chunk, chunks_per_shard, row_bytes(dtype, trailing))
        };
        let step = GrowthArray::create(
            store,
            &join_path(TRAJECTORY_GROUP, STEP_ARRAY),
            DType::Int64,
            &[],
            dense(DType::Int64, &[]),
            serde_json::Map::new(),
            false,
        )?;
        let time = if with_time {
            Some(GrowthArray::create(
                store,
                &join_path(TRAJECTORY_GROUP, TIME_ARRAY),
                DType::Float,
                &[],
                dense(DType::Float, &[]),
                serde_json::Map::new(),
                false,
            )?)
        } else {
            None
        };

        let mut meta = BTreeMap::new();
        if !schema.meta.is_empty() {
            GroupBuilder::new()
                .build(store.clone(), &join_path(TRAJECTORY_GROUP, META_GROUP))?
                .store_metadata()?;
        }
        for (key, declared) in &schema.meta {
            let (dtype, trailing) = meta_layout(&declared.dtype).ok_or_else(|| {
                MolRsError::zarr(format!(
                    "meta key {key:?} declares unknown dtype {:?}",
                    declared.dtype
                ))
            })?;
            // The dtype rides with the array, which is what makes exactness
            // free: no convention has to be remembered to read it back.
            let mut attributes = serde_json::Map::new();
            attributes.insert(
                META_DTYPE_ATTRIBUTE.to_string(),
                serde_json::Value::String(declared.dtype.clone()),
            );
            meta.insert(
                key.clone(),
                GrowthArray::create(
                    store,
                    &join_path(&join_path(TRAJECTORY_GROUP, META_GROUP), key),
                    dtype,
                    &trailing,
                    dense(dtype, &trailing),
                    attributes,
                    false,
                )?,
            );
        }

        let mut blocks = BTreeMap::new();
        for (name, declared) in &schema.blocks {
            let path = join_path(TRAJECTORY_GROUP, name);
            let mut attributes = serde_json::Map::new();
            if let Some(shape) = &declared.structural_shape {
                // Mirrored onto the section for foreign readers of the
                // contract; molrs itself reads the schema attribute.
                attributes.insert(
                    STRUCTURAL_SHAPE_ATTRIBUTE.to_string(),
                    serde_json::Value::Array(
                        shape
                            .iter()
                            .map(|axis| serde_json::Value::from(*axis as u64))
                            .collect(),
                    ),
                );
            }
            GroupBuilder::new()
                .attributes(attributes)
                .build(store.clone(), &path)?
                .store_metadata()?;

            // Every column of a block shares one rows-per-chunk, set by the
            // widest of them, so a frame's rows stay aligned across the
            // section.
            let widest = declared
                .columns
                .values()
                .map(|column| {
                    dtype_from_tag(&column.dtype).map(|dtype| row_bytes(dtype, &column.trailing))
                })
                .collect::<Result<Vec<u64>, MolRsError>>()?
                .into_iter()
                .max()
                .unwrap_or(8);
            let extents = derive_extents(rows_per_chunk, chunks_per_shard, widest);
            let mut columns = BTreeMap::new();
            for (column, schema) in &declared.columns {
                columns.insert(
                    column.clone(),
                    GrowthArray::create(
                        store,
                        &join_path(&path, column),
                        dtype_from_tag(&schema.dtype)?,
                        &schema.trailing,
                        extents,
                        serde_json::Map::new(),
                        true,
                    )?,
                );
            }
            blocks.insert(
                name.clone(),
                BlockArrays {
                    columns,
                    offset: GrowthArray::create(
                        store,
                        &join_path(&path, OFFSET_ARRAY),
                        DType::UInt,
                        &[],
                        dense(DType::UInt, &[]),
                        serde_json::Map::new(),
                        false,
                    )?,
                    step_index: GrowthArray::create(
                        store,
                        &join_path(&path, STEP_INDEX_ARRAY),
                        DType::UInt,
                        &[],
                        dense(DType::UInt, &[]),
                        serde_json::Map::new(),
                        false,
                    )?,
                    total_rows: 0,
                    updates: 0,
                },
            );
        }

        Ok(Self {
            step,
            time,
            meta,
            blocks,
            cell: None,
        })
    }

    /// Reopen every array the schema declares, validating each against it.
    fn open(
        store: &ReadableWritableListableStorage,
        schema: &SequenceSchema,
    ) -> Result<Self, MolRsError> {
        let step = GrowthArray::open(store, &join_path(TRAJECTORY_GROUP, STEP_ARRAY))?;
        let time_path = join_path(TRAJECTORY_GROUP, TIME_ARRAY);
        let time = if array_exists(store, &time_path)? {
            Some(GrowthArray::open(store, &time_path)?)
        } else {
            None
        };

        let mut meta = BTreeMap::new();
        for key in schema.meta.keys() {
            meta.insert(
                key.clone(),
                GrowthArray::open(
                    store,
                    &join_path(&join_path(TRAJECTORY_GROUP, META_GROUP), key),
                )?,
            );
        }

        let mut blocks = BTreeMap::new();
        for (name, declared) in &schema.blocks {
            let path = join_path(TRAJECTORY_GROUP, name);
            let mut columns = BTreeMap::new();
            for (column, column_schema) in &declared.columns {
                let column_path = join_path(&path, column);
                let opened = GrowthArray::open(store, &column_path)?;
                validate_column(&opened, &column_path, column_schema)?;
                columns.insert(column.clone(), opened);
            }
            let offset = GrowthArray::open(store, &join_path(&path, OFFSET_ARRAY))?;
            let step_index = GrowthArray::open(store, &join_path(&path, STEP_INDEX_ARRAY))?;
            let total_rows = if step_index.rows == 0 {
                0
            } else {
                *read_whole::<u64, _>(store, &join_path(&path, OFFSET_ARRAY))?
                    .last()
                    .unwrap_or(&0)
            };
            blocks.insert(
                name.clone(),
                BlockArrays {
                    columns,
                    updates: step_index.rows,
                    offset,
                    step_index,
                    total_rows,
                },
            );
        }

        let box_prefix = join_path(TRAJECTORY_GROUP, BOX_GROUP);
        let cell = if array_exists(store, &join_path(&box_prefix, VECTORS_ARRAY))? {
            let optional = |name: &str| -> Result<Option<GrowthArray>, MolRsError> {
                let path = join_path(&box_prefix, name);
                if array_exists(store, &path)? {
                    Ok(Some(GrowthArray::open(store, &path)?))
                } else {
                    Ok(None)
                }
            };
            Some(BoxArrays {
                step_index: optional(STEP_INDEX_ARRAY)?,
                vectors: GrowthArray::open(store, &join_path(&box_prefix, VECTORS_ARRAY))?,
                origin: optional(ORIGIN_ARRAY)?,
                boundary: optional(BOUNDARY_ARRAY)?,
            })
        } else {
            None
        };

        Ok(Self {
            step,
            time,
            meta,
            blocks,
            cell,
        })
    }

    /// Create the `box/` section on the first cell-carrying flush.
    fn ensure_cell(
        &mut self,
        store: &ReadableWritableListableStorage,
        rows_per_chunk: Option<u64>,
        chunks_per_shard: Option<u64>,
        first: &SimBox,
    ) -> Result<&mut BoxArrays, MolRsError> {
        if self.cell.is_none() {
            let prefix = join_path(TRAJECTORY_GROUP, BOX_GROUP);
            let mut attributes = serde_json::Map::new();
            // Written only when false, exactly as `write_simbox` writes it for
            // the frame section: every store predating the attribute carries a
            // defined cell, so absent has to keep meaning `true`.
            if !first.is_cell_defined() {
                attributes.insert(
                    CELL_DEFINED_ATTRIBUTE.to_string(),
                    serde_json::Value::Bool(false),
                );
            }
            GroupBuilder::new()
                .attributes(attributes)
                .build(store.clone(), &prefix)?
                .store_metadata()?;
            let dense = |dtype: DType, trailing: &[u64]| {
                derive_extents(rows_per_chunk, chunks_per_shard, row_bytes(dtype, trailing))
            };
            self.cell = Some(BoxArrays {
                step_index: None,
                vectors: GrowthArray::create(
                    store,
                    &join_path(&prefix, VECTORS_ARRAY),
                    DType::Float,
                    &[3, 3],
                    dense(DType::Float, &[3, 3]),
                    serde_json::Map::new(),
                    false,
                )?,
                origin: None,
                boundary: None,
            });
        }
        Ok(self
            .cell
            .as_mut()
            .expect("the box section was just created"))
    }

    /// Every growth array, for the sweep `close` makes over all of them.
    fn all_mut(&mut self) -> Vec<&mut GrowthArray> {
        let mut all: Vec<&mut GrowthArray> = vec![&mut self.step];
        all.extend(self.time.as_mut());
        all.extend(self.meta.values_mut());
        for block in self.blocks.values_mut() {
            all.extend(block.columns.values_mut());
            all.push(&mut block.offset);
            all.push(&mut block.step_index);
        }
        if let Some(cell) = self.cell.as_mut() {
            all.extend(cell.step_index.as_mut());
            all.push(&mut cell.vectors);
            all.extend(cell.origin.as_mut());
            all.extend(cell.boundary.as_mut());
        }
        all
    }
}

/// Check one reopened column array against what the schema pinned for it.
fn validate_column(
    opened: &GrowthArray,
    path: &str,
    schema: &ColumnSchema,
) -> Result<(), MolRsError> {
    let found = dtype_of_stored(opened.array.data_type()).map_or_else(
        || format!("{:?}", opened.array.data_type()),
        |dtype| dtype_tag(dtype).to_string(),
    );
    if found != schema.dtype {
        return Err(MolRsError::zarr(format!(
            "sequence schema mismatch at {path}: dtype expected {}, found {found}",
            schema.dtype
        )));
    }
    let trailing: Vec<u64> = opened.array.shape().iter().skip(1).copied().collect();
    if trailing != schema.trailing {
        return Err(MolRsError::zarr(format!(
            "sequence schema mismatch at {path}: trailing shape expected {:?}, found {trailing:?}",
            schema.trailing
        )));
    }
    Ok(())
}

/// One appended frame, buffered until `flush` lands it.
struct PendingFrame {
    step: i64,
    time: Option<F>,
    /// Sections this frame changed: `Some` is new content, `None` is the
    /// zero-row update that marks the section gone.
    blocks: BTreeMap<String, Option<Block>>,
    /// The cell, when it changed.
    cell: Option<SimBox>,
    /// Every declared meta key, resolved to a value or its declared fill.
    meta: BTreeMap<String, MetaValue>,
}

/// The streaming producer of a frame sequence: one frame per
/// [`append`](Self::append), [`flush`](Self::flush) commits, `close(self)`
/// seals.
///
/// One of the three access forms of a single object — the eager in-memory
/// carrier is `Trajectory`, the lazy store cursor is [`FrameSequence`], and
/// this is the writer that produces what that cursor reads.
///
/// **No `Drop`.** Closing is [`close`](Self::close), which consumes the writer
/// and can therefore return the IO error a `Drop` would have had to swallow —
/// and a scientific writer that swallows an IO error is silent data loss. The
/// lifecycle is the explicit one `FrameIndexBuilder` already uses in this
/// crate: create / append / flush / close.
///
/// **Error vocabulary.** Every door on this type yields [`MolRsError`], and the
/// storage-backed ones arrive as the [`MolRsError::Zarr`] variant carrying a
/// message that names the block, column, metadata key or array path that
/// disagreed.
pub struct FrameSequenceWriter {
    store: ReadableWritableListableStorage,
    schema: SequenceSchema,
    rows_per_chunk: Option<u64>,
    chunks_per_shard: Option<u64>,
    /// `None` until the first append freezes the extents and creates them.
    arrays: Option<SequenceArrays>,
    /// Highest step number landed or buffered.
    last_step: Option<i64>,
    /// Whether this run carries times. Fixed by the first append.
    uses_time: Option<bool>,
    /// Frames committed by a flush.
    committed: u64,
    pending: Vec<PendingFrame>,
    /// Last content landed or buffered per section, for the change detection
    /// that keeps an unchanging section at one `step_index` entry.
    landed_blocks: BTreeMap<String, Block>,
    landed_cell: Option<SimBox>,
}

impl FrameSequenceWriter {
    /// Mint a new sequence at `trajectory/` and pin `schema` to it.
    ///
    /// Writes the group and its schema attributes and nothing else: the arrays
    /// are created by the first [`append`](Self::append), which is what leaves
    /// the extent knobs a window to act in. A store that already holds a
    /// sequence is an `Err` naming it — never a silent overwrite, because that
    /// path is somebody's run.
    ///
    /// **`trajectory/` is cleared first.** Zarr stores are written key by key,
    /// so a mint that only wrote its own keys would inherit every leftover
    /// child of whatever stood at that node; this door therefore erases the
    /// whole `trajectory/` prefix before writing. The check above guards the
    /// case that matters — a *molrs sequence*, recognised by its
    /// `trajectory/step` array, is refused rather than erased — but anything
    /// else under that node is destroyed, including the
    /// `trajectory/frames/<i>/` tree written by molrs <= 0.13, which carries no
    /// `trajectory/step` array to be recognised by. Reopen with
    /// [`open`](Self::open) instead when the store may already hold data:
    /// that door refuses the legacy layout by name.
    ///
    /// # Errors
    ///
    /// A [`MolRsError::Zarr`] naming `trajectory/step` when that array already
    /// exists, so nothing is erased and nothing is written. Otherwise any
    /// storage error raised while erasing the prefix or writing the group
    /// metadata.
    pub fn create(
        store: ReadableWritableListableStorage,
        schema: SequenceSchema,
    ) -> Result<Self, MolRsError> {
        let step_path = join_path(TRAJECTORY_GROUP, STEP_ARRAY);
        if array_exists(&store, &step_path)? {
            return Err(MolRsError::zarr(format!(
                "a frame sequence already exists at {step_path:?}; refusing to overwrite it"
            )));
        }
        // This writer's target node is `trajectory/`, so whatever it held
        // before is not part of the sequence being minted.
        store.erase_prefix(&node_prefix(TRAJECTORY_GROUP)?)?;
        let mut attributes = serde_json::Map::new();
        attributes.insert(
            SCHEMA_ATTRIBUTE.to_string(),
            serde_json::to_value(&schema).map_err(zerr)?,
        );
        GroupBuilder::new()
            .attributes(attributes)
            .build(store.clone(), TRAJECTORY_GROUP)?
            .store_metadata()?;
        Ok(Self {
            store,
            schema,
            rows_per_chunk: None,
            chunks_per_shard: None,
            arrays: None,
            last_step: None,
            uses_time: None,
            committed: 0,
            pending: Vec::new(),
            landed_blocks: BTreeMap::new(),
            landed_cell: None,
        })
    }

    /// Mint a new sequence at `path` — the path-taking door over the fast
    /// positional-write store.
    ///
    /// Builds the [`PositionalWriteStore`] that keeps an append a tail write
    /// rather than a whole-shard rewrite, so a streaming producer reaches the
    /// fast path in one call instead of reconstructing the store's write
    /// policy. The store-taking [`create`](Self::create) stays for callers with
    /// a store of their own (in-memory, packed).
    ///
    /// [`PositionalWriteStore`]: crate::io::zarr::store::PositionalWriteStore
    ///
    /// # Errors
    ///
    /// The store-root errors of [`PositionalWriteStore::new`], then every error
    /// [`create`](Self::create) can raise.
    #[cfg(feature = "filesystem")]
    pub fn create_at(
        path: impl AsRef<std::path::Path>,
        schema: SequenceSchema,
    ) -> Result<Self, MolRsError> {
        let store: ReadableWritableListableStorage =
            Arc::new(crate::io::zarr::store::PositionalWriteStore::new(path)?);
        Self::create(store, schema)
    }

    /// Reattach to the sequence at `path` and continue appending — the
    /// path-taking counterpart of [`open`](Self::open), over the same
    /// positional-write store [`create_at`](Self::create_at) uses.
    ///
    /// # Errors
    ///
    /// The store-root errors of `PositionalWriteStore::new`, then every error
    /// [`open`](Self::open) can raise.
    #[cfg(feature = "filesystem")]
    pub fn open_at(path: impl AsRef<std::path::Path>) -> Result<Self, MolRsError> {
        let store: ReadableWritableListableStorage =
            Arc::new(crate::io::zarr::store::PositionalWriteStore::new(path)?);
        Self::open(store)
    }

    /// Reattach to an existing sequence and continue appending to it.
    ///
    /// The schema comes off the group attributes and every array is checked
    /// against it; the extents come off the arrays themselves, because a chunk
    /// grid cannot be re-planned under live data.
    ///
    /// Reopening for append requires the writer's own schema pin — the read
    /// door's derivation from the store is not a substitute for it.
    ///
    /// # Errors
    ///
    /// A [`MolRsError::Zarr`] when the store holds the `trajectory/frames/<i>/`
    /// layout written by molrs <= 0.13, which this layout replaced and does not
    /// migrate; the message says to re-write such a store with 0.13. Also when
    /// `trajectory/` carries no `molrs_sequence_schema` attribute — a store
    /// that a foreign writer produced can be *read* without the pin (see
    /// [`FrameSequence::open`]) but not appended to, because appending needs
    /// the declared union a later frame is checked against and the fills for
    /// metadata keys a step omits, neither of which is recoverable from data
    /// that was never written. Finally, when a reopened column array disagrees
    /// with the pinned schema on dtype or trailing shape, the error names the
    /// array path and both sides.
    pub fn open(store: ReadableWritableListableStorage) -> Result<Self, MolRsError> {
        ensure_not_legacy(&store)?;
        let schema = schema_of(&store)?;
        if !array_exists(&store, &join_path(TRAJECTORY_GROUP, STEP_ARRAY))? {
            // Created but never appended: nothing is frozen yet.
            return Ok(Self {
                store,
                schema,
                rows_per_chunk: None,
                chunks_per_shard: None,
                arrays: None,
                last_step: None,
                uses_time: None,
                committed: 0,
                pending: Vec::new(),
                landed_blocks: BTreeMap::new(),
                landed_cell: None,
            });
        }

        let arrays = SequenceArrays::open(&store, &schema)?;
        let committed = arrays.step.rows;
        let last_step = if committed == 0 {
            None
        } else {
            read_whole::<i64, _>(&store, &join_path(TRAJECTORY_GROUP, STEP_ARRAY))?
                .last()
                .copied()
        };
        let uses_time = Some(arrays.time.is_some());

        // Recover the change-detection state, so a section that does not move
        // across the reopen still costs one `step_index` entry rather than two.
        let mut landed_blocks = BTreeMap::new();
        for (name, block) in &arrays.blocks {
            if block.updates == 0 {
                continue;
            }
            let path = join_path(TRAJECTORY_GROUP, name);
            let offsets = read_whole::<u64, _>(&store, &join_path(&path, OFFSET_ARRAY))?;
            let (Some(&end), Some(&start)) =
                (offsets.last(), offsets.get(offsets.len().saturating_sub(2)))
            else {
                continue;
            };
            if end == start {
                // The last update marks the section absent.
                continue;
            }
            let schema = &schema.blocks[name];
            landed_blocks.insert(
                name.clone(),
                read_block_rows(&store, &path, schema, start, end - start)?,
            );
        }
        let landed_cell = match arrays.cell.as_ref() {
            Some(cell) if cell.vectors.rows > 0 => Some(read_box_row(
                &store,
                cell.vectors.rows - 1,
                cell_defined_of(&store)?,
            )?),
            _ => None,
        };

        Ok(Self {
            store,
            schema,
            rows_per_chunk: None,
            chunks_per_shard: None,
            arrays: Some(arrays),
            last_step,
            uses_time,
            committed,
            pending: Vec::new(),
            landed_blocks,
            landed_cell,
        })
    }

    /// Rows one inner chunk of every growth array holds.
    ///
    /// Legal only before the first append, which freezes the extents. Its
    /// consumer is the lifecycle test: a chunk boundary in a handful of frames
    /// instead of 512 KiB of them. Left unset, the value is derived from that
    /// 512 KiB target and the row width.
    ///
    /// # Errors
    ///
    /// A [`MolRsError::Zarr`] when `rows` is 0 — a chunk extent must be at
    /// least one row on every axis — or when the first
    /// [`append`](Self::append) has already run, since `zarrs` cannot re-plan a
    /// chunk grid under live data.
    pub fn with_rows_per_chunk(mut self, rows: u64) -> Result<Self, MolRsError> {
        self.refuse_after_first_append("with_rows_per_chunk")?;
        if rows == 0 {
            return Err(MolRsError::zarr("rows_per_chunk must be at least 1"));
        }
        self.rows_per_chunk = Some(rows);
        Ok(self)
    }

    /// Inner chunks one shard file holds.
    ///
    /// Legal only before the first append. Its consumer is the file-count
    /// bound: a small `chunks` makes even a small store span several shard
    /// files, so that bound is measured rather than assumed. Left unset, the
    /// value is derived from a 256 MiB shard target and the chunk size.
    ///
    /// # Errors
    ///
    /// A [`MolRsError::Zarr`] when `chunks` is 0 — a shard spans at least one
    /// inner chunk — or when the first [`append`](Self::append) has already
    /// run, since `zarrs` cannot re-plan a chunk grid under live data.
    pub fn with_chunks_per_shard(mut self, chunks: u64) -> Result<Self, MolRsError> {
        self.refuse_after_first_append("with_chunks_per_shard")?;
        if chunks == 0 {
            return Err(MolRsError::zarr("chunks_per_shard must be at least 1"));
        }
        self.chunks_per_shard = Some(chunks);
        Ok(self)
    }

    fn refuse_after_first_append(&self, knob: &str) -> Result<(), MolRsError> {
        if self.arrays.is_some() {
            return Err(MolRsError::zarr(format!(
                "{knob} is legal only before the first append: the chunk and shard extents are \
                 frozen when the first append creates the arrays, and zarrs cannot re-plan a grid \
                 under live data"
            )));
        }
        Ok(())
    }

    /// Buffer `frame` at the next step number, with no time.
    ///
    /// Step numbers start at 0 and rise by one — or, after
    /// [`open`](Self::open) has reattached to an existing sequence, continue
    /// from the last step already landed. A run that carries real MD (molecular
    /// dynamics) step numbers or times uses [`append_at`](Self::append_at)
    /// instead.
    ///
    /// # Errors
    ///
    /// [`append_at`](Self::append_at)'s, with one of them unreachable: this
    /// door picks the step number itself, so it cannot go backwards. It passes
    /// no time, so on a sequence started *with* times it errs on the
    /// all-or-nothing rule.
    pub fn append(&mut self, frame: &Frame) -> Result<(), MolRsError> {
        self.append_at(frame, self.last_step.map_or(0, |step| step + 1), None)
    }

    /// Buffer `frame` at an explicit step number and optional time.
    ///
    /// `step` is the run's own dimensionless step counter. It must be strictly
    /// greater than the previous one — a sequence is ordered, and a step that
    /// goes backwards would make the `step_index` binary search meaningless.
    ///
    /// `time` is the frame's physical time in **femtoseconds (fs)**, molrs's
    /// time unit throughout. It is stored verbatim; this door converts nothing,
    /// so a caller working in another unit converts before calling. `time` is
    /// also all-or-nothing across a run: the first append decides whether the
    /// `time` array exists at all, and mixing afterwards is an `Err` naming it.
    ///
    /// No frame *data* reaches the store here: the frame is checked and
    /// buffered, and the next [`flush`](Self::flush) lands it. What gets
    /// buffered is only what *changed* — a block or a cell whose bytes are
    /// identical to the previous step's earns no new update, which is why a run
    /// with fixed topology costs one `step_index` entry rather than one per
    /// step. The very first append is the exception in one respect: it creates
    /// every array the schema declares, and that is the moment the chunk and
    /// shard extents freeze.
    ///
    /// # Errors
    ///
    /// Every case is a [`MolRsError::Zarr`] naming what disagreed:
    ///
    /// - `step` is not strictly greater than the previous step;
    /// - `time` is supplied on a run started without times, or omitted on a run
    ///   started with them;
    /// - the frame carries a block, a column, or a `meta` key the pinned schema
    ///   does not declare — a schema is pinned at create and cannot grow
    ///   mid-run;
    /// - a column or `meta` key is declared with one dtype and this frame
    ///   carries another, or a column's trailing shape disagrees;
    /// - a block is present at this step but omits one of its declared columns:
    ///   sparsity is per block, not per column, so a block appears whole or not
    ///   at all;
    /// - the frame omits a declared `meta` key for which no fill was declared
    ///   through [`SequenceSchema::declare_meta`].
    ///
    /// Because the first append creates the arrays, it can additionally surface
    /// a storage error that later appends cannot.
    pub fn append_at(
        &mut self,
        frame: &Frame,
        step: i64,
        time: Option<F>,
    ) -> Result<(), MolRsError> {
        if let Some(previous) = self.last_step
            && step <= previous
        {
            return Err(MolRsError::zarr(format!(
                "step numbers must increase strictly: {step} follows {previous}"
            )));
        }
        match self.uses_time {
            None => self.uses_time = Some(time.is_some()),
            Some(expected) if expected != time.is_some() => {
                return Err(MolRsError::zarr(format!(
                    "the {TIME_ARRAY:?} array is all-or-nothing across a run: this sequence was \
                     started {}, and this frame supplies {}",
                    if expected {
                        "with times"
                    } else {
                        "without times"
                    },
                    if time.is_some() { "one" } else { "none" }
                )));
            }
            Some(_) => {}
        }

        self.validate(frame)?;
        let meta = self.resolve_meta(frame)?;
        self.ensure_arrays()?;

        let mut blocks = BTreeMap::new();
        for name in self.schema.blocks.keys() {
            match frame.get(name) {
                Some(block) => {
                    let changed = self
                        .landed_blocks
                        .get(name)
                        .is_none_or(|landed| !same_block(landed, block));
                    if changed {
                        blocks.insert(name.clone(), Some(block.clone()));
                        self.landed_blocks.insert(name.clone(), block.clone());
                    }
                }
                None => {
                    if self.landed_blocks.remove(name).is_some() {
                        // The section was present and is not any more; a
                        // zero-row update is how the layout says so.
                        blocks.insert(name.clone(), None);
                    }
                }
            }
        }

        let cell = match &frame.simbox {
            Some(simbox) => {
                let changed = self
                    .landed_cell
                    .as_ref()
                    .is_none_or(|landed| !same_simbox(landed, simbox));
                if changed {
                    self.landed_cell = Some(simbox.clone());
                    Some(simbox.clone())
                } else {
                    None
                }
            }
            // The box section has no absence marker; see the module doc.
            None => None,
        };

        self.last_step = Some(step);
        self.pending.push(PendingFrame {
            step,
            time,
            blocks,
            cell,
            meta,
        });
        Ok(())
    }

    /// Check `frame` against the pinned schema.
    ///
    /// Schema only: `Validator::canonical` stays the read/write doors' job, as
    /// it is today, and append does not smuggle in a second semantic gate.
    fn validate(&self, frame: &Frame) -> Result<(), MolRsError> {
        for (name, block) in frame.iter() {
            let Some(declared) = self.schema.blocks.get(name) else {
                return Err(MolRsError::zarr(format!(
                    "block {name:?} is not declared by this sequence: a schema is pinned at \
                     create and cannot grow mid-run"
                )));
            };
            for (column, values) in block.iter() {
                let Some(pinned) = declared.columns.get(column) else {
                    return Err(MolRsError::zarr(format!(
                        "column {column:?} of block {name:?} is not declared by this sequence: a \
                         schema is pinned at create and cannot grow mid-run"
                    )));
                };
                let dtype = dtype_tag(values.dtype());
                if pinned.dtype != dtype {
                    return Err(MolRsError::zarr(format!(
                        "column {column:?} of block {name:?} is declared {} but this frame carries \
                         {dtype}",
                        pinned.dtype
                    )));
                }
                let trailing: Vec<u64> = values.shape().iter().skip(1).map(|&n| n as u64).collect();
                if pinned.trailing != trailing {
                    return Err(MolRsError::zarr(format!(
                        "column {column:?} of block {name:?} is declared with trailing shape {:?} \
                         but this frame carries {trailing:?}",
                        pinned.trailing
                    )));
                }
            }
            for column in declared.columns.keys() {
                if !block.contains_key(column) {
                    return Err(MolRsError::zarr(format!(
                        "block {name:?} is present at this step but omits its declared column \
                         {column:?}: sparsity is per block, not per column"
                    )));
                }
            }
        }
        Ok(())
    }

    /// Resolve every declared meta key to the value this step stores.
    fn resolve_meta(&self, frame: &Frame) -> Result<BTreeMap<String, MetaValue>, MolRsError> {
        for key in frame.meta.keys() {
            if key == STEP_ARRAY || key == TIME_ARRAY {
                continue;
            }
            if !self.schema.meta.contains_key(key) {
                return Err(MolRsError::zarr(format!(
                    "meta key {key:?} is not declared by this sequence: a schema is pinned at \
                     create and cannot grow mid-run"
                )));
            }
        }
        let mut resolved = BTreeMap::new();
        for (key, declared) in &self.schema.meta {
            let value = match frame.meta.get(key) {
                Some(value) => {
                    if value.dtype() != declared.dtype {
                        return Err(MolRsError::zarr(format!(
                            "meta key {key:?} is declared {} but this frame carries {}",
                            declared.dtype,
                            value.dtype()
                        )));
                    }
                    value.clone()
                }
                None => {
                    let fill = declared.fill.as_ref().ok_or_else(|| {
                        MolRsError::zarr(format!(
                            "this frame omits the declared meta key {key:?} and no fill value was \
                             declared for it; there is no implicit fill"
                        ))
                    })?;
                    MetaValue::from_json_value(fill).map_err(MolRsError::zarr)?
                }
            };
            resolved.insert(key.clone(), value);
        }
        Ok(resolved)
    }

    /// Create every array on the first append, freezing the extents.
    fn ensure_arrays(&mut self) -> Result<(), MolRsError> {
        if self.arrays.is_none() {
            self.arrays = Some(SequenceArrays::create(
                &self.store,
                &self.schema,
                self.rows_per_chunk,
                self.chunks_per_shard,
                self.uses_time == Some(true),
            )?);
        }
        Ok(())
    }

    /// Commit every buffered frame.
    ///
    /// **What `len()` counts.** Every frame appended before this call, ragged
    /// trailing chunk included — commit granularity is any step, not a chunk
    /// boundary (branch A, defined in this module's own docs). `step` is
    /// extended **last**, and it is the commit marker: a reader sees `nstep`
    /// frames only once this call has written it.
    ///
    /// **What a crash loses.** Exactly the frames appended after the last
    /// `flush`, no more and no less. A crash between two of this call's writes
    /// leaves the data arrays long and `step` short, which reads back as the
    /// previous commit; the residual window is a crash *inside* one of the small
    /// `step` / `zarr.json` writes, and inside the shard re-encode described
    /// next.
    ///
    /// **What the active shard carries.** Branch A pays for arbitrary commit
    /// granularity in dead bytes: each rewrite of a partially filled trailing
    /// inner chunk leaves its superseded copy in the shard file, so a
    /// frequently flushed store's *active* shard grows beyond its live size.
    /// Nothing is hidden and nothing accumulates: a flush that carries the run
    /// across a shard boundary re-encodes the completed shard once as a clean
    /// full write — one extra sequential write of that shard, paid once per
    /// shard — and [`close`](Self::close) does the same for the final partial
    /// shard. A closed store therefore carries no dead bytes, whatever the
    /// flush cadence was.
    ///
    /// A flush with nothing buffered is `Ok(())` and writes nothing, so calling
    /// it on a cadence costs nothing on the steps that added no frame.
    ///
    /// # Errors
    ///
    /// A [`MolRsError::Zarr`] wrapping whatever the store or the codec raised
    /// while growing an array, encoding a chunk or re-encoding a completed
    /// shard. A handful of consistency checks also fire here rather than at
    /// append, because they compare the buffer against the pinned schema at the
    /// moment of writing: a buffered block that no longer carries a declared
    /// column, a buffered column whose width is not the declared one, and a
    /// `meta` key that was resolved at append but is missing, carries the wrong
    /// variant, or declares a dtype no per-step array can store. Each names the
    /// block, column or key.
    ///
    /// A flush that fails partway leaves the arrays it already extended longer
    /// than `step`; since `step` is the commit marker, such a store reads back
    /// as the *previous* commit rather than as a partial one.
    pub fn flush(&mut self) -> Result<(), MolRsError> {
        if self.pending.is_empty() {
            return Ok(());
        }
        let options = partial_encoding_options();
        let Self {
            store,
            schema,
            rows_per_chunk,
            chunks_per_shard,
            arrays,
            committed,
            pending,
            ..
        } = self;
        let Some(arrays) = arrays.as_mut() else {
            return Ok(());
        };
        let base = *committed;

        // 1. Block sections: columns first, then this section's own index.
        for (name, declared) in &schema.blocks {
            let Some(section) = arrays.blocks.get_mut(name) else {
                continue;
            };
            let updates: Vec<(u64, Option<&Block>)> = pending
                .iter()
                .enumerate()
                .filter_map(|(index, frame)| {
                    frame
                        .blocks
                        .get(name)
                        .map(|update| (base + index as u64, update.as_ref()))
                })
                .collect();
            if updates.is_empty() {
                continue;
            }

            let landed: Vec<&Block> = updates.iter().filter_map(|(_, block)| *block).collect();
            let added: u64 = landed
                .iter()
                .map(|block| block.nrows().unwrap_or(0) as u64)
                .sum();
            if added > 0 {
                for (column, pinned) in &declared.columns {
                    let Some(array) = section.columns.get_mut(column) else {
                        continue;
                    };
                    let before = array.rows;
                    let subset = array.extend(added)?;
                    let values: Vec<&Column> = landed
                        .iter()
                        .map(|block| {
                            block.get(column).ok_or_else(|| {
                                MolRsError::zarr(format!(
                                    "block {name:?} lost its declared column {column:?} between \
                                     append and flush"
                                ))
                            })
                        })
                        .collect::<Result<_, MolRsError>>()?;
                    array.store_columns(
                        &subset,
                        dtype_from_tag(&pinned.dtype)?,
                        &values,
                        &options,
                    )?;
                    array.seal(before)?;
                }
            }

            let mut offsets = Vec::with_capacity(updates.len() + 1);
            if section.updates == 0 {
                // The CSR row pointer opens at zero.
                offsets.push(0u64);
            }
            let mut running = section.total_rows;
            for (_, update) in &updates {
                running += update.map_or(0, |block| block.nrows().unwrap_or(0) as u64);
                offsets.push(running);
            }
            let before = section.offset.rows;
            let subset = section.offset.extend(offsets.len() as u64)?;
            section
                .offset
                .array
                .store_array_subset_opt(&subset, offsets, &options)?;
            section.offset.seal(before)?;

            let ordinals: Vec<u64> = updates.iter().map(|(ordinal, _)| *ordinal).collect();
            let before = section.step_index.rows;
            let subset = section.step_index.extend(ordinals.len() as u64)?;
            section
                .step_index
                .array
                .store_array_subset_opt(&subset, ordinals, &options)?;
            section.step_index.seal(before)?;

            section.updates += updates.len() as u64;
            section.total_rows = running;
        }

        // 2. The cell.
        let cells: Vec<(u64, &SimBox)> = pending
            .iter()
            .enumerate()
            .filter_map(|(index, frame)| {
                frame.cell.as_ref().map(|cell| (base + index as u64, cell))
            })
            .collect();
        if let Some((_, first)) = cells.first() {
            let section = arrays.ensure_cell(store, *rows_per_chunk, *chunks_per_shard, first)?;
            let mut vectors = Vec::with_capacity(cells.len() * 9);
            let mut origins = Vec::with_capacity(cells.len() * 3);
            let mut boundaries = Vec::with_capacity(cells.len() * 3);
            let mut need_origin = section.origin.is_some();
            let mut need_boundary = section.boundary.is_some();
            for (_, cell) in &cells {
                vectors.extend(cell.h_view().iter().copied());
                origins.extend(cell.origin_view().iter().copied());
                boundaries.extend(cell.pbc_view().iter().copied());
                need_origin |= !origin_is_default(cell);
                need_boundary |= !boundary_is_default(cell);
            }
            let count = cells.len() as u64;
            let previous = section.vectors.rows;
            let ordinals: Vec<u64> = cells.iter().map(|(ordinal, _)| *ordinal).collect();
            let trivial_index = previous == 0 && matches!(ordinals.as_slice(), [0]);

            let before = section.vectors.rows;
            let subset = section.vectors.extend(count)?;
            section
                .vectors
                .array
                .store_array_subset_opt(&subset, vectors, &options)?;
            section.vectors.seal(before)?;

            if need_origin {
                let origin = section.ensure_origin(
                    store,
                    *rows_per_chunk,
                    *chunks_per_shard,
                    previous,
                    &options,
                )?;
                let before = origin.rows;
                let subset = origin.extend(count)?;
                origin
                    .array
                    .store_array_subset_opt(&subset, origins, &options)?;
                origin.seal(before)?;
            }
            if need_boundary {
                let boundary = section.ensure_boundary(
                    store,
                    *rows_per_chunk,
                    *chunks_per_shard,
                    previous,
                    &options,
                )?;
                let before = boundary.rows;
                let subset = boundary.extend(count)?;
                boundary
                    .array
                    .store_array_subset_opt(&subset, boundaries, &options)?;
                boundary.seal(before)?;
            }
            if section.step_index.is_some() || !trivial_index {
                let step_index = section.ensure_step_index(
                    store,
                    *rows_per_chunk,
                    *chunks_per_shard,
                    previous,
                    &options,
                )?;
                let before = step_index.rows;
                let subset = step_index.extend(count)?;
                step_index
                    .array
                    .store_array_subset_opt(&subset, ordinals, &options)?;
                step_index.seal(before)?;
            }
        }

        // 3. Per-step metadata.
        for (key, declared) in &schema.meta {
            let Some(array) = arrays.meta.get_mut(key) else {
                continue;
            };
            let values: Vec<&MetaValue> = pending
                .iter()
                .map(|frame| {
                    frame.meta.get(key).ok_or_else(|| {
                        MolRsError::zarr(format!(
                            "meta key {key:?} was resolved at append but is missing at flush"
                        ))
                    })
                })
                .collect::<Result<_, MolRsError>>()?;
            let before = array.rows;
            let subset = array.extend(values.len() as u64)?;
            array.store_meta(&subset, key, &declared.dtype, &values, &options)?;
            array.seal(before)?;
        }

        // 4. Time.
        if let Some(array) = arrays.time.as_mut() {
            let times: Vec<F> = pending
                .iter()
                .map(|frame| frame.time.unwrap_or(0.0))
                .collect();
            let before = array.rows;
            let subset = array.extend(times.len() as u64)?;
            array
                .array
                .store_array_subset_opt(&subset, times, &options)?;
            array.seal(before)?;
        }

        // 5. The commit marker, last: everything above is invisible until it
        //    lands, which is what makes a crash between two writes harmless.
        let steps: Vec<i64> = pending.iter().map(|frame| frame.step).collect();
        let before = arrays.step.rows;
        let subset = arrays.step.extend(steps.len() as u64)?;
        arrays
            .step
            .array
            .store_array_subset_opt(&subset, steps, &options)?;
        arrays.step.seal(before)?;

        *committed += pending.len() as u64;
        pending.clear();
        Ok(())
    }

    /// Flush, seal the final partial shard of every array, and consume the
    /// writer.
    ///
    /// Consuming rather than dropping is the point: this is the only place an
    /// IO error at the end of a run can still be returned to the caller.
    ///
    /// A writer that was created but never appended to closes cleanly: there
    /// are no arrays yet, so there is nothing to seal.
    ///
    /// # Errors
    ///
    /// [`flush`](Self::flush)'s, since this flushes first, plus a
    /// [`MolRsError::Zarr`] wrapping whatever the store raised while
    /// re-encoding a final partial shard. The writer is consumed either way, so
    /// a failed close cannot be retried; whatever had already been committed
    /// stays readable, and a failure in the sealing pass costs only the dead
    /// bytes that pass would have removed.
    pub fn close(mut self) -> Result<(), MolRsError> {
        self.flush()?;
        if let Some(arrays) = self.arrays.as_mut() {
            for array in arrays.all_mut() {
                array.compact_tail()?;
            }
        }
        Ok(())
    }
}

/// The `box/` section's `cell_defined` attribute; absent means a defined cell.
fn cell_defined_of<S>(store: &Arc<S>) -> Result<bool, MolRsError>
where
    S: ?Sized + ReadableStorageTraits + 'static,
{
    let prefix = join_path(TRAJECTORY_GROUP, BOX_GROUP);
    let Ok(group) = Group::open(store.clone(), &prefix) else {
        return Ok(true);
    };
    Ok(group
        .attributes()
        .get(CELL_DEFINED_ATTRIBUTE)
        .and_then(serde_json::Value::as_bool)
        .unwrap_or(true))
}

// ---------------------------------------------------------------------------
// Reader
// ---------------------------------------------------------------------------

/// The index of one block section, as [`FrameSequence::open`] cached it.
struct BlockIndex {
    /// Frame ordinals at which the section was updated, ascending.
    step_index: Vec<u64>,
    /// CSR row pointer, one longer than `step_index`.
    offset: Vec<u64>,
}

/// The index of the `box/` section.
struct BoxIndex {
    step_index: Vec<u64>,
    cell_defined: bool,
}

/// The lazy store cursor over a frame sequence: index-only open, one frame per
/// read.
///
/// One of the three access forms of a single object — `Trajectory` is the eager
/// in-memory carrier that materializes every frame, [`FrameSequenceWriter`] is
/// the streaming producer, and this reads what that writer produced without
/// materializing anything it was not asked for. [`to_trajectory`](Self::to_trajectory)
/// is the named lazy → eager conversion.
///
/// **Error vocabulary.** Every inherent door on this type yields
/// [`MolRsError`], as the [`MolRsError::Zarr`] variant naming the section or
/// array path at fault. The three [`TrajectoryReader`] methods keep
/// `std::io::Result` and flatten a `MolRsError` into an `io::Error` at that
/// boundary — lossy, and deliberately so: the trait is the shape every
/// trajectory backend shares and must not be captured by one backend's error
/// type.
pub struct FrameSequence {
    /// The read-only view of the store. A read door keeps no write capability,
    /// which is also what lets a packed `.mrec.zip` — readable and listable and
    /// nothing more — be opened through the same door as a directory store.
    store: ReadableListableStorage,
    schema: SequenceSchema,
    /// Step numbers of the committed frames; its length is `nstep`.
    steps: Vec<i64>,
    times: Option<Vec<F>>,
    blocks: BTreeMap<String, BlockIndex>,
    cell: Option<BoxIndex>,
}

impl FrameSequence {
    /// Open a sequence for reading, taking only its indices.
    ///
    /// Index-only: the schema attributes plus each section's `step_index` and
    /// `offset`. No frame data is touched until [`frame`](Self::frame) asks
    /// for a frame.
    ///
    /// Any readable, listable store opens: a directory store, an in-memory one,
    /// or the read-only `zarrs_zip` adapter that `open_packed` (the
    /// `filesystem`-gated function in [`crate::io::mrec`]) hands back for a
    /// packed `.mrec.zip`. Whatever arrives is kept as a
    /// [`ReadableListableStorage`] — a reader that could still write would be a
    /// write door wearing the wrong name.
    ///
    /// A conforming store needs no writer pin — the schema is derived from the
    /// store itself when the attribute is absent. That is the one fact this
    /// door and [`FrameSequenceWriter::open`] answer differently, and
    /// deliberately: reading needs only what is on disk, appending needs the
    /// declared union and the metadata fills, which are not.
    ///
    /// A store created and closed without a single append opens as a legal,
    /// empty sequence rather than as an error.
    ///
    /// # Errors
    ///
    /// Every case is a [`MolRsError::Zarr`] naming the path it failed on:
    ///
    /// - the store holds the `trajectory/frames/<i>/` layout written by molrs
    ///   <= 0.13, which this layout replaced and does not migrate;
    /// - a schema pin is present but is not a schema this build can
    ///   deserialize;
    /// - no pin is present and the derivation from the store fails, either
    ///   because a column array is stored as a Zarr dtype molrs has no column
    ///   width for, or because a `trajectory/meta/<key>` array carries no
    ///   `molrs_meta_dtype` attribute to say how to read its values back.
    ///
    /// Also any storage error raised while reading the index arrays.
    pub fn open<S>(store: Arc<S>) -> Result<Self, MolRsError>
    where
        S: ?Sized + ReadableStorageTraits + ListableStorageTraits + 'static,
    {
        // `StorageHandle` is how an unsized store becomes an owned, sized one:
        // `Arc<dyn ReadableWritableListableStorageTraits>` does not coerce to
        // `Arc<dyn ReadableListableStorageTraits>` — the latter is not a
        // supertrait of the former — so the read-only view is taken here.
        let store: ReadableListableStorage = Arc::new(StorageHandle::new(store));
        ensure_not_legacy(&store)?;
        let schema = match pinned_schema(&store)? {
            Some(pinned) => pinned,
            None => schema_from_store(&store)?,
        };
        let steps = if array_exists(&store, &join_path(TRAJECTORY_GROUP, STEP_ARRAY))? {
            read_whole::<i64, _>(&store, &join_path(TRAJECTORY_GROUP, STEP_ARRAY))?
        } else {
            // Created and closed without an append: a legal, empty sequence.
            Vec::new()
        };
        let time_path = join_path(TRAJECTORY_GROUP, TIME_ARRAY);
        let times = if array_exists(&store, &time_path)? {
            Some(read_whole::<F, _>(&store, &time_path)?)
        } else {
            None
        };

        let mut blocks = BTreeMap::new();
        for name in schema.blocks.keys() {
            let path = join_path(TRAJECTORY_GROUP, name);
            if !array_exists(&store, &join_path(&path, STEP_INDEX_ARRAY))? {
                continue;
            }
            blocks.insert(
                name.clone(),
                BlockIndex {
                    step_index: read_whole::<u64, _>(&store, &join_path(&path, STEP_INDEX_ARRAY))?,
                    offset: read_whole::<u64, _>(&store, &join_path(&path, OFFSET_ARRAY))?,
                },
            );
        }

        let box_prefix = join_path(TRAJECTORY_GROUP, BOX_GROUP);
        let vectors_path = join_path(&box_prefix, VECTORS_ARRAY);
        let cell = if array_exists(&store, &vectors_path)? {
            let step_path = join_path(&box_prefix, STEP_INDEX_ARRAY);
            let step_index = if array_exists(&store, &step_path)? {
                read_whole::<u64, _>(&store, &step_path)?
            } else {
                let shape = Array::open(store.clone(), &vectors_path)?.shape().to_vec();
                let n = *shape.first().ok_or_else(|| {
                    MolRsError::zarr("box vectors array has an empty shape".to_string())
                })?;
                if n == 1 {
                    vec![0]
                } else {
                    return Err(MolRsError::zarr(format!(
                        "box has {n} updates but no step_index array"
                    )));
                }
            };
            Some(BoxIndex {
                step_index,
                cell_defined: cell_defined_of(&store)?,
            })
        } else {
            None
        };

        Ok(Self {
            store,
            schema,
            steps,
            times,
            blocks,
            cell,
        })
    }

    /// Read frame `index`, or `None` when it is past the commit marker.
    ///
    /// Asking for a frame that was appended but never flushed is `Ok(None)`,
    /// not an error: an uncommitted frame is simply not in the sequence. So is
    /// any `index` at or past the committed count, which is what lets a caller
    /// iterate until `None` without asking the length first.
    ///
    /// The frame is assembled from whichever sections resolve at `index`: a
    /// block whose `step_index` has no entry at or before `index`, or whose
    /// most recent update holds zero rows, is left out of the frame entirely.
    /// The cell has no absence marker, so once a run has written one, every
    /// later frame carries the most recent one.
    ///
    /// # Errors
    ///
    /// A [`MolRsError::Zarr`] naming the section when the store is internally
    /// inconsistent: a block whose `offset` array is too short for the update
    /// its `step_index` points at, a block present on disk but absent from the
    /// schema, a `box/` update that does not hold nine cell values, three
    /// origin values and three boundary flags, or a `meta` array whose row at
    /// `index` does not hold the value count its declared dtype requires. Also
    /// any storage or codec error raised while reading the rows themselves.
    pub fn frame(&mut self, index: u64) -> Result<Option<Frame>, MolRsError> {
        if index as usize >= self.steps.len() {
            return Ok(None);
        }
        let mut frame = Frame::new();

        for (name, block_index) in &self.blocks {
            let Some(update) = latest_update(&block_index.step_index, index) else {
                // No entry at or before this step: the section does not exist
                // here.
                continue;
            };
            let (Some(&start), Some(&end)) = (
                block_index.offset.get(update),
                block_index.offset.get(update + 1),
            ) else {
                return Err(MolRsError::zarr(format!(
                    "block {name:?} has {} offsets for update {update}",
                    block_index.offset.len()
                )));
            };
            // A hostile or corrupt store can carry a non-monotonic `offset`
            // array; `end - start` would then wrap to a near-`u64::MAX` row
            // count and drive an unbounded allocation. Reject it instead.
            let count = end.checked_sub(start).ok_or_else(|| {
                MolRsError::zarr(format!(
                    "block {name:?} update {update} has non-monotonic offsets ({start} > {end})"
                ))
            })?;
            if count == 0 {
                // A zero-row update is how the layout says "gone from here on".
                continue;
            }
            let schema = self.schema.blocks.get(name).ok_or_else(|| {
                MolRsError::zarr(format!("block {name:?} is on disk but not in the schema"))
            })?;
            let path = join_path(TRAJECTORY_GROUP, name);
            frame.insert(
                name.clone(),
                read_block_rows(&self.store, &path, schema, start, count)?,
            );
        }

        if let Some(cell) = &self.cell
            && let Some(update) = latest_update(&cell.step_index, index)
        {
            frame.simbox = Some(read_box_row(&self.store, update as u64, cell.cell_defined)?);
        }

        for (key, declared) in &self.schema.meta {
            let path = join_path(&join_path(TRAJECTORY_GROUP, META_GROUP), key);
            frame.meta.insert(
                key.clone(),
                read_meta_value(&self.store, &path, key, &declared.dtype, index)?,
            );
        }

        Ok(Some(frame))
    }

    /// Materialize the whole sequence: the named lazy → eager conversion.
    ///
    /// Reads every committed frame through [`frame`](Self::frame) and hands
    /// back a `Trajectory` — the eager in-memory carrier — carrying those
    /// frames, the sequence's step numbers, and its times when the run wrote
    /// any. `step` always comes back as `Some`: it is the commit marker and is
    /// therefore always on disk, so a `Trajectory` written with `step: None`
    /// returns with the numbering the writer assigned (`0, 1, 2, …`).
    ///
    /// Every frame is materialized at once. A run too large for memory is what
    /// [`frame`](Self::frame) is for.
    ///
    /// # Errors
    ///
    /// [`frame`](Self::frame)'s, on the first frame that raises one.
    // `to_trajectory` reads `&mut self` because the cursor owns mutable read
    // state; the name is the maintainer's vocabulary for lazy -> eager and is
    // not up for renaming by a lint.
    #[allow(clippy::wrong_self_convention)]
    pub fn to_trajectory(&mut self) -> Result<Trajectory, MolRsError> {
        let mut frames = Vec::with_capacity(self.steps.len());
        for index in 0..self.steps.len() as u64 {
            let Some(frame) = self.frame(index)? else {
                break;
            };
            frames.push(frame);
        }
        Ok(Trajectory {
            frames,
            step: Some(self.steps.clone()),
            time: self.times.clone(),
        })
    }

    /// The committed frames' step numbers, as [`open`](Self::open) read them.
    ///
    /// Already in memory — no frame decode. Its length is [`len`](Self::len).
    pub fn steps(&self) -> &[i64] {
        &self.steps
    }

    /// The committed frames' times, when the run wrote a `time` array.
    pub fn times(&self) -> Option<&[F]> {
        self.times.as_deref()
    }

    /// The sequence schema pinned or derived at [`open`](Self::open).
    pub fn schema(&self) -> &SequenceSchema {
        &self.schema
    }

    /// Whether a block section is present in the store (carries an on-disk
    /// `step_index`), so a reader can decide once — e.g. `has_block("bonds")`
    /// — instead of probing every frame for it.
    pub fn has_block(&self, name: &str) -> bool {
        self.blocks.contains_key(name)
    }

    /// The names of every block section present in the store.
    pub fn block_names(&self) -> impl Iterator<Item = &str> {
        self.blocks.keys().map(String::as_str)
    }
}

/// Index of the latest entry at or before frame `index`, or `None` when the
/// section had not appeared yet.
///
/// `step_index` is ascending by construction, so this is a binary search: the
/// cost of resolving a section is logarithmic in its *changes*, not in the run
/// length.
fn latest_update(step_index: &[u64], index: u64) -> Option<usize> {
    match step_index.binary_search(&index) {
        Ok(found) => Some(found),
        Err(0) => None,
        Err(after) => Some(after - 1),
    }
}

impl TrajectoryReader for FrameSequence {
    /// Already cached: [`FrameSequence::open`] is the index-only read, so
    /// there is nothing left for this to build.
    fn build_index(&mut self) -> std::io::Result<()> {
        Ok(())
    }

    /// [`FrameSequence::frame`] behind the backend-neutral trait: the same
    /// frame, and the same `Ok(None)` past the commit marker.
    ///
    /// The [`MolRsError`] is flattened into an `io::Error` here — lossy, and
    /// deliberately so, because the trait is the shape every backend shares and
    /// must not be captured by one backend's error type. A caller who needs the
    /// structured error calls [`FrameSequence::frame`] directly.
    fn read_step(&mut self, step: usize) -> std::io::Result<Option<Frame>> {
        self.frame(step as u64).map_err(std::io::Error::other)
    }

    /// Committed frames — the length of `trajectory/step` as
    /// [`FrameSequence::open`] read it.
    ///
    /// Infallible in practice: the count was taken at open, so this returns
    /// `Ok` unconditionally and is `io::Result` only to match the trait.
    fn len(&mut self) -> std::io::Result<usize> {
        Ok(self.steps.len())
    }
}

/// The whole `trajectory/` sequence contract, pinned before the code exists.
///
/// **Fixtures.** Every test writes a real store under a `tempfile::TempDir`;
/// nothing here is a mock except [`RecordingStore`], which exists only to
/// observe *write order* and delegates every byte to a real `FilesystemStore`.
/// The stock fixture frame is one `atoms` block carrying one `f64` column
/// [`X`], with non-repeating values, so a mis-sliced CSR row range is a
/// mismatch rather than a coincidence.
///
/// **What is pinned on disk.** The layout of Design §一 is the contract, so
/// these tests read `trajectory/step`, `trajectory/<block>/offset`,
/// `trajectory/<block>/step_index`, `trajectory/box/step_index` and
/// `trajectory/meta/<key>` directly through `zarrs`, in addition to reading
/// frames back through [`FrameSequence`]. A round trip that agreed with itself
/// while writing a private layout would satisfy neither molrec nor molpy.
///
/// **No tolerances anywhere.** Every value assertion is `assert_eq!` on the
/// bits that arrived: the downstream precision study forbids lossy encoding,
/// so a comparison with a tolerance would hide exactly the defect this suite
/// is for.
///
/// **Extents are read from the on-disk array metadata**, because the growth
/// arrays' chunk and shard extents are frozen at the first append and have no
/// other observation point ([`extents`]).
#[cfg(all(test, feature = "filesystem"))]
mod tests {
    use std::collections::BTreeMap;
    use std::path::{Path, PathBuf};
    use std::sync::{Arc, Mutex};

    use molrs::spatial::simbox::SimBox;
    use molrs::store::block::{Block, Column, DType};
    use molrs::store::frame::Frame;
    use molrs::store::meta::MetaValue;
    use ndarray::{ArrayD, array};
    use tempfile::TempDir;
    use zarrs::array::{Array, ArrayBuilder, ArraySubset, data_type};
    use zarrs::filesystem::FilesystemStore;
    use zarrs::group::{Group, GroupBuilder};
    use zarrs::storage::byte_range::{ByteRange, ByteRangeIterator};
    use zarrs::storage::{
        Bytes, ListableStorageTraits, MaybeBytes, MaybeBytesIterator, OffsetBytesIterator,
        ReadableStorageTraits, ReadableWritableListableStorage, StorageError, StoreKey, StoreKeys,
        StoreKeysPrefixes, StorePrefix, WritableStorageTraits,
    };

    use crate::io::reader::TrajectoryReader;
    use crate::io::zarr::store::PositionalWriteStore;

    use super::{FrameSequence, FrameSequenceWriter, SequenceSchema};

    // -- fixtures -----------------------------------------------------------

    /// The sequence root group: the on-disk name of the whole layout.
    const TRAJ: &str = "/trajectory";
    /// The `trajectory/` group attribute the **reference writer** pins its
    /// declared schema in.
    ///
    /// Spelled out rather than imported from `super`: this is a name on disk
    /// that a foreign writer reads and writes, so a rename of the Rust constant
    /// must break these tests rather than travel silently into them — the same
    /// reading `every_meta_variant_round_trips_bit_exact_with_its_dtype_tag`
    /// takes on `molrs_meta_dtype`.
    const SCHEMA_ATTRIBUTE: &str = "molrs_sequence_schema";
    /// The block every stock fixture frame carries.
    const ATOMS: &str = "atoms";
    /// The one `f64` column in that block.
    const X: &str = "x";
    /// A second block, used wherever a frame must present a *subset* of the
    /// declared union.
    const BONDS: &str = "bonds";
    /// The `u64` column of [`BONDS`].
    const I: &str = "i";
    /// A column name outside the canonical schema vocabulary
    /// (`core/store/schema/mod.rs`), so the same name can carry two dtypes.
    ///
    /// [`X`] cannot: `x` is pinned `Float` there, and `Block::insert_column`
    /// rejects a `u64` under that name before any sequence code runs — the
    /// dtype-conflict fixtures would then be testing `Block`, not the schema.
    /// Same reason `frame_io`'s round-trip tests probe on a name of their own.
    const PROBE: &str = "probe";

    fn store_in(dir: &TempDir) -> ReadableWritableListableStorage {
        Arc::new(FilesystemStore::new(dir.path()).unwrap())
    }

    fn float_column(values: &[f64]) -> Column {
        Column::from_float(ArrayD::from_shape_vec(vec![values.len()], values.to_vec()).unwrap())
    }

    fn uint_column(values: &[u64]) -> Column {
        Column::from_uint(ArrayD::from_shape_vec(vec![values.len()], values.to_vec()).unwrap())
    }

    fn block_with(column: &str, col: Column) -> Block {
        let mut block = Block::new();
        block.insert_column(column, col).unwrap();
        block
    }

    /// The stock fixture frame: one `atoms` block, one `f64` column.
    fn atoms_frame(values: &[f64]) -> Frame {
        let mut frame = Frame::new();
        frame.insert(ATOMS, block_with(X, float_column(values)));
        frame
    }

    /// Column [`X`] of the `atoms` block, as it came back from the store.
    fn atoms_x(frame: &Frame) -> Vec<f64> {
        frame
            .get(ATOMS)
            .expect("the frame carries an atoms block")
            .get(X)
            .expect("the atoms block carries column x")
            .as_float()
            .expect("column x arrived as f64")
            .iter()
            .copied()
            .collect()
    }

    /// The ragged golden run: 3, 5 and 4 rows, every value distinct so a row
    /// range taken from the wrong frame cannot match by accident.
    fn ragged_frames() -> Vec<Frame> {
        vec![
            atoms_frame(&[1.0, 2.0, 3.0]),
            atoms_frame(&[10.0, 11.0, 12.0, 13.0, 14.0]),
            atoms_frame(&[20.0, 21.0, 22.0, 23.0]),
        ]
    }

    /// The cell every box fixture carries — default origin and all-periodic
    /// boundary, so those optional arrays are omitted.
    fn fixed_cell() -> SimBox {
        SimBox::new(
            array![[10.0, 0.0, 0.0], [0.0, 11.0, 0.0], [0.0, 0.0, 12.0]],
            array![0.0, 0.0, 0.0],
            [true, true, true],
        )
        .unwrap()
    }

    /// Mint from the union of `frames`, append them all in order, and close.
    fn write_all(store: &ReadableWritableListableStorage, frames: &[Frame]) {
        let schema = SequenceSchema::from_frames(frames).unwrap();
        let mut writer = FrameSequenceWriter::create(store.clone(), schema).unwrap();
        for frame in frames {
            writer.append(frame).unwrap();
        }
        writer.close().unwrap();
    }

    /// [`write_all`] with both extent knobs pinned, for the tests that need a
    /// tiny `R` to reach a chunk boundary in a handful of frames.
    fn write_all_with(
        store: &ReadableWritableListableStorage,
        frames: &[Frame],
        rows_per_chunk: u64,
        chunks_per_shard: u64,
    ) {
        let schema = SequenceSchema::from_frames(frames).unwrap();
        let mut writer = FrameSequenceWriter::create(store.clone(), schema)
            .unwrap()
            .with_rows_per_chunk(rows_per_chunk)
            .unwrap()
            .with_chunks_per_shard(chunks_per_shard)
            .unwrap();
        for frame in frames {
            writer.append(frame).unwrap();
        }
        writer.close().unwrap();
    }

    /// Drop [`SCHEMA_ATTRIBUTE`] from `trajectory/`'s `zarr.json`, leaving every
    /// array, every array attribute and every byte of data untouched.
    ///
    /// That is exactly the store a conforming foreign writer leaves behind:
    /// molrec's `docs/spec/trajectory.md` makes the attribute the **reference
    /// writer's** own reopen pin, and resolution needs only `step_index` /
    /// `offset` plus the array metadata. The removal is asserted rather than
    /// assumed, so a renamed attribute cannot turn either caller into a test of
    /// an unmodified store.
    fn strip_schema_attribute(store: &ReadableWritableListableStorage) {
        let mut group = Group::open(store.clone(), TRAJ).expect("the sequence group exists");
        assert!(
            group.attributes_mut().remove(SCHEMA_ATTRIBUTE).is_some(),
            "the writer must have pinned {SCHEMA_ATTRIBUTE} for this to be a removal"
        );
        group
            .store_metadata()
            .expect("the group metadata rewrites without the attribute");
    }

    fn open_sequence(store: &ReadableWritableListableStorage) -> FrameSequence {
        FrameSequence::open(store.clone()).unwrap()
    }

    /// `frame(i)`, asserting the frame is present.
    fn frame_at(seq: &mut FrameSequence, index: u64) -> Frame {
        seq.frame(index)
            .expect("reading a frame must not error")
            .expect("frame is present")
    }

    /// The committed frame count, taken through the `TrajectoryReader` door —
    /// the surface `FrameIterator` and every generic consumer already use.
    fn committed_len(seq: &mut FrameSequence) -> usize {
        TrajectoryReader::len(seq).expect("len must not error")
    }

    fn u64_array(store: &ReadableWritableListableStorage, path: &str) -> Vec<u64> {
        let arr = Array::open(store.clone(), path).expect("array exists");
        let subset = ArraySubset::new_with_shape(arr.shape().to_vec());
        arr.retrieve_array_subset(&subset)
            .expect("array reads back")
    }

    fn i64_array(store: &ReadableWritableListableStorage, path: &str) -> Vec<i64> {
        let arr = Array::open(store.clone(), path).expect("array exists");
        let subset = ArraySubset::new_with_shape(arr.shape().to_vec());
        arr.retrieve_array_subset(&subset)
            .expect("array reads back")
    }

    /// The frozen extents of a growth array: `(shard extent, inner chunk
    /// extent)`, both in rows of the leading axis and whatever the trailing
    /// axes are.
    ///
    /// The array's *shape* is deliberately not returned: it grows with every
    /// append, while the two extents are frozen by the first append and must
    /// never move again (`zarrs::Array::set_shape` rebuilds the grid from the
    /// frozen chunk metadata). Under sharding the array's own chunk extent is
    /// the shard, and the planned chunk is the sharding codec's subchunk —
    /// the same reading `frame_io`'s `a_five_chunk_column_shards_and_round_trips`
    /// takes.
    fn extents(store: &ReadableWritableListableStorage, path: &str) -> (Vec<u64>, Vec<u64>) {
        let arr = Array::open(store.clone(), path).expect("array exists");
        let metadata = serde_json::to_value(arr.metadata()).unwrap();
        let shard: Vec<u64> =
            serde_json::from_value(metadata["chunk_grid"]["configuration"]["chunk_shape"].clone())
                .expect("a regular chunk grid carries a chunk_shape");
        let sharding = metadata["codecs"]
            .as_array()
            .expect("codecs is an array")
            .iter()
            .find(|codec| codec["name"] == "sharding_indexed")
            .expect("a growth array is sharded, so file count follows bytes")
            .clone();
        let inner: Vec<u64> =
            serde_json::from_value(sharding["configuration"]["chunk_shape"].clone())
                .expect("the sharding codec carries its subchunk shape");
        (shard, inner)
    }

    /// Every file under `dir`, keyed by its path relative to `dir`.
    fn file_map(dir: &Path) -> BTreeMap<PathBuf, Vec<u8>> {
        let mut out = BTreeMap::new();
        let mut stack = vec![dir.to_path_buf()];
        while let Some(current) = stack.pop() {
            for entry in std::fs::read_dir(&current).unwrap().flatten() {
                let path = entry.path();
                if path.is_dir() {
                    stack.push(path);
                } else {
                    let key = path.strip_prefix(dir).unwrap().to_path_buf();
                    out.insert(key, std::fs::read(&path).unwrap());
                }
            }
        }
        out
    }

    /// The chunk (shard) files under `dir`, without the `zarr.json` metadata —
    /// the bytes a shard actually costs.
    fn chunk_files(dir: &Path) -> BTreeMap<PathBuf, Vec<u8>> {
        file_map(dir)
            .into_iter()
            .filter(|(path, _)| path.file_name() != Some(std::ffi::OsStr::new("zarr.json")))
            .collect()
    }

    fn chunk_bytes(dir: &Path) -> u64 {
        chunk_files(dir).values().map(|v| v.len() as u64).sum()
    }

    /// Deterministic, poorly compressible `f64` payload in `[1, 2)`: a
    /// splitmix hash supplies 52 random mantissa bits, so gzip lands near
    /// 0.85x and every value is finite and bit-exactly comparable. Same shape
    /// as the `zarrs_pins` payload, for the same reason: a compressible
    /// fixture would make a write-amplification bound pass vacuously.
    fn noise(index: u64) -> f64 {
        let mut x = index.wrapping_add(0x9e37_79b9_7f4a_7c15);
        x = (x ^ (x >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        x = (x ^ (x >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        x ^= x >> 31;
        f64::from_bits((0x3ffu64 << 52) | (x & 0x000f_ffff_ffff_ffff))
    }

    // =======================================================================
    // A. Layout primitives — CSR, per-section step_index, derived extents
    // =======================================================================

    /// `offset` is the CSR row pointer of the ragged run and nothing else:
    /// `offset[0] = 0` and the prefix sums of 3, 5 and 4 rows (ac-014).
    ///
    /// The row *counts* are never stored — they are `diff(offset)`, decision 2
    /// — so this array is the only place the raggedness lives.
    #[test]
    fn csr_offset_is_the_prefix_sum_of_a_ragged_run() {
        let dir = TempDir::new().unwrap();
        let store = store_in(&dir);
        write_all(&store, &ragged_frames());

        assert_eq!(
            u64_array(&store, &format!("{TRAJ}/{ATOMS}/offset")),
            vec![0, 3, 8, 12],
            "offset is the prefix sum of 3, 5 and 4 rows"
        );
    }

    /// `step` is the commit marker: one entry per appended frame, in append
    /// order (ac-014).
    #[test]
    fn step_counts_every_appended_frame() {
        let dir = TempDir::new().unwrap();
        let store = store_in(&dir);
        write_all(&store, &ragged_frames());

        assert_eq!(
            i64_array(&store, &format!("{TRAJ}/step")),
            vec![0, 1, 2],
            "three appends commit three steps"
        );
    }

    /// Each frame of the ragged run reads back its own rows, bit exact
    /// (ac-014).
    #[test]
    fn each_frame_of_a_ragged_run_reads_back_its_own_rows() {
        let dir = TempDir::new().unwrap();
        let store = store_in(&dir);
        let frames = ragged_frames();
        write_all(&store, &frames);

        let mut seq = open_sequence(&store);
        for (index, frame) in frames.iter().enumerate() {
            assert_eq!(
                atoms_x(&frame_at(&mut seq, index as u64)),
                atoms_x(frame),
                "frame {index} must come back bit exact"
            );
        }
    }

    /// A block with no `step_index` entry `<= i` is **absent** at step `i`,
    /// not present-and-empty (ac-014).
    ///
    /// Absence is how the layout expresses "this section does not exist yet";
    /// a zero-row block would be a different claim (it exists and holds
    /// nothing), and molrec's conformance suite distinguishes the two.
    #[test]
    fn a_block_absent_at_a_step_reads_as_absent_not_empty() {
        let dir = TempDir::new().unwrap();
        let store = store_in(&dir);
        let mut with_bonds = atoms_frame(&[10.0, 11.0]);
        with_bonds.insert(BONDS, block_with(I, uint_column(&[0, 1])));
        let frames = vec![atoms_frame(&[1.0, 2.0]), with_bonds];
        write_all(&store, &frames);

        let mut seq = open_sequence(&store);
        assert!(
            frame_at(&mut seq, 0).get(BONDS).is_none(),
            "a block whose first step_index entry is 1 does not exist at step 0"
        );
        assert_eq!(
            frame_at(&mut seq, 1)
                .get(BONDS)
                .expect("bonds exists at step 1")
                .nrows(),
            Some(2),
            "and at step 1 it carries the rows it was appended with"
        );
    }

    /// A fixed cell over a 20-step run writes the cell once, and omits the
    /// optional box arrays that would only restate defaults (decision 7).
    #[test]
    fn a_fixed_cell_omits_default_origin_boundary_and_trivial_step_index() {
        const STEPS: usize = 20;

        let dir = TempDir::new().unwrap();
        let store = store_in(&dir);
        let frames: Vec<Frame> = (0..STEPS)
            .map(|step| {
                let mut frame = atoms_frame(&[step as f64, step as f64 + 0.5]);
                frame.simbox = Some(fixed_cell());
                frame
            })
            .collect();
        write_all(&store, &frames);

        assert!(
            Array::open(store.clone(), &format!("{TRAJ}/box/step_index")).is_err(),
            "a single update at ordinal 0 does not write step_index"
        );
        assert!(
            Array::open(store.clone(), &format!("{TRAJ}/box/origin")).is_err(),
            "a zero origin is the default and is not written"
        );
        assert!(
            Array::open(store.clone(), &format!("{TRAJ}/box/boundary")).is_err(),
            "all-periodic boundary is the default and is not written"
        );
        let mut seq = open_sequence(&store);
        let cell = frame_at(&mut seq, 19)
            .simbox
            .expect("later frames still carry the cell");
        assert_eq!(
            cell.origin_view().iter().copied().collect::<Vec<_>>(),
            vec![0.0, 0.0, 0.0]
        );
        assert_eq!(cell.pbc(), [true, true, true]);
    }

    /// A non-default origin is written; the default-omission only applies
    /// when every update is zeros.
    #[test]
    fn a_nonzero_origin_is_written() {
        let dir = TempDir::new().unwrap();
        let store = store_in(&dir);
        let mut frame = atoms_frame(&[1.0]);
        frame.simbox = Some(
            SimBox::new(
                array![[10.0, 0.0, 0.0], [0.0, 11.0, 0.0], [0.0, 0.0, 12.0]],
                array![1.0, 2.0, 3.0],
                [true, true, true],
            )
            .unwrap(),
        );
        write_all(&store, &[frame]);

        let origin: Vec<f64> = {
            let arr = Array::open(store.clone(), &format!("{TRAJ}/box/origin")).unwrap();
            let subset = ArraySubset::new_with_shape(arr.shape().to_vec());
            arr.retrieve_array_subset(&subset).unwrap()
        };
        assert_eq!(origin, vec![1.0, 2.0, 3.0]);
        assert!(
            Array::open(store.clone(), &format!("{TRAJ}/box/boundary")).is_err(),
            "still-default boundary is omitted"
        );
    }

    /// A `step` key on frame.meta does not mint `trajectory/meta/step`.
    #[test]
    fn frame_meta_step_is_not_a_per_step_meta_array() {
        let dir = TempDir::new().unwrap();
        let store = store_in(&dir);
        let mut frame = atoms_frame(&[1.0]);
        frame.meta.insert("step", MetaValue::I64(0));
        write_all(&store, &[frame]);

        assert!(
            Array::open(store.clone(), &format!("{TRAJ}/meta/step")).is_err(),
            "the commit marker is trajectory/step, not meta/step"
        );
        let mut seq = open_sequence(&store);
        assert!(
            frame_at(&mut seq, 0).meta.get("step").is_none(),
            "layout-owned keys are not echoed back as frame meta"
        );
    }

    /// A block whose content never changes writes exactly one `step_index`
    /// entry over a 20-step run (decision 7, ac-014).
    #[test]
    fn a_constant_block_writes_one_step_index_entry() {
        const STEPS: usize = 20;

        let dir = TempDir::new().unwrap();
        let store = store_in(&dir);
        let frames: Vec<Frame> = (0..STEPS)
            .map(|step| {
                let mut frame = atoms_frame(&[step as f64, step as f64 + 0.5]);
                frame.insert(BONDS, block_with(I, uint_column(&[0, 1])));
                frame
            })
            .collect();
        write_all(&store, &frames);

        assert_eq!(
            u64_array(&store, &format!("{TRAJ}/{BONDS}/step_index")),
            vec![0],
            "a constant topology is one entry, however long the run"
        );
    }

    /// A block that changes at every step carries one `step_index` entry per
    /// step, `0..n` (decision 7, ac-014) — the other end of the same rule.
    #[test]
    fn a_block_changing_every_step_indexes_every_step() {
        const STEPS: usize = 20;

        let dir = TempDir::new().unwrap();
        let store = store_in(&dir);
        let frames: Vec<Frame> = (0..STEPS)
            .map(|step| atoms_frame(&[step as f64, step as f64 + 0.5]))
            .collect();
        write_all(&store, &frames);

        assert_eq!(
            u64_array(&store, &format!("{TRAJ}/{ATOMS}/step_index")),
            (0..STEPS as u64).collect::<Vec<u64>>(),
            "moving coordinates are indexed at every step"
        );
    }

    /// A frame whose rows straddle an inner chunk boundary reads back bit
    /// exact (ac-014).
    ///
    /// Frames are **not** rounded up to whole chunks (Design §一: "帧可以跨
    /// chunk"), which is what lets a million-atom frame work at all. With
    /// `R = 4` and three-row frames, frame 1 occupies rows 3..6 and therefore
    /// spans the boundary at row 4.
    #[test]
    fn a_frame_straddling_an_inner_chunk_boundary_reads_back_bit_exact() {
        let dir = TempDir::new().unwrap();
        let store = store_in(&dir);
        let frames = vec![
            atoms_frame(&[1.0, 2.0, 3.0]),
            atoms_frame(&[4.0, 5.0, 6.0]),
            atoms_frame(&[7.0, 8.0, 9.0]),
        ];
        write_all_with(&store, &frames, 4, 2);

        let mut seq = open_sequence(&store);
        assert_eq!(
            atoms_x(&frame_at(&mut seq, 1)),
            vec![4.0, 5.0, 6.0],
            "the frame at rows 3..6 crosses the chunk boundary at row 4"
        );
    }

    /// The worked example of Design §一 lever 1, on disk: a 3000-atom `f64`
    /// xyz block derives `R = 21 845` rows per inner chunk and `k = 512`
    /// chunks per shard (ac-023).
    ///
    /// Hard-coded from the formulas, not from a measurement:
    /// row = 3 * 8 = 24 B; `R = max(1, floor(512 KiB / 24)) = 21 845`;
    /// chunk = 21 845 * 24 = 524 280 B; `k = max(1, floor(256 MiB / 524 280))
    /// = 512`; shard = 21 845 * 512 = 11 184 640 rows (about 3728 frames, so
    /// 10^4 frames are 3 shard files).
    #[test]
    fn the_worked_example_derives_r_21845_and_k_512() {
        const ATOM_COUNT: usize = 3000;
        const ROWS_PER_CHUNK: u64 = 21_845;
        const CHUNKS_PER_SHARD: u64 = 512;
        const ROWS_PER_SHARD: u64 = 11_184_640;

        let dir = TempDir::new().unwrap();
        let store = store_in(&dir);
        let values: Vec<f64> = (0..ATOM_COUNT * 3).map(|i| i as f64 * 0.5).collect();
        let mut frame = Frame::new();
        frame.insert(
            ATOMS,
            block_with(
                "xyz",
                Column::from_float(ArrayD::from_shape_vec(vec![ATOM_COUNT, 3], values).unwrap()),
            ),
        );
        write_all(&store, &[frame]);

        let (shard, inner) = extents(&store, &format!("{TRAJ}/{ATOMS}/xyz"));
        assert_eq!(
            inner,
            vec![ROWS_PER_CHUNK, 3],
            "R = floor(512 KiB / 24 B row) = 21 845 rows, trailing axis whole"
        );
        assert_eq!(
            shard,
            vec![ROWS_PER_SHARD, 3],
            "the shard spans R * k = 11 184 640 rows"
        );
        assert_eq!(
            shard[0] / inner[0],
            CHUNKS_PER_SHARD,
            "k = floor(256 MiB / 524 280 B chunk) = 512"
        );
    }

    // =======================================================================
    // B. Schema — derived union, enforcement at append, reserved names
    // =======================================================================

    /// A union mint over heterogeneous frames writes a store in which every
    /// frame reads back **only** the blocks it presented (ac-020).
    ///
    /// This is how the new layout expresses what the old per-frame group
    /// layout got for free: `from_frames` unions the blocks, and per-section
    /// `step_index` keeps each frame's own set.
    #[test]
    fn a_union_mint_lets_each_frame_read_back_only_its_own_blocks() {
        let dir = TempDir::new().unwrap();
        let store = store_in(&dir);

        let plain = atoms_frame(&[1.0, 2.0]);
        let mut bonded = atoms_frame(&[10.0, 11.0]);
        bonded.insert(BONDS, block_with(I, uint_column(&[0, 1])));
        let mut charged = atoms_frame(&[20.0, 21.0]);
        charged.insert("charges", block_with("q", float_column(&[-0.5, 0.5])));
        let frames = vec![plain, bonded, charged];
        write_all(&store, &frames);

        let mut seq = open_sequence(&store);
        let step0 = frame_at(&mut seq, 0);
        assert_eq!(step0.len(), 1, "step 0 presented only atoms");
        assert_eq!(atoms_x(&step0), vec![1.0, 2.0]);

        let step1 = frame_at(&mut seq, 1);
        assert_eq!(step1.len(), 2, "step 1 presented atoms and bonds");
        assert!(step1.get(BONDS).is_some());
        assert!(step1.get("charges").is_none());

        let step2 = frame_at(&mut seq, 2);
        assert_eq!(step2.len(), 2, "step 2 presented atoms and charges");
        assert!(step2.get("charges").is_some());
        assert!(step2.get(BONDS).is_none());
    }

    /// The same column arriving with two dtypes is a mint-time `Err` naming
    /// the column and both values (ac-020).
    ///
    /// Both values are spelled with the schema dtype tag — the same vocabulary
    /// the store is written in, which for the domain aliases is molrec's
    /// concrete-width spelling (`f64`/`u64`), not [`DType`]'s alias name.
    #[test]
    fn from_frames_errs_naming_the_column_and_both_dtypes() {
        let mut as_float = Frame::new();
        as_float.insert(ATOMS, block_with(PROBE, float_column(&[1.0, 2.0])));
        let mut as_uint = Frame::new();
        as_uint.insert(ATOMS, block_with(PROBE, uint_column(&[1, 2])));
        let frames = vec![as_float, as_uint];

        let message = SequenceSchema::from_frames(&frames)
            .expect_err("a conflicting dtype must be rejected at mint")
            .to_string();
        assert!(message.contains(PROBE), "must name the column: {message}");
        assert!(
            message.contains("f64"),
            "must carry the first dtype (schema tag): {message}"
        );
        assert!(
            message.contains("u64"),
            "must carry the second dtype (schema tag): {message}"
        );
    }

    /// The same column arriving with two trailing shapes is a mint-time `Err`
    /// naming the column and both values (ac-020).
    #[test]
    fn from_frames_errs_naming_the_column_and_both_trailing_shapes() {
        let shaped = |trailing: usize| {
            let mut frame = Frame::new();
            frame.insert(
                ATOMS,
                block_with(
                    "pos",
                    Column::from_float(
                        ArrayD::from_shape_vec(vec![2, trailing], vec![0.5; 2 * trailing]).unwrap(),
                    ),
                ),
            );
            frame
        };
        let frames = vec![shaped(3), shaped(9)];

        let message = SequenceSchema::from_frames(&frames)
            .expect_err("a conflicting trailing shape must be rejected at mint")
            .to_string();
        assert!(message.contains("pos"), "must name the column: {message}");
        assert!(
            message.contains('3'),
            "must carry the first shape: {message}"
        );
        assert!(
            message.contains('9'),
            "must carry the second shape: {message}"
        );
    }

    /// Appending a block the schema never declared is an `Err` naming it
    /// (ac-020).
    #[test]
    fn an_undeclared_block_errs_at_append_naming_it() {
        let dir = TempDir::new().unwrap();
        let store = store_in(&dir);
        let declared = atoms_frame(&[1.0, 2.0]);
        let schema = SequenceSchema::from_frame(&declared).unwrap();
        let mut writer = FrameSequenceWriter::create(store, schema).unwrap();

        let mut extra = atoms_frame(&[3.0, 4.0]);
        extra.insert(BONDS, block_with(I, uint_column(&[0, 1])));
        let message = writer
            .append(&extra)
            .expect_err("an undeclared block must not be written")
            .to_string();
        assert!(message.contains(BONDS), "must name the block: {message}");
    }

    /// Appending a column the schema never declared is an `Err` naming it
    /// (ac-020) — no mid-run schema extension, decision 6.
    #[test]
    fn an_undeclared_column_errs_at_append_naming_it() {
        let dir = TempDir::new().unwrap();
        let store = store_in(&dir);
        let schema = SequenceSchema::from_frame(&atoms_frame(&[1.0, 2.0])).unwrap();
        let mut writer = FrameSequenceWriter::create(store, schema).unwrap();

        let mut wider = Block::new();
        wider.insert_column(X, float_column(&[3.0, 4.0])).unwrap();
        wider
            .insert_column("fx", float_column(&[0.0, 0.0]))
            .unwrap();
        let mut frame = Frame::new();
        frame.insert(ATOMS, wider);
        let message = writer
            .append(&frame)
            .expect_err("an undeclared column must not be written")
            .to_string();
        assert!(message.contains("fx"), "must name the column: {message}");
    }

    /// Appending a declared column at a different dtype is an `Err` naming it
    /// (ac-020).
    #[test]
    fn a_changed_dtype_errs_at_append_naming_the_column() {
        let dir = TempDir::new().unwrap();
        let store = store_in(&dir);
        let mut declared = Frame::new();
        declared.insert(ATOMS, block_with(PROBE, float_column(&[1.0, 2.0])));
        let schema = SequenceSchema::from_frame(&declared).unwrap();
        let mut writer = FrameSequenceWriter::create(store, schema).unwrap();

        let mut frame = Frame::new();
        frame.insert(ATOMS, block_with(PROBE, uint_column(&[3, 4])));
        let message = writer
            .append(&frame)
            .expect_err("a dtype change must not be written")
            .to_string();
        assert!(message.contains(PROBE), "must name the column: {message}");
    }

    /// Appending a declared column at a different trailing shape is an `Err`
    /// naming it (ac-020).
    #[test]
    fn a_changed_trailing_shape_errs_at_append_naming_the_column() {
        let dir = TempDir::new().unwrap();
        let store = store_in(&dir);
        let mut declared = Frame::new();
        declared.insert(
            ATOMS,
            block_with(
                "pos",
                Column::from_float(ArrayD::from_shape_vec(vec![2, 3], vec![0.5; 6]).unwrap()),
            ),
        );
        let schema = SequenceSchema::from_frame(&declared).unwrap();
        let mut writer = FrameSequenceWriter::create(store, schema).unwrap();

        let mut widened = Frame::new();
        widened.insert(
            ATOMS,
            block_with(
                "pos",
                Column::from_float(ArrayD::from_shape_vec(vec![2, 9], vec![0.5; 18]).unwrap()),
            ),
        );
        let message = writer
            .append(&widened)
            .expect_err("a trailing shape change must not be written")
            .to_string();
        assert!(message.contains("pos"), "must name the column: {message}");
    }

    /// A frame presenting a strict subset of the declared union is accepted —
    /// that *is* the sparse expression (ac-020).
    #[test]
    fn a_frame_presenting_a_subset_of_the_declared_blocks_is_accepted() {
        let dir = TempDir::new().unwrap();
        let store = store_in(&dir);
        let mut bonded = atoms_frame(&[10.0, 11.0]);
        bonded.insert(BONDS, block_with(I, uint_column(&[0, 1])));
        let schema = SequenceSchema::from_frames(&[atoms_frame(&[1.0, 2.0]), bonded]).unwrap();
        let mut writer = FrameSequenceWriter::create(store, schema).unwrap();

        writer
            .append(&atoms_frame(&[1.0, 2.0]))
            .expect("a frame carrying only atoms is a legal subset of the union");
        writer.close().unwrap();
    }

    /// `trajectory/`'s reserved child names are refused when the schema is
    /// **minted**, not at the first append (ac-020).
    ///
    /// Minting is the only door: a schema that cannot be minted cannot reach
    /// `create`, so this is also why no test hands `create` a reserved-name
    /// schema — the type system makes that unreachable.
    #[test]
    fn a_reserved_block_name_is_rejected_at_mint() {
        for reserved in ["step", "time", "meta", "box"] {
            let mut frame = Frame::new();
            frame.insert(reserved, block_with(X, float_column(&[1.0])));
            let message = SequenceSchema::from_frame(&frame)
                .expect_err("a reserved block name must be rejected at mint")
                .to_string();
            assert!(
                message.contains(reserved),
                "must name the reserved block {reserved}: {message}"
            );
        }
    }

    /// A block's reserved column names (`offset`, `step_index`) are refused
    /// at mint too (ac-020).
    #[test]
    fn a_reserved_column_name_is_rejected_at_mint() {
        for reserved in ["offset", "step_index"] {
            let mut frame = Frame::new();
            frame.insert(ATOMS, block_with(reserved, uint_column(&[0, 1])));
            let message = SequenceSchema::from_frame(&frame)
                .expect_err("a reserved column name must be rejected at mint")
                .to_string();
            assert!(
                message.contains(reserved),
                "must name the reserved column {reserved}: {message}"
            );
        }
    }

    // =======================================================================
    // C. Per-step meta — typed, exact, never implicitly filled
    // =======================================================================

    /// One value of every [`MetaValue`] variant, at range extremes where the
    /// variant has any: a per-step meta array that silently narrowed a width
    /// would come back changed.
    fn meta_variants() -> Vec<MetaValue> {
        vec![
            MetaValue::Bool(true),
            MetaValue::I32(i32::MIN),
            MetaValue::I64(i64::MIN),
            MetaValue::U32(u32::MAX),
            MetaValue::U64(u64::MAX),
            MetaValue::F32(f32::MIN_POSITIVE),
            MetaValue::F64(f64::MIN_POSITIVE),
            MetaValue::String("gamma-phase".to_string()),
            MetaValue::Bool3([true, false, true]),
            MetaValue::I32x3([i32::MIN, 0, i32::MAX]),
            MetaValue::I64x3([i64::MIN, 0, i64::MAX]),
            MetaValue::U32x3([0, 1, u32::MAX]),
            MetaValue::U64x3([0, 1, u64::MAX]),
            MetaValue::F32x3([1.0, -0.5, f32::MAX]),
            MetaValue::F64x3([1.0, -0.5, f64::MAX]),
            MetaValue::F32x6([1.0, 2.0, 3.0, -4.0, 5.5, 6.25]),
            MetaValue::F64x6([1.0, 2.0, 3.0, -4.0, 5.5, 6.25]),
            MetaValue::F32x9([1.0, 2.0, 3.0, -4.0, 5.5, 6.25, 7.0, -8.0, 9.5]),
            MetaValue::F64x9([1.0, 2.0, 3.0, -4.0, 5.5, 6.25, 7.0, -8.0, 9.5]),
            MetaValue::Json(serde_json::json!({"basis": "def2-TZVP", "scf": [1, 2, 3]})),
        ]
    }

    /// Every `MetaValue` variant round-trips bit exact through
    /// `trajectory/meta/<key>`, and the array carries its `dtype()` tag in
    /// `molrs_meta_dtype` (ac-019).
    ///
    /// The dtype travels with the array rather than with a convention, which
    /// is what makes exactness free (decision 8). Written for two steps so the
    /// array is a real per-step array and not a scalar attribute.
    #[test]
    fn column_dtype_tags_are_molrec_names_and_legacy_tags_still_read() {
        use super::{dtype_from_tag, dtype_tag};
        // The three domain aliases are written under molrec's concrete-width
        // spelling so a schema validates against molrec's dtype enum.
        assert_eq!(dtype_tag(DType::Float), "f64");
        assert_eq!(dtype_tag(DType::Int), "i32");
        assert_eq!(dtype_tag(DType::UInt), "u64");
        assert_eq!(dtype_from_tag("f64").unwrap(), DType::Float);
        assert_eq!(dtype_from_tag("i32").unwrap(), DType::Int);
        assert_eq!(dtype_from_tag("u64").unwrap(), DType::UInt);
        // Stores written by molrs < 0.14 tagged them `float`/`int`/`uint`;
        // those must stay readable forever (e.g. the driving `growth.mrec`).
        assert_eq!(dtype_from_tag("float").unwrap(), DType::Float);
        assert_eq!(dtype_from_tag("int").unwrap(), DType::Int);
        assert_eq!(dtype_from_tag("uint").unwrap(), DType::UInt);
    }

    #[test]
    fn every_meta_variant_round_trips_bit_exact_with_its_dtype_tag() {
        const KEY: &str = "probe";

        for value in meta_variants() {
            let dir = TempDir::new().unwrap();
            let store = store_in(&dir);
            let frames: Vec<Frame> = [1.0f64, 2.0]
                .iter()
                .map(|v| {
                    let mut frame = atoms_frame(&[*v]);
                    frame.meta.insert(KEY, value.clone());
                    frame
                })
                .collect();
            write_all(&store, &frames);

            let array = Array::open(store.clone(), &format!("{TRAJ}/meta/{KEY}"))
                .expect("a declared meta key is a per-step array");
            assert_eq!(
                array.attributes().get("molrs_meta_dtype"),
                Some(&serde_json::json!(value.dtype())),
                "the array must carry the {} tag",
                value.dtype()
            );

            let mut seq = open_sequence(&store);
            for step in 0..2u64 {
                assert_eq!(
                    frame_at(&mut seq, step).meta.get(KEY),
                    Some(&value),
                    "{} must survive step {step} bit exact",
                    value.dtype()
                );
            }
        }
    }

    /// Omitting a declared meta key that has no declared fill is an `Err`
    /// naming the key — there is no implicit NaN (ac-019).
    #[test]
    fn omitting_a_declared_meta_key_without_a_fill_errs_naming_it() {
        const KEY: &str = "temperature";

        let dir = TempDir::new().unwrap();
        let store = store_in(&dir);
        let mut declaring = atoms_frame(&[1.0]);
        declaring.meta.insert(KEY, MetaValue::F64(300.0));
        let schema = SequenceSchema::from_frame(&declaring).unwrap();
        let mut writer = FrameSequenceWriter::create(store, schema).unwrap();
        writer.append(&declaring).unwrap();

        let message = writer
            .append(&atoms_frame(&[2.0]))
            .expect_err("a declared meta key with no fill may not be silently filled")
            .to_string();
        assert!(message.contains(KEY), "must name the key: {message}");
    }

    /// A meta key declared with an explicit fill writes that fill for the
    /// steps that omit it, and leaves the steps that carry a value alone
    /// (ac-019).
    #[test]
    fn a_declared_fill_is_written_for_an_omitted_meta_key() {
        const KEY: &str = "temperature";
        const FILL: f64 = -1.0;

        let dir = TempDir::new().unwrap();
        let store = store_in(&dir);
        let mut declaring = atoms_frame(&[1.0]);
        declaring.meta.insert(KEY, MetaValue::F64(300.0));
        let mut schema = SequenceSchema::from_frame(&declaring).unwrap();
        schema.declare_meta(KEY, MetaValue::F64(FILL)).unwrap();

        let mut writer = FrameSequenceWriter::create(store.clone(), schema).unwrap();
        writer.append(&declaring).unwrap();
        writer.append(&atoms_frame(&[2.0])).unwrap();
        writer.close().unwrap();

        let mut seq = open_sequence(&store);
        assert_eq!(
            frame_at(&mut seq, 1).meta.get(KEY),
            Some(&MetaValue::F64(FILL)),
            "the omitting step reads the declared fill, not NaN"
        );
        assert_eq!(
            frame_at(&mut seq, 0).meta.get(KEY),
            Some(&MetaValue::F64(300.0)),
            "and the step that carried a value keeps it"
        );
    }

    // =======================================================================
    // D. Lifecycle — create, open, commit, reopen, the knob window
    // =======================================================================

    /// `create` on a store that already holds a sequence is an `Err` naming
    /// the path, and the existing store is left byte-unchanged (ac-018).
    ///
    /// Never a silent overwrite: the store on that path is somebody's run.
    #[test]
    fn create_on_an_occupied_store_errs_and_changes_nothing() {
        let dir = TempDir::new().unwrap();
        let store = store_in(&dir);
        let frames = ragged_frames();
        write_all(&store, &frames);
        let before = file_map(dir.path());

        let schema = SequenceSchema::from_frames(&frames).unwrap();
        let message = FrameSequenceWriter::create(store, schema)
            .err()
            .expect("create must refuse an occupied store")
            .to_string();

        assert!(
            message.contains("trajectory"),
            "must name the occupied path: {message}"
        );
        assert_eq!(
            file_map(dir.path()),
            before,
            "a refused create must not have touched a byte"
        );
    }

    /// `open` refuses a store whose arrays disagree with the schema it minted,
    /// naming what differed and both values (ac-018).
    ///
    /// The store is corrupted the way a foreign writer would corrupt it — the
    /// declared column `x` is replaced by an `i64` array — because the schema
    /// pinned at `create` is exactly what `open` has to check the arrays
    /// against.
    #[test]
    fn open_errs_naming_what_differed_and_both_values() {
        let dir = TempDir::new().unwrap();
        let store = store_in(&dir);
        write_all(&store, &ragged_frames());

        let column = format!("{TRAJ}/{ATOMS}/{X}");
        store
            .erase_prefix(&StorePrefix::new(format!("trajectory/{ATOMS}/{X}/")).unwrap())
            .unwrap();
        ArrayBuilder::new(vec![12], vec![12], data_type::int64(), 0i64)
            .build(store.clone(), &column)
            .unwrap()
            .store_metadata()
            .unwrap();

        let message = FrameSequenceWriter::open(store)
            .err()
            .expect("open must refuse a store that no longer matches its schema")
            .to_string();
        assert!(
            message.contains("sequence schema mismatch"),
            "the pinned message shape: {message}"
        );
        assert!(
            message.contains(ATOMS),
            "must name what differed: {message}"
        );
        assert!(
            message.contains("expected") && message.contains("found"),
            "must carry both values: {message}"
        );
        assert!(
            message.contains("f64"),
            "must carry the declared dtype (schema tag): {message}"
        );
        assert!(
            message.contains(&DType::Int64.to_string()),
            "must carry the dtype found on disk: {message}"
        );
    }

    /// A store that conforms to the layout but carries no [`SCHEMA_ATTRIBUTE`]
    /// still **reads**: the read door derives the schema from the store.
    ///
    /// The attribute is the reference writer's own reopen pin
    /// (molrec `docs/spec/trajectory.md`), not a read requirement — resolving a
    /// frame needs `step_index`, `offset` and the array metadata, all of which
    /// are on disk regardless. Refusing a conforming store for a missing
    /// convenience pin locks every foreign writer out of the format.
    ///
    /// The fixture is written by the real writer and then *stripped*, so the
    /// only difference from a store this suite already reads is the one
    /// attribute. It carries a ragged block column **and** a per-step meta key,
    /// because the derivation has two halves — blocks/columns from the group
    /// tree and array metadata, meta keys from `meta/`'s children and their
    /// `molrs_meta_dtype` tags — and one half working is not the claim.
    #[test]
    fn a_conforming_store_without_the_schema_attribute_still_reads() {
        const TEMPERATURE: &str = "temperature";

        let dir = TempDir::new().unwrap();
        let store = store_in(&dir);
        let frames: Vec<Frame> = ragged_frames()
            .into_iter()
            .enumerate()
            .map(|(step, mut frame)| {
                frame
                    .meta
                    .insert(TEMPERATURE, MetaValue::F64(300.0 + step as f64));
                frame
            })
            .collect();
        write_all(&store, &frames);
        strip_schema_attribute(&store);

        let mut seq = FrameSequence::open(store.clone())
            .expect("a conforming store reads without the writer's schema pin");
        assert_eq!(
            committed_len(&mut seq),
            frames.len(),
            "the derived sequence is as long as the run that was written"
        );
        for (index, frame) in frames.iter().enumerate() {
            let read = frame_at(&mut seq, index as u64);
            assert_eq!(
                atoms_x(&read),
                atoms_x(frame),
                "frame {index} must come back bit exact from a derived schema"
            );
            assert_eq!(
                read.meta.get(TEMPERATURE),
                frame.meta.get(TEMPERATURE),
                "the per-step meta key of frame {index} must survive the derivation"
            );
        }
    }

    /// The strict half: [`FrameSequenceWriter::open`] on that same stripped
    /// store still errs, naming [`SCHEMA_ATTRIBUTE`].
    ///
    /// A writer needs the *pinned* declaration — the fills for omitted meta keys
    /// and the union a later frame is checked against — and neither can be
    /// derived from data that was never written. Pinned next to the read door so
    /// relaxing the reader cannot quietly relax the writer with it.
    #[test]
    fn a_writer_reopen_still_requires_the_schema_attribute() {
        let dir = TempDir::new().unwrap();
        let store = store_in(&dir);
        write_all(&store, &ragged_frames());
        strip_schema_attribute(&store);

        let message = FrameSequenceWriter::open(store)
            .err()
            .expect("a writer may not reopen a sequence whose schema pin is gone")
            .to_string();
        assert!(
            message.contains(SCHEMA_ATTRIBUTE),
            "must name the missing pin: {message}"
        );
    }

    /// `flush()` is the commit point, and the crash-loss boundary is exactly
    /// "everything appended after it" — branch A (ac-015).
    ///
    /// `std::mem::forget` is the crash: the writer never runs another line, so
    /// what the reopened store shows is what a killed process would have left.
    /// It is also why the writer must have no `Drop` — a `Drop` that flushed
    /// would make this test's boundary unobservable, and would swallow IO
    /// errors in the bargain.
    #[test]
    fn a_flush_commits_every_frame_appended_before_it() {
        let dir = TempDir::new().unwrap();
        let store = store_in(&dir);
        let frames: Vec<Frame> = (0..5)
            .map(|step| atoms_frame(&[step as f64, step as f64 + 0.5]))
            .collect();

        let schema = SequenceSchema::from_frames(&frames).unwrap();
        let mut writer = FrameSequenceWriter::create(store.clone(), schema).unwrap();
        for frame in &frames[..3] {
            writer.append(frame).unwrap();
        }
        writer.flush().unwrap();
        for frame in &frames[3..] {
            writer.append(frame).unwrap();
        }
        std::mem::forget(writer);

        let mut seq = open_sequence(&store);
        assert_eq!(
            committed_len(&mut seq),
            3,
            "branch A commits every buffered row, ragged tail included"
        );
        for (index, frame) in frames[..3].iter().enumerate() {
            assert_eq!(
                atoms_x(&frame_at(&mut seq, index as u64)),
                atoms_x(frame),
                "committed frame {index} must read bit exact"
            );
        }
        assert!(
            seq.frame(3).unwrap().is_none(),
            "frames appended after the last flush are absent, not partial"
        );
    }

    /// `close(self)` then `open` then appends that carry the run past an inner
    /// chunk boundary: every frame reads bit exact and the extents are the
    /// ones creation froze (ac-016).
    #[test]
    fn a_reopened_writer_appends_across_an_inner_chunk_boundary() {
        let dir = TempDir::new().unwrap();
        let store = store_in(&dir);
        let frames: Vec<Frame> = (0..4)
            .map(|step| atoms_frame(&[step as f64, step as f64 + 0.25, step as f64 + 0.5]))
            .collect();

        let schema = SequenceSchema::from_frames(&frames).unwrap();
        let mut writer = FrameSequenceWriter::create(store.clone(), schema)
            .unwrap()
            .with_rows_per_chunk(4)
            .unwrap()
            .with_chunks_per_shard(2)
            .unwrap();
        for frame in &frames[..2] {
            writer.append(frame).unwrap();
        }
        writer.close().unwrap();
        let column = format!("{TRAJ}/{ATOMS}/{X}");
        let frozen = extents(&store, &column);

        let mut writer = FrameSequenceWriter::open(store.clone()).unwrap();
        for frame in &frames[2..] {
            writer.append(frame).unwrap();
        }
        writer.close().unwrap();

        let mut seq = open_sequence(&store);
        assert_eq!(committed_len(&mut seq), 4);
        for (index, frame) in frames.iter().enumerate() {
            assert_eq!(
                atoms_x(&frame_at(&mut seq, index as u64)),
                atoms_x(frame),
                "frame {index} must survive the reopen bit exact"
            );
        }
        assert_eq!(
            extents(&store, &column),
            frozen,
            "a reopened writer may not re-plan the extents"
        );
    }

    /// Both knobs are legal after `create` and before the first append, and
    /// the first append freezes exactly what they asked for (decision 13).
    ///
    /// This is the knobs' whole reason to exist: `with_rows_per_chunk`'s
    /// consumer is the lifecycle tests (a boundary in a handful of frames
    /// instead of 512 KiB of them) and `with_chunks_per_shard`'s is the file
    /// count bound.
    #[test]
    fn the_knobs_set_the_extents_the_first_append_freezes() {
        const ROWS_PER_CHUNK: u64 = 4;
        const CHUNKS_PER_SHARD: u64 = 2;

        let dir = TempDir::new().unwrap();
        let store = store_in(&dir);
        write_all_with(
            &store,
            &[atoms_frame(&[1.0, 2.0, 3.0])],
            ROWS_PER_CHUNK,
            CHUNKS_PER_SHARD,
        );

        assert_eq!(
            extents(&store, &format!("{TRAJ}/{ATOMS}/{X}")),
            (
                vec![ROWS_PER_CHUNK * CHUNKS_PER_SHARD],
                vec![ROWS_PER_CHUNK]
            ),
            "the knobs decide the frozen extents, not the byte target"
        );
    }

    /// A knob after the first append is an `Err` naming the phase: the extents
    /// are frozen and `zarrs` cannot re-plan a grid under live data
    /// (decision 3, decision 13).
    #[test]
    fn a_knob_after_the_first_append_errs_naming_the_phase() {
        let dir = TempDir::new().unwrap();
        let store = store_in(&dir);
        let schema = SequenceSchema::from_frame(&atoms_frame(&[1.0, 2.0])).unwrap();
        let mut writer = FrameSequenceWriter::create(store, schema).unwrap();
        writer.append(&atoms_frame(&[1.0, 2.0])).unwrap();

        let message = writer
            .with_rows_per_chunk(8)
            .err()
            .expect("the knob window closes at the first append")
            .to_string();
        assert!(
            message.contains("append"),
            "must name the phase that closed the window: {message}"
        );
    }

    /// A store created and closed without a single append reopens as an empty
    /// sequence (the knob window's consequence: the arrays are not created
    /// until the first append, so such a store may hold only the group and its
    /// schema attributes).
    #[test]
    fn a_created_but_never_appended_store_reopens_empty() {
        let dir = TempDir::new().unwrap();
        let store = store_in(&dir);
        let schema = SequenceSchema::from_frame(&atoms_frame(&[1.0])).unwrap();
        FrameSequenceWriter::create(store.clone(), schema)
            .unwrap()
            .close()
            .unwrap();

        let mut seq = open_sequence(&store);
        assert_eq!(committed_len(&mut seq), 0, "no append, no committed frame");
        assert!(seq.frame(0).unwrap().is_none());
    }

    // =======================================================================
    // E. Ordering — step is extended last, and that is observable
    // =======================================================================

    /// Rolling `step` back to its pre-flush length hides the uncommitted
    /// frames and errors at nothing (ac-017).
    ///
    /// This is the crash window made deterministic: `step` is extended last,
    /// so a crash anywhere before that leaves the data arrays long and `step`
    /// short — exactly the state this test builds through the store API.
    #[test]
    fn a_step_array_rolled_back_hides_the_uncommitted_frames() {
        let dir = TempDir::new().unwrap();
        let store = store_in(&dir);
        let frames: Vec<Frame> = (0..5)
            .map(|step| atoms_frame(&[step as f64, step as f64 + 0.5]))
            .collect();
        write_all(&store, &frames);

        let mut step = Array::open(store.clone(), &format!("{TRAJ}/step")).unwrap();
        step.set_shape(vec![3]).unwrap();
        step.store_metadata().unwrap();

        let mut seq = open_sequence(&store);
        assert_eq!(
            committed_len(&mut seq),
            3,
            "len follows the commit marker, not the data arrays"
        );
        for (index, frame) in frames[..3].iter().enumerate() {
            assert_eq!(
                atoms_x(&frame_at(&mut seq, index as u64)),
                atoms_x(frame),
                "committed frame {index} still reads"
            );
        }
        assert!(
            seq.frame(3).unwrap().is_none(),
            "the uncommitted frames are invisible, and asking for one is not an error"
        );
    }

    /// A `FilesystemStore` that records the order of the keys written through
    /// it. Reads and listings are the wrapped store's; only the write order is
    /// this type's business.
    ///
    /// It records `set` and `set_partial_many` — the two doors a value reaches
    /// the disk through — and deliberately not `erase`, which is not a write
    /// of content.
    #[derive(Debug)]
    struct RecordingStore {
        inner: FilesystemStore,
        written: Mutex<Vec<String>>,
    }

    impl RecordingStore {
        fn new(path: &Path) -> Self {
            Self {
                inner: FilesystemStore::new(path).unwrap(),
                written: Mutex::new(Vec::new()),
            }
        }

        fn record(&self, key: &StoreKey) {
            self.written.lock().unwrap().push(key.as_str().to_string());
        }

        fn written_keys(&self) -> Vec<String> {
            self.written.lock().unwrap().clone()
        }

        fn clear(&self) {
            self.written.lock().unwrap().clear();
        }
    }

    impl ReadableStorageTraits for RecordingStore {
        fn get(&self, key: &StoreKey) -> Result<MaybeBytes, StorageError> {
            self.inner.get(key)
        }

        fn get_partial(
            &self,
            key: &StoreKey,
            byte_range: ByteRange,
        ) -> Result<MaybeBytes, StorageError> {
            self.inner.get_partial(key, byte_range)
        }

        fn get_partial_many<'a>(
            &'a self,
            key: &StoreKey,
            byte_ranges: ByteRangeIterator<'a>,
        ) -> Result<MaybeBytesIterator<'a>, StorageError> {
            self.inner.get_partial_many(key, byte_ranges)
        }

        fn size_key(&self, key: &StoreKey) -> Result<Option<u64>, StorageError> {
            self.inner.size_key(key)
        }

        fn supports_get_partial(&self) -> bool {
            self.inner.supports_get_partial()
        }
    }

    impl ListableStorageTraits for RecordingStore {
        fn list(&self) -> Result<StoreKeys, StorageError> {
            self.inner.list()
        }

        fn list_prefix(&self, prefix: &StorePrefix) -> Result<StoreKeys, StorageError> {
            self.inner.list_prefix(prefix)
        }

        fn list_dir(&self, prefix: &StorePrefix) -> Result<StoreKeysPrefixes, StorageError> {
            self.inner.list_dir(prefix)
        }

        fn size(&self) -> Result<u64, StorageError> {
            self.inner.size()
        }

        fn size_prefix(&self, prefix: &StorePrefix) -> Result<u64, StorageError> {
            self.inner.size_prefix(prefix)
        }
    }

    impl WritableStorageTraits for RecordingStore {
        fn set(&self, key: &StoreKey, value: Bytes) -> Result<(), StorageError> {
            self.record(key);
            self.inner.set(key, value)
        }

        fn set_partial_many(
            &self,
            key: &StoreKey,
            offset_values: OffsetBytesIterator,
        ) -> Result<(), StorageError> {
            self.record(key);
            self.inner.set_partial_many(key, offset_values)
        }

        fn erase(&self, key: &StoreKey) -> Result<(), StorageError> {
            self.inner.erase(key)
        }

        fn erase_prefix(&self, prefix: &StorePrefix) -> Result<(), StorageError> {
            self.inner.erase_prefix(prefix)
        }

        fn supports_set_partial(&self) -> bool {
            self.inner.supports_set_partial()
        }
    }

    /// The write order is a property, not a comment: within one `flush()`
    /// every key under `trajectory/step/` is written **after** every other key
    /// (ac-017).
    ///
    /// That ordering is what makes a crash between two writes invisible — the
    /// commit marker is the last thing to move — and it is unobservable from
    /// the outside, so the observation is a recording store.
    #[test]
    fn step_is_written_after_every_other_key() {
        let dir = TempDir::new().unwrap();
        let recorder = Arc::new(RecordingStore::new(dir.path()));
        let store: ReadableWritableListableStorage = recorder.clone();

        let frame = atoms_frame(&[1.0, 2.0]);
        let schema = SequenceSchema::from_frame(&frame).unwrap();
        let mut writer = FrameSequenceWriter::create(store, schema).unwrap();
        writer.append(&frame).unwrap();
        recorder.clear();
        writer.flush().unwrap();

        let log = recorder.written_keys();
        let first_step = log
            .iter()
            .position(|key| key.starts_with("trajectory/step/"))
            .unwrap_or_else(|| panic!("a flush must extend step; wrote {log:?}"));
        assert!(
            log[..first_step]
                .iter()
                .any(|key| key.starts_with(&format!("trajectory/{ATOMS}/"))),
            "the data arrays must be written before step: {log:?}"
        );
        assert!(
            log[first_step..]
                .iter()
                .all(|key| key.starts_with("trajectory/step/")),
            "nothing may be written after the commit marker: {log:?}"
        );
    }

    // =======================================================================
    // F. Seal on complete — a closed store carries no dead bytes
    // =======================================================================

    /// After `close(self)` the shard files are the same bytes whatever the
    /// flush cadence was, and before the seal the frequently flushed store is
    /// measurably bigger (ac-034).
    ///
    /// Branch A pays for arbitrary commit granularity with dead bytes: every
    /// rewrite of the trailing inner chunk leaves the superseded copy in the
    /// shard file. The seal is what closes that gap, and the mid-run
    /// assertion is what proves the comparison is not vacuous — without it,
    /// two stores that never accumulated dead bytes in the first place would
    /// pass.
    #[test]
    fn flush_cadence_does_not_change_the_closed_shard_bytes() {
        const ROWS_PER_CHUNK: u64 = 64;
        const CHUNKS_PER_SHARD: u64 = 8;
        const FRAMES: usize = 24;

        // 24 frames of 5 rows = 120 rows: two inner chunks inside one shard,
        // so the seal under test is close()'s compaction of the final partial
        // shard, with no boundary crossing in the way.
        let frames: Vec<Frame> = (0..FRAMES)
            .map(|step| {
                let base = step as f64 * 10.0;
                atoms_frame(&[base, base + 1.0, base + 2.0, base + 3.0, base + 4.0])
            })
            .collect();
        let schema = SequenceSchema::from_frames(&frames).unwrap();

        let eager_dir = TempDir::new().unwrap();
        let eager_store = store_in(&eager_dir);
        let mut eager = FrameSequenceWriter::create(eager_store, schema)
            .unwrap()
            .with_rows_per_chunk(ROWS_PER_CHUNK)
            .unwrap()
            .with_chunks_per_shard(CHUNKS_PER_SHARD)
            .unwrap();
        for frame in &frames {
            eager.append(frame).unwrap();
            eager.flush().unwrap();
        }
        let column_dir = PathBuf::from("trajectory").join(ATOMS).join(X);
        let active = chunk_bytes(&eager_dir.path().join(&column_dir));

        let lazy_dir = TempDir::new().unwrap();
        let lazy_store = store_in(&lazy_dir);
        let mut lazy =
            FrameSequenceWriter::create(lazy_store, SequenceSchema::from_frames(&frames).unwrap())
                .unwrap()
                .with_rows_per_chunk(ROWS_PER_CHUNK)
                .unwrap()
                .with_chunks_per_shard(CHUNKS_PER_SHARD)
                .unwrap();
        for frame in &frames {
            lazy.append(frame).unwrap();
        }
        lazy.flush().unwrap();
        lazy.close().unwrap();
        let sealed = chunk_bytes(&lazy_dir.path().join(&column_dir));

        assert!(
            active > sealed,
            "before the seal the active shard must carry the superseded copies \
             it is about to drop: {active} B vs {sealed} B"
        );

        eager.close().unwrap();
        assert_eq!(
            chunk_files(&eager_dir.path().join(&column_dir)),
            chunk_files(&lazy_dir.path().join(&column_dir)),
            "a closed store's shard files must not remember how often it flushed"
        );
    }

    // =======================================================================
    // G. Write amplification — the store half of the append fast path
    // =======================================================================

    /// A steady-state flush writes about one compressed chunk plus one shard
    /// index **at the disk**, and does not grow with the shard file it lands
    /// in (ac-011, store half).
    ///
    /// This is the only thing standing between a 256 MiB shard default and a
    /// silent O(shard) rewrite per flush: `zarrs_filesystem` 0.3.12 reports
    /// `supports_set_partial() == true` while reading, patching and rewriting
    /// the whole value, so the codec-layer measurement in `zarrs_pins` is
    /// structurally blind to it. [`PositionalWriteStore::bytes_written`] is
    /// the observation that is not.
    #[test]
    fn a_steady_state_flush_writes_a_chunk_not_the_shard_file() {
        /// Rows per frame, and rows per inner chunk: one frame is one chunk,
        /// so one flush is one chunk write.
        const ROWS: u64 = 4096;
        /// 4096 f64 rows = 32 768 B uncompressed.
        const CHUNK_BYTES: u64 = ROWS * 8;
        /// A shard extent of 256 chunks = 1 048 576 rows = 8 MiB, so all 64
        /// flushes land in shard 0 and no seal-on-complete rewrite is in play.
        const CHUNKS_PER_SHARD: u64 = 256;
        /// Two `u64` per inner chunk plus the `crc32c` checksum.
        const INDEX_BYTES: u64 = CHUNKS_PER_SHARD * 16 + 4;
        /// The dense per-step arrays (`step`, the block's `offset` and
        /// `step_index`) and the metadata each flush restates. Their cost is
        /// bounded by the number of arrays, not by the run length; 64 KiB is
        /// a hard ceiling on it, three orders below the shard file the naive
        /// path would have rewritten.
        const AUXILIARY_BYTES: u64 = 64 * 1024;
        const FLUSHES: u64 = 64;

        let dir = TempDir::new().unwrap();
        let positional = Arc::new(PositionalWriteStore::new(dir.path()).unwrap());
        let store: ReadableWritableListableStorage = positional.clone();

        let frame_of = |step: u64| {
            atoms_frame(
                &(0..ROWS)
                    .map(|row| noise(step * ROWS + row))
                    .collect::<Vec<f64>>(),
            )
        };
        let schema = SequenceSchema::from_frame(&frame_of(0)).unwrap();
        let mut writer = FrameSequenceWriter::create(store, schema)
            .unwrap()
            .with_rows_per_chunk(ROWS)
            .unwrap()
            .with_chunks_per_shard(CHUNKS_PER_SHARD)
            .unwrap();

        let mut early = 0u64;
        let mut late = 0u64;
        for step in 0..FLUSHES {
            let before = positional.bytes_written();
            writer.append(&frame_of(step)).unwrap();
            writer.flush().unwrap();
            let delta = positional.bytes_written() - before;
            if step == 3 {
                early = delta;
            }
            if step == FLUSHES - 1 {
                late = delta;
            }
        }
        let file = chunk_bytes(&dir.path().join("trajectory").join(ATOMS).join(X));
        writer.close().unwrap();

        assert!(
            late <= CHUNK_BYTES + INDEX_BYTES + AUXILIARY_BYTES,
            "a flush must cost one chunk plus one shard index, not {late} B"
        );
        assert!(
            late <= 2 * early,
            "the per-flush cost must not grow with the file: {early} B at flush 4, \
             {late} B at flush {FLUSHES}"
        );
        assert!(
            file > 16 * late,
            "the measurement is only meaningful while the file dwarfs the write: \
             {file} B file, {late} B flush"
        );
    }

    // =======================================================================
    // H. File count — bounded by bytes, not by frames
    // =======================================================================

    /// A store's file count follows its **bytes**, not its frame count
    /// (ac-023, decision 13).
    ///
    /// Every number below is hard-coded from the layout rather than measured.
    /// [`FrameSequenceWriter::with_chunks_per_shard`] exists for this test: a
    /// 256 MiB shard would swallow the whole fixture, so `k` is pressed down
    /// to 4 and the store really does span several shards. With `R = 8` rows
    /// per inner chunk, one shard spans `R * k = 32` rows, and every array
    /// here is 8 B per row (`f64` x; `i64` step; `u64` offset and
    /// `step_index`), so one shard file holds `S = 256` B.
    ///
    /// Per growth array the file count is `ceil(rows / 32)` shard files plus
    /// one `zarr.json`; the two groups (`trajectory/` and `trajectory/atoms/`)
    /// add one `zarr.json` each. Eight 8-row frames are 64 rows of `x` — two
    /// full shards — while `step` and `step_index` are 8 rows and `offset` is
    /// 9 (the opening zero of the CSR pointer), one shard each: 11 files.
    /// Sixteen frames take `x` to 128 rows, four shards, and leave the other
    /// three arrays inside their first shard: 13 files. Doubling the frames
    /// adds two files.
    ///
    /// The counted set is **every** file under the store root, `zarr.json`
    /// included — the claim is about what lands on a filesystem, so excluding
    /// the metadata would be excluding the part that scales with arrays.
    #[test]
    fn file_count_scales_with_bytes_not_frames() {
        /// Rows one fixture frame carries, and rows one inner chunk holds:
        /// one frame is exactly one chunk of the `x` column.
        const ROWS_PER_FRAME: u64 = 8;
        const ROWS_PER_CHUNK: u64 = 8;
        /// Pressed far below the derived default so a small store spans
        /// several shards.
        const CHUNKS_PER_SHARD: u64 = 4;
        /// Bytes one row of every array in this store occupies.
        const ROW_BYTES: u64 = 8;
        /// `S`: bytes one shard file holds, `R * k * row`.
        const SHARD_BYTES: u64 = ROWS_PER_CHUNK * CHUNKS_PER_SHARD * ROW_BYTES;
        /// The growth arrays: `step`, and the block's `x`, `offset` and
        /// `step_index`. No `time`, no `meta`, no `box` — the fixture frames
        /// carry none.
        const ARRAYS: u64 = 4;
        /// `trajectory/` and `trajectory/atoms/`.
        const GROUPS: u64 = 2;
        const FRAMES: u64 = 8;

        /// `O(arrays)` of the bound: each array costs its own `zarr.json` and
        /// rounds its last shard up, each group costs a `zarr.json`, and
        /// unsharded index arrays may add a chunk file per inner chunk.
        const ARRAY_FLOOR: u64 = 2 * ARRAYS + GROUPS + 4;
        /// Rows across the four arrays at `FRAMES`: 8 * 8 of `x`, 8 of `step`,
        /// 8 of `step_index`, 8 + 1 of `offset`.
        const SINGLE_ROWS: u64 = 64 + 8 + 8 + 9;
        /// The same at `2 * FRAMES`.
        const DOUBLE_ROWS: u64 = 128 + 16 + 16 + 17;
        /// 2 group `zarr.json` + 4 array `zarr.json` + 2 `x` shards + 2 offset
        /// chunks (unsharded) + 1 step_index + 1 step.
        const SINGLE_FILES: usize = 12;
        /// `x` on four shards; unsharded index arrays pick up extra chunks.
        const DOUBLE_FILES: usize = 17;

        // Every frame carries distinct values, so every step earns its own
        // `step_index` entry: a repeated block would be stored once and the
        // row counts above would not be the ones on disk.
        let store_files = |count: u64| -> Vec<PathBuf> {
            let dir = TempDir::new().unwrap();
            let store = store_in(&dir);
            let frames: Vec<Frame> = (0..count)
                .map(|step| {
                    atoms_frame(
                        &(0..ROWS_PER_FRAME)
                            .map(|row| noise(step * ROWS_PER_FRAME + row))
                            .collect::<Vec<f64>>(),
                    )
                })
                .collect();
            write_all_with(&store, &frames, ROWS_PER_CHUNK, CHUNKS_PER_SHARD);
            file_map(dir.path()).into_keys().collect()
        };

        let single = store_files(FRAMES);
        let double = store_files(2 * FRAMES);

        assert_eq!(
            single.len(),
            SINGLE_FILES,
            "{FRAMES} frames must cost {SINGLE_FILES} files: {single:?}"
        );
        assert_eq!(
            double.len(),
            DOUBLE_FILES,
            "{} frames must cost {DOUBLE_FILES} files: {double:?}",
            2 * FRAMES
        );

        let bound = |rows: u64| ((rows * ROW_BYTES) / SHARD_BYTES + ARRAY_FLOOR) as usize;
        assert!(
            single.len() <= bound(SINGLE_ROWS),
            "file count must stay under total_bytes/S + O(arrays): {} > {}",
            single.len(),
            bound(SINGLE_ROWS)
        );
        assert!(
            double.len() <= bound(DOUBLE_ROWS),
            "file count must stay under total_bytes/S + O(arrays): {} > {}",
            double.len(),
            bound(DOUBLE_ROWS)
        );
        assert!(
            double.len() < 2 * single.len(),
            "twice the frames must not be twice the files: {} then {}",
            single.len(),
            double.len()
        );
    }

    // =======================================================================
    // I. Reader — one cursor, two access shapes
    // =======================================================================

    /// `FrameIterator` over a `FrameSequence` yields exactly what a `frame(i)`
    /// loop yields (ac-014).
    ///
    /// This is the payoff of dropping the `Reader` supertrait from
    /// `TrajectoryReader`: the existing generic consumers work against a
    /// store-backed sequence with no `BufRead` anywhere.
    #[test]
    fn the_frame_iterator_yields_what_frame_yields() {
        let dir = TempDir::new().unwrap();
        let store = store_in(&dir);
        let frames = ragged_frames();
        write_all(&store, &frames);

        let mut iterated = open_sequence(&store);
        let streamed: Vec<Vec<f64>> = iterated
            .iter()
            .map(|frame| atoms_x(&frame.unwrap()))
            .collect();

        let mut indexed = open_sequence(&store);
        let random: Vec<Vec<f64>> = (0..frames.len() as u64)
            .map(|index| atoms_x(&frame_at(&mut indexed, index)))
            .collect();

        assert_eq!(streamed, random, "the iterator is the frame(i) loop");
        assert_eq!(
            streamed,
            frames.iter().map(atoms_x).collect::<Vec<Vec<f64>>>(),
            "and both are what was appended"
        );
    }

    /// `to_trajectory()` is the named lazy-to-eager conversion: the same
    /// frames, materialized.
    #[test]
    fn to_trajectory_carries_the_same_frames_as_frame() {
        let dir = TempDir::new().unwrap();
        let store = store_in(&dir);
        let frames = ragged_frames();
        write_all(&store, &frames);

        let mut seq = open_sequence(&store);
        let eager = seq.to_trajectory().unwrap();

        assert_eq!(eager.frames.len(), frames.len());
        assert_eq!(
            eager.frames.iter().map(atoms_x).collect::<Vec<Vec<f64>>>(),
            frames.iter().map(atoms_x).collect::<Vec<Vec<f64>>>(),
            "the eager carrier holds the same bits as the lazy cursor"
        );
    }

    // =======================================================================
    // J. Legacy layout — loud, not silently empty
    // =======================================================================

    /// A store written by molrs <= 0.13 — the per-frame `trajectory/frames/`
    /// groups — is refused by name, not read as an empty sequence
    /// (decision 10's one free quadrant).
    ///
    /// The fixture is hand-built rather than produced by an old writer,
    /// because the old writer is gone: what identifies the layout is the
    /// `trajectory/frames/` group beside `trajectory/step`.
    #[test]
    fn a_trajectory_frames_group_is_refused_as_a_legacy_layout() {
        let dir = TempDir::new().unwrap();
        let store = store_in(&dir);
        for group in [TRAJ, "/trajectory/frames", "/trajectory/frames/0"] {
            GroupBuilder::new()
                .build(store.clone(), group)
                .unwrap()
                .store_metadata()
                .unwrap();
        }
        let step = ArrayBuilder::new(vec![1], vec![1], data_type::int64(), 0i64)
            .build(store.clone(), &format!("{TRAJ}/step"))
            .unwrap();
        step.store_metadata().unwrap();
        step.store_array_subset(&ArraySubset::new_with_shape(vec![1]), &[0i64])
            .unwrap();

        let message = FrameSequence::open(store)
            .err()
            .expect("the old layout must be refused, not read as empty")
            .to_string();
        // The comparison glyph is the implementer's (the Design writes "≤",
        // the task line "<="); everything around it is the pinned message.
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
