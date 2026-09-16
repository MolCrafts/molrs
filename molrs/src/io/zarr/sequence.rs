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
//!   per [`append`](FrameSequenceWriter::append); complete inner chunks land
//!   on their own, [`flush`](FrameSequenceWriter::flush) commits whatever is
//!   still buffered.
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
//! /                              root group
//! meta/                          the record's identity document (attributes)
//! trajectory/                    group; attrs: sequence_schema (the pin),
//!                                nstep (the commit marker, written last),
//!                                step_progression / time_progression
//!                                ({start, stride} while the series is regular)
//!   step     i64  [nstep]        only once the numbering is not a progression
//!   time     f64  [nstep]        only when supplied and not a progression
//!   meta/<key>    typed [nstep] (or [nstep][3|6|9]); attr meta_dtype
//!   box/                         attrs: cell_defined; vectors / origin /
//!                                boundary while the cell is fixed from ordinal 0
//!     step_index u64  [n_updates]     the arrays, once the cell changes
//!     vectors    f64  [n_updates][3][3]
//!     origin     f64  [n_updates][3]
//!     boundary   bool [n_updates][3]
//!   <block>/                     attrs structural_shape, uniform_rows, dense_updates
//!     step_index u64  [n_updates]     absent while the block is regular
//!     offset     u64  [n_updates+1]   CSR row pointer, offset[0] = 0
//!     <column>        [total_rows][...trailing]
//! ```
//!
//! The layout is built so the common run — a fixed number of atoms moving
//! every frame, a topology written once, a fixed cell, frames dumped every
//! `k` steps — costs one array per column and nothing else. A **regular**
//! block (a fixed, non-zero row count at ordinals `0, 1, 2, …`) writes no
//! index: its group hints plus any column's length resolve every frame. The
//! first irregular update materializes `offset` and `step_index`, backfilled
//! with the regular history, and withdraws the hints for good.
//!
//! `offset` is a **CSR** (compressed sparse row) row pointer, the standard way
//! to store a ragged sequence of row groups in one flat array: entry `j` holds
//! the index of the first row of update `j`, so update `j` owns the half-open
//! row range `offset[j]..offset[j+1]`, and the array is therefore one entry
//! longer than `step_index`.
//!
//! # Three states of a block at a frame
//!
//! Resolving block `B` at frame `i`: binary-search `B/step_index` for the
//! largest entry `<= i`, giving update `j`; the frame's rows are
//! `offset[j]..offset[j+1]`. The three cases are the layout's whole
//! presence vocabulary:
//!
//! - **No entry `<= i`** — the block does not exist at frame `i` (absence).
//! - **An update of zero rows** — the block is present and empty from that
//!   frame until its next update. It reads back as an empty [`Block`] with the
//!   declared columns.
//! - **A frame that omits a declared block** earns *no* update: the block
//!   carries forward from its latest update. That is what makes a constant
//!   topology cost one `step_index` entry whether the producer re-presents it
//!   every frame (bit-identical content earns no update either) or presents
//!   it once.
//!
//! Once a block has appeared it never becomes absent again; there is no
//! tombstone. The `box/` section has no `offset` and follows the same
//! carry-forward rule: once a run writes a cell, every later frame resolves to
//! the most recent one.
//!
//! # Chunks, shards, and the commit
//!
//! Every array of the sequence is a Zarr V3 `sharding_indexed` array whose
//! shard **index sits at the start** of the shard file. An **inner chunk** is
//! the unit the codec compresses; a **shard** is one file holding a fixed
//! number of consecutive inner chunks plus that index. With the index at the
//! start, appending a complete inner chunk is a write at the tail of the file
//! plus an in-place rewrite of the fixed-size index — nothing is re-encoded and
//! no dead bytes are left behind.
//!
//! Block columns are **frame-aligned**: their inner chunk holds a whole number
//! of frames of the representative row count (see [`SequenceSchema`]), so a
//! frame decodes from exactly its own chunk and a commit at a chunk boundary
//! rewrites nothing. The writer lands complete chunks on its own cadence
//! ([`FrameSequenceWriter::with_flush_every`] overrides it); an explicit
//! [`flush`](FrameSequenceWriter::flush) may land a partially filled trailing
//! chunk, whose superseded copy then stays in the shard as dead bytes — bounded
//! by one chunk per column per flush, and never cleaned up, because the
//! whole-shard rewrite that would clean it is the one write that can destroy
//! committed data on a crash.
//!
//! **Commit protocol.** For every array: data is written, then the array's
//! `zarr.json` (its new shape) is replaced atomically. Last comes the
//! trajectory group's metadata — the `nstep` marker and the progression
//! attributes — in one atomic replace. A reader takes `nstep` from that
//! attribute; anything a crash left longer is invisible, and a reopened
//! writer rolls it back before appending. An explicit `flush` / `close` is durable (the touched
//! files are synced before `step` moves) unless
//! [`with_durable(false)`](FrameSequenceWriter::with_durable) says otherwise;
//! the automatic chunk-boundary landings are not synced.
//!
//! [`Trajectory`]: molrs::store::trajectory::Trajectory
//! [`TrajectoryReader`]: crate::io::reader::TrajectoryReader

use std::collections::{BTreeMap, VecDeque};
use std::num::NonZeroU64;
use std::sync::{Arc, Mutex, Once};

use ndarray::{Array1, Array2, ArrayD, Axis, Slice};
use serde::{Deserialize, Serialize};
use zarrs::array::codec::GzipCodec;
use zarrs::array::codec::array_to_bytes::sharding::{
    ShardingCodecBuilder, ShardingCodecOptions, ShardingIndexLocation, SubchunkWriteOrder,
};
use zarrs::array::codec::bytes_to_bytes::crc32c::Crc32cCodec;
use zarrs::array::{
    Array, ArrayBuilder, ArraySubset, BytesToBytesCodecTraits, CodecOptions, CodecSpecificOptions,
};
use zarrs::config::{global_config, global_config_mut};
use zarrs::group::{Group, GroupBuilder};
use zarrs::node::{Node, NodeMetadata, NodePath, get_child_nodes};
use zarrs::storage::{
    ListableStorageTraits, ReadableListableStorage, ReadableListableStorageTraits,
    ReadableStorageTraits, ReadableWritableListableStorage, ReadableWritableListableStorageTraits,
    StorageHandle, StorePrefix, WritableStorageTraits,
};

use molrs::MolRsError;
use molrs::spatial::simbox::SimBox;
use molrs::store::block::{Block, Column, DType};
use molrs::store::frame::Frame;
use molrs::store::meta::MetaValue;
use molrs::store::trajectory::Trajectory;
use molrs::types::F;

use crate::io::reader::TrajectoryReader;

use super::frame_io::{
    BOX_GROUP, insert_column_into_block, join_path, node_prefix, read_column_array, zarr_dtype,
};
use super::record_io::zerr;

// ---------------------------------------------------------------------------
// Layout vocabulary
// ---------------------------------------------------------------------------

/// The record root.
const ROOT_GROUP: &str = "/";
/// The identity document's group.
const META_ROOT_GROUP: &str = "/meta";
/// The sequence root. Absolute, because every array path is built from it.
const TRAJECTORY_GROUP: &str = "/trajectory";
/// The commit marker: `i64[nstep]`, extended last.
const STEP_ARRAY: &str = "step";
/// Physical time per frame, `f64[nstep]`, only when the run supplies times.
const TIME_ARRAY: &str = "time";
/// The per-step metadata group under `trajectory/`.
const META_GROUP: &str = "meta";
/// A block section's CSR row pointer.
const OFFSET_ARRAY: &str = "offset";
/// A section's sparse update index: the frame ordinals it changed at.
const STEP_INDEX_ARRAY: &str = "step_index";
/// The cell matrices of the `box/` section.
const VECTORS_ARRAY: &str = "vectors";
/// The cell origins of the `box/` section (omitted when all zero).
const ORIGIN_ARRAY: &str = "origin";
/// The per-axis periodic flags of the `box/` section (omitted when all true).
const BOUNDARY_ARRAY: &str = "boundary";
/// The child molrs <= 0.13 wrote one group per frame under.
const LEGACY_FRAMES_GROUP: &str = "frames";
/// The `trajectory/` group attribute the pinned schema lives in.
const SCHEMA_ATTRIBUTE: &str = "sequence_schema";
/// The attribute every per-step meta array carries: its exact dtype tag.
const META_DTYPE_ATTRIBUTE: &str = "meta_dtype";
/// The `box/` group attribute recording an undefined cell (absent = defined).
const CELL_DEFINED_ATTRIBUTE: &str = "cell_defined";
/// Trajectory-group attribute: the commit marker — frames fully on disk.
const NSTEP_ATTRIBUTE: &str = "nstep";
/// Trajectory-group attribute: `step` as `{start, stride}` while it is regular.
const STEP_PROGRESSION_ATTRIBUTE: &str = "step_progression";
/// Trajectory-group attribute: `time` as `{start, stride}` while it is regular.
const TIME_PROGRESSION_ATTRIBUTE: &str = "time_progression";
/// A block section's mirrored structural shape.
const STRUCTURAL_SHAPE_ATTRIBUTE: &str = "structural_shape";
/// Block-section hint: every update so far holds this many rows.
const UNIFORM_ROWS_ATTRIBUTE: &str = "uniform_rows";
/// Block-section hint: `step_index[j] == j` for every update so far.
const DENSE_UPDATES_ATTRIBUTE: &str = "dense_updates";

/// Children of `trajectory/` a block may not be named after.
const RESERVED_BLOCK_NAMES: [&str; 4] = [STEP_ARRAY, TIME_ARRAY, META_GROUP, BOX_GROUP];
/// Children of a block section a column may not be named after.
const RESERVED_COLUMN_NAMES: [&str; 2] = [OFFSET_ARRAY, STEP_INDEX_ARRAY];

// ---------------------------------------------------------------------------
// Extent policy
// ---------------------------------------------------------------------------

/// The fewest bytes one inner chunk of a block column aims for.
///
/// A chunk is the unit the codec compresses and the shard index points at;
/// below this the per-chunk overhead (a 16 B index entry, a codec header) is
/// no longer negligible against the payload. A frame larger than this is its
/// own chunk.
const MIN_CHUNK_BYTES: u64 = 16 * 1024;

/// Bytes one shard file aims for, and therefore the file-count lever.
const SHARD_TARGET_BYTES: u64 = 256 * 1024 * 1024;

/// The most inner chunks one shard holds, whatever the byte target says.
///
/// The shard index is 16 B per inner chunk and is rewritten in place on every
/// landing, so this caps that rewrite at 64 KiB.
const MAX_CHUNKS_PER_SHARD: u64 = 4096;

/// Rows per inner chunk of the dense per-frame / per-update arrays (`step`,
/// `time`, `meta/*`, `offset`, `step_index`, `box/*`): 8 KiB of `u64`.
const DENSE_ROWS_PER_CHUNK: u64 = 1024;

/// Inner chunks per shard of the dense arrays: a 4 KiB shard index, so the
/// in-place index rewrite every landing pays on each of them stays small
/// (a flush-per-frame producer rewrites every dense array's index per frame).
/// 256 × 1024 rows is still 262 144 frames per shard file.
const DENSE_CHUNKS_PER_SHARD: u64 = 256;

/// Bytes of frame payload the writer buffers before landing on its own.
const FLUSH_TARGET_BYTES: u64 = 4 * 1024 * 1024;

/// The most frames the writer buffers before landing on its own.
const MAX_FLUSH_EVERY: u64 = 4096;

/// Bytes assumed for one element of a variable-width ([`DType::String`])
/// column when sizing a growth array.
///
/// [`DType::itemsize`] reports `None` there, but a growth array's extents are
/// frozen at creation, so a number has to be chosen. It moves a chunk boundary
/// and nothing else.
const ASSUMED_STRING_ITEMSIZE: u64 = 16;

/// `gzip` level for the columns and arrays that compress (integers, booleans,
/// strings, every index array). Level 1: these compress by structure, not by
/// effort, and the write path pays the codec on every landing.
const GZIP_LEVEL: u32 = 1;

/// Inner chunks a column reader keeps decoded per column.
///
/// Playback reads the same chunk `frames_per_chunk` times in a row; a small
/// LRU turns those repeats into slices. Two entries cover forward play and a
/// one-step scrub back without holding more than two chunks of a huge column.
const CHUNK_CACHE_ENTRIES: usize = 2;

/// How the floating-point columns of a sequence are compressed.
///
/// Integer, boolean and string columns, and every index array, always carry
/// `gzip` level 1: they compress by structure. Floating-point coordinates do
/// not — 52 random mantissa bits gzip to about 95 % of their size at a real
/// CPU cost — so their compression is a producer's choice, `None` by default.
/// Every choice is lossless; a precision study admits nothing else.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Compression {
    /// Raw little-endian bytes. The default for floating-point columns.
    #[default]
    None,
    /// `gzip` at this level (1–9). Every reader of these stores decodes it.
    Gzip(u32),
    /// `zstd` at this level. Native builds only (`zarr-codecs`); wasm32
    /// readers do not decode it, so a store meant for the browser stays on
    /// `None` or `Gzip`.
    #[cfg(feature = "zarr-codecs")]
    Zstd(i32),
}

/// The frozen extents of one growth array.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Extents {
    rows_per_chunk: u64,
    chunks_per_shard: u64,
}

impl Extents {
    /// Rows one inner chunk of a block column holds.
    ///
    /// Frame-aligned: the smallest whole number of frames of `frame_rows` rows
    /// whose narrowest column reaches [`MIN_CHUNK_BYTES`]. A block whose
    /// representative frame carries no rows (or is unknown) falls back to the
    /// byte floor alone.
    fn block_rows_per_chunk(frame_rows: u64, min_row_bytes: u64) -> u64 {
        let floor = (MIN_CHUNK_BYTES / min_row_bytes.max(1)).max(1);
        if frame_rows == 0 {
            return floor;
        }
        frame_rows.max(floor).div_ceil(frame_rows) * frame_rows
    }

    /// Inner chunks one shard of an array with `chunk_bytes`-byte chunks holds.
    fn chunks_per_shard_for(chunk_bytes: u64) -> u64 {
        (SHARD_TARGET_BYTES / chunk_bytes.max(1)).clamp(1, MAX_CHUNKS_PER_SHARD)
    }

    /// Extents of one block column: `rows_per_chunk` shared by the block,
    /// `chunks_per_shard` from this column's own width.
    fn for_column(rows_per_chunk: u64, row_bytes: u64, chunks_per_shard: Option<u64>) -> Self {
        Self {
            rows_per_chunk,
            chunks_per_shard: chunks_per_shard
                .unwrap_or_else(|| Self::chunks_per_shard_for(rows_per_chunk * row_bytes)),
        }
    }
}

/// The options every data write of this module carries.
///
/// Built explicitly and threaded into the `_opt` methods on purpose: the
/// non-`_opt` methods construct their own defaults, where
/// `experimental_partial_encoding` is `false` and every landing silently
/// becomes an O(shard) rewrite.
fn partial_encoding_options() -> CodecOptions {
    global_config()
        .codec_options()
        .with_experimental_partial_encoding(true)
}

/// Codec options that pin a shard's subchunk layout to C (row-major) order, so
/// a store's bytes do not depend on thread scheduling.
fn deterministic_shard_layout() -> CodecSpecificOptions {
    CodecSpecificOptions::default().with_option(
        ShardingCodecOptions::default().with_subchunk_write_order(SubchunkWriteOrder::C),
    )
}

/// Switch off zarrs' `_zarrs` provenance attribute on every array this process
/// creates. Done once: the attribute is bundle noise on a contract layout, and
/// the flag is process-wide by the library's design.
fn quiet_zarrs_metadata() {
    static ONCE: Once = Once::new();
    ONCE.call_once(|| {
        global_config_mut().set_include_zarrs_metadata(false);
    });
}

/// The bytes-to-bytes codecs of one inner chunk: the optional compressor, then
/// `crc32c` so a torn chunk is a checksum error rather than garbage rows.
fn inner_codecs(
    compression: Compression,
) -> Result<Vec<Arc<dyn BytesToBytesCodecTraits>>, MolRsError> {
    let mut codecs: Vec<Arc<dyn BytesToBytesCodecTraits>> = Vec::with_capacity(2);
    match compression {
        Compression::None => {}
        Compression::Gzip(level) => codecs
            .push(Arc::new(GzipCodec::new(level).map_err(|e| {
                MolRsError::zarr(format!("gzip level {level}: {e}"))
            })?)),
        #[cfg(feature = "zarr-codecs")]
        Compression::Zstd(level) => codecs.push(Arc::new(
            zarrs::array::codec::bytes_to_bytes::zstd::ZstdCodec::new(level, false),
        )),
    }
    codecs.push(Arc::new(Crc32cCodec::new()));
    Ok(codecs)
}

/// Whether a column width is floating point — the widths whose compression is
/// the producer's [`Compression`] choice rather than always-`gzip`.
fn is_float_width(dtype: DType) -> bool {
    matches!(
        dtype,
        DType::Float16 | DType::Float32 | DType::Float | DType::Complex64 | DType::Complex128
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

/// Every width a column may take, in the order molrec's dtype enum lists them.
const SCHEMA_WIDTHS: [DType; 15] = [
    DType::Float16,
    DType::Float32,
    DType::Float,
    DType::Int8,
    DType::Int16,
    DType::Int,
    DType::Int64,
    DType::U8,
    DType::UInt16,
    DType::UInt32,
    DType::UInt,
    DType::Bool,
    DType::String,
    DType::Complex64,
    DType::Complex128,
];

/// The column width a schema dtype tag names — the public spelling of the
/// closed dtype set (`f16` … `c128`) for callers that declare a schema from
/// text rather than from a frame.
///
/// # Errors
///
/// A [`MolRsError::Zarr`] naming an unknown tag.
pub fn column_dtype(tag: &str) -> Result<DType, MolRsError> {
    dtype_from_tag(tag)
}

/// The column width a schema tag names.
///
/// Reads the fifteen concrete-width tags [`dtype_tag`] writes, plus the three
/// legacy aliases (`float`, `int`, `uint`) an early store may carry.
fn dtype_from_tag(tag: &str) -> Result<DType, MolRsError> {
    match tag {
        "float" => return Ok(DType::Float),
        "int" => return Ok(DType::Int),
        "uint" => return Ok(DType::UInt),
        _ => {}
    }
    SCHEMA_WIDTHS
        .iter()
        .copied()
        .find(|&dtype| dtype_tag(dtype) == tag)
        .ok_or_else(|| MolRsError::zarr(format!("unknown column dtype tag {tag:?}")))
}

/// The column width a stored Zarr data type maps to, if any.
fn dtype_of_stored(stored: &zarrs::array::DataType) -> Option<DType> {
    SCHEMA_WIDTHS
        .iter()
        .copied()
        .find(|&dtype| zarr_dtype(dtype).0 == *stored)
}

/// The column width and trailing shape a per-step meta tag is stored as.
///
/// `None` for a tag no per-step array can store.
fn meta_layout(tag: &str) -> Option<(DType, Vec<u64>)> {
    let scalar = |dtype| Some((dtype, Vec::new()));
    let vector = |dtype, n: u64| Some((dtype, vec![n]));
    match tag {
        "bool" => scalar(DType::Bool),
        "i32" => scalar(DType::Int),
        "i64" => scalar(DType::Int64),
        "u32" => scalar(DType::UInt32),
        "u64" => scalar(DType::UInt),
        "f32" => scalar(DType::Float32),
        "f64" => scalar(DType::Float),
        // A JSON document per step rides in a string array; the tag says how
        // to read it back.
        "string" | "json" => scalar(DType::String),
        "bool3" => vector(DType::Bool, 3),
        "i32x3" => vector(DType::Int, 3),
        "i64x3" => vector(DType::Int64, 3),
        "u32x3" => vector(DType::UInt32, 3),
        "u64x3" => vector(DType::UInt, 3),
        "f32x3" => vector(DType::Float32, 3),
        "f64x3" => vector(DType::Float, 3),
        "f32x6" => vector(DType::Float32, 6),
        "f64x6" => vector(DType::Float, 6),
        "f32x9" => vector(DType::Float32, 9),
        "f64x9" => vector(DType::Float, 9),
        _ => None,
    }
}

/// Bytes one row of a column occupies, trailing axes included.
fn row_bytes(dtype: DType, trailing: &[u64]) -> u64 {
    let width = dtype
        .itemsize()
        .map_or(ASSUMED_STRING_ITEMSIZE, |width| width as u64);
    trailing
        .iter()
        .fold(width, |acc, &axis| acc.saturating_mul(axis.max(1)))
        .max(1)
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
/// Bitwise rather than numeric: a NaN coordinate has to count as *unchanged*,
/// or a section that never moves would earn an entry at every step. Compared
/// element by element on the typed arrays, so a comparison allocates nothing.
fn same_column(left: &Column, right: &Column) -> bool {
    if left.dtype() != right.dtype() || left.shape() != right.shape() {
        return false;
    }
    macro_rules! bits {
        ($a:expr, $b:expr) => {
            $a.iter()
                .zip($b.iter())
                .all(|(x, y)| x.to_bits() == y.to_bits())
        };
    }
    macro_rules! exact {
        ($a:expr, $b:expr) => {
            $a.iter().zip($b.iter()).all(|(x, y)| x == y)
        };
    }
    match (left, right) {
        (Column::Float16(a), Column::Float16(b)) => bits!(a, b),
        (Column::Float32(a), Column::Float32(b)) => bits!(a, b),
        (Column::Float(a), Column::Float(b)) => bits!(a, b),
        (Column::Complex64(a), Column::Complex64(b)) => a
            .iter()
            .zip(b.iter())
            .all(|(x, y)| x.re.to_bits() == y.re.to_bits() && x.im.to_bits() == y.im.to_bits()),
        (Column::Complex128(a), Column::Complex128(b)) => a
            .iter()
            .zip(b.iter())
            .all(|(x, y)| x.re.to_bits() == y.re.to_bits() && x.im.to_bits() == y.im.to_bits()),
        (Column::Int8(a), Column::Int8(b)) => exact!(a, b),
        (Column::Int16(a), Column::Int16(b)) => exact!(a, b),
        (Column::Int(a), Column::Int(b)) => exact!(a, b),
        (Column::Int64(a), Column::Int64(b)) => exact!(a, b),
        (Column::U8(a), Column::U8(b)) => exact!(a, b),
        (Column::UInt16(a), Column::UInt16(b)) => exact!(a, b),
        (Column::UInt32(a), Column::UInt32(b)) => exact!(a, b),
        (Column::UInt(a), Column::UInt(b)) => exact!(a, b),
        (Column::Bool(a), Column::Bool(b)) => exact!(a, b),
        (Column::String(a), Column::String(b)) => exact!(a, b),
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

/// Rows `start..end` of `column`, as a new column.
///
/// The whole column comes back as a cheap Arc clone; any other range is one
/// copy of exactly those rows.
fn column_rows(column: &Column, start: usize, end: usize) -> Column {
    let rows = column.nrows().unwrap_or(0);
    if start == 0 && end == rows {
        return column.clone();
    }
    macro_rules! slice {
        ($holder:expr, $ctor:ident) => {
            Column::$ctor(
                $holder
                    .slice_axis(Axis(0), Slice::from(start..end))
                    .to_owned(),
            )
        };
    }
    match column {
        Column::Float16(h) => slice!(h, from_f16),
        Column::Float32(h) => slice!(h, from_f32),
        Column::Float(h) => slice!(h, from_float),
        Column::Int8(h) => slice!(h, from_i8),
        Column::Int16(h) => slice!(h, from_i16),
        Column::Int(h) => slice!(h, from_int),
        Column::Int64(h) => slice!(h, from_i64),
        Column::U8(h) => slice!(h, from_u8),
        Column::UInt16(h) => slice!(h, from_u16),
        Column::UInt32(h) => slice!(h, from_u32),
        Column::UInt(h) => slice!(h, from_uint),
        Column::Bool(h) => slice!(h, from_bool),
        Column::String(h) => slice!(h, from_string),
        Column::Complex64(h) => slice!(h, from_c64),
        Column::Complex128(h) => slice!(h, from_c128),
    }
}

/// A zero-row column of `dtype` with `trailing` axes — the columns of a block
/// that is present and empty.
fn empty_column(dtype: DType, trailing: &[u64]) -> Result<Column, MolRsError> {
    let mut shape = Vec::with_capacity(trailing.len() + 1);
    shape.push(0usize);
    shape.extend(trailing.iter().map(|&n| n as usize));
    macro_rules! empty {
        ($ctor:ident, $ty:ty) => {
            Column::$ctor(ArrayD::<$ty>::from_shape_vec(shape, Vec::new()).map_err(zerr)?)
        };
    }
    Ok(match dtype {
        DType::Float16 => empty!(from_f16, half::f16),
        DType::Float32 => empty!(from_f32, f32),
        DType::Float => empty!(from_float, f64),
        DType::Int8 => empty!(from_i8, i8),
        DType::Int16 => empty!(from_i16, i16),
        DType::Int => empty!(from_int, i32),
        DType::Int64 => empty!(from_i64, i64),
        DType::U8 => empty!(from_u8, u8),
        DType::UInt16 => empty!(from_u16, u16),
        DType::UInt32 => empty!(from_u32, u32),
        DType::UInt => empty!(from_uint, u64),
        DType::Bool => empty!(from_bool, bool),
        DType::String => empty!(from_string, String),
        DType::Complex64 => empty!(from_c64, num_complex::Complex<f32>),
        DType::Complex128 => empty!(from_c128, num_complex::Complex<f64>),
    })
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
    /// Value written for a step that omits the key, as the plain JSON value
    /// ([`MetaValue::to_attr_value`]) — `dtype` beside it says how to read it
    /// back. `None` makes the omission an error — there is no implicit fill.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    fill: Option<serde_json::Value>,
}

/// The blocks, columns, dtypes and trailing shapes a sequence is pinned to.
///
/// Two ways to mint one: **derive** it from representative frames
/// ([`from_frame`](Self::from_frame) / [`from_frames`](Self::from_frames)),
/// or **declare** it column by column ([`new`](Self::new) then
/// [`declare_block`](Self::declare_block) / [`declare_column`](Self::declare_column)
/// / [`declare_meta`](Self::declare_meta)). A producer that knows its columns
/// declares them; one that has a frame in hand derives.
///
/// A later frame may present a *subset* of the declaration — that is how the
/// layout expresses sparsity — but never anything outside it: a run that
/// decides halfway through to record forces needs a new store.
///
/// The declaration also carries a **representative row count** per block
/// (the frame's row count, or the `rows` given to `declare_block`). It is not
/// part of the pin written to the store: it sizes the frame-aligned inner
/// chunk of the block's columns, nothing else, and a ragged run whose row
/// count wanders away from it still reads back exactly.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SequenceSchema {
    blocks: BTreeMap<String, BlockSchema>,
    #[serde(default)]
    meta: BTreeMap<String, MetaSchema>,
    /// Representative rows per block, for chunk sizing. Never pinned.
    #[serde(skip)]
    rows_hint: BTreeMap<String, u64>,
}

impl PartialEq for SequenceSchema {
    /// The pin — blocks and meta — is the identity; the row hints are sizing
    /// advice and two schemas that differ only there are the same schema.
    fn eq(&self, other: &Self) -> bool {
        self.blocks == other.blocks && self.meta == other.meta
    }
}

fn check_block_name(name: &str) -> Result<(), MolRsError> {
    if RESERVED_BLOCK_NAMES.contains(&name) {
        return Err(MolRsError::zarr(format!(
            "{name:?} is a reserved child of the trajectory group; a block cannot take it"
        )));
    }
    Ok(())
}

fn check_column_name(column: &str) -> Result<(), MolRsError> {
    if RESERVED_COLUMN_NAMES.contains(&column) {
        return Err(MolRsError::zarr(format!(
            "{column:?} is a reserved child of a block section; a column cannot take it"
        )));
    }
    Ok(())
}

impl SequenceSchema {
    /// An empty declaration, to be filled with `declare_*`.
    pub fn new() -> Self {
        Self::default()
    }

    /// Mint from one frame's own columns.
    ///
    /// # Errors
    ///
    /// The single-frame case of [`from_frames`](Self::from_frames), and it
    /// yields the same [`MolRsError::Zarr`] for the same reasons: a block or
    /// column that takes one of the layout's reserved names.
    pub fn from_frame(frame: &Frame) -> Result<Self, MolRsError> {
        Self::from_frames(std::slice::from_ref(frame))
    }

    /// Mint from the **union** of `frames`.
    ///
    /// This is how a heterogeneous run is expressed: blocks and columns are
    /// unioned across the frames, and each frame later presents whichever
    /// subset it has. A column appearing twice with a conflicting dtype or
    /// trailing shape is an error *here*, at mint, rather than at the append
    /// that would have discovered it. The largest row count a block shows
    /// across the frames becomes its representative row count.
    ///
    /// # Errors
    ///
    /// Every case is a [`MolRsError::Zarr`] naming the offending block, column
    /// or metadata key:
    ///
    /// - a block named `step`, `time`, `meta` or `box`;
    /// - a column named `offset` or `step_index`;
    /// - the same block declaring two different structural shapes;
    /// - the same column declaring two different dtypes or trailing shapes;
    /// - the same `meta` key carrying two different `MetaValue` dtypes.
    pub fn from_frames(frames: &[Frame]) -> Result<Self, MolRsError> {
        let mut schema = Self::new();
        let mut shapes: BTreeMap<String, Option<Vec<usize>>> = BTreeMap::new();

        for frame in frames {
            for (name, block) in frame.iter() {
                let rows = block.nrows().unwrap_or(0) as u64;
                schema.declare_block(name, Some(rows))?;
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
                for (column, values) in block.iter() {
                    let trailing: Vec<u64> =
                        values.shape().iter().skip(1).map(|&n| n as u64).collect();
                    schema.declare_column(name, column, values.dtype(), &trailing)?;
                }
            }

            for (key, value) in frame.meta.iter() {
                // `step` / `time` are the sequence's own arrays, not per-step
                // meta. A producer that stashed the commit marker on the frame
                // must not mint a duplicate `trajectory/meta/step`.
                if key == STEP_ARRAY || key == TIME_ARRAY {
                    continue;
                }
                schema.declare_meta(key, value.dtype())?;
            }
        }

        for (name, shape) in shapes {
            if let Some(shape) = shape {
                schema.declare_structural_shape(&name, &shape)?;
            }
        }
        Ok(schema)
    }

    /// Declare a block, with the row count a typical frame of it carries.
    ///
    /// Declaring an already-declared block only raises its representative
    /// row count. `rows` sizes the block's frame-aligned inner chunk; `None`
    /// (or `0`) leaves the byte floor to decide.
    ///
    /// # Errors
    ///
    /// A [`MolRsError::Zarr`] when `name` is `step`, `time`, `meta` or `box`.
    pub fn declare_block(&mut self, name: &str, rows: Option<u64>) -> Result<(), MolRsError> {
        check_block_name(name)?;
        self.blocks
            .entry(name.to_string())
            .or_insert_with(|| BlockSchema {
                columns: BTreeMap::new(),
                structural_shape: None,
            });
        if let Some(rows) = rows {
            let hint = self.rows_hint.entry(name.to_string()).or_insert(0);
            *hint = (*hint).max(rows);
        }
        Ok(())
    }

    /// Declare a column of `block` — its width and trailing shape.
    ///
    /// Declares the block too when it is new. Declaring the same column twice
    /// with the same width and shape is a no-op.
    ///
    /// # Errors
    ///
    /// A [`MolRsError::Zarr`] when `column` is `offset` or `step_index`, when
    /// `block` takes a reserved name, or when the column is already declared
    /// with another width or trailing shape — a declaration is a pin, and one
    /// column cannot mean two widths in one sequence.
    pub fn declare_column(
        &mut self,
        block: &str,
        column: &str,
        dtype: DType,
        trailing: &[u64],
    ) -> Result<(), MolRsError> {
        check_column_name(column)?;
        self.declare_block(block, None)?;
        let declared = ColumnSchema {
            dtype: dtype_tag(dtype).to_string(),
            trailing: trailing.to_vec(),
        };
        let entry = self
            .blocks
            .get_mut(block)
            .expect("declare_block just inserted it");
        match entry.columns.get(column) {
            None => {
                entry.columns.insert(column.to_string(), declared);
            }
            Some(existing) if existing.dtype != declared.dtype => {
                return Err(MolRsError::zarr(format!(
                    "sequence schema conflict: column {column:?} of block {block:?} is {} in one \
                     declaration and {} in another",
                    existing.dtype, declared.dtype
                )));
            }
            Some(existing) if existing.trailing != declared.trailing => {
                return Err(MolRsError::zarr(format!(
                    "sequence schema conflict: column {column:?} of block {block:?} has trailing \
                     shape {:?} in one declaration and {:?} in another",
                    existing.trailing, declared.trailing
                )));
            }
            Some(_) => {}
        }
        Ok(())
    }

    /// Declare the structural shape of `block` (a volumetric `[nx][ny][nz]`).
    ///
    /// A block with a structural shape carries exactly `product(shape)` rows
    /// in every update; the writer refuses any other row count.
    ///
    /// # Errors
    ///
    /// A [`MolRsError::Zarr`] when `block` is not declared, or is already
    /// declared with a different shape.
    pub fn declare_structural_shape(
        &mut self,
        block: &str,
        shape: &[usize],
    ) -> Result<(), MolRsError> {
        let entry = self.blocks.get_mut(block).ok_or_else(|| {
            MolRsError::zarr(format!(
                "block {block:?} is not declared; declare it before its structural shape"
            ))
        })?;
        match &entry.structural_shape {
            Some(existing) if existing.as_slice() != shape => Err(MolRsError::zarr(format!(
                "sequence schema conflict: block {block:?} declares structural shape {existing:?} \
                 and {shape:?}"
            ))),
            _ => {
                entry.structural_shape = Some(shape.to_vec());
                let rows = shape.iter().product::<usize>() as u64;
                self.rows_hint.insert(block.to_string(), rows);
                Ok(())
            }
        }
    }

    /// Declare a per-step metadata key by its dtype tag.
    ///
    /// The tag is one of the scalar forms `bool`, `i32`, `i64`, `u32`, `u64`,
    /// `f32`, `f64`, `string`, `json`, the three-component forms `bool3`,
    /// `i32x3`, `i64x3`, `u32x3`, `u64x3`, `f32x3`, `f64x3`, or the six- and
    /// nine-component float forms `f32x6`, `f64x6`, `f32x9`, `f64x9`. A
    /// frame that omits a key declared this way is an error at append; declare
    /// a fill with [`declare_meta_with_fill`](Self::declare_meta_with_fill)
    /// to make omission legal.
    ///
    /// # Errors
    ///
    /// A [`MolRsError::Zarr`] naming `key` when the tag is unknown, or when the
    /// key is already declared with a different dtype.
    pub fn declare_meta(&mut self, key: &str, dtype: &str) -> Result<(), MolRsError> {
        if meta_layout(dtype).is_none() {
            return Err(MolRsError::zarr(format!(
                "meta key {key:?} cannot be stored per step: unknown dtype {dtype}"
            )));
        }
        match self.meta.get(key) {
            Some(existing) if existing.dtype != dtype => Err(MolRsError::zarr(format!(
                "sequence schema conflict: meta key {key:?} is {} in one declaration and {dtype} \
                 in another",
                existing.dtype
            ))),
            Some(_) => Ok(()),
            None => {
                self.meta.insert(
                    key.to_string(),
                    MetaSchema {
                        dtype: dtype.to_string(),
                        fill: None,
                    },
                );
                Ok(())
            }
        }
    }

    /// Declare `key` with the value written for the steps that omit it.
    ///
    /// The declared dtype is the fill's own. There is deliberately no implicit
    /// NaN: a NaN nobody asked for is a measurement nobody made. The fill is
    /// recorded in the pinned schema, so a reopened writer keeps honouring it.
    ///
    /// # Errors
    ///
    /// A [`MolRsError::Zarr`] naming `key` when `fill`'s dtype is one no
    /// per-step array can store, or when `key` is already declared with a
    /// different dtype.
    pub fn declare_meta_with_fill(&mut self, key: &str, fill: MetaValue) -> Result<(), MolRsError> {
        let tag = fill.dtype();
        self.declare_meta(key, tag)?;
        self.meta
            .get_mut(key)
            .expect("declare_meta just inserted it")
            .fill = Some(fill.to_attr_value());
        Ok(())
    }

    /// The declared block names.
    pub fn block_names(&self) -> impl Iterator<Item = &str> {
        self.blocks.keys().map(String::as_str)
    }

    /// The declared column names of `block`, or `None` when it is not declared.
    pub fn column_names(&self, block: &str) -> Option<impl Iterator<Item = &str>> {
        self.blocks
            .get(block)
            .map(|declared| declared.columns.keys().map(String::as_str))
    }

    /// The declared per-step metadata keys with their dtype tags.
    pub fn meta_keys(&self) -> impl Iterator<Item = (&str, &str)> {
        self.meta
            .iter()
            .map(|(key, declared)| (key.as_str(), declared.dtype.as_str()))
    }

    /// The representative row count of `block`, `0` when none was given.
    fn frame_rows(&self, block: &str) -> u64 {
        self.rows_hint.get(block).copied().unwrap_or(0)
    }

    /// Bytes one frame of the representative row counts occupies across every
    /// declared column — the payload estimate the automatic landing cadence
    /// is derived from.
    fn frame_bytes(&self) -> u64 {
        self.blocks
            .iter()
            .map(|(name, block)| {
                let per_row: u64 = block
                    .columns
                    .values()
                    .map(|column| {
                        dtype_from_tag(&column.dtype)
                            .map_or(8, |dtype| row_bytes(dtype, &column.trailing))
                    })
                    .sum();
                per_row.saturating_mul(self.frame_rows(name))
            })
            .sum()
    }
}

/// Read the schema back out of the `trajectory/` group attributes, if one is
/// pinned there.
///
/// A missing pin is `Ok(None)`, not an error, because it is the one fact the
/// two doors answer differently: [`schema_of`] refuses it, and
/// [`FrameSequence::open`] derives from the store instead.
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

/// [`pinned_schema`], requiring the pin — the writer's half of the seam:
/// appending needs the *declared* union and the meta fills, neither of which
/// is recoverable from data that was never written.
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
/// only. A path holding no children yields an empty vector.
fn children<S>(store: &Arc<S>, path: &str) -> Result<Vec<Node>, MolRsError>
where
    S: ?Sized + ReadableStorageTraits + ListableStorageTraits + 'static,
{
    let path: NodePath = path.try_into().map_err(zerr)?;
    Ok(get_child_nodes(store, &path, false)?)
}

/// The schema of a store that pins none: derived from the layout itself.
///
/// A block is a child *group* of `trajectory/` that is not one of
/// [`RESERVED_BLOCK_NAMES`]; its columns are that group's arrays that are not
/// one of [`RESERVED_COLUMN_NAMES`]; each column's width and trailing shape
/// are its own array metadata. Per-step metadata keys are
/// `trajectory/meta/`'s arrays, each carrying its own
/// [`META_DTYPE_ATTRIBUTE`] tag. Every derived fill is `None`: a fill is what
/// a *writer* chose, and a reader that was never told cannot say more.
fn schema_from_store<S>(store: &Arc<S>) -> Result<SequenceSchema, MolRsError>
where
    S: ?Sized + ReadableStorageTraits + ListableStorageTraits + 'static,
{
    let mut schema = SequenceSchema::new();
    for section in children(store, TRAJECTORY_GROUP)? {
        if !matches!(section.metadata(), NodeMetadata::Group(_)) {
            continue;
        }
        let name = section.name().as_str().to_string();
        if RESERVED_BLOCK_NAMES.contains(&name.as_str()) {
            continue;
        }
        let path = join_path(TRAJECTORY_GROUP, &name);
        schema.declare_block(&name, None)?;

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
            let trailing: Vec<u64> = array.shape().iter().skip(1).copied().collect();
            schema.declare_column(&name, &column, dtype, &trailing)?;
        }

        // The section mirrors its structural shape for exactly this reader.
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
        if let Some(shape) = structural_shape {
            schema.declare_structural_shape(&name, &shape)?;
        }
    }

    let meta_path = join_path(TRAJECTORY_GROUP, META_GROUP);
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
        schema.declare_meta(&key, tag)?;
    }

    Ok(schema)
}

/// Refuse a store written by molrs <= 0.13, whose `trajectory/frames/<i>/`
/// groups this layout replaced.
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

/// Whether a group exists at `path`.
fn group_exists<S>(store: &Arc<S>, path: &str) -> Result<bool, MolRsError>
where
    S: ?Sized + ReadableStorageTraits + 'static,
{
    match Group::open(store.clone(), path) {
        Ok(_) => Ok(true),
        Err(zarrs::group::GroupCreateError::MissingMetadata) => Ok(false),
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

/// Rows one inner chunk of `array` holds: the sharding subchunk's leading
/// extent, or the chunk grid's when the array is not sharded.
fn inner_rows_of<S: ?Sized>(array: &Array<S>) -> u64 {
    let metadata = serde_json::to_value(array.metadata()).unwrap_or_default();
    let sharded = metadata["codecs"]
        .as_array()
        .and_then(|codecs| {
            codecs
                .iter()
                .find(|codec| codec["name"] == "sharding_indexed")
        })
        .and_then(|codec| codec["configuration"]["chunk_shape"][0].as_u64());
    sharded
        .or_else(|| metadata["chunk_grid"]["configuration"]["chunk_shape"][0].as_u64())
        .unwrap_or(1)
        .max(1)
}

/// Ensure the record root and its `meta/` group exist, writing `meta` when
/// one is handed in.
fn ensure_root_and_meta(
    store: &ReadableWritableListableStorage,
    meta: Option<&serde_json::Map<String, serde_json::Value>>,
) -> Result<(), MolRsError> {
    if !group_exists(store, ROOT_GROUP)? {
        GroupBuilder::new()
            .build(store.clone(), ROOT_GROUP)?
            .store_metadata()?;
    }
    if meta.is_some() || !group_exists(store, META_ROOT_GROUP)? {
        GroupBuilder::new()
            .attributes(meta.cloned().unwrap_or_default())
            .build(store.clone(), META_ROOT_GROUP)?
            .store_metadata()?;
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Growth arrays
// ---------------------------------------------------------------------------

/// A `zarrs` array whose leading axis grows and whose extents never do.
///
/// `zarrs::Array::set_shape` grows the leading axis and rebuilds the chunk grid
/// from *frozen* chunk metadata, so the inner-chunk and shard extents are
/// decided once, at creation. A landing is three steps in a fixed order —
/// [`reserve`](Self::reserve) the rows, write them, then
/// [`commit_shape`](Self::commit_shape) — so the array's metadata never
/// advertises rows that are not on disk yet.
struct GrowthArray {
    array: Array<dyn ReadableWritableListableStorageTraits>,
    /// Rows landed so far (the leading extent the metadata will publish).
    rows: u64,
}

impl GrowthArray {
    /// Create the array at `path`, zero rows long, extents frozen.
    ///
    /// Every growth array is a `sharding_indexed` array with its index at the
    /// **start** of the shard, so appending a chunk is a tail write plus an
    /// in-place index rewrite. The inner chunk carries the compressor
    /// `compression` names, then `crc32c`.
    fn create(
        store: &ReadableWritableListableStorage,
        path: &str,
        dtype: DType,
        trailing: &[u64],
        extents: Extents,
        attributes: serde_json::Map<String, serde_json::Value>,
        compression: Compression,
    ) -> Result<Self, MolRsError> {
        quiet_zarrs_metadata();
        let (data_type, fill) = zarr_dtype(dtype);
        let mut shape = Vec::with_capacity(trailing.len() + 1);
        shape.push(0u64);
        shape.extend_from_slice(trailing);
        // A chunk extent must be non-zero on every axis, including a trailing
        // axis that happens to be empty.
        let mut inner: Vec<u64> = shape.iter().map(|&axis| axis.max(1)).collect();
        inner[0] = extents.rows_per_chunk.max(1);
        let mut shard = inner.clone();
        shard[0] = inner[0].saturating_mul(extents.chunks_per_shard.max(1));

        let subchunk: Vec<NonZeroU64> = inner
            .iter()
            .map(|&axis| NonZeroU64::new(axis).expect("chunk extents are floored at one"))
            .collect();
        let mut sharding = ShardingCodecBuilder::new(subchunk, &data_type);
        sharding
            .bytes_to_bytes_codecs(inner_codecs(compression)?)
            .index_location(ShardingIndexLocation::Start);

        let mut builder = ArrayBuilder::new(shape, shard, data_type, fill);
        builder.array_to_bytes_codec(sharding.build_arc());
        builder.attributes(attributes);
        let array = builder
            .build(store.clone(), path)?
            .with_codec_specific_options(&deterministic_shard_layout());
        array.store_metadata()?;
        Ok(Self { array, rows: 0 })
    }

    /// Reopen an existing array, adopting the extents it was created with.
    fn open(store: &ReadableWritableListableStorage, path: &str) -> Result<Self, MolRsError> {
        let array = Array::open(store.clone(), path)?
            .with_codec_specific_options(&deterministic_shard_layout());
        let rows = array.shape().first().copied().unwrap_or(0);
        Ok(Self { array, rows })
    }

    /// The array's axes after the leading row axis.
    fn trailing(&self) -> Vec<u64> {
        self.array.shape().iter().skip(1).copied().collect()
    }

    /// Rows one inner chunk holds.
    fn rows_per_chunk(&self) -> u64 {
        inner_rows_of(&self.array)
    }

    /// Grow the leading axis by `count` rows **in memory** and return the
    /// subset they land in. The caller writes the data, then
    /// [`commit_shape`](Self::commit_shape) publishes the new extent.
    fn reserve(&mut self, count: u64) -> Result<ArraySubset, MolRsError> {
        let start = self.rows;
        self.rows += count;
        let mut shape = self.array.shape().to_vec();
        shape[0] = self.rows;
        self.array.set_shape(shape)?;
        rows_subset(start, count, &self.trailing())
    }

    /// Publish the reserved extent: one atomic `zarr.json` replacement.
    fn commit_shape(&self) -> Result<(), MolRsError> {
        self.array.store_metadata()?;
        Ok(())
    }

    /// Roll the leading axis back to `rows` — what a reopen does to whatever a
    /// crash left longer than the commit marker.
    fn truncate_to(&mut self, rows: u64) -> Result<(), MolRsError> {
        if rows >= self.rows {
            return Ok(());
        }
        let mut shape = self.array.shape().to_vec();
        shape[0] = rows;
        self.array.set_shape(shape)?;
        self.array.store_metadata()?;
        self.rows = rows;
        Ok(())
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

/// Reserve, write, publish — the landing order every plain-valued array uses.
macro_rules! land_values {
    ($array:expr, $count:expr, $values:expr, $options:expr) => {{
        let subset = $array.reserve($count)?;
        $array
            .array
            .store_array_subset_opt(&subset, $values, $options)?;
        $array.commit_shape()?;
    }};
}

/// One value of a per-step meta array, at step `index`.
fn read_meta_value<S>(
    array: &Array<S>,
    key: &str,
    tag: &str,
    index: u64,
) -> Result<MetaValue, MolRsError>
where
    S: ?Sized + ReadableStorageTraits + 'static,
{
    let (_, trailing) = meta_layout(tag).ok_or_else(|| {
        MolRsError::zarr(format!("meta key {key:?} declares unknown dtype {tag:?}"))
    })?;
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

/// A declared block with no rows: every declared column, zero rows long.
fn empty_block(schema: &BlockSchema) -> Result<Block, MolRsError> {
    let mut block = Block::new();
    for (column, declared) in &schema.columns {
        let dtype = dtype_from_tag(&declared.dtype)?;
        insert_column_into_block(&mut block, column, empty_column(dtype, &declared.trailing)?)?;
    }
    Ok(block)
}

/// Apply a declared structural shape to a block whose row count matches it.
fn apply_structural_shape(
    block: &mut Block,
    schema: &BlockSchema,
    path: &str,
) -> Result<(), MolRsError> {
    if let Some(shape) = &schema.structural_shape
        && block.nrows() == Some(shape.iter().product::<usize>())
    {
        block
            .set_shape(shape)
            .map_err(|e| MolRsError::zarr(format!("block {path:?} shape {shape:?}: {e}")))?;
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Block hints
// ---------------------------------------------------------------------------

/// Whether every update of a block so far carried the same row count.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Uniform {
    /// No update yet.
    Unknown,
    /// Every update so far carried this many rows.
    Rows(u64),
    /// Two updates disagreed; the hint is gone for good.
    Broken,
}

/// The two block-section hints the writer maintains as group attributes.
///
/// `uniform_rows` says every update carried the same row count; `dense_updates`
/// says `step_index[j] == j` for every update. A reader holding both resolves
/// any frame arithmetically without decoding the index arrays. They are
/// **monotone**: once a hint breaks it is removed and never returns, so a
/// reader that sees one may trust it for every committed frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct BlockHints {
    uniform: Uniform,
    dense: bool,
    /// What the group attributes currently say, so a rewrite happens only on
    /// a change.
    stored: (Option<u64>, bool),
}

impl BlockHints {
    fn fresh() -> Self {
        Self {
            uniform: Uniform::Unknown,
            dense: true,
            stored: (None, false),
        }
    }

    /// Rebuild from a section's index arrays — what a reopen does.
    fn from_index(step_index: &[u64], offset: &[u64]) -> Self {
        let mut hints = Self::fresh();
        for (update, &ordinal) in step_index.iter().enumerate() {
            let rows = offset
                .get(update + 1)
                .zip(offset.get(update))
                .map_or(0, |(end, start)| end.saturating_sub(*start));
            hints.observe(update as u64, ordinal, rows);
        }
        hints
    }

    /// Fold one more update in.
    fn observe(&mut self, update: u64, ordinal: u64, rows: u64) {
        self.dense &= ordinal == update;
        self.uniform = match self.uniform {
            Uniform::Unknown => Uniform::Rows(rows),
            Uniform::Rows(n) if n == rows => Uniform::Rows(n),
            _ => Uniform::Broken,
        };
    }

    /// Whether every update so far is regular: a fixed, non-zero row count
    /// at ordinals `0, 1, 2, …` — the state under which the CSR index is
    /// not written.
    fn is_regular(&self) -> bool {
        matches!(self.uniform, Uniform::Rows(n) if n > 0) && self.dense
    }

    /// The attributes these hints spell: `(uniform_rows, dense_updates)`.
    fn attributes(&self) -> (Option<u64>, bool) {
        match self.uniform {
            Uniform::Rows(n) if n > 0 => (Some(n), self.dense),
            Uniform::Rows(_) | Uniform::Unknown | Uniform::Broken => (None, false),
        }
    }
}

/// The group attributes of a block section: its mirrored structural shape
/// plus whichever hints currently hold.
fn block_attributes(
    structural_shape: Option<&[usize]>,
    hints: (Option<u64>, bool),
) -> serde_json::Map<String, serde_json::Value> {
    let mut attributes = serde_json::Map::new();
    if let Some(shape) = structural_shape {
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
    if let Some(rows) = hints.0 {
        attributes.insert(UNIFORM_ROWS_ATTRIBUTE.to_string(), rows.into());
    }
    if hints.1 {
        attributes.insert(DENSE_UPDATES_ATTRIBUTE.to_string(), true.into());
    }
    attributes
}

/// Read a block section's hints back off its group attributes.
fn stored_hints<S>(store: &Arc<S>, path: &str) -> Result<(Option<u64>, bool), MolRsError>
where
    S: ?Sized + ReadableStorageTraits + 'static,
{
    let group = Group::open(store.clone(), path)?;
    let attributes = group.attributes();
    Ok((
        attributes
            .get(UNIFORM_ROWS_ATTRIBUTE)
            .and_then(serde_json::Value::as_u64),
        attributes
            .get(DENSE_UPDATES_ATTRIBUTE)
            .and_then(serde_json::Value::as_bool)
            .unwrap_or(false),
    ))
}

/// Rewrite a block section's group attributes.
fn store_block_attributes(
    store: &ReadableWritableListableStorage,
    path: &str,
    structural_shape: Option<&[usize]>,
    hints: (Option<u64>, bool),
) -> Result<(), MolRsError> {
    let mut group = Group::open(store.clone(), path)?;
    *group.attributes_mut() = block_attributes(structural_shape, hints);
    group.store_metadata()?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Section arrays
// ---------------------------------------------------------------------------

/// The two test knobs that override the derived extents of every array.
#[derive(Debug, Clone, Copy, Default)]
struct Knobs {
    rows_per_chunk: Option<u64>,
    chunks_per_shard: Option<u64>,
}

impl Knobs {
    /// Extents of a dense array under these knobs.
    fn dense(self) -> Extents {
        Extents {
            rows_per_chunk: self.rows_per_chunk.unwrap_or(DENSE_ROWS_PER_CHUNK),
            chunks_per_shard: self.chunks_per_shard.unwrap_or(DENSE_CHUNKS_PER_SHARD),
        }
    }
}

/// The compression every dense array carries.
const DENSE_COMPRESSION: Compression = Compression::Gzip(GZIP_LEVEL);

/// The arrays of one block section.
///
/// The CSR index — `offset` and `step_index` — is **not** written while the
/// block is regular (every update carries the same non-zero row count at
/// ordinals `0, 1, 2, …`): the block group's `uniform_rows` /
/// `dense_updates` hints let a reader resolve any frame arithmetically, and
/// the two arrays would only restate what the column shapes already say. The
/// first update that breaks either rule materializes both arrays, backfilling
/// the regular history, and withdraws the hints for good.
struct BlockArrays {
    columns: BTreeMap<String, GrowthArray>,
    /// The CSR index, once the block stopped being regular.
    index: Option<IndexArrays>,
    /// Rows landed across every update.
    total_rows: u64,
    /// Updates landed.
    updates: u64,
    /// Rows one inner chunk of this block's columns holds.
    rows_per_chunk: u64,
    hints: BlockHints,
}

/// A block section's CSR index arrays.
struct IndexArrays {
    offset: GrowthArray,
    step_index: GrowthArray,
}

impl IndexArrays {
    /// Create both arrays and backfill the `history` regular updates of
    /// `rows` rows each — the history the hints described until now.
    fn materialize(
        store: &ReadableWritableListableStorage,
        path: &str,
        knobs: Knobs,
        history: u64,
        rows: u64,
        options: &CodecOptions,
    ) -> Result<Self, MolRsError> {
        let mut offset = GrowthArray::create(
            store,
            &join_path(path, OFFSET_ARRAY),
            DType::UInt,
            &[],
            knobs.dense(),
            serde_json::Map::new(),
            DENSE_COMPRESSION,
        )?;
        let mut step_index = GrowthArray::create(
            store,
            &join_path(path, STEP_INDEX_ARRAY),
            DType::UInt,
            &[],
            knobs.dense(),
            serde_json::Map::new(),
            DENSE_COMPRESSION,
        )?;
        // The CSR row pointer opens at zero; every regular update added
        // `rows` rows at the ordinal equal to its own index.
        let offsets: Vec<u64> = (0..=history).map(|update| update * rows).collect();
        let count = offsets.len() as u64;
        land_values!(offset, count, offsets, options);
        if history > 0 {
            let ordinals: Vec<u64> = (0..history).collect();
            land_values!(step_index, history, ordinals, options);
        }
        Ok(Self { offset, step_index })
    }
}

/// Where the cell of a sequence lives.
///
/// A cell that arrives at ordinal 0 and never changes is recorded as the
/// `box/` group's own attributes — no array, no extra file. The first change
/// migrates it into the `box/` arrays.
enum CellStore {
    /// One cell, at ordinal 0, in the group attributes.
    Attributes(Box<SimBox>),
    /// The per-update arrays.
    Arrays(Box<BoxArrays>),
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
    fn create(store: &ReadableWritableListableStorage, knobs: Knobs) -> Result<Self, MolRsError> {
        Ok(Self {
            step_index: None,
            vectors: Self::create_optional(store, VECTORS_ARRAY, DType::Float, &[3, 3], knobs)?,
            origin: None,
            boundary: None,
        })
    }

    fn create_optional(
        store: &ReadableWritableListableStorage,
        name: &str,
        dtype: DType,
        trailing: &[u64],
        knobs: Knobs,
    ) -> Result<GrowthArray, MolRsError> {
        let prefix = join_path(TRAJECTORY_GROUP, BOX_GROUP);
        GrowthArray::create(
            store,
            &join_path(&prefix, name),
            dtype,
            trailing,
            knobs.dense(),
            serde_json::Map::new(),
            DENSE_COMPRESSION,
        )
    }

    fn ensure_origin(
        &mut self,
        store: &ReadableWritableListableStorage,
        knobs: Knobs,
        previous: u64,
        options: &CodecOptions,
    ) -> Result<&mut GrowthArray, MolRsError> {
        if self.origin.is_none() {
            let mut origin = Self::create_optional(store, ORIGIN_ARRAY, DType::Float, &[3], knobs)?;
            if previous > 0 {
                let fill = vec![0.0 as F; (previous * 3) as usize];
                land_values!(origin, previous, fill, options);
            }
            self.origin = Some(origin);
        }
        Ok(self.origin.as_mut().expect("origin was just created"))
    }

    fn ensure_boundary(
        &mut self,
        store: &ReadableWritableListableStorage,
        knobs: Knobs,
        previous: u64,
        options: &CodecOptions,
    ) -> Result<&mut GrowthArray, MolRsError> {
        if self.boundary.is_none() {
            let mut boundary =
                Self::create_optional(store, BOUNDARY_ARRAY, DType::Bool, &[3], knobs)?;
            if previous > 0 {
                let fill = vec![true; (previous * 3) as usize];
                land_values!(boundary, previous, fill, options);
            }
            self.boundary = Some(boundary);
        }
        Ok(self.boundary.as_mut().expect("boundary was just created"))
    }

    fn ensure_step_index(
        &mut self,
        store: &ReadableWritableListableStorage,
        knobs: Knobs,
        previous: u64,
        options: &CodecOptions,
    ) -> Result<&mut GrowthArray, MolRsError> {
        if self.step_index.is_none() {
            let mut step_index =
                Self::create_optional(store, STEP_INDEX_ARRAY, DType::UInt, &[], knobs)?;
            if previous > 0 {
                // The omitted form is one update at ordinal 0.
                let fill = vec![0u64; previous as usize];
                land_values!(step_index, previous, fill, options);
            }
            self.step_index = Some(step_index);
        }
        Ok(self
            .step_index
            .as_mut()
            .expect("step_index was just created"))
    }

    /// Land `cells` — `(ordinal, cell)` pairs — as updates.
    fn land(
        &mut self,
        store: &ReadableWritableListableStorage,
        knobs: Knobs,
        cells: &[(u64, &SimBox)],
        options: &CodecOptions,
    ) -> Result<(), MolRsError> {
        let mut vectors = Vec::with_capacity(cells.len() * 9);
        let mut origins = Vec::with_capacity(cells.len() * 3);
        let mut boundaries = Vec::with_capacity(cells.len() * 3);
        let mut need_origin = self.origin.is_some();
        let mut need_boundary = self.boundary.is_some();
        for (_, cell) in cells {
            vectors.extend(cell.h_view().iter().copied());
            origins.extend(cell.origin_view().iter().copied());
            boundaries.extend(cell.pbc_view().iter().copied());
            need_origin |= !origin_is_default(cell);
            need_boundary |= !boundary_is_default(cell);
        }
        let count = cells.len() as u64;
        let previous = self.vectors.rows;
        let ordinals: Vec<u64> = cells.iter().map(|(ordinal, _)| *ordinal).collect();
        let trivial_index = previous == 0 && matches!(ordinals.as_slice(), [0]);

        land_values!(self.vectors, count, vectors, options);
        if need_origin {
            let origin = self.ensure_origin(store, knobs, previous, options)?;
            land_values!(origin, count, origins, options);
        }
        if need_boundary {
            let boundary = self.ensure_boundary(store, knobs, previous, options)?;
            land_values!(boundary, count, boundaries, options);
        }
        if self.step_index.is_some() || !trivial_index {
            let step_index = self.ensure_step_index(store, knobs, previous, options)?;
            land_values!(step_index, count, ordinals, options);
        }
        Ok(())
    }

    /// Roll every array back to `updates` committed cell updates.
    fn truncate_to(&mut self, updates: u64) -> Result<(), MolRsError> {
        self.vectors.truncate_to(updates)?;
        for array in [&mut self.step_index, &mut self.origin, &mut self.boundary]
            .into_iter()
            .flatten()
        {
            array.truncate_to(updates)?;
        }
        Ok(())
    }
}

/// The `box/` group attributes: `cell_defined` when false, plus the fixed
/// cell itself (`vectors`, and `origin` / `boundary` off their defaults) when
/// the cell lives in the attributes.
fn box_attributes(
    cell_defined: bool,
    fixed: Option<&SimBox>,
) -> serde_json::Map<String, serde_json::Value> {
    let mut attributes = serde_json::Map::new();
    if !cell_defined {
        attributes.insert(CELL_DEFINED_ATTRIBUTE.to_string(), false.into());
    }
    if let Some(cell) = fixed {
        let h = cell.h_view();
        let rows: Vec<serde_json::Value> = (0..3)
            .map(|i| serde_json::json!([h[[i, 0]], h[[i, 1]], h[[i, 2]]]))
            .collect();
        attributes.insert(VECTORS_ARRAY.to_string(), serde_json::Value::Array(rows));
        if !origin_is_default(cell) {
            let o = cell.origin_view();
            attributes.insert(
                ORIGIN_ARRAY.to_string(),
                serde_json::json!([o[0], o[1], o[2]]),
            );
        }
        if !boundary_is_default(cell) {
            attributes.insert(BOUNDARY_ARRAY.to_string(), serde_json::json!(cell.pbc()));
        }
    }
    attributes
}

/// The fixed cell recorded in the `box/` group attributes, if any.
fn cell_from_attributes(
    attributes: &serde_json::Map<String, serde_json::Value>,
) -> Result<Option<SimBox>, MolRsError> {
    let Some(vectors) = attributes.get(VECTORS_ARRAY) else {
        return Ok(None);
    };
    let flat: Vec<F> = match vectors.as_array() {
        Some(rows) if rows.len() == 3 && rows.iter().all(serde_json::Value::is_array) => rows
            .iter()
            .flat_map(|row| row.as_array().into_iter().flatten())
            .filter_map(serde_json::Value::as_f64)
            .collect(),
        Some(values) => values
            .iter()
            .filter_map(serde_json::Value::as_f64)
            .collect(),
        None => Vec::new(),
    };
    if flat.len() != 9 {
        return Err(MolRsError::zarr(format!(
            "box attribute {VECTORS_ARRAY:?} must hold a 3x3 cell, found {vectors}"
        )));
    }
    let origin: Vec<F> = match attributes.get(ORIGIN_ARRAY).and_then(|v| v.as_array()) {
        Some(values) => values
            .iter()
            .filter_map(serde_json::Value::as_f64)
            .collect(),
        None => vec![0.0, 0.0, 0.0],
    };
    let boundary: Vec<bool> = match attributes.get(BOUNDARY_ARRAY).and_then(|v| v.as_array()) {
        Some(values) => values
            .iter()
            .filter_map(serde_json::Value::as_bool)
            .collect(),
        None => vec![true, true, true],
    };
    if origin.len() != 3 || boundary.len() != 3 {
        return Err(MolRsError::zarr(
            "box attributes origin / boundary must hold three values each".to_string(),
        ));
    }
    let cell_defined = attributes
        .get(CELL_DEFINED_ATTRIBUTE)
        .and_then(serde_json::Value::as_bool)
        .unwrap_or(true);
    SimBox::new_cell(
        Array2::from_shape_vec((3, 3), flat).map_err(zerr)?,
        Array1::from(origin),
        [boundary[0], boundary[1], boundary[2]],
        cell_defined,
    )
    .map(Some)
    .map_err(|e| MolRsError::zarr(format!("box attributes are not a valid cell: {e:?}")))
}

/// Write the `box/` group with these attributes (creating it if needed).
fn store_box_attributes(
    store: &ReadableWritableListableStorage,
    attributes: serde_json::Map<String, serde_json::Value>,
) -> Result<(), MolRsError> {
    let prefix = join_path(TRAJECTORY_GROUP, BOX_GROUP);
    GroupBuilder::new()
        .attributes(attributes)
        .build(store.clone(), &prefix)?
        .store_metadata()?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Dense per-frame series: progression or array
// ---------------------------------------------------------------------------

/// A value type a dense per-frame series (`step`, `time`) holds.
trait Series: Copy + PartialEq + zarrs::array::Element + zarrs::array::ElementOwned {
    const DTYPE: DType;
    /// `start + i * stride`, or `None` when it does not fit the type.
    fn nth(start: Self, stride: Self, i: u64) -> Option<Self>;
    /// The stride from `a` to `b`, or `None` when it does not fit the type.
    fn stride_between(a: Self, b: Self) -> Option<Self>;
    fn to_json(self) -> Option<serde_json::Value>;
    fn from_json(value: &serde_json::Value) -> Option<Self>;
}

impl Series for i64 {
    const DTYPE: DType = DType::Int64;
    fn nth(start: Self, stride: Self, i: u64) -> Option<Self> {
        let i = i64::try_from(i).ok()?;
        start.checked_add(stride.checked_mul(i)?)
    }
    fn stride_between(a: Self, b: Self) -> Option<Self> {
        b.checked_sub(a)
    }
    fn to_json(self) -> Option<serde_json::Value> {
        Some(self.into())
    }
    fn from_json(value: &serde_json::Value) -> Option<Self> {
        value.as_i64()
    }
}

impl Series for f64 {
    const DTYPE: DType = DType::Float;
    fn nth(start: Self, stride: Self, i: u64) -> Option<Self> {
        Some(start + (i as f64) * stride)
    }
    fn stride_between(a: Self, b: Self) -> Option<Self> {
        Some(b - a)
    }
    fn to_json(self) -> Option<serde_json::Value> {
        // JSON has no NaN / infinity; such a series goes to an array.
        self.is_finite().then(|| self.into())
    }
    fn from_json(value: &serde_json::Value) -> Option<Self> {
        value.as_f64()
    }
}

/// An arithmetic progression `start + i * stride` recorded as a trajectory
/// group attribute — how `step` and `time` are kept while they are regular.
#[derive(Debug, Clone, Copy, PartialEq)]
struct Progression<T> {
    start: T,
    /// Unknown until the second value.
    stride: Option<T>,
}

impl<T: Series> Progression<T> {
    /// Parse the attribute form `{"start": …, "stride": …}`.
    fn from_attribute(value: &serde_json::Value) -> Option<Self> {
        let start = T::from_json(value.get("start")?)?;
        let stride = value.get("stride").and_then(T::from_json);
        Some(Self { start, stride })
    }

    fn to_attribute(self) -> Option<serde_json::Value> {
        let mut object = serde_json::Map::new();
        object.insert("start".to_string(), self.start.to_json()?);
        if let Some(stride) = self.stride {
            object.insert("stride".to_string(), stride.to_json()?);
        }
        Some(serde_json::Value::Object(object))
    }

    /// The `i`-th value, or `None` past what the progression can say.
    fn nth(&self, i: u64) -> Option<T> {
        if i == 0 {
            return Some(self.start);
        }
        T::nth(self.start, self.stride?, i)
    }

    /// The first `count` values.
    fn values(&self, count: u64) -> Result<Vec<T>, MolRsError> {
        (0..count)
            .map(|i| {
                self.nth(i).ok_or_else(|| {
                    MolRsError::zarr(format!(
                        "progression attribute cannot produce value {i}: stride unknown or overflow"
                    ))
                })
            })
            .collect()
    }
}

/// Where a dense per-frame series lives: in the trajectory group's attribute
/// while it is regular, in its own array once it is not.
enum Track<T: Series> {
    Regular {
        progression: Option<Progression<T>>,
        /// Values landed so far.
        count: u64,
    },
    Array(Box<GrowthArray>),
}

impl<T: Series> Track<T> {
    fn fresh() -> Self {
        Self::Regular {
            progression: None,
            count: 0,
        }
    }

    /// Whether appending `values` keeps the series regular, and the
    /// progression it becomes.
    fn extended(&self, values: &[T]) -> Option<Progression<T>> {
        let Self::Regular { progression, count } = self else {
            return None;
        };
        let mut progression = *progression;
        for (index, &value) in (*count..).zip(values.iter()) {
            progression = match progression {
                None => Some(Progression {
                    start: value,
                    stride: None,
                }),
                Some(Progression {
                    start,
                    stride: None,
                }) => Some(Progression {
                    start,
                    stride: Some(T::stride_between(start, value)?),
                }),
                Some(p) => {
                    if p.nth(index)? != value {
                        return None;
                    }
                    Some(p)
                }
            };
            // The attribute must be able to spell it.
            progression?.to_attribute()?;
        }
        progression
    }

    /// The last value landed.
    fn last(
        &self,
        store: &ReadableWritableListableStorage,
        path: &str,
    ) -> Result<Option<T>, MolRsError> {
        match self {
            Self::Regular { progression, count } => Ok(match (progression, *count) {
                (Some(p), n) if n > 0 => p.nth(n - 1),
                _ => None,
            }),
            Self::Array(array) => {
                if array.rows == 0 {
                    return Ok(None);
                }
                let values: Vec<T> = read_whole(store, path)?;
                Ok(values.get(array.rows as usize - 1).copied())
            }
        }
    }

    /// Land `values`: extend the progression, or materialize the array (backfilling the regular history) and append.
    fn land(
        &mut self,
        store: &ReadableWritableListableStorage,
        path: &str,
        knobs: Knobs,
        values: Vec<T>,
        options: &CodecOptions,
    ) -> Result<(), MolRsError> {
        if let Some(progression) = self.extended(&values) {
            let Self::Regular { count, .. } = self else {
                unreachable!("extended() is Some only for a regular track")
            };
            let count = *count + values.len() as u64;
            *self = Self::Regular {
                progression: Some(progression),
                count,
            };
            return Ok(());
        }
        if let Self::Regular { progression, count } = self {
            let mut array = GrowthArray::create(
                store,
                path,
                T::DTYPE,
                &[],
                knobs.dense(),
                serde_json::Map::new(),
                DENSE_COMPRESSION,
            )?;
            if *count > 0 {
                let history = progression
                    .as_ref()
                    .ok_or_else(|| MolRsError::zarr("regular track without a progression"))?
                    .values(*count)?;
                let n = history.len() as u64;
                land_values!(array, n, history, options);
            }
            *self = Self::Array(Box::new(array));
        }
        let Self::Array(array) = self else {
            unreachable!("materialized above")
        };
        let n = values.len() as u64;
        land_values!(array, n, values, options);
        Ok(())
    }

    /// The attribute this track wants on the trajectory group, if any.
    fn attribute(&self) -> Option<serde_json::Value> {
        match self {
            Self::Regular {
                progression: Some(p),
                count,
            } if *count > 0 => p.to_attribute(),
            _ => None,
        }
    }
}

/// Read a dense series back: the progression attribute when present, else
/// the array cut to `nstep`, else `None`.
fn read_track<T: Series, S>(
    store: &Arc<S>,
    attributes: &serde_json::Map<String, serde_json::Value>,
    attribute: &str,
    path: &str,
    nstep: u64,
) -> Result<Option<Vec<T>>, MolRsError>
where
    S: ?Sized + ReadableStorageTraits + 'static,
{
    if let Some(value) = attributes.get(attribute) {
        let progression = Progression::<T>::from_attribute(value).ok_or_else(|| {
            MolRsError::zarr(format!(
                "trajectory attribute {attribute:?} is not a progression"
            ))
        })?;
        return Ok(Some(progression.values(nstep)?));
    }
    if array_exists(store, path)? {
        let mut values: Vec<T> = read_whole(store, path)?;
        values.truncate(nstep as usize);
        return Ok(Some(values));
    }
    Ok(None)
}

/// The commit marker and the regular series, read off the trajectory group.
struct TrajectoryAttributes {
    nstep: Option<u64>,
    all: serde_json::Map<String, serde_json::Value>,
}

fn trajectory_attributes<S>(store: &Arc<S>) -> Result<TrajectoryAttributes, MolRsError>
where
    S: ?Sized + ReadableStorageTraits + 'static,
{
    let group = Group::open(store.clone(), TRAJECTORY_GROUP)?;
    let all = group.attributes().clone();
    let nstep = all.get(NSTEP_ATTRIBUTE).and_then(serde_json::Value::as_u64);
    Ok(TrajectoryAttributes { nstep, all })
}

/// The committed frame count: the `nstep` attribute, or — for a store from a
/// writer that kept no marker attribute — the length of the `step` array.
fn committed_frames<S>(store: &Arc<S>, attributes: &TrajectoryAttributes) -> Result<u64, MolRsError>
where
    S: ?Sized + ReadableStorageTraits + 'static,
{
    if let Some(nstep) = attributes.nstep {
        return Ok(nstep);
    }
    let path = join_path(TRAJECTORY_GROUP, STEP_ARRAY);
    if array_exists(store, &path)? {
        return Ok(Array::open(store.clone(), &path)?
            .shape()
            .first()
            .copied()
            .unwrap_or(0));
    }
    Ok(0)
}

/// Every array of one sequence, created at the first append and never
/// re-planned afterwards.
struct SequenceArrays {
    step: Track<i64>,
    time: Option<Track<f64>>,
    meta: BTreeMap<String, GrowthArray>,
    blocks: BTreeMap<String, BlockArrays>,
    cell: Option<CellStore>,
    /// Frames committed — the `nstep` attribute.
    nstep: u64,
}

impl SequenceArrays {
    /// Create every zero-length array the schema declares.
    fn create(
        store: &ReadableWritableListableStorage,
        schema: &SequenceSchema,
        knobs: Knobs,
        compression: Compression,
        with_time: bool,
    ) -> Result<Self, MolRsError> {
        let dense = |store: &ReadableWritableListableStorage,
                     path: &str,
                     dtype: DType,
                     trailing: &[u64],
                     attributes: serde_json::Map<String, serde_json::Value>| {
            GrowthArray::create(
                store,
                path,
                dtype,
                trailing,
                knobs.dense(),
                attributes,
                DENSE_COMPRESSION,
            )
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
                dense(
                    store,
                    &join_path(&join_path(TRAJECTORY_GROUP, META_GROUP), key),
                    dtype,
                    &trailing,
                    attributes,
                )?,
            );
        }

        let mut blocks = BTreeMap::new();
        for (name, declared) in &schema.blocks {
            let path = join_path(TRAJECTORY_GROUP, name);
            let hints = BlockHints::fresh();
            GroupBuilder::new()
                .attributes(block_attributes(
                    declared.structural_shape.as_deref(),
                    hints.attributes(),
                ))
                .build(store.clone(), &path)?
                .store_metadata()?;

            // Every column of a block shares one rows-per-chunk — a whole
            // number of representative frames, sized so the narrowest column
            // still reaches the byte floor — so a frame's rows stay aligned
            // across the section.
            let widths = declared
                .columns
                .values()
                .map(|column| {
                    dtype_from_tag(&column.dtype).map(|dtype| row_bytes(dtype, &column.trailing))
                })
                .collect::<Result<Vec<u64>, MolRsError>>()?;
            let narrowest = widths.iter().copied().min().unwrap_or(8);
            let rows_per_chunk = knobs.rows_per_chunk.unwrap_or_else(|| {
                Extents::block_rows_per_chunk(schema.frame_rows(name), narrowest)
            });
            let mut columns = BTreeMap::new();
            for ((column, column_schema), width) in declared.columns.iter().zip(widths) {
                let dtype = dtype_from_tag(&column_schema.dtype)?;
                let compression = if is_float_width(dtype) {
                    compression
                } else {
                    Compression::Gzip(GZIP_LEVEL)
                };
                columns.insert(
                    column.clone(),
                    GrowthArray::create(
                        store,
                        &join_path(&path, column),
                        dtype,
                        &column_schema.trailing,
                        Extents::for_column(rows_per_chunk, width, knobs.chunks_per_shard),
                        serde_json::Map::new(),
                        compression,
                    )?,
                );
            }
            blocks.insert(
                name.clone(),
                BlockArrays {
                    columns,
                    index: None,
                    total_rows: 0,
                    updates: 0,
                    rows_per_chunk,
                    hints,
                },
            );
        }

        Ok(Self {
            step: Track::fresh(),
            time: with_time.then(Track::fresh),
            meta,
            blocks,
            cell: None,
            nstep: 0,
        })
    }

    /// Reopen every array the schema declares, validate each against it, and
    /// **roll back** anything a crash left longer than the commit marker.
    ///
    /// The `nstep` attribute is the truth: every per-frame series is cut to
    /// it, every section's index to the updates at ordinals below it, and
    /// every column to the rows those updates own.
    fn open(
        store: &ReadableWritableListableStorage,
        schema: &SequenceSchema,
    ) -> Result<Self, MolRsError> {
        let attributes = trajectory_attributes(store)?;
        let nstep = committed_frames(store, &attributes)?;

        let step_path = join_path(TRAJECTORY_GROUP, STEP_ARRAY);
        let step = if let Some(value) = attributes.all.get(STEP_PROGRESSION_ATTRIBUTE) {
            Track::Regular {
                progression: Progression::<i64>::from_attribute(value),
                count: nstep,
            }
        } else if array_exists(store, &step_path)? {
            let mut array = GrowthArray::open(store, &step_path)?;
            array.truncate_to(nstep)?;
            Track::Array(Box::new(array))
        } else {
            Track::fresh()
        };
        let time_path = join_path(TRAJECTORY_GROUP, TIME_ARRAY);
        let time = if let Some(value) = attributes.all.get(TIME_PROGRESSION_ATTRIBUTE) {
            Some(Track::Regular {
                progression: Progression::<f64>::from_attribute(value),
                count: nstep,
            })
        } else if array_exists(store, &time_path)? {
            let mut array = GrowthArray::open(store, &time_path)?;
            array.truncate_to(nstep)?;
            Some(Track::Array(Box::new(array)))
        } else {
            None
        };

        let mut meta = BTreeMap::new();
        for key in schema.meta.keys() {
            let mut array = GrowthArray::open(
                store,
                &join_path(&join_path(TRAJECTORY_GROUP, META_GROUP), key),
            )?;
            array.truncate_to(nstep)?;
            meta.insert(key.clone(), array);
        }

        let mut blocks = BTreeMap::new();
        for (name, declared) in &schema.blocks {
            let path = join_path(TRAJECTORY_GROUP, name);
            let mut columns = BTreeMap::new();
            let mut rows_per_chunk = None;
            for (column, column_schema) in &declared.columns {
                let column_path = join_path(&path, column);
                let opened = GrowthArray::open(store, &column_path)?;
                validate_column(&opened, &column_path, column_schema)?;
                rows_per_chunk.get_or_insert_with(|| opened.rows_per_chunk());
                columns.insert(column.clone(), opened);
            }
            let stored = stored_hints(store, &path)?;

            let (index, updates, total_rows, mut hints) =
                if array_exists(store, &join_path(&path, STEP_INDEX_ARRAY))? {
                    let mut step_index =
                        GrowthArray::open(store, &join_path(&path, STEP_INDEX_ARRAY))?;
                    let mut offset = GrowthArray::open(store, &join_path(&path, OFFSET_ARRAY))?;
                    let ordinals =
                        read_whole::<u64, _>(store, &join_path(&path, STEP_INDEX_ARRAY))?;
                    let updates = ordinals
                        .iter()
                        .take_while(|&&ordinal| ordinal < nstep)
                        .count();
                    step_index.truncate_to(updates as u64)?;
                    offset.truncate_to(if updates == 0 { 0 } else { updates as u64 + 1 })?;
                    let mut offsets = read_whole::<u64, _>(store, &join_path(&path, OFFSET_ARRAY))?;
                    offsets.truncate(if updates == 0 { 0 } else { updates + 1 });
                    let total_rows = offsets.last().copied().unwrap_or(0);
                    (
                        Some(IndexArrays { offset, step_index }),
                        updates as u64,
                        total_rows,
                        BlockHints::from_index(&ordinals[..updates], &offsets),
                    )
                } else {
                    match stored {
                        (Some(rows), true) if rows > 0 && !columns.is_empty() => {
                            let landed = columns.values().map(|c| c.rows).min().unwrap_or(0);
                            let updates = (landed / rows).min(nstep);
                            let mut hints = BlockHints::fresh();
                            for update in 0..updates {
                                hints.observe(update, update, rows);
                            }
                            (None, updates, updates * rows, hints)
                        }
                        // No committed update: whatever the columns hold is
                        // a torn landing.
                        _ => (None, 0, 0, BlockHints::fresh()),
                    }
                };
            for column in columns.values_mut() {
                column.truncate_to(total_rows)?;
            }
            hints.stored = stored;
            let wanted = if index.is_some() {
                (None, false)
            } else {
                hints.attributes()
            };
            if wanted != hints.stored {
                store_block_attributes(store, &path, declared.structural_shape.as_deref(), wanted)?;
                hints.stored = wanted;
            }

            blocks.insert(
                name.clone(),
                BlockArrays {
                    columns,
                    index,
                    total_rows,
                    updates,
                    rows_per_chunk: rows_per_chunk.unwrap_or(1),
                    hints,
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
            let mut arrays = BoxArrays {
                step_index: optional(STEP_INDEX_ARRAY)?,
                vectors: GrowthArray::open(store, &join_path(&box_prefix, VECTORS_ARRAY))?,
                origin: optional(ORIGIN_ARRAY)?,
                boundary: optional(BOUNDARY_ARRAY)?,
            };
            let committed = match &arrays.step_index {
                Some(_) => read_whole::<u64, _>(store, &join_path(&box_prefix, STEP_INDEX_ARRAY))?
                    .iter()
                    .take_while(|&&ordinal| ordinal < nstep)
                    .count() as u64,
                // The omitted index is one update at ordinal 0.
                None => u64::from(arrays.vectors.rows > 0 && nstep > 0),
            };
            arrays.truncate_to(committed)?;
            Some(CellStore::Arrays(Box::new(arrays)))
        } else if group_exists(store, &box_prefix)? {
            let group = Group::open(store.clone(), &box_prefix)?;
            match cell_from_attributes(group.attributes())? {
                Some(cell) if nstep > 0 => Some(CellStore::Attributes(Box::new(cell))),
                _ => None,
            }
        } else {
            None
        };

        Ok(Self {
            step,
            time,
            meta,
            blocks,
            cell,
            nstep,
        })
    }

    /// The trajectory group attributes a commit publishes: the pin, the
    /// marker, and the regular series.
    fn store_marker(
        &self,
        store: &ReadableWritableListableStorage,
        nstep: u64,
    ) -> Result<(), MolRsError> {
        let mut group = Group::open(store.clone(), TRAJECTORY_GROUP)?;
        let attributes = group.attributes_mut();
        attributes.insert(NSTEP_ATTRIBUTE.to_string(), nstep.into());
        match self.step.attribute() {
            Some(value) => attributes.insert(STEP_PROGRESSION_ATTRIBUTE.to_string(), value),
            None => attributes.remove(STEP_PROGRESSION_ATTRIBUTE),
        };
        match self.time.as_ref().and_then(Track::attribute) {
            Some(value) => attributes.insert(TIME_PROGRESSION_ATTRIBUTE.to_string(), value),
            None => attributes.remove(TIME_PROGRESSION_ATTRIBUTE),
        };
        group.store_metadata()?;
        Ok(())
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

// ---------------------------------------------------------------------------
// Writer
// ---------------------------------------------------------------------------

/// One appended frame, buffered until a landing writes it.
struct PendingFrame {
    step: i64,
    time: Option<F>,
    /// Sections this frame changed. A block omitted by the frame is not here:
    /// it carries forward. A present block with zero rows is here, as the
    /// zero-row update that means present-and-empty.
    blocks: BTreeMap<String, Block>,
    /// The cell, when it changed.
    cell: Option<SimBox>,
    /// Every declared meta key, resolved to a value or its declared fill.
    meta: BTreeMap<String, MetaValue>,
}

/// The streaming producer of a frame sequence: one frame per
/// [`append`](Self::append); complete inner chunks land on their own,
/// [`flush`](Self::flush) commits the rest, `close(self)` ends the run.
///
/// One of the three access forms of a single object — the eager in-memory
/// carrier is `Trajectory`, the lazy store cursor is [`FrameSequence`], and
/// this is the writer that produces what that cursor reads.
///
/// **No `Drop`.** Closing is [`close`](Self::close), which consumes the writer
/// and can therefore return the IO error a `Drop` would have had to swallow —
/// and a scientific writer that swallows an IO error is silent data loss.
///
/// **What lands when.** Every append is checked and buffered. When the buffer
/// reaches the landing cadence — derived from the frame size so that block
/// columns land whole inner chunks and roughly 4 MiB at a
/// time, or set by [`with_flush_every`](Self::with_flush_every) — the buffered
/// frames are committed. [`flush`](Self::flush) commits at any time and is
/// **durable** by default (the touched files are synced before the commit
/// marker moves); the automatic landings are not synced.
///
/// **Error vocabulary.** Every door on this type yields [`MolRsError`], and the
/// storage-backed ones arrive as the [`MolRsError::Zarr`] variant carrying a
/// message that names the block, column, metadata key or array path that
/// disagreed.
pub struct FrameSequenceWriter {
    store: ReadableWritableListableStorage,
    /// The concrete positional-write store behind `store`, when this writer
    /// was opened by path — the only store that can be asked to sync.
    #[cfg(feature = "filesystem")]
    positional: Option<Arc<crate::io::zarr::store::PositionalWriteStore>>,
    schema: SequenceSchema,
    knobs: Knobs,
    compression: Compression,
    /// Landing cadence in frames; `None` derives it at the first append.
    flush_every: Option<u64>,
    /// Whether an explicit `flush` / `close` syncs the touched files.
    durable: bool,
    /// `None` until the first append freezes the extents and creates them.
    arrays: Option<SequenceArrays>,
    /// The landing cadence in force, fixed at the first append.
    auto_flush_every: u64,
    /// Highest step number landed or buffered.
    last_step: Option<i64>,
    /// Whether this run carries times. Fixed by the first append.
    uses_time: Option<bool>,
    /// Frames committed.
    committed: u64,
    pending: Vec<PendingFrame>,
    /// Last content landed or buffered per section, for the change detection
    /// that keeps an unchanging section at one `step_index` entry.
    landed_blocks: BTreeMap<String, Block>,
    landed_cell: Option<SimBox>,
}

/// What a writer knows about the sequence it is attached to: nothing for a
/// fresh mint, the recovered state for a reopen.
struct Attached {
    arrays: Option<SequenceArrays>,
    last_step: Option<i64>,
    uses_time: Option<bool>,
    committed: u64,
    landed_blocks: BTreeMap<String, Block>,
    landed_cell: Option<SimBox>,
}

impl Attached {
    /// A sequence with no arrays yet: created, never appended.
    fn fresh() -> Self {
        Self {
            arrays: None,
            last_step: None,
            uses_time: None,
            committed: 0,
            landed_blocks: BTreeMap::new(),
            landed_cell: None,
        }
    }
}

impl FrameSequenceWriter {
    fn assemble(
        store: ReadableWritableListableStorage,
        #[cfg(feature = "filesystem")] positional: Option<
            Arc<crate::io::zarr::store::PositionalWriteStore>,
        >,
        schema: SequenceSchema,
        attached: Attached,
    ) -> Self {
        let mut writer = Self {
            store,
            #[cfg(feature = "filesystem")]
            positional,
            schema,
            knobs: Knobs::default(),
            compression: Compression::None,
            flush_every: None,
            durable: true,
            arrays: attached.arrays,
            auto_flush_every: 1,
            last_step: attached.last_step,
            uses_time: attached.uses_time,
            committed: attached.committed,
            pending: Vec::new(),
            landed_blocks: attached.landed_blocks,
            landed_cell: attached.landed_cell,
        };
        if writer.arrays.is_some() {
            writer.auto_flush_every = writer.derive_flush_every();
        }
        writer
    }

    /// Mint a new sequence at `trajectory/` and pin `schema` to it.
    ///
    /// Writes the record root and its `meta/` group when the store has none,
    /// then the `trajectory/` group with its schema attribute — and nothing
    /// else: the arrays are created by the first [`append`](Self::append). A
    /// store that already holds a sequence is an `Err` naming it, never a
    /// silent overwrite.
    ///
    /// **`trajectory/` is cleared first.** Zarr stores are written key by key,
    /// so a mint that only wrote its own keys would inherit every leftover
    /// child of whatever stood at that node. Reopen with [`open`](Self::open)
    /// instead when the store may already hold data.
    ///
    /// # Errors
    ///
    /// A [`MolRsError::Zarr`] naming `trajectory/step` when that array already
    /// exists, so nothing is erased and nothing is written. Otherwise any
    /// storage error raised while writing the groups.
    pub fn create(
        store: ReadableWritableListableStorage,
        schema: SequenceSchema,
    ) -> Result<Self, MolRsError> {
        Self::create_with(
            store,
            #[cfg(feature = "filesystem")]
            None,
            schema,
        )
    }

    fn create_with(
        store: ReadableWritableListableStorage,
        #[cfg(feature = "filesystem")] positional: Option<
            Arc<crate::io::zarr::store::PositionalWriteStore>,
        >,
        schema: SequenceSchema,
    ) -> Result<Self, MolRsError> {
        if group_exists(&store, TRAJECTORY_GROUP)? {
            let attributes = trajectory_attributes(&store)?;
            if attributes.nstep.is_some()
                || array_exists(&store, &join_path(TRAJECTORY_GROUP, STEP_ARRAY))?
            {
                return Err(MolRsError::zarr(format!(
                    "a frame sequence already exists at {TRAJECTORY_GROUP:?}; refusing to \
                     overwrite it"
                )));
            }
        }
        ensure_root_and_meta(&store, None)?;
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
        Ok(Self::assemble(
            store,
            #[cfg(feature = "filesystem")]
            positional,
            schema,
            Attached::fresh(),
        ))
    }

    /// Mint a new sequence at `path` — the path-taking door over the fast
    /// positional-write store, which is also the store that makes
    /// [`flush`](Self::flush) durable.
    ///
    /// # Errors
    ///
    /// The store-root errors of the positional store, then every error
    /// [`create`](Self::create) can raise.
    #[cfg(feature = "filesystem")]
    pub fn create_at(
        path: impl AsRef<std::path::Path>,
        schema: SequenceSchema,
    ) -> Result<Self, MolRsError> {
        let positional = Arc::new(crate::io::zarr::store::PositionalWriteStore::new(path)?);
        let store: ReadableWritableListableStorage = positional.clone();
        Self::create_with(store, Some(positional), schema)
    }

    /// Reattach to the sequence at `path` and continue appending — the
    /// path-taking counterpart of [`open`](Self::open).
    ///
    /// # Errors
    ///
    /// The store-root errors of the positional store, then every error
    /// [`open`](Self::open) can raise.
    #[cfg(feature = "filesystem")]
    pub fn open_at(path: impl AsRef<std::path::Path>) -> Result<Self, MolRsError> {
        let positional = Arc::new(crate::io::zarr::store::PositionalWriteStore::new(path)?);
        let store: ReadableWritableListableStorage = positional.clone();
        Self::open_with(store, Some(positional))
    }

    /// Reattach to an existing sequence and continue appending to it.
    ///
    /// The schema comes off the group attributes and every array is checked
    /// against it; the extents come off the arrays themselves. Whatever a
    /// crash left longer than the commit marker is rolled back first, so the
    /// next append continues from the last committed frame.
    ///
    /// # Errors
    ///
    /// A [`MolRsError::Zarr`] when the store holds the `trajectory/frames/<i>/`
    /// layout written by molrs <= 0.13; when `trajectory/` carries no
    /// `sequence_schema` attribute (a foreign store can be *read* without the
    /// pin but not appended to); or when a reopened column array disagrees
    /// with the pinned schema on dtype or trailing shape.
    pub fn open(store: ReadableWritableListableStorage) -> Result<Self, MolRsError> {
        Self::open_with(
            store,
            #[cfg(feature = "filesystem")]
            None,
        )
    }

    fn open_with(
        store: ReadableWritableListableStorage,
        #[cfg(feature = "filesystem")] positional: Option<
            Arc<crate::io::zarr::store::PositionalWriteStore>,
        >,
    ) -> Result<Self, MolRsError> {
        ensure_not_legacy(&store)?;
        let schema = schema_of(&store)?;
        if trajectory_attributes(&store)?.nstep.is_none()
            && !array_exists(&store, &join_path(TRAJECTORY_GROUP, STEP_ARRAY))?
        {
            // Created but never appended: nothing is frozen yet.
            return Ok(Self::assemble(
                store,
                #[cfg(feature = "filesystem")]
                positional,
                schema,
                Attached::fresh(),
            ));
        }

        let arrays = SequenceArrays::open(&store, &schema)?;
        let committed = arrays.nstep;
        let last_step = arrays
            .step
            .last(&store, &join_path(TRAJECTORY_GROUP, STEP_ARRAY))?;
        let uses_time = Some(arrays.time.is_some());

        // Recover the change-detection state, so a section that does not move
        // across the reopen still costs one `step_index` entry rather than two.
        let mut landed_blocks = BTreeMap::new();
        for (name, block) in &arrays.blocks {
            if block.updates == 0 {
                continue;
            }
            let path = join_path(TRAJECTORY_GROUP, name);
            let end = block.total_rows;
            let start = match &block.index {
                Some(_) => {
                    let offsets = read_whole::<u64, _>(&store, &join_path(&path, OFFSET_ARRAY))?;
                    offsets
                        .get(block.updates as usize - 1)
                        .copied()
                        .unwrap_or(end)
                }
                None => match block.hints.uniform {
                    Uniform::Rows(rows) => end.saturating_sub(rows),
                    _ => end,
                },
            };
            let declared = &schema.blocks[name];
            let landed = if end > start {
                let mut landed = Block::new();
                for (column, column_schema) in &declared.columns {
                    let subset = rows_subset(start, end - start, &column_schema.trailing)?;
                    let array = &block.columns[column].array;
                    insert_column_into_block(
                        &mut landed,
                        column,
                        read_column_array(array, &subset)?,
                    )?;
                }
                apply_structural_shape(&mut landed, declared, &path)?;
                landed
            } else {
                empty_block(declared)?
            };
            landed_blocks.insert(name.clone(), landed);
        }
        let landed_cell = match arrays.cell.as_ref() {
            Some(CellStore::Attributes(cell)) => Some((**cell).clone()),
            Some(CellStore::Arrays(section)) if section.vectors.rows > 0 => {
                let reader: ReadableListableStorage = Arc::new(StorageHandle::new(store.clone()));
                let mut boxes = BoxReader::open(&reader)?;
                Some(boxes.cell_at(section.vectors.rows - 1, cell_defined_of(&store)?)?)
            }
            _ => None,
        };

        Ok(Self::assemble(
            store,
            #[cfg(feature = "filesystem")]
            positional,
            schema,
            Attached {
                arrays: Some(arrays),
                last_step,
                uses_time,
                committed,
                landed_blocks,
                landed_cell,
            },
        ))
    }

    /// Rows one inner chunk of every growth array holds — a test knob.
    ///
    /// Legal only before the first append, which freezes the extents. Left
    /// unset, block columns are frame-aligned and dense arrays take
    /// [`DENSE_ROWS_PER_CHUNK`].
    #[cfg(test)]
    pub(crate) fn with_rows_per_chunk(mut self, rows: u64) -> Result<Self, MolRsError> {
        self.refuse_after_first_append("with_rows_per_chunk")?;
        if rows == 0 {
            return Err(MolRsError::zarr("rows_per_chunk must be at least 1"));
        }
        self.knobs.rows_per_chunk = Some(rows);
        Ok(self)
    }

    /// Inner chunks one shard file holds — a test knob.
    ///
    /// Legal only before the first append.
    #[cfg(test)]
    pub(crate) fn with_chunks_per_shard(mut self, chunks: u64) -> Result<Self, MolRsError> {
        self.refuse_after_first_append("with_chunks_per_shard")?;
        if chunks == 0 {
            return Err(MolRsError::zarr("chunks_per_shard must be at least 1"));
        }
        self.knobs.chunks_per_shard = Some(chunks);
        Ok(self)
    }

    /// How the floating-point columns are compressed. Legal only before the
    /// first append, which freezes every array's codecs.
    ///
    /// # Errors
    ///
    /// A [`MolRsError::Zarr`] when the first append has already run.
    pub fn with_compression(mut self, compression: Compression) -> Result<Self, MolRsError> {
        self.refuse_after_first_append("with_compression")?;
        self.compression = compression;
        Ok(self)
    }

    /// Land every `frames` appended frames, instead of the derived cadence.
    ///
    /// A cadence that is a whole multiple of the frames one inner chunk holds
    /// lands whole chunks; any other cadence re-encodes a partial trailing
    /// chunk per landing (bounded, but real). `1` lands every frame.
    ///
    /// # Errors
    ///
    /// A [`MolRsError::Zarr`] when `frames` is 0.
    pub fn with_flush_every(mut self, frames: u64) -> Result<Self, MolRsError> {
        if frames == 0 {
            return Err(MolRsError::zarr("flush_every must be at least 1"));
        }
        self.flush_every = Some(frames);
        if self.arrays.is_some() {
            self.auto_flush_every = frames;
        }
        Ok(self)
    }

    /// Whether an explicit [`flush`](Self::flush) / [`close`](Self::close)
    /// syncs the touched files before moving the commit marker. `true` by
    /// default; `false` trades power-loss safety for throughput on a store
    /// where every flush is a checkpoint anyway.
    pub fn with_durable(mut self, durable: bool) -> Self {
        self.durable = durable;
        self
    }

    /// Write `meta` as the record's identity document (`meta/` attributes).
    ///
    /// Replaces whatever the group held. A record needs no particular key
    /// here during development; `molrec_version`, when present, must be a
    /// positive integer.
    ///
    /// # Errors
    ///
    /// Any storage error raised while writing the group.
    pub fn with_meta(
        self,
        meta: &serde_json::Map<String, serde_json::Value>,
    ) -> Result<Self, MolRsError> {
        ensure_root_and_meta(&self.store, Some(meta))?;
        Ok(self)
    }

    /// The landing cadence in force: frames per automatic commit.
    pub fn flush_every(&self) -> u64 {
        self.auto_flush_every
    }

    /// Frames committed so far.
    pub fn committed(&self) -> u64 {
        self.committed
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

    /// The landing cadence: enough frames for 4 MiB of
    /// payload, rounded up to a whole number of the coarsest block's
    /// frames-per-chunk so every block column lands whole chunks.
    fn derive_flush_every(&self) -> u64 {
        if let Some(frames) = self.flush_every {
            return frames;
        }
        let frame_bytes = self.schema.frame_bytes().max(1);
        let base = (FLUSH_TARGET_BYTES / frame_bytes).clamp(1, MAX_FLUSH_EVERY);
        let coarsest = self
            .arrays
            .as_ref()
            .map(|arrays| {
                arrays
                    .blocks
                    .iter()
                    .filter_map(|(name, block)| {
                        let rows = self.schema.frame_rows(name);
                        (rows > 0).then(|| (block.rows_per_chunk / rows).max(1))
                    })
                    .max()
                    .unwrap_or(1)
            })
            .unwrap_or(1);
        base.div_ceil(coarsest) * coarsest
    }

    /// Buffer `frame` at the next step number, with no time.
    ///
    /// Step numbers start at 0 and rise by one — or, after
    /// [`open`](Self::open) has reattached to an existing sequence, continue
    /// from the last step already landed. A run that carries real MD step
    /// numbers or times uses [`append_at`](Self::append_at) instead.
    ///
    /// # Errors
    ///
    /// [`append_at`](Self::append_at)'s.
    pub fn append(&mut self, frame: &Frame) -> Result<(), MolRsError> {
        self.append_at(frame, self.last_step.map_or(0, |step| step + 1), None)
    }

    /// Buffer `frame` at the next step number, carrying a physical time.
    ///
    /// # Errors
    ///
    /// [`append_at`](Self::append_at)'s.
    pub fn append_timed(&mut self, frame: &Frame, time: F) -> Result<(), MolRsError> {
        self.append_at(frame, self.last_step.map_or(0, |step| step + 1), Some(time))
    }

    /// Buffer `frame` at an explicit step number and optional time.
    ///
    /// `step` is the run's own dimensionless step counter. It must be strictly
    /// greater than the previous one. `time` is the frame's physical time in
    /// **femtoseconds (fs)**, stored verbatim, and all-or-nothing across a
    /// run: the first append decides whether the `time` array exists.
    ///
    /// What gets buffered is only what *changed*. A block whose bytes are
    /// identical to its previous update earns no new update; a block the frame
    /// **omits** earns none either and carries forward; a block presented with
    /// zero rows earns a zero-row update (present and empty). The very first
    /// append creates every array the schema declares, freezing the extents.
    /// When the buffer reaches the landing cadence it is committed here.
    ///
    /// # Errors
    ///
    /// Every case is a [`MolRsError::Zarr`] naming what disagreed:
    ///
    /// - `step` is not strictly greater than the previous step;
    /// - `time` is supplied on a run started without times, or omitted on a run
    ///   started with them;
    /// - the frame carries a block, a column, or a `meta` key the pinned schema
    ///   does not declare;
    /// - a column or `meta` key is declared with one dtype and this frame
    ///   carries another, or a column's trailing shape disagrees;
    /// - a block is present but omits one of its declared columns: sparsity is
    ///   per block, not per column;
    /// - a block with a declared structural shape carries another row count;
    /// - the frame omits a declared `meta` key for which no fill was declared.
    ///
    /// Also any storage error raised by an automatic landing.
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
            // An omitted block carries forward: no update, no state change.
            let Some(block) = frame.get(name) else {
                continue;
            };
            let changed = self
                .landed_blocks
                .get(name)
                .is_none_or(|landed| !same_block(landed, block));
            if changed {
                blocks.insert(name.clone(), block.clone());
                self.landed_blocks.insert(name.clone(), block.clone());
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
            // The cell carries forward like any other section.
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
        if self.pending.len() as u64 >= self.auto_flush_every {
            self.commit(false)?;
        }
        Ok(())
    }

    /// Check `frame` against the pinned schema.
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
            if let Some(shape) = &declared.structural_shape {
                let expected = shape.iter().product::<usize>();
                if block.nrows() != Some(expected) {
                    return Err(MolRsError::zarr(format!(
                        "block {name:?} declares structural shape {shape:?} ({expected} rows) but \
                         this frame carries {} rows: a shaped block keeps its row count",
                        block.nrows().unwrap_or(0)
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
                Some(value) if value.dtype() == declared.dtype => value.clone(),
                // A value that arrived at another width — a Python float for a
                // declared `f32`, a JSON list for a declared `f64x3` — is
                // re-read at the declared width, exactly; a value that cannot
                // be is the error.
                Some(value) => MetaValue::from_json_value(&serde_json::json!({
                    "dtype": declared.dtype,
                    "value": value.to_attr_value(),
                }))
                .map_err(|e| {
                    MolRsError::zarr(format!(
                        "meta key {key:?} is declared {} but this frame carries {}: {e}",
                        declared.dtype,
                        value.dtype()
                    ))
                })?,
                None => {
                    let fill = declared.fill.as_ref().ok_or_else(|| {
                        MolRsError::zarr(format!(
                            "this frame omits the declared meta key {key:?} and no fill value was \
                             declared for it; there is no implicit fill"
                        ))
                    })?;
                    // The pin stores the plain value; the declared tag says
                    // how to read it back.
                    MetaValue::from_json_value(&serde_json::json!({
                        "dtype": declared.dtype,
                        "value": fill,
                    }))
                    .map_err(|e| MolRsError::zarr(format!("meta key {key:?} fill: {e}")))?
                }
            };
            resolved.insert(key.clone(), value);
        }
        Ok(resolved)
    }

    /// Create every array on the first append, freezing the extents and the
    /// landing cadence.
    fn ensure_arrays(&mut self) -> Result<(), MolRsError> {
        if self.arrays.is_none() {
            self.arrays = Some(SequenceArrays::create(
                &self.store,
                &self.schema,
                self.knobs,
                self.compression,
                self.uses_time == Some(true),
            )?);
            self.auto_flush_every = self.derive_flush_every();
        }
        Ok(())
    }

    /// Commit every buffered frame, durably.
    ///
    /// Every frame appended before this call lands, whether or not it fills a
    /// chunk. `step` is extended **last** and is the commit marker: a reader
    /// sees `nstep` frames only once this call has written it. Unless
    /// [`with_durable(false)`](Self::with_durable) was set, the touched files
    /// are synced before the marker moves and again after it, so a returned
    /// `Ok` means the frames survive a power loss.
    ///
    /// A flush that lands a partially filled trailing inner chunk re-encodes
    /// that chunk and leaves its superseded copy in the shard as dead bytes —
    /// at most one chunk per column per flush. A flush with nothing buffered
    /// is `Ok(())` and writes nothing.
    ///
    /// # Errors
    ///
    /// A [`MolRsError::Zarr`] wrapping whatever the store or the codec raised.
    /// A flush that fails partway leaves arrays longer than `step`; such a
    /// store reads back as the previous commit, and a reopened writer rolls
    /// the excess back.
    pub fn flush(&mut self) -> Result<(), MolRsError> {
        self.commit(self.durable)
    }

    /// Land every buffered frame; `durable` syncs the touched files around
    /// the commit marker.
    fn commit(&mut self, durable: bool) -> Result<(), MolRsError> {
        if self.pending.is_empty() {
            return Ok(());
        }
        let options = partial_encoding_options();
        let Self {
            store,
            schema,
            knobs,
            arrays,
            committed,
            pending,
            ..
        } = self;
        let Some(arrays) = arrays.as_mut() else {
            return Ok(());
        };
        let base = *committed;

        // 1. Block sections: columns first, then — for a block that is no
        //    longer regular — its own index, then its hints.
        for (name, declared) in &schema.blocks {
            let Some(section) = arrays.blocks.get_mut(name) else {
                continue;
            };
            let updates: Vec<(u64, &Block)> = pending
                .iter()
                .enumerate()
                .filter_map(|(index, frame)| {
                    frame
                        .blocks
                        .get(name)
                        .map(|update| (base + index as u64, update))
                })
                .collect();
            if updates.is_empty() {
                continue;
            }

            let landed: Vec<&Block> = updates.iter().map(|(_, block)| *block).collect();
            let added: u64 = landed
                .iter()
                .map(|block| block.nrows().unwrap_or(0) as u64)
                .sum();
            if added > 0 {
                for (column, pinned) in &declared.columns {
                    let Some(array) = section.columns.get_mut(column) else {
                        continue;
                    };
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
                    let subset = array.reserve(added)?;
                    array.store_columns(
                        &subset,
                        dtype_from_tag(&pinned.dtype)?,
                        &values,
                        &options,
                    )?;
                    array.commit_shape()?;
                }
            }

            let mut hints = section.hints;
            for (k, (ordinal, block)) in updates.iter().enumerate() {
                hints.observe(
                    section.updates + k as u64,
                    *ordinal,
                    block.nrows().unwrap_or(0) as u64,
                );
            }
            let regular = hints.is_regular() && !declared.columns.is_empty();

            let mut running = section.total_rows;
            let mut offsets = Vec::with_capacity(updates.len());
            for (_, update) in &updates {
                running += update.nrows().unwrap_or(0) as u64;
                offsets.push(running);
            }
            let ordinals: Vec<u64> = updates.iter().map(|(ordinal, _)| *ordinal).collect();

            if !regular {
                if section.index.is_none() {
                    let history_rows = match section.hints.uniform {
                        Uniform::Rows(rows) => rows,
                        _ => 0,
                    };
                    section.index = Some(IndexArrays::materialize(
                        store,
                        &join_path(TRAJECTORY_GROUP, name),
                        *knobs,
                        section.updates,
                        history_rows,
                        &options,
                    )?);
                }
                let index = section.index.as_mut().expect("materialized above");
                let count = offsets.len() as u64;
                land_values!(index.offset, count, offsets, &options);
                let count = ordinals.len() as u64;
                land_values!(index.step_index, count, ordinals, &options);
            }

            section.hints = hints;
            section.updates += updates.len() as u64;
            section.total_rows = running;

            // Once the index arrays exist they are authoritative: the hints
            // are withdrawn for good.
            let wanted = if section.index.is_some() {
                (None, false)
            } else {
                section.hints.attributes()
            };
            if wanted != section.hints.stored {
                store_block_attributes(
                    store,
                    &join_path(TRAJECTORY_GROUP, name),
                    declared.structural_shape.as_deref(),
                    wanted,
                )?;
                section.hints.stored = wanted;
            }
        }

        // 2. The cell: attributes while it is one fixed cell from ordinal 0,
        //    arrays from the first change on.
        let cells: Vec<(u64, &SimBox)> = pending
            .iter()
            .enumerate()
            .filter_map(|(index, frame)| {
                frame.cell.as_ref().map(|cell| (base + index as u64, cell))
            })
            .collect();
        if let Some((first_ordinal, first)) = cells.first().copied() {
            match arrays.cell.take() {
                None if cells.len() == 1 && first_ordinal == 0 => {
                    store_box_attributes(
                        store,
                        box_attributes(first.is_cell_defined(), Some(first)),
                    )?;
                    arrays.cell = Some(CellStore::Attributes(Box::new(first.clone())));
                }
                None => {
                    store_box_attributes(store, box_attributes(first.is_cell_defined(), None))?;
                    let mut section = BoxArrays::create(store, *knobs)?;
                    section.land(store, *knobs, &cells, &options)?;
                    arrays.cell = Some(CellStore::Arrays(Box::new(section)));
                }
                Some(CellStore::Attributes(fixed)) => {
                    let mut section = BoxArrays::create(store, *knobs)?;
                    let mut all: Vec<(u64, &SimBox)> = Vec::with_capacity(cells.len() + 1);
                    all.push((0, fixed.as_ref()));
                    all.extend(cells.iter().copied());
                    section.land(store, *knobs, &all, &options)?;
                    // The arrays now carry the cell; the attributes keep only
                    // `cell_defined`.
                    store_box_attributes(store, box_attributes(fixed.is_cell_defined(), None))?;
                    arrays.cell = Some(CellStore::Arrays(Box::new(section)));
                }
                Some(CellStore::Arrays(mut section)) => {
                    section.land(store, *knobs, &cells, &options)?;
                    arrays.cell = Some(CellStore::Arrays(section));
                }
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
            let subset = array.reserve(values.len() as u64)?;
            array.store_meta(&subset, key, &declared.dtype, &values, &options)?;
            array.commit_shape()?;
        }

        // 4. Time and step: a progression attribute while they are regular,
        //    an array once they are not.
        if let Some(track) = arrays.time.as_mut() {
            let times: Vec<F> = pending
                .iter()
                .map(|frame| frame.time.unwrap_or(0.0))
                .collect();
            track.land(
                store,
                &join_path(TRAJECTORY_GROUP, TIME_ARRAY),
                *knobs,
                times,
                &options,
            )?;
        }
        let steps: Vec<i64> = pending.iter().map(|frame| frame.step).collect();
        arrays.step.land(
            store,
            &join_path(TRAJECTORY_GROUP, STEP_ARRAY),
            *knobs,
            steps,
            &options,
        )?;

        // 5. Everything above is on disk (and, when durable, synced) before
        //    the commit marker moves.
        #[cfg(feature = "filesystem")]
        if durable && let Some(positional) = &self.positional {
            positional.sync_dirty()?;
        }
        #[cfg(not(feature = "filesystem"))]
        let _ = durable;

        // 6. The commit marker, last: the trajectory group's `nstep`
        //    attribute, one atomic metadata replace.
        let nstep = base + pending.len() as u64;
        arrays.store_marker(store, nstep)?;
        arrays.nstep = nstep;

        #[cfg(feature = "filesystem")]
        if durable && let Some(positional) = &self.positional {
            positional.sync_dirty()?;
        }

        *committed += pending.len() as u64;
        pending.clear();
        Ok(())
    }

    /// Commit whatever is still buffered and consume the writer.
    ///
    /// Consuming rather than dropping is the point: this is the only place an
    /// IO error at the end of a run can still be returned to the caller. No
    /// shard is rewritten on close — the store's bytes are exactly what the
    /// landings wrote.
    ///
    /// # Errors
    ///
    /// [`flush`](Self::flush)'s. The writer is consumed either way; whatever
    /// had already been committed stays readable.
    pub fn close(mut self) -> Result<(), MolRsError> {
        self.commit(self.durable)
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
enum BlockIndex {
    /// The general case: the section's own `step_index` and CSR `offset`.
    Sparse {
        /// Frame ordinals at which the section was updated, ascending.
        step_index: Vec<u64>,
        /// CSR row pointer, one longer than `step_index`.
        offset: Vec<u64>,
    },
    /// The hinted case (`dense_updates` + `uniform_rows`): update `j` sits at
    /// ordinal `j` and owns rows `j*rows..(j+1)*rows`, so nothing was read.
    Regular {
        /// Updates committed.
        updates: u64,
        /// Rows every update carries.
        rows: u64,
    },
}

impl BlockIndex {
    /// `(update, first row, rows)` of the section at frame `index`, or `None`
    /// when the section has not appeared yet.
    fn resolve(&self, index: u64) -> Result<Option<(u64, u64, u64)>, MolRsError> {
        match self {
            Self::Sparse { step_index, offset } => {
                let Some(update) = latest_update(step_index, index) else {
                    return Ok(None);
                };
                let (Some(&start), Some(&end)) = (offset.get(update), offset.get(update + 1))
                else {
                    return Err(MolRsError::zarr(format!(
                        "block section has {} offsets for update {update}",
                        offset.len()
                    )));
                };
                // A hostile or corrupt store can carry a non-monotonic
                // `offset` array; `end - start` would then wrap to a
                // near-`u64::MAX` row count and drive an unbounded allocation.
                let count = end.checked_sub(start).ok_or_else(|| {
                    MolRsError::zarr(format!(
                        "block section update {update} has non-monotonic offsets ({start} > {end})"
                    ))
                })?;
                Ok(Some((update as u64, start, count)))
            }
            Self::Regular { updates, rows } => {
                if *updates == 0 {
                    return Ok(None);
                }
                let update = index.min(updates - 1);
                Ok(Some((update, update * rows, *rows)))
            }
        }
    }
}

/// The index of the `box/` section.
struct BoxIndex {
    step_index: Vec<u64>,
    cell_defined: bool,
}

/// The `box/` section's cells: the fixed cell from the group attributes, or
/// the per-update arrays, with a per-update cache.
struct BoxReader {
    fixed: Option<SimBox>,
    vectors: Option<Array<dyn ReadableListableStorageTraits>>,
    origin: Option<Array<dyn ReadableListableStorageTraits>>,
    boundary: Option<Array<dyn ReadableListableStorageTraits>>,
    cells: BTreeMap<u64, SimBox>,
}

impl BoxReader {
    fn open(store: &ReadableListableStorage) -> Result<Self, MolRsError> {
        let prefix = join_path(TRAJECTORY_GROUP, BOX_GROUP);
        let optional =
            |name: &str| -> Result<Option<Array<dyn ReadableListableStorageTraits>>, MolRsError> {
                let path = join_path(&prefix, name);
                if array_exists(store, &path)? {
                    Ok(Some(Array::open(store.clone(), &path)?))
                } else {
                    Ok(None)
                }
            };
        let vectors = optional(VECTORS_ARRAY)?;
        let fixed = if vectors.is_none() {
            cell_from_attributes(Group::open(store.clone(), &prefix)?.attributes())?
        } else {
            None
        };
        Ok(Self {
            fixed,
            vectors,
            origin: optional(ORIGIN_ARRAY)?,
            boundary: optional(BOUNDARY_ARRAY)?,
            cells: BTreeMap::new(),
        })
    }

    /// Cell update `index`, decoded once.
    fn cell_at(&mut self, index: u64, cell_defined: bool) -> Result<SimBox, MolRsError> {
        if let Some(fixed) = &self.fixed {
            return Ok(fixed.clone());
        }
        if let Some(cell) = self.cells.get(&index) {
            return Ok(cell.clone());
        }
        let vectors = self
            .vectors
            .as_ref()
            .ok_or_else(|| MolRsError::zarr("box section has neither a fixed cell nor arrays"))?;
        let cell: Vec<F> = vectors.retrieve_array_subset(&rows_subset(index, 1, &[3, 3])?)?;
        let origin: Vec<F> = match &self.origin {
            Some(array) => array.retrieve_array_subset(&rows_subset(index, 1, &[3])?)?,
            None => vec![0.0, 0.0, 0.0],
        };
        let boundary: Vec<bool> = match &self.boundary {
            Some(array) => array.retrieve_array_subset(&rows_subset(index, 1, &[3])?)?,
            None => vec![true, true, true],
        };
        if cell.len() != 9 || origin.len() != 3 || boundary.len() != 3 {
            return Err(MolRsError::zarr(format!(
                "box update {index} is malformed: {} cell values, {} origin values, {} boundary \
                 flags",
                cell.len(),
                origin.len(),
                boundary.len()
            )));
        }
        let simbox = SimBox::new_cell(
            Array2::from_shape_vec((3, 3), cell).map_err(zerr)?,
            Array1::from(origin),
            [boundary[0], boundary[1], boundary[2]],
            cell_defined,
        )
        .map_err(|e| MolRsError::zarr(format!("box update {index} is not a valid cell: {e:?}")))?;
        if self.cells.len() >= 64 {
            self.cells.clear();
        }
        self.cells.insert(index, simbox.clone());
        Ok(simbox)
    }
}

/// One open column array plus its most recently decoded inner chunks.
struct ColumnReader {
    array: Array<dyn ReadableListableStorageTraits>,
    /// Rows one inner chunk holds.
    chunk_rows: u64,
    /// Decoded whole chunks, most recently used first.
    chunks: VecDeque<(u64, Column)>,
}

impl ColumnReader {
    fn open(store: &ReadableListableStorage, path: &str) -> Result<Self, MolRsError> {
        let array = Array::open(store.clone(), path)?;
        let chunk_rows = inner_rows_of(&array);
        Ok(Self {
            array,
            chunk_rows,
            chunks: VecDeque::with_capacity(CHUNK_CACHE_ENTRIES + 1),
        })
    }

    /// Rows `start..start+rows`, decoding at most the chunks they touch.
    ///
    /// A range inside one chunk comes out of the chunk cache (decoding the
    /// chunk whole on a miss, so the next frame in it is a slice); a range
    /// straddling two chunks — a ragged run — is read directly.
    fn rows(&mut self, start: u64, rows: u64, trailing: &[u64]) -> Result<Column, MolRsError> {
        let first = start / self.chunk_rows;
        let last = (start + rows - 1) / self.chunk_rows;
        if first != last {
            return read_column_array(&self.array, &rows_subset(start, rows, trailing)?);
        }
        let within = (start - first * self.chunk_rows) as usize;
        let chunk = self.chunk(first, trailing)?;
        Ok(column_rows(chunk, within, within + rows as usize))
    }

    /// Inner chunk `index`, decoded whole and kept.
    fn chunk(&mut self, index: u64, trailing: &[u64]) -> Result<&Column, MolRsError> {
        if let Some(position) = self.chunks.iter().position(|(i, _)| *i == index) {
            if position != 0 {
                let hit = self.chunks.remove(position).expect("position is in range");
                self.chunks.push_front(hit);
            }
        } else {
            let total = self.array.shape().first().copied().unwrap_or(0);
            let chunk_start = index * self.chunk_rows;
            let chunk_len = self.chunk_rows.min(total.saturating_sub(chunk_start));
            let column =
                read_column_array(&self.array, &rows_subset(chunk_start, chunk_len, trailing)?)?;
            self.chunks.push_front((index, column));
            self.chunks.truncate(CHUNK_CACHE_ENTRIES);
        }
        Ok(&self
            .chunks
            .front()
            .expect("just inserted or moved to front")
            .1)
    }
}

/// The lazily opened arrays and caches behind a [`FrameSequence`].
#[derive(Default)]
struct ReadState {
    columns: BTreeMap<String, ColumnReader>,
    metas: BTreeMap<String, Array<dyn ReadableListableStorageTraits>>,
    boxes: Option<BoxReader>,
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
/// Every read door takes `&self`: the cursor holds no cursor state, only
/// caches (open array handles, the last decoded inner chunk per column, the
/// decoded cells) behind a lock, so a sequence can be shared and read from
/// several places.
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
    /// which is also what lets a packed `.mrec.zip` be opened through it.
    store: ReadableListableStorage,
    schema: SequenceSchema,
    /// Step numbers of the committed frames; its length is `nstep`.
    steps: Vec<i64>,
    times: Option<Vec<F>>,
    blocks: BTreeMap<String, BlockIndex>,
    cell: Option<BoxIndex>,
    state: Mutex<ReadState>,
}

impl FrameSequence {
    /// Open a sequence for reading, taking only its indices.
    ///
    /// Index-only: the schema attributes plus each section's `step_index` and
    /// `offset` — or, for a section whose hints say every update is regular,
    /// nothing but its array metadata. No frame data is touched until a read
    /// asks for it. The commit marker bounds everything: index entries at or
    /// past `len(step)` (a crash's leftovers) are ignored.
    ///
    /// Any readable, listable store opens: a directory store, an in-memory one,
    /// or the read-only zip adapter `open_packed` hands back. A conforming
    /// store needs no writer pin — the schema is derived from the store itself
    /// when the attribute is absent. A store created and closed without a
    /// single append opens as a legal, empty sequence.
    ///
    /// # Errors
    ///
    /// Every case is a [`MolRsError::Zarr`] naming the path it failed on:
    ///
    /// - the store holds the `trajectory/frames/<i>/` layout written by molrs
    ///   <= 0.13;
    /// - a schema pin is present but is not a schema this build can
    ///   deserialize;
    /// - no pin is present and the derivation from the store fails.
    ///
    /// Also any storage error raised while reading the index arrays.
    pub fn open<S>(store: Arc<S>) -> Result<Self, MolRsError>
    where
        S: ?Sized + ReadableStorageTraits + ListableStorageTraits + 'static,
    {
        // `StorageHandle` is how an unsized store becomes an owned, sized one:
        // `Arc<dyn ReadableWritableListableStorageTraits>` does not coerce to
        // `Arc<dyn ReadableListableStorageTraits>`, so the read-only view is
        // taken here.
        let store: ReadableListableStorage = Arc::new(StorageHandle::new(store));
        ensure_not_legacy(&store)?;
        let schema = match pinned_schema(&store)? {
            Some(pinned) => pinned,
            None => schema_from_store(&store)?,
        };
        let attributes = trajectory_attributes(&store)?;
        let nstep = committed_frames(&store, &attributes)?;
        // Created and closed without an append: a legal, empty sequence.
        let steps = read_track::<i64, _>(
            &store,
            &attributes.all,
            STEP_PROGRESSION_ATTRIBUTE,
            &join_path(TRAJECTORY_GROUP, STEP_ARRAY),
            nstep,
        )?
        .unwrap_or_default();
        if steps.len() as u64 != nstep {
            return Err(MolRsError::zarr(format!(
                "trajectory claims {nstep} committed frames but carries {} step numbers",
                steps.len()
            )));
        }
        let times = read_track::<F, _>(
            &store,
            &attributes.all,
            TIME_PROGRESSION_ATTRIBUTE,
            &join_path(TRAJECTORY_GROUP, TIME_ARRAY),
            nstep,
        )?;

        let mut blocks = BTreeMap::new();
        for (name, declared) in &schema.blocks {
            let path = join_path(TRAJECTORY_GROUP, name);
            if !group_exists(&store, &path)? {
                // Declared, never appended: the section does not exist yet.
                continue;
            }
            let index_path = join_path(&path, STEP_INDEX_ARRAY);
            let index = if array_exists(&store, &index_path)? {
                let mut step_index = read_whole::<u64, _>(&store, &index_path)?;
                let updates = step_index
                    .iter()
                    .take_while(|&&ordinal| ordinal < nstep)
                    .count();
                step_index.truncate(updates);
                let mut offset = read_whole::<u64, _>(&store, &join_path(&path, OFFSET_ARRAY))?;
                offset.truncate(if updates == 0 { 0 } else { updates + 1 });
                BlockIndex::Sparse { step_index, offset }
            } else {
                match stored_hints(&store, &path)? {
                    (Some(rows), true) if rows > 0 => {
                        // Regular: the columns' own length says how many
                        // updates landed; the marker bounds them.
                        let Some(column) = declared.columns.keys().next() else {
                            continue;
                        };
                        let landed = Array::open(store.clone(), &join_path(&path, column))?
                            .shape()
                            .first()
                            .copied()
                            .unwrap_or(0);
                        BlockIndex::Regular {
                            updates: (landed / rows).min(nstep),
                            rows,
                        }
                    }
                    _ => continue,
                }
            };
            let empty = match &index {
                BlockIndex::Regular { updates, .. } => *updates == 0,
                BlockIndex::Sparse { step_index, .. } => step_index.is_empty(),
            };
            if !empty {
                blocks.insert(name.clone(), index);
            }
        }

        let box_prefix = join_path(TRAJECTORY_GROUP, BOX_GROUP);
        let vectors_path = join_path(&box_prefix, VECTORS_ARRAY);
        let cell = if array_exists(&store, &vectors_path)? {
            let step_path = join_path(&box_prefix, STEP_INDEX_ARRAY);
            let mut step_index = if array_exists(&store, &step_path)? {
                read_whole::<u64, _>(&store, &step_path)?
            } else {
                let n = Array::open(store.clone(), &vectors_path)?
                    .shape()
                    .first()
                    .copied()
                    .unwrap_or(0);
                // The omitted index is one update at ordinal 0.
                if n == 0 { Vec::new() } else { vec![0] }
            };
            let committed = step_index
                .iter()
                .take_while(|&&ordinal| ordinal < nstep)
                .count();
            step_index.truncate(committed);
            if step_index.is_empty() {
                None
            } else {
                Some(BoxIndex {
                    step_index,
                    cell_defined: cell_defined_of(&store)?,
                })
            }
        } else if group_exists(&store, &box_prefix)? {
            let group = Group::open(store.clone(), &box_prefix)?;
            match cell_from_attributes(group.attributes())? {
                Some(cell) if nstep > 0 => Some(BoxIndex {
                    step_index: vec![0],
                    cell_defined: cell.is_cell_defined(),
                }),
                _ => None,
            }
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
            state: Mutex::new(ReadState::default()),
        })
    }

    /// Read frame `index`, or `None` when it is past the commit marker.
    ///
    /// The frame is assembled from whichever sections resolve at `index`: a
    /// block whose `step_index` has no entry at or before `index` is left out
    /// (absent); a block whose latest update holds zero rows comes back as an
    /// empty block with its declared columns (present and empty); every other
    /// block carries its latest update's rows. The cell and the per-step
    /// metadata come along the same way.
    ///
    /// # Errors
    ///
    /// A [`MolRsError::Zarr`] naming the section when the store is internally
    /// inconsistent — a non-monotonic `offset`, a block on disk but absent
    /// from the schema, a malformed `box/` update, a `meta` row of the wrong
    /// width — and any storage or codec error raised while reading rows.
    pub fn frame(&self, index: u64) -> Result<Option<Frame>, MolRsError> {
        self.assemble(index, None)
    }

    /// Read frame `index` carrying only the named `(block, column)` pairs.
    ///
    /// A viewer that needs coordinates decodes `x`, `y`, `z` and nothing
    /// else. A block none of whose columns is named is left out; a named
    /// block still resolves to absent / empty / rows exactly as in
    /// [`frame`](Self::frame). The cell and per-step metadata are always
    /// carried — they are cheap.
    ///
    /// # Errors
    ///
    /// [`frame`](Self::frame)'s, plus a [`MolRsError::Zarr`] naming a pair
    /// the schema does not declare.
    pub fn frame_columns(
        &self,
        index: u64,
        columns: &[(&str, &str)],
    ) -> Result<Option<Frame>, MolRsError> {
        let mut wanted: BTreeMap<&str, Vec<&str>> = BTreeMap::new();
        for (block, column) in columns {
            let declared = self.schema.blocks.get(*block).ok_or_else(|| {
                MolRsError::zarr(format!("block {block:?} is not declared by this sequence"))
            })?;
            if !declared.columns.contains_key(*column) {
                return Err(MolRsError::zarr(format!(
                    "column {column:?} of block {block:?} is not declared by this sequence"
                )));
            }
            wanted.entry(block).or_default().push(column);
        }
        self.assemble(index, Some(&wanted))
    }

    fn assemble(
        &self,
        index: u64,
        wanted: Option<&BTreeMap<&str, Vec<&str>>>,
    ) -> Result<Option<Frame>, MolRsError> {
        if index as usize >= self.steps.len() {
            return Ok(None);
        }
        let mut state = self.state.lock().map_err(|_| {
            MolRsError::zarr("frame sequence read cache is poisoned by an earlier panic")
        })?;
        let mut frame = Frame::new();

        for (name, block_index) in &self.blocks {
            let selection = match wanted {
                Some(wanted) => match wanted.get(name.as_str()) {
                    Some(columns) => Some(columns.as_slice()),
                    None => continue,
                },
                None => None,
            };
            let Some((_, start, rows)) = block_index.resolve(index)? else {
                // No entry at or before this step: the section does not exist
                // here.
                continue;
            };
            let schema = self.schema.blocks.get(name).ok_or_else(|| {
                MolRsError::zarr(format!("block {name:?} is on disk but not in the schema"))
            })?;
            let path = join_path(TRAJECTORY_GROUP, name);
            let mut block = Block::new();
            for (column, declared) in &schema.columns {
                if let Some(selection) = selection
                    && !selection.contains(&column.as_str())
                {
                    continue;
                }
                let dtype = dtype_from_tag(&declared.dtype)?;
                let values = if rows == 0 {
                    empty_column(dtype, &declared.trailing)?
                } else {
                    let column_path = join_path(&path, column);
                    if !state.columns.contains_key(&column_path) {
                        state.columns.insert(
                            column_path.clone(),
                            ColumnReader::open(&self.store, &column_path)?,
                        );
                    }
                    state
                        .columns
                        .get_mut(&column_path)
                        .expect("just inserted")
                        .rows(start, rows, &declared.trailing)?
                };
                insert_column_into_block(&mut block, column, values)?;
            }
            if block.is_empty() {
                // A declared block with no (selected) columns is still a block
                // with a row count.
                block.resize(rows as usize)?;
            }
            apply_structural_shape(&mut block, schema, &path)?;
            frame.insert(name.clone(), block);
        }

        if let Some(cell) = &self.cell
            && let Some(update) = latest_update(&cell.step_index, index)
        {
            if state.boxes.is_none() {
                state.boxes = Some(BoxReader::open(&self.store)?);
            }
            let boxes = state.boxes.as_mut().expect("just opened");
            frame.simbox = Some(boxes.cell_at(update as u64, cell.cell_defined)?);
        }

        for (key, declared) in &self.schema.meta {
            let path = join_path(&join_path(TRAJECTORY_GROUP, META_GROUP), key);
            if !state.metas.contains_key(key) {
                state
                    .metas
                    .insert(key.clone(), Array::open(self.store.clone(), &path)?);
            }
            let array = &state.metas[key];
            frame.meta.insert(
                key.clone(),
                read_meta_value(array, key, &declared.dtype, index)?,
            );
        }

        Ok(Some(frame))
    }

    /// The update of block `name` that frame `index` resolves to, or `None`
    /// when the block is absent there (or not in the store).
    ///
    /// Two consecutive frames resolving to the same update carry the same
    /// rows — a consumer can skip re-uploading a block whose update did not
    /// change without comparing a single value.
    pub fn block_update_at(&self, name: &str, index: u64) -> Result<Option<u64>, MolRsError> {
        if index as usize >= self.steps.len() {
            return Ok(None);
        }
        let Some(block_index) = self.blocks.get(name) else {
            return Ok(None);
        };
        Ok(block_index.resolve(index)?.map(|(update, _, _)| update))
    }

    /// The cell at frame `index`, or `None` when no cell has been written at
    /// or before it (or `index` is past the commit marker).
    ///
    /// # Errors
    ///
    /// A [`MolRsError::Zarr`] when the `box/` update is malformed.
    pub fn box_at(&self, index: u64) -> Result<Option<SimBox>, MolRsError> {
        if index as usize >= self.steps.len() {
            return Ok(None);
        }
        let Some(cell) = &self.cell else {
            return Ok(None);
        };
        let Some(update) = latest_update(&cell.step_index, index) else {
            return Ok(None);
        };
        let mut state = self.state.lock().map_err(|_| {
            MolRsError::zarr("frame sequence read cache is poisoned by an earlier panic")
        })?;
        if state.boxes.is_none() {
            state.boxes = Some(BoxReader::open(&self.store)?);
        }
        let boxes = state.boxes.as_mut().expect("just opened");
        Ok(Some(boxes.cell_at(update as u64, cell.cell_defined)?))
    }

    /// Materialize the whole sequence: the named lazy → eager conversion.
    ///
    /// Reads every committed frame through [`frame`](Self::frame) and hands
    /// back a `Trajectory` carrying those frames, the sequence's step numbers,
    /// and its times when the run wrote any. A run too large for memory is
    /// what [`frame`](Self::frame) is for.
    ///
    /// # Errors
    ///
    /// [`frame`](Self::frame)'s, on the first frame that raises one.
    pub fn to_trajectory(&self) -> Result<Trajectory, MolRsError> {
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

    /// Whether a block section is present in the store with at least one
    /// committed update, so a reader can decide once — e.g.
    /// `has_block("bonds")` — instead of probing every frame for it.
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
    fn read_step(&mut self, step: usize) -> std::io::Result<Option<Frame>> {
        self.frame(step as u64).map_err(std::io::Error::other)
    }

    /// Committed frames — the length of `trajectory/step` as
    /// [`FrameSequence::open`] read it.
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
    /// takes on `meta_dtype`.
    const SCHEMA_ATTRIBUTE: &str = "sequence_schema";
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

    /// `nstep` counts every appended frame, and a regular numbering lives in
    /// the `step_progression` attribute rather than in an array.
    #[test]
    fn step_counts_every_appended_frame() {
        let dir = TempDir::new().unwrap();
        let store = store_in(&dir);
        write_all(&store, &ragged_frames());
        let group = Group::open(store.clone(), TRAJ).unwrap();
        assert_eq!(group.attributes()["nstep"], 3);
        assert_eq!(
            group.attributes()["step_progression"],
            serde_json::json!({"start": 0, "stride": 1})
        );
        assert!(
            Array::open(store.clone(), &format!("{TRAJ}/step")).is_err(),
            "a regular step series needs no array"
        );
        let seq = open_sequence(&store);
        assert_eq!(seq.steps(), &[0, 1, 2]);
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

    /// A non-default origin is recorded — as a `box/` attribute while the
    /// cell is fixed, as an array once the cell changes; the still-default
    /// boundary is omitted either way.
    #[test]
    fn a_nonzero_origin_is_written() {
        let cell = |lengths: f64| {
            SimBox::new(
                array![[lengths, 0.0, 0.0], [0.0, 11.0, 0.0], [0.0, 0.0, 12.0]],
                array![1.0, 2.0, 3.0],
                [true, true, true],
            )
            .unwrap()
        };
        let dir = TempDir::new().unwrap();
        let store = store_in(&dir);
        let mut frame = atoms_frame(&[1.0]);
        frame.simbox = Some(cell(10.0));
        write_all(&store, &[frame]);
        let group = Group::open(store.clone(), &format!("{TRAJ}/box")).unwrap();
        assert_eq!(
            group.attributes()["origin"],
            serde_json::json!([1.0, 2.0, 3.0])
        );
        assert!(
            group.attributes().get("boundary").is_none(),
            "still-default boundary is omitted"
        );
        assert!(
            Array::open(store.clone(), &format!("{TRAJ}/box/vectors")).is_err(),
            "a fixed cell is attributes, not arrays"
        );
        let seq = open_sequence(&store);
        let back = seq.box_at(0).unwrap().unwrap();
        assert_eq!(back.origin_view().to_vec(), vec![1.0, 2.0, 3.0]);

        // The first change migrates the cell into arrays.
        let dir = TempDir::new().unwrap();
        let store = store_in(&dir);
        let mut first = atoms_frame(&[1.0]);
        first.simbox = Some(cell(10.0));
        let mut second = atoms_frame(&[2.0]);
        second.simbox = Some(cell(10.5));
        write_all(&store, &[first, second]);
        let origin: Vec<f64> = {
            let arr = Array::open(store.clone(), &format!("{TRAJ}/box/origin")).unwrap();
            let subset = ArraySubset::new_with_shape(arr.shape().to_vec());
            arr.retrieve_array_subset(&subset).unwrap()
        };
        assert_eq!(origin, vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0]);
        assert_eq!(
            u64_array(&store, &format!("{TRAJ}/box/step_index")),
            vec![0, 1]
        );
        assert!(Array::open(store.clone(), &format!("{TRAJ}/box/boundary")).is_err());
        let group = Group::open(store.clone(), &format!("{TRAJ}/box")).unwrap();
        assert!(
            group.attributes().get("vectors").is_none(),
            "the attribute form is withdrawn"
        );
        let seq = open_sequence(&store);
        assert_eq!(seq.box_at(1).unwrap().unwrap().h_view()[[0, 0]], 10.5);
        assert_eq!(seq.box_at(0).unwrap().unwrap().h_view()[[0, 0]], 10.0);
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

    /// A block that never changes costs one update — and, being regular
    /// (one update at ordinal 0), no index arrays at all: the group hints
    /// carry it and every frame resolves to update 0.
    #[test]
    fn a_constant_block_costs_one_update_and_no_index_arrays() {
        const STEPS: usize = 5;
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

        let group = Group::open(store.clone(), &format!("{TRAJ}/{BONDS}")).unwrap();
        assert_eq!(group.attributes()["uniform_rows"], 2);
        assert_eq!(group.attributes()["dense_updates"], true);
        for name in ["step_index", "offset"] {
            assert!(
                Array::open(store.clone(), &format!("{TRAJ}/{BONDS}/{name}")).is_err(),
                "a regular block writes no {name} array"
            );
        }
        let rows = Array::open(store.clone(), &format!("{TRAJ}/{BONDS}/{I}"))
            .unwrap()
            .shape()[0];
        assert_eq!(rows, 2, "one update's rows, not one per frame");
        let seq = open_sequence(&store);
        for index in 0..STEPS as u64 {
            assert_eq!(seq.block_update_at(BONDS, index).unwrap(), Some(0));
            assert_eq!(
                seq.frame(index)
                    .unwrap()
                    .unwrap()
                    .get(BONDS)
                    .unwrap()
                    .nrows(),
                Some(2)
            );
        }
    }

    /// A block that changes every step is regular too — update `i` at ordinal
    /// `i` with a fixed row count — so it costs no index arrays either; a
    /// ragged run is what materializes `step_index` and `offset`.
    #[test]
    fn a_block_changing_every_step_is_regular_and_a_ragged_one_is_indexed() {
        let dir = TempDir::new().unwrap();
        let store = store_in(&dir);
        let frames: Vec<Frame> = (0..5)
            .map(|i| atoms_frame(&[i as f64, i as f64 + 0.5]))
            .collect();
        write_all(&store, &frames);
        assert!(Array::open(store.clone(), &format!("{TRAJ}/{ATOMS}/step_index")).is_err());
        let seq = open_sequence(&store);
        for index in 0..5u64 {
            assert_eq!(seq.block_update_at(ATOMS, index).unwrap(), Some(index));
        }

        let dir = TempDir::new().unwrap();
        let store = store_in(&dir);
        write_all(&store, &ragged_frames());
        assert_eq!(
            u64_array(&store, &format!("{TRAJ}/{ATOMS}/step_index")),
            vec![0, 1, 2]
        );
        assert_eq!(
            u64_array(&store, &format!("{TRAJ}/{ATOMS}/offset")),
            vec![0, 3, 8, 12]
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

    /// The worked example, on disk: a 3000-atom `f64` xyz block derives a
    /// **frame-aligned** inner chunk of exactly one frame and a shard of
    /// 3728 such chunks.
    ///
    /// Hard-coded from the formulas, not from a measurement: row = 3 * 8 =
    /// 24 B; the byte floor is `16 KiB / 24 = 682` rows, which 3000 exceeds,
    /// so `R = 3000` (one frame); chunk = 72 000 B;
    /// `k = clamp(floor(256 MiB / 72 000), 1, 4096) = 3728`; shard =
    /// 3000 * 3728 = 11 184 000 rows.
    #[test]
    fn the_worked_example_derives_one_frame_per_chunk_and_k_3728() {
        const ATOM_COUNT: usize = 3000;
        const ROWS_PER_CHUNK: u64 = 3000;
        const CHUNKS_PER_SHARD: u64 = 3728;
        const ROWS_PER_SHARD: u64 = 11_184_000;

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
            "R = one 3000-row frame (the 16 KiB floor is 682 rows), trailing axis whole"
        );
        assert_eq!(
            shard,
            vec![ROWS_PER_SHARD, 3],
            "the shard spans R * k = 11 184 000 rows"
        );
        assert_eq!(
            shard[0] / inner[0],
            CHUNKS_PER_SHARD,
            "k = clamp(floor(256 MiB / 72 000 B chunk), 1, 4096) = 3728"
        );
    }

    /// A small frame is rounded up to the byte floor in whole frames: 100
    /// `f64` rows (800 B) need 21 frames to pass 16 KiB, so `R = 2100`.
    #[test]
    fn a_small_frame_is_chunked_in_whole_frames_up_to_the_byte_floor() {
        let dir = TempDir::new().unwrap();
        let store = store_in(&dir);
        let values: Vec<f64> = (0..100).map(|i| i as f64).collect();
        write_all(&store, &[atoms_frame(&values)]);
        let (_, inner) = extents(&store, &format!("{TRAJ}/{ATOMS}/{X}"));
        assert_eq!(inner, vec![2100], "ceil(2048 / 100) * 100 = 2100 rows");
    }

    /// The dense per-frame arrays — written once a series stops being
    /// regular — share one small extent whatever the frames look like: 1024
    /// rows per inner chunk, 256 chunks per shard.
    #[test]
    fn dense_arrays_take_the_dense_extents() {
        let dir = TempDir::new().unwrap();
        let store = store_in(&dir);
        let frames = ragged_frames();
        let mut writer = FrameSequenceWriter::create(
            store.clone(),
            SequenceSchema::from_frames(&frames).unwrap(),
        )
        .unwrap();
        // Steps 0, 1, 5: not a progression, so `step` becomes an array.
        for (frame, step) in frames.iter().zip([0i64, 1, 5]) {
            writer.append_at(frame, step, None).unwrap();
        }
        writer.close().unwrap();
        for path in [
            format!("{TRAJ}/step"),
            format!("{TRAJ}/{ATOMS}/offset"),
            format!("{TRAJ}/{ATOMS}/step_index"),
        ] {
            let (shard, inner) = extents(&store, &path);
            assert_eq!(inner, vec![1024], "{path}");
            assert_eq!(shard, vec![1024 * 256], "{path}");
        }
        assert_eq!(i64_array(&store, &format!("{TRAJ}/step")), vec![0, 1, 5]);
        let seq = open_sequence(&store);
        assert_eq!(seq.steps(), &[0, 1, 5]);
    }

    // =======================================================================
    // B. Schema — derived union, enforcement at append, reserved names
    // =======================================================================

    /// A union mint over heterogeneous frames writes a store in which every
    /// block **carries forward** from its latest update: a frame reads back
    /// what it presented plus whatever earlier frames left standing.
    ///
    /// `from_frames` unions the blocks, per-section `step_index` records when
    /// each one changed, and omission means "unchanged" — the reading that
    /// makes a topology written once cost one entry.
    #[test]
    fn a_union_mint_carries_each_block_forward_from_its_latest_update() {
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
        assert_eq!(
            step0.len(),
            1,
            "step 0 presented only atoms; nothing to carry"
        );
        assert_eq!(atoms_x(&step0), vec![1.0, 2.0]);

        let step1 = frame_at(&mut seq, 1);
        assert_eq!(step1.len(), 2, "step 1 presented atoms and bonds");
        assert!(step1.get(BONDS).is_some());
        assert!(step1.get("charges").is_none());

        let step2 = frame_at(&mut seq, 2);
        assert_eq!(
            step2.len(),
            3,
            "step 2 presented atoms and charges and carries bonds forward"
        );
        assert!(step2.get("charges").is_some());
        assert_eq!(
            step2.get(BONDS).expect("bonds carried forward").nrows(),
            Some(2)
        );
        assert_eq!(
            u64_array(&store, &format!("{TRAJ}/{BONDS}/step_index")),
            vec![1],
            "a block omitted after it appeared earns no update"
        );
    }

    /// The three presence states, on one store: omitted = carried forward,
    /// zero rows = present and empty, no entry yet = absent.
    #[test]
    fn omitted_zero_row_and_unseen_blocks_read_back_as_carried_empty_and_absent() {
        let dir = TempDir::new().unwrap();
        let store = store_in(&dir);

        let mut with_bonds = atoms_frame(&[1.0]);
        with_bonds.insert(BONDS, block_with(I, uint_column(&[0, 1])));
        let omits_bonds = atoms_frame(&[2.0]);
        let mut empties_bonds = atoms_frame(&[3.0]);
        empties_bonds.insert(BONDS, block_with(I, uint_column(&[])));
        let omits_again = atoms_frame(&[4.0]);
        // Frame 0 has no bonds yet: absent there.
        let frames = vec![
            atoms_frame(&[0.5]),
            with_bonds,
            omits_bonds,
            empties_bonds,
            omits_again,
        ];
        write_all(&store, &frames);

        let mut seq = open_sequence(&store);
        assert!(
            frame_at(&mut seq, 0).get(BONDS).is_none(),
            "no update at or before frame 0: absent"
        );
        assert_eq!(frame_at(&mut seq, 1).get(BONDS).unwrap().nrows(), Some(2));
        assert_eq!(
            frame_at(&mut seq, 2).get(BONDS).unwrap().nrows(),
            Some(2),
            "an omitted block carries forward"
        );
        let empty = frame_at(&mut seq, 3);
        let bonds = empty.get(BONDS).expect("a zero-row update is present");
        assert_eq!(bonds.nrows(), Some(0), "and empty");
        assert!(bonds.get(I).is_some(), "with its declared columns");
        assert_eq!(
            frame_at(&mut seq, 4).get(BONDS).unwrap().nrows(),
            Some(0),
            "an omission after the empty update carries the empty block forward"
        );
        assert_eq!(
            u64_array(&store, &format!("{TRAJ}/{BONDS}/step_index")),
            vec![1, 3]
        );
        assert_eq!(
            u64_array(&store, &format!("{TRAJ}/{BONDS}/offset")),
            vec![0, 2, 2]
        );
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
    /// `meta_dtype` (ac-019).
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
                array.attributes().get("meta_dtype"),
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
        schema
            .declare_meta_with_fill(KEY, MetaValue::F64(FILL))
            .unwrap();

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
    /// `meta_dtype` tags — and one half working is not the claim.
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
    /// "everything appended after it" (ac-015).
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
            "a flush commits every buffered row, ragged tail included"
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

    /// Rolling the `nstep` marker back to its pre-flush value hides the
    /// uncommitted frames and errors at nothing (ac-017).
    ///
    /// This is the crash window made deterministic: the marker is written
    /// last, so a crash anywhere before that leaves the data arrays long and
    /// the marker short — exactly the state this test builds.
    #[test]
    fn a_marker_rolled_back_hides_the_uncommitted_frames() {
        let dir = TempDir::new().unwrap();
        let store = store_in(&dir);
        let frames: Vec<Frame> = (0..5)
            .map(|step| atoms_frame(&[step as f64, step as f64 + 0.5]))
            .collect();
        write_all(&store, &frames);

        let mut group = Group::open(store.clone(), TRAJ).unwrap();
        group.attributes_mut().insert("nstep".to_string(), 3.into());
        group.store_metadata().unwrap();

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
    /// the trajectory group's metadata — the `nstep` marker — is written
    /// **after** every other key (ac-017).
    ///
    /// That ordering is what makes a crash between two writes invisible — the
    /// commit marker is the last thing to move — and it is unobservable from
    /// the outside, so the observation is a recording store.
    #[test]
    fn the_marker_is_written_after_every_other_key() {
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
        let marker = log
            .iter()
            .rposition(|key| key == "trajectory/zarr.json")
            .unwrap_or_else(|| panic!("a flush must write the marker; wrote {log:?}"));
        assert_eq!(
            marker,
            log.len() - 1,
            "nothing may be written after the commit marker: {log:?}"
        );
        assert!(
            log[..marker]
                .iter()
                .any(|key| key.starts_with(&format!("trajectory/{ATOMS}/"))),
            "the data arrays must be written before the marker: {log:?}"
        );
    }

    // =======================================================================
    // F. Flush cadence — values never depend on it; dead bytes stay bounded
    // =======================================================================

    /// Whatever the flush cadence, the frames read back identical, and the
    /// dead bytes an eager flusher leaves behind are bounded by one
    /// re-encoded trailing chunk per flush — nothing is compacted at close,
    /// because a whole-shard rewrite is the one write that can destroy
    /// committed data on a crash.
    #[test]
    fn flush_cadence_changes_dead_bytes_but_never_what_reads_back() {
        const ROWS_PER_CHUNK: u64 = 64;
        const CHUNKS_PER_SHARD: u64 = 8;
        const FRAMES: usize = 24;
        /// One 64-row `f64` chunk, its `crc32c`, and the shard index slack.
        const CHUNK_BYTES_UPPER_BOUND: u64 = 64 * 8 + 4 + 16;

        // 24 frames of 5 rows = 120 rows: two inner chunks inside one shard.
        let frames: Vec<Frame> = (0..FRAMES)
            .map(|step| {
                let base = step as f64 * 10.0;
                atoms_frame(&[base, base + 1.0, base + 2.0, base + 3.0, base + 4.0])
            })
            .collect();
        let schema = SequenceSchema::from_frames(&frames).unwrap();

        let eager_dir = TempDir::new().unwrap();
        let eager_store = store_in(&eager_dir);
        let mut eager = FrameSequenceWriter::create(eager_store.clone(), schema)
            .unwrap()
            .with_rows_per_chunk(ROWS_PER_CHUNK)
            .unwrap()
            .with_chunks_per_shard(CHUNKS_PER_SHARD)
            .unwrap();
        for frame in &frames {
            eager.append(frame).unwrap();
            eager.flush().unwrap();
        }
        eager.close().unwrap();
        let column_dir = PathBuf::from("trajectory").join(ATOMS).join(X);
        let eager_bytes = chunk_bytes(&eager_dir.path().join(&column_dir));

        let lazy_dir = TempDir::new().unwrap();
        let lazy_store = store_in(&lazy_dir);
        let mut lazy = FrameSequenceWriter::create(
            lazy_store.clone(),
            SequenceSchema::from_frames(&frames).unwrap(),
        )
        .unwrap()
        .with_rows_per_chunk(ROWS_PER_CHUNK)
        .unwrap()
        .with_chunks_per_shard(CHUNKS_PER_SHARD)
        .unwrap()
        .with_flush_every(FRAMES as u64)
        .unwrap();
        for frame in &frames {
            lazy.append(frame).unwrap();
        }
        lazy.close().unwrap();
        let lazy_bytes = chunk_bytes(&lazy_dir.path().join(&column_dir));

        assert!(
            eager_bytes > lazy_bytes,
            "flushing every frame re-encodes the trailing chunk and leaves dead bytes: \
             {eager_bytes} B vs {lazy_bytes} B"
        );
        assert!(
            eager_bytes <= lazy_bytes + FRAMES as u64 * CHUNK_BYTES_UPPER_BOUND,
            "dead bytes are bounded by one chunk per flush: {eager_bytes} B vs {lazy_bytes} B"
        );

        let mut eager_seq = open_sequence(&eager_store);
        let mut lazy_seq = open_sequence(&lazy_store);
        assert_eq!(committed_len(&mut eager_seq), FRAMES);
        assert_eq!(committed_len(&mut lazy_seq), FRAMES);
        for index in 0..FRAMES as u64 {
            assert_eq!(
                atoms_x(&frame_at(&mut eager_seq, index)),
                atoms_x(&frame_at(&mut lazy_seq, index)),
                "frame {index} must not remember the flush cadence"
            );
        }
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
    /// one `zarr.json`; the four groups (root, `meta/`, `trajectory/` and
    /// `trajectory/atoms/`) add one `zarr.json` each. A regular run has
    /// exactly one growth array: `x`. `step` is the `step_progression`
    /// attribute and the block's CSR index is implied by its hints, so
    /// neither costs a file. Eight 8-row frames are 64 rows of `x` — two
    /// full shards — 7 files; sixteen frames take `x` to 128 rows, four
    /// shards, 9 files. Doubling the frames adds two files.
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
        /// The growth arrays: the block's `x` alone — `step` is a progression
        /// attribute and a regular block writes no index arrays. No `time`, no
        /// `meta`, no `box` — the fixture frames carry none.
        const ARRAYS: u64 = 1;
        /// The root, `meta/`, `trajectory/` and `trajectory/atoms/`.
        const GROUPS: u64 = 4;
        const FRAMES: u64 = 8;

        /// `O(arrays)` of the bound: each array costs its own `zarr.json` and
        /// rounds its last shard up, and each group costs a `zarr.json`.
        const ARRAY_FLOOR: u64 = 2 * ARRAYS + GROUPS;
        /// Rows of `x` at `FRAMES`: 8 * 8.
        const SINGLE_ROWS: u64 = 64;
        /// The same at `2 * FRAMES`.
        const DOUBLE_ROWS: u64 = 128;
        /// 4 group `zarr.json` + 1 array `zarr.json` + 2 `x` shards.
        const SINGLE_FILES: usize = 7;
        /// `x` on four shards.
        const DOUBLE_FILES: usize = 9;

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

        let seq = open_sequence(&store);
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

    // =======================================================================
    // H. The 2026-09 contract — root/meta, cadence, hints, rollback, reads
    // =======================================================================

    /// The streaming writer mints a record, not a bare `trajectory/`: the
    /// root group and an (empty) `meta/` group exist before the first append,
    /// and no version key is stamped.
    #[test]
    fn create_writes_the_root_and_meta_groups_without_a_version_key() {
        let dir = TempDir::new().unwrap();
        let store = store_in(&dir);
        let schema = SequenceSchema::from_frame(&atoms_frame(&[1.0])).unwrap();
        let writer = FrameSequenceWriter::create(store.clone(), schema).unwrap();
        assert!(dir.path().join("zarr.json").is_file(), "root group");
        let meta = Group::open(store.clone(), "/meta").expect("meta group exists");
        assert!(meta.attributes().is_empty(), "{:?}", meta.attributes());
        drop(writer);
    }

    /// `with_meta` writes the identity document the producer hands in.
    #[test]
    fn with_meta_writes_the_identity_document() {
        let dir = TempDir::new().unwrap();
        let store = store_in(&dir);
        let schema = SequenceSchema::from_frame(&atoms_frame(&[1.0])).unwrap();
        let mut doc = serde_json::Map::new();
        doc.insert("creator".into(), serde_json::json!({"name": "test"}));
        let writer = FrameSequenceWriter::create(store.clone(), schema)
            .unwrap()
            .with_meta(&doc)
            .unwrap();
        drop(writer);
        let meta = Group::open(store.clone(), "/meta").unwrap();
        assert_eq!(meta.attributes()["creator"]["name"], "test");
    }

    /// The landing cadence follows the frame size: a 1000-row `f64` frame
    /// (8 KB) lands every 512 frames (4 MiB), rounded to whole chunks of
    /// three frames; `with_flush_every` overrides it.
    #[test]
    fn the_landing_cadence_is_derived_from_the_frame_size() {
        let dir = TempDir::new().unwrap();
        let store = store_in(&dir);
        let values: Vec<f64> = (0..1000).map(|i| i as f64).collect();
        let frame = atoms_frame(&values);
        let schema = SequenceSchema::from_frame(&frame).unwrap();
        let mut writer = FrameSequenceWriter::create(store.clone(), schema.clone()).unwrap();
        writer.append(&frame).unwrap();
        // 4 MiB / 8000 B = 524, rounded up to a multiple of 3 frames per
        // 2100-row... no: 1000-row frames reach the 16 KiB floor at 3 frames
        // (3000 rows), and ceil(524 / 3) * 3 = 525.
        assert_eq!(writer.flush_every(), 525);
        assert_eq!(writer.committed(), 0, "one frame is below the cadence");

        let dir2 = TempDir::new().unwrap();
        let store2 = store_in(&dir2);
        let mut writer = FrameSequenceWriter::create(store2.clone(), schema)
            .unwrap()
            .with_flush_every(2)
            .unwrap();
        for _ in 0..5 {
            writer.append(&frame).unwrap();
        }
        assert_eq!(
            writer.committed(),
            4,
            "two landings of two frames, one buffered"
        );
        writer.close().unwrap();
        let mut seq = open_sequence(&store2);
        assert_eq!(committed_len(&mut seq), 5);
    }

    /// A block whose updates all carry the same row count at ordinals
    /// 0, 1, 2, … advertises `uniform_rows` and `dense_updates` and writes no
    /// index arrays; a reader resolves it arithmetically. The moment a frame
    /// breaks either rule the arrays are materialized and the hints withdrawn.
    #[test]
    fn regular_blocks_carry_hints_and_read_back_through_them() {
        let dir = TempDir::new().unwrap();
        let store = store_in(&dir);
        let frames: Vec<Frame> = (0..5)
            .map(|i| atoms_frame(&[i as f64, i as f64 + 0.5]))
            .collect();
        write_all(&store, &frames);
        let group = Group::open(store.clone(), &format!("{TRAJ}/{ATOMS}")).unwrap();
        assert_eq!(group.attributes()["uniform_rows"], 2);
        assert_eq!(group.attributes()["dense_updates"], true);

        // There are no index arrays to decode: a reader that trusts the hints
        // resolves arithmetically off the column length alone.
        for name in ["step_index", "offset"] {
            assert!(
                Array::open(store.clone(), &format!("{TRAJ}/{ATOMS}/{name}")).is_err(),
                "a regular block writes no {name} array"
            );
        }
        let mut seq = open_sequence(&store);
        assert_eq!(atoms_x(&frame_at(&mut seq, 3)), vec![3.0, 3.5]);
        assert_eq!(seq.block_update_at(ATOMS, 4).unwrap(), Some(4));

        // A ragged run withdraws `uniform_rows`; a skipped ordinal withdraws
        // `dense_updates`.
        let dir = TempDir::new().unwrap();
        let store = store_in(&dir);
        let mut frames = ragged_frames();
        frames.push(frames[2].clone()); // identical: no update at ordinal 3
        frames.push(atoms_frame(&[99.0]));
        write_all(&store, &frames);
        let group = Group::open(store.clone(), &format!("{TRAJ}/{ATOMS}")).unwrap();
        assert!(group.attributes().get("uniform_rows").is_none());
        assert!(group.attributes().get("dense_updates").is_none());
        let mut seq = open_sequence(&store);
        assert_eq!(
            atoms_x(&frame_at(&mut seq, 3)),
            vec![20.0, 21.0, 22.0, 23.0]
        );
        assert_eq!(atoms_x(&frame_at(&mut seq, 4)), vec![99.0]);
        assert_eq!(seq.block_update_at(ATOMS, 3).unwrap(), Some(2));
    }

    /// A crash that left every array longer than `step` is rolled back on
    /// reopen, and appending continues from the committed frame with the CSR
    /// pointer intact.
    #[test]
    fn a_reopened_writer_rolls_back_to_the_commit_marker() {
        let dir = TempDir::new().unwrap();
        let store = store_in(&dir);
        let frames = ragged_frames();
        let mut writer = FrameSequenceWriter::create(
            store.clone(),
            SequenceSchema::from_frames(&frames).unwrap(),
        )
        .unwrap()
        .with_flush_every(1)
        .unwrap();
        for frame in &frames {
            writer.append(frame).unwrap();
        }
        writer.close().unwrap();

        // Simulate a torn commit: `step` stays at 3 frames while the block's
        // arrays claim a fourth update of 7 rows.
        for (path, extra) in [
            (format!("{TRAJ}/{ATOMS}/{X}"), 7u64),
            (format!("{TRAJ}/{ATOMS}/offset"), 1),
            (format!("{TRAJ}/{ATOMS}/step_index"), 1),
        ] {
            let mut array = Array::open(store.clone(), &path).unwrap();
            let mut shape = array.shape().to_vec();
            shape[0] += extra;
            array.set_shape(shape).unwrap();
            array.store_metadata().unwrap();
        }
        {
            let array = Array::open(store.clone(), &format!("{TRAJ}/{ATOMS}/step_index")).unwrap();
            array
                .store_array_subset(
                    &ArraySubset::new_with_start_shape(vec![3], vec![1]).unwrap(),
                    vec![3u64],
                )
                .unwrap();
        }

        let mut seq = open_sequence(&store);
        assert_eq!(committed_len(&mut seq), 3, "the reader is bounded by step");
        assert_eq!(
            atoms_x(&frame_at(&mut seq, 2)),
            vec![20.0, 21.0, 22.0, 23.0]
        );

        let mut writer = FrameSequenceWriter::open(store.clone()).unwrap();
        writer.append(&atoms_frame(&[30.0, 31.0])).unwrap();
        writer.close().unwrap();

        assert_eq!(
            u64_array(&store, &format!("{TRAJ}/{ATOMS}/offset")),
            vec![0, 3, 8, 12, 14],
            "the rolled-back CSR pointer continues from the committed total"
        );
        assert_eq!(
            u64_array(&store, &format!("{TRAJ}/{ATOMS}/step_index")),
            vec![0, 1, 2, 3]
        );
        let mut seq = open_sequence(&store);
        assert_eq!(atoms_x(&frame_at(&mut seq, 3)), vec![30.0, 31.0]);
        assert_eq!(
            atoms_x(&frame_at(&mut seq, 2)),
            vec![20.0, 21.0, 22.0, 23.0]
        );
    }

    /// `frame_columns` decodes only the named columns; the block still
    /// resolves through the same three states.
    #[test]
    fn frame_columns_reads_only_the_named_columns() {
        let dir = TempDir::new().unwrap();
        let store = store_in(&dir);
        let mut frame = atoms_frame(&[1.0, 2.0]);
        frame
            .get_mut(ATOMS)
            .unwrap()
            .insert_column("y", float_column(&[10.0, 20.0]))
            .unwrap();
        frame.insert(BONDS, block_with(I, uint_column(&[0, 1])));
        write_all(&store, &[frame]);

        let seq = open_sequence(&store);
        let picked = seq
            .frame_columns(0, &[(ATOMS, "y")])
            .unwrap()
            .expect("frame 0 exists");
        let atoms = picked.get(ATOMS).expect("the named block is present");
        assert!(atoms.get("y").is_some());
        assert!(atoms.get(X).is_none(), "an unnamed column is not decoded");
        assert!(picked.get(BONDS).is_none(), "an unnamed block is left out");
        assert_eq!(atoms.nrows(), Some(2));

        let err = seq
            .frame_columns(0, &[(ATOMS, "nope")])
            .unwrap_err()
            .to_string();
        assert!(err.contains("nope"), "{err}");
    }

    /// `box_at` resolves the cell through the box index and caches it.
    #[test]
    fn box_at_resolves_the_latest_cell() {
        let dir = TempDir::new().unwrap();
        let store = store_in(&dir);
        let mut with_cell = atoms_frame(&[1.0]);
        with_cell.simbox = Some(fixed_cell());
        write_all(
            &store,
            &[atoms_frame(&[0.0]), with_cell, atoms_frame(&[2.0])],
        );
        let seq = open_sequence(&store);
        assert!(seq.box_at(0).unwrap().is_none(), "no cell yet");
        let cell = seq.box_at(2).unwrap().expect("the cell carries forward");
        assert_eq!(cell.h_view()[[1, 1]], 11.0);
        assert!(seq.box_at(3).unwrap().is_none(), "past the commit marker");
    }

    /// A shaped block keeps its row count: any other count is refused at
    /// append, naming the block and both counts.
    #[test]
    fn a_shaped_block_with_another_row_count_is_refused_at_append() {
        let dir = TempDir::new().unwrap();
        let store = store_in(&dir);
        let mut grid = Block::new();
        grid.insert_column("rho", float_column(&[1.0, 2.0, 3.0, 4.0]))
            .unwrap();
        grid.set_shape(&[2, 2]).unwrap();
        let mut frame = Frame::new();
        frame.insert("grid", grid);
        let schema = SequenceSchema::from_frame(&frame).unwrap();
        let mut writer = FrameSequenceWriter::create(store, schema).unwrap();
        writer.append(&frame).unwrap();
        let mut wrong = Frame::new();
        wrong.insert("grid", block_with("rho", float_column(&[1.0, 2.0])));
        let err = writer.append(&wrong).unwrap_err().to_string();
        assert!(err.contains("grid") && err.contains("4 rows"), "{err}");
    }

    /// A meta value that arrives at another width is re-read at the declared
    /// width — a JSON list becomes the declared `f64x3`, an `f64` the declared
    /// `f32` — and one that cannot be is refused naming both.
    #[test]
    fn a_meta_value_at_another_width_is_read_at_the_declared_one() {
        let dir = TempDir::new().unwrap();
        let store = store_in(&dir);
        let mut schema = SequenceSchema::from_frame(&atoms_frame(&[1.0])).unwrap();
        schema.declare_meta("com", "f64x3").unwrap();
        schema.declare_meta("scale", "f32").unwrap();
        let mut writer = FrameSequenceWriter::create(store.clone(), schema).unwrap();
        let mut frame = atoms_frame(&[1.0]);
        frame
            .meta
            .insert("com", MetaValue::Json(serde_json::json!([1.0, 2.0, 3.0])));
        frame.meta.insert("scale", MetaValue::F64(0.5));
        writer.append(&frame).unwrap();
        let mut wrong = atoms_frame(&[2.0]);
        wrong
            .meta
            .insert("com", MetaValue::Json(serde_json::json!([1.0, 2.0])));
        wrong.meta.insert("scale", MetaValue::F64(0.25));
        let err = writer.append(&wrong).unwrap_err().to_string();
        assert!(err.contains("com") && err.contains("f64x3"), "{err}");
        writer.close().unwrap();
        let mut seq = open_sequence(&store);
        let back = frame_at(&mut seq, 0);
        assert_eq!(
            back.meta.get("com"),
            Some(&MetaValue::F64x3([1.0, 2.0, 3.0]))
        );
        assert_eq!(back.meta.get("scale"), Some(&MetaValue::F32(0.5)));
    }

    /// A schema declared column by column is the same pin a derived one is.
    #[test]
    fn a_declared_schema_equals_the_derived_one() {
        let mut frame = atoms_frame(&[1.0, 2.0]);
        frame.insert(BONDS, block_with(I, uint_column(&[0])));
        frame.meta.insert("energy", MetaValue::F64(1.5));
        let derived = SequenceSchema::from_frame(&frame).unwrap();

        let mut declared = SequenceSchema::new();
        declared
            .declare_column(ATOMS, X, DType::Float, &[])
            .unwrap();
        declared.declare_column(BONDS, I, DType::UInt, &[]).unwrap();
        declared.declare_meta("energy", "f64").unwrap();
        assert_eq!(declared, derived);

        let err = declared
            .declare_column(ATOMS, X, DType::Float32, &[])
            .unwrap_err()
            .to_string();
        assert!(err.contains("f64") && err.contains("f32"), "{err}");
        let err = declared
            .declare_column("step", "a", DType::Float, &[])
            .unwrap_err()
            .to_string();
        assert!(err.contains("reserved"), "{err}");
        let err = declared
            .declare_meta("bad", "f128")
            .unwrap_err()
            .to_string();
        assert!(err.contains("f128"), "{err}");
    }

    /// Every inner chunk ends in `crc32c`, floats stay raw and everything
    /// else carries `gzip` level 1.
    #[test]
    fn inner_codecs_follow_the_width_policy() {
        let dir = TempDir::new().unwrap();
        let store = store_in(&dir);
        let mut frame = atoms_frame(&[1.0]);
        frame.insert(BONDS, block_with(I, uint_column(&[0])));
        let mut second = atoms_frame(&[2.0]);
        second.insert(BONDS, block_with(I, uint_column(&[0])));
        let mut third = atoms_frame(&[3.0]);
        third.insert(BONDS, block_with(I, uint_column(&[0])));
        let mut writer = FrameSequenceWriter::create(
            store.clone(),
            SequenceSchema::from_frames(&[frame.clone(), second.clone()]).unwrap(),
        )
        .unwrap();
        // Steps 0, 1, 5: not a progression (two values always are one), so
        // `step` is an array to inspect.
        writer.append_at(&frame, 0, None).unwrap();
        writer.append_at(&second, 1, None).unwrap();
        writer.append_at(&third, 5, None).unwrap();
        writer.close().unwrap();
        let inner_codecs = |path: &str| -> Vec<String> {
            let arr = Array::open(store.clone(), path).unwrap();
            let metadata = serde_json::to_value(arr.metadata()).unwrap();
            let sharding = metadata["codecs"]
                .as_array()
                .unwrap()
                .iter()
                .find(|codec| codec["name"] == "sharding_indexed")
                .unwrap()
                .clone();
            assert_eq!(sharding["configuration"]["index_location"], "start");
            sharding["configuration"]["codecs"]
                .as_array()
                .unwrap()
                .iter()
                .map(|codec| codec["name"].as_str().unwrap().to_string())
                .collect()
        };
        assert_eq!(
            inner_codecs(&format!("{TRAJ}/{ATOMS}/{X}")),
            vec!["bytes", "crc32c"]
        );
        assert_eq!(
            inner_codecs(&format!("{TRAJ}/{BONDS}/{I}")),
            vec!["bytes", "gzip", "crc32c"]
        );
        assert_eq!(
            inner_codecs(&format!("{TRAJ}/step")),
            vec!["bytes", "gzip", "crc32c"]
        );
    }
}
