//! Low-level Zarr ↔ Block/Column/SimBox/Frame helpers.
//!
//! These functions are shared by the frame/trajectory Zarr backend. They convert between
//! molrs in-memory types and
//! Zarr V3 arrays/groups, always relative to a caller-supplied path prefix.

#[cfg(feature = "zarr")]
use zarrs::array::codec::bytes_to_bytes::crc32c::Crc32cCodec;
use zarrs::array::data_type::{
    BoolDataType, Complex64DataType, Complex128DataType, Float16DataType, Float32DataType,
    Float64DataType, Int8DataType, Int16DataType, Int32DataType, Int64DataType, StringDataType,
    UInt8DataType, UInt16DataType, UInt32DataType, UInt64DataType,
};
use zarrs::array::{Array, ArraySubset};
#[cfg(feature = "zarr")]
use zarrs::array::{ArrayBuilder, BytesToBytesCodecTraits, codec::GzipCodec, data_type};
#[cfg(feature = "zarr")]
use zarrs::group::GroupBuilder;
use zarrs::node::{Node, NodeMetadata};
use zarrs::storage::{ReadableStorageTraits, ReadableWritableListableStorage};
#[cfg(feature = "zarr")]
use zarrs::storage::{StorePrefix, WritableStorageTraits};

use ndarray::ArrayD;
#[cfg(feature = "zarr")]
use ndarray::ArrayViewD;
use std::sync::Arc;

use molrs::error::MolRsError;
use molrs::spatial::simbox::SimBox;
#[cfg(feature = "zarr")]
use molrs::store::block::DType;
use molrs::store::block::{Block, Column};
use molrs::store::frame::Frame;
use molrs::store::meta::MetaValue;
use molrs::types::F;

#[cfg(feature = "zarr")]
use super::chunking::{ChunkPlan, plan};

/// `gzip` level for the fixed-size arrays that compress: integer, boolean and
/// string columns. Level 1 — these compress by structure, not by effort.
///
/// Floating-point columns are stored raw: 52 random mantissa bits gzip to
/// about 95 % of their size at a real CPU cost, and a precision study admits
/// no lossy codec that would do better. Every array carries `crc32c` so a torn
/// chunk is a checksum error rather than garbage rows.
#[cfg(feature = "zarr")]
pub(in crate::io::zarr) const GZIP_LEVEL: u32 = 1;

// ---------------------------------------------------------------------------
// Column write
// ---------------------------------------------------------------------------

#[cfg(feature = "zarr")]
pub(crate) fn write_column(
    store: &ReadableWritableListableStorage,
    path: &str,
    col: &Column,
) -> Result<(), MolRsError> {
    let shape: Vec<u64> = col.shape().iter().map(|&s| s as u64).collect();
    let chunking = plan(&shape, col.dtype().itemsize());
    let (dt, fill) = dtype_of(col);
    match col {
        Column::Float(a) => write_typed_array(store, path, a.view(), dt, fill, chunking),
        Column::Int8(a) => write_typed_array(store, path, a.view(), dt, fill, chunking),
        Column::Int16(a) => write_typed_array(store, path, a.view(), dt, fill, chunking),
        Column::Int(a) => write_typed_array(store, path, a.view(), dt, fill, chunking),
        Column::Int64(a) => write_typed_array(store, path, a.view(), dt, fill, chunking),
        Column::UInt(a) => write_typed_array(store, path, a.view(), dt, fill, chunking),
        Column::U8(a) => write_typed_array(store, path, a.view(), dt, fill, chunking),
        Column::UInt16(a) => write_typed_array(store, path, a.view(), dt, fill, chunking),
        Column::UInt32(a) => write_typed_array(store, path, a.view(), dt, fill, chunking),
        Column::Bool(a) => write_typed_array(store, path, a.view(), dt, fill, chunking),
        Column::String(a) => write_typed_array(store, path, a.view(), dt, fill, chunking),
        Column::Complex64(a) => write_typed_array(store, path, a.view(), dt, fill, chunking),
        Column::Complex128(a) => write_typed_array(store, path, a.view(), dt, fill, chunking),
    }
}

/// The Zarr data type and fill value a column of this dtype is stored as.
#[cfg(feature = "zarr")]
fn dtype_of(col: &Column) -> (zarrs::array::DataType, zarrs::array::FillValue) {
    zarr_dtype(col.dtype())
}

/// The Zarr data type and fill value a column of `dtype` is stored as.
///
/// The fill value is always the dtype's zero — [`DType::itemsize`] zero bytes,
/// and no bytes (the empty string) for the variable-width variant. The frame
/// path writes its arrays whole, so no reader there ever observes a fill; the
/// zero is what the stores written before this function existed already
/// carried, and it is also what a sequence's growth array is padded with
/// beyond its live rows.
///
/// Keyed on [`DType`] rather than on a [`Column`] because the sequence writer
/// creates arrays from a *schema*, where no column value exists yet — this is
/// the one dtype table, and [`dtype_of`] is its column-shaped door.
#[cfg(feature = "zarr")]
pub(in crate::io::zarr) fn zarr_dtype(
    dtype: DType,
) -> (zarrs::array::DataType, zarrs::array::FillValue) {
    let dt = match dtype {
        DType::Float => data_type::float64(),
        DType::Int8 => data_type::int8(),
        DType::Int16 => data_type::int16(),
        DType::Int => data_type::int32(),
        DType::Int64 => data_type::int64(),
        DType::Bool => data_type::bool(),
        DType::UInt => data_type::uint64(),
        DType::U8 => data_type::uint8(),
        DType::UInt16 => data_type::uint16(),
        DType::UInt32 => data_type::uint32(),
        DType::String => data_type::string(),
        DType::Complex64 => data_type::complex64(),
        DType::Complex128 => data_type::complex128(),
    };
    let fill = zarrs::array::FillValue::new(vec![0u8; dtype.itemsize().unwrap_or(0)]);
    (dt, fill)
}

/// The one array writer: every array this backend stores is created and filled
/// here, laid out by `chunking` and compressed losslessly.
///
/// `chunking.chunks` of `None` — a variable-width dtype or an empty leading
/// axis, both of which molrec declines to size — keeps the pre-plan layout of
/// one chunk spanning the whole array. `chunking.shards` of `Some` packs those
/// chunks into one shard file per shard extent, with `gzip` on the inner
/// chunks and the shard index at the end (zarrs' default); `None` gzips the
/// chunks directly.
#[cfg(feature = "zarr")]
pub(in crate::io::zarr) fn write_typed_array<T>(
    store: &ReadableWritableListableStorage,
    path: &str,
    a: ArrayViewD<'_, T>,
    dt: zarrs::array::DataType,
    fill: impl Into<zarrs::array::builder::ArrayBuilderFillValue>,
    chunking: ChunkPlan,
) -> Result<(), MolRsError>
where
    T: zarrs::array::Element + Clone,
{
    let data = a.as_standard_layout();
    let shape: Vec<u64> = data.shape().iter().map(|&s| s as u64).collect();
    let chunk = chunking.chunks.unwrap_or_else(|| shape.clone());
    // Sharding makes the array's own chunk extent the *shard*, and the planned
    // chunk the subchunk inside it.
    let (extent, subchunk) = match chunking.shards {
        Some(shards) => (shards, Some(chunk)),
        None => (chunk, None),
    };
    let is_float = dt.is::<Float16DataType>()
        || dt.is::<Float32DataType>()
        || dt.is::<Float64DataType>()
        || dt.is::<Complex64DataType>()
        || dt.is::<Complex128DataType>();
    let mut builder = ArrayBuilder::new(shape.clone(), extent, dt, fill);
    // Lossless throughout: integers, booleans and strings gzip (they compress
    // by structure); floating-point payloads stay raw (they do not); every
    // chunk ends in `crc32c`. Under sharding these codecs encode the
    // subchunks, inside the shard.
    let mut codecs: Vec<Arc<dyn BytesToBytesCodecTraits>> = Vec::with_capacity(2);
    if !is_float {
        codecs.push(Arc::new(GzipCodec::new(GZIP_LEVEL).map_err(|e| {
            MolRsError::zarr(format!("gzip level {GZIP_LEVEL}: {e}"))
        })?));
    }
    codecs.push(Arc::new(Crc32cCodec::new()));
    builder.bytes_to_bytes_codecs(codecs);
    if let Some(subchunk) = subchunk {
        builder.subchunk_shape(subchunk);
    }
    let arr = builder.build(store.clone(), path)?;
    arr.store_metadata()?;
    arr.store_array_subset(
        &ArraySubset::new_with_shape(shape),
        data.as_slice().unwrap(),
    )?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Column read
// ---------------------------------------------------------------------------

/// Read the `subset` of the array at `path` back as a [`Column`] — at the
/// width it was stored in. Narrow floats are refused, not widened: the record
/// has one float ([`F`]).
///
/// The column's shape is the subset's, not the array's: a caller reading one
/// frame out of a sequence array passes that frame's subset and gets a column
/// shaped like the frame.
///
/// Generic over the store, and asking for reads only, so the read-write store
/// the record doors hold and the read-only one [`FrameSequence`] holds share
/// this one dtype dispatch.
///
/// [`F`]: crate::types::F
/// [`FrameSequence`]: super::FrameSequence
pub(crate) fn read_column<S>(
    store: &Arc<S>,
    path: &str,
    subset: &ArraySubset,
) -> Result<Column, MolRsError>
where
    S: ?Sized + ReadableStorageTraits + 'static,
{
    let arr = Array::open(store.clone(), path)?;
    read_column_array(&arr, subset)
}

/// [`read_column`] over an already-open array — the form a reader that keeps
/// its array handles across frames uses, so a frame read is one chunk fetch
/// rather than a metadata round trip plus a chunk fetch.
pub(crate) fn read_column_array<S>(
    arr: &Array<S>,
    subset: &ArraySubset,
) -> Result<Column, MolRsError>
where
    S: ?Sized + ReadableStorageTraits + 'static,
{
    let shape: Vec<usize> = subset.shape().iter().map(|&s| s as usize).collect();

    let dt = arr.data_type();

    // Narrow floats are refused, not promoted: the record has one float
    // (`F = f64`), so a `float16`/`float32` array on disk is an error naming
    // the array and its stored type. Same rule as `sequence.rs::dtype_of_stored`.
    if dt.is::<Float16DataType>() || dt.is::<Float32DataType>() {
        Err(MolRsError::zarr(format!(
            "{} is stored as {dt:?}: narrow floats are not read; \
             the record has one float, `F = f64`",
            arr.path()
        )))
    } else if dt.is::<Float64DataType>() {
        let data: Vec<f64> = arr.retrieve_array_subset(subset)?;
        Ok(Column::from_float(
            ArrayD::from_shape_vec(shape, data).map_err(shape_err)?,
        ))
    } else if dt.is::<Int8DataType>() {
        let data: Vec<i8> = arr.retrieve_array_subset(subset)?;
        Ok(Column::from_i8(
            ArrayD::from_shape_vec(shape, data).map_err(shape_err)?,
        ))
    } else if dt.is::<Int16DataType>() {
        let data: Vec<i16> = arr.retrieve_array_subset(subset)?;
        Ok(Column::from_i16(
            ArrayD::from_shape_vec(shape, data).map_err(shape_err)?,
        ))
    } else if dt.is::<Int32DataType>() {
        let data: Vec<i32> = arr.retrieve_array_subset(subset)?;
        Ok(Column::from_int(
            ArrayD::from_shape_vec(shape, data).map_err(shape_err)?,
        ))
    } else if dt.is::<Int64DataType>() {
        let data: Vec<i64> = arr.retrieve_array_subset(subset)?;
        Ok(Column::from_i64(
            ArrayD::from_shape_vec(shape, data).map_err(shape_err)?,
        ))
    } else if dt.is::<UInt16DataType>() {
        let data: Vec<u16> = arr.retrieve_array_subset(subset)?;
        Ok(Column::from_u16(
            ArrayD::from_shape_vec(shape, data).map_err(shape_err)?,
        ))
    } else if dt.is::<UInt32DataType>() {
        let data: Vec<u32> = arr.retrieve_array_subset(subset)?;
        Ok(Column::from_u32(
            ArrayD::from_shape_vec(shape, data).map_err(shape_err)?,
        ))
    } else if dt.is::<UInt64DataType>() {
        let data: Vec<u64> = arr.retrieve_array_subset(subset)?;
        Ok(Column::from_uint(
            ArrayD::from_shape_vec(shape, data).map_err(shape_err)?,
        ))
    } else if dt.is::<BoolDataType>() {
        let data: Vec<bool> = arr.retrieve_array_subset(subset)?;
        Ok(Column::from_bool(
            ArrayD::from_shape_vec(shape, data).map_err(shape_err)?,
        ))
    } else if dt.is::<UInt8DataType>() {
        let data: Vec<u8> = arr.retrieve_array_subset(subset)?;
        Ok(Column::from_u8(
            ArrayD::from_shape_vec(shape, data).map_err(shape_err)?,
        ))
    } else if dt.is::<StringDataType>() {
        let data: Vec<String> = arr.retrieve_array_subset(subset)?;
        Ok(Column::from_string(
            ArrayD::from_shape_vec(shape, data).map_err(shape_err)?,
        ))
    } else if dt.is::<Complex64DataType>() {
        let data: Vec<num_complex::Complex<f32>> = arr.retrieve_array_subset(subset)?;
        Ok(Column::from_c64(
            ArrayD::from_shape_vec(shape, data).map_err(shape_err)?,
        ))
    } else if dt.is::<Complex128DataType>() {
        let data: Vec<num_complex::Complex<f64>> = arr.retrieve_array_subset(subset)?;
        Ok(Column::from_c128(
            ArrayD::from_shape_vec(shape, data).map_err(shape_err)?,
        ))
    } else {
        Err(MolRsError::zarr(format!("unsupported dtype: {:?}", dt)))
    }
}

/// Insert a column read from a store into `block`.
///
/// The store read path is strict where the in-memory API is lenient:
/// [`Block::insert_column`] widens a narrow unsigned array under a canonical
/// identifier key to `u64`, but a store that holds one at another width is
/// refused here (see [`check_canonical_width`]) rather than silently widened.
pub(crate) fn insert_column_into_block(
    block: &mut Block,
    name: &str,
    col: Column,
) -> Result<(), MolRsError> {
    check_canonical_width(name, col.dtype())?;
    // Zero-copy insert: hand the Arc-backed Column directly to the Block.
    block.insert_column(name, col).map_err(MolRsError::Block)
}

/// Refuse a canonical identifier / endpoint key (`id`, `atomic_number`,
/// `mol_id`, `res_id`, `type_id`, `atomi` … `atoml`, `bond_type`,
/// `bond_number`) stored at any width but `u64`.
///
/// Every molrs writer stores those keys as `u64` — the in-memory block widens
/// them on insert — so a narrower stored array came from a producer that broke
/// the contract, and reading it back as `u64` would hide that.
pub(crate) fn check_canonical_width(
    name: &str,
    stored: molrs::store::block::DType,
) -> Result<(), MolRsError> {
    use molrs::store::block::DType;
    match molrs::store::schema::column(name) {
        Some(spec) if spec.dtype == DType::UInt && stored != DType::UInt => {
            Err(MolRsError::zarr(format!(
                "column {name:?} is stored as {}; the canonical key {name:?} is u64, \
                 and a store is not widened on read",
                stored.name()
            )))
        }
        _ => Ok(()),
    }
}

// ---------------------------------------------------------------------------
// SimBox write / read
// ---------------------------------------------------------------------------

#[cfg(feature = "zarr")]
pub(crate) fn write_simbox(
    store: &ReadableWritableListableStorage,
    prefix: &str,
    simbox: &SimBox,
) -> Result<(), MolRsError> {
    let mut attrs = serde_json::Map::new();
    // `cell_defined` is written only when it is *false*: absent means a
    // defined cell.
    if !simbox.is_cell_defined() {
        attrs.insert("cell_defined".to_string(), serde_json::Value::Bool(false));
    }
    GroupBuilder::new()
        .attributes(attrs)
        .build(store.clone(), prefix)?
        .store_metadata()?;

    // vectors: [3,3] Float64 — lattice vectors are the COLUMNS, per contract.
    // Geometry stays f64 to match the in-memory science contract.
    let h_view = simbox.h_view();
    let h_data: Vec<F> = h_view.iter().copied().collect();
    write_f64_array(store, &format!("{}/vectors", prefix), &[3, 3], &h_data)?;

    // origin: [3] Float64, written only when it carries information (the
    // normative default is the coordinate origin).
    let origin_view = simbox.origin_view();
    if origin_view.iter().any(|&v| v != 0.0) {
        let origin_data: Vec<F> = origin_view.iter().copied().collect();
        write_f64_array(store, &format!("{}/origin", prefix), &[3], &origin_data)?;
    }

    // boundary: [3] bool, the same array form the trajectory path writes;
    // omitted for the all-periodic default.
    let pbc = simbox.pbc();
    if pbc != [true, true, true] {
        let flags = ndarray::ArrayD::from_shape_vec(vec![3], pbc.to_vec()).map_err(shape_err)?;
        write_column(
            store,
            &format!("{}/boundary", prefix),
            &Column::from_bool(flags),
        )?;
    }

    Ok(())
}

pub(crate) fn read_simbox(
    store: &ReadableWritableListableStorage,
    prefix: &str,
) -> Result<SimBox, MolRsError> {
    use ndarray::{Array2, array};

    let group = zarrs::group::Group::open(store.clone(), prefix)?;

    // `cell_defined` is optional and only ever written when false, so absent
    // means a defined cell.
    let cell_defined = group
        .attributes()
        .get("cell_defined")
        .and_then(|v| v.as_bool())
        .unwrap_or(true);

    // An undefined cell's `vectors` mean nothing, so they are not read:
    // whatever a store holds there (zeros included) is accepted, never
    // inverted, and `SimBox::new_cell` carries the identity instead.
    let h = if cell_defined {
        // Narrow float arrays are refused; see `read_simbox_float_path`.
        let vectors_path = format!("{}/vectors", prefix);
        let h_data = read_simbox_float_path(store, &vectors_path)?.ok_or_else(|| {
            MolRsError::zarr(format!("box vectors array is missing: {vectors_path}"))
        })?;
        if h_data.len() != 9 {
            return Err(MolRsError::zarr(format!(
                "box vectors expected 9 values, got {}",
                h_data.len()
            )));
        }
        Array2::from_shape_vec((3, 3), h_data).map_err(shape_err)?
    } else {
        Array2::eye(3)
    };

    // An absent `origin` array is the zero origin: molrec's codec omits it for
    // a cell anchored at the coordinate origin, and that store is a well-formed
    // cell rather than a malformed one.
    let o_data = read_simbox_float_path(store, &format!("{}/origin", prefix))?
        .unwrap_or_else(|| vec![0.0; 3]);
    if o_data.len() != 3 {
        return Err(MolRsError::zarr(format!(
            "box origin expected 3 values, got {}",
            o_data.len()
        )));
    }
    let origin = array![o_data[0], o_data[1], o_data[2]];

    // Boundary flags are a `bool[3]` array. An absent array is fully
    // periodic, the normative default -- reading it as vacuum would make one
    // store two different physical systems depending on which implementation
    // opened it. A store from before the array form carried the flags as a
    // group attribute; that is still honoured.
    let boundary_path = format!("{}/boundary", prefix);
    let pbc = match Array::open(store.clone(), &boundary_path) {
        Ok(arr) => {
            let flags: Vec<bool> =
                arr.retrieve_array_subset(&ArraySubset::new_with_shape(arr.shape().to_vec()))?;
            if flags.len() != 3 {
                return Err(MolRsError::zarr(format!(
                    "box boundary expected 3 flags, got {}",
                    flags.len()
                )));
            }
            [flags[0], flags[1], flags[2]]
        }
        Err(zarrs::array::ArrayCreateError::MissingMetadata) => match group
            .attributes()
            .get("boundary")
            .and_then(|v| v.as_array())
        {
            Some(flags) if flags.len() == 3 => [
                flags[0].as_bool().unwrap_or(true),
                flags[1].as_bool().unwrap_or(true),
                flags[2].as_bool().unwrap_or(true),
            ],
            _ => [true, true, true],
        },
        Err(e) => return Err(e.into()),
    };

    SimBox::new_cell(h, origin, pbc, cell_defined)
        .map_err(|e| MolRsError::zarr(format!("invalid box: {:?}", e)))
}

/// Read a simbox float array as `Vec<F>` (narrow float arrays are refused —
/// the record has one float), or `None` when the store holds no array at `path`.
///
/// Absence is reported to the caller instead of being raised, because the
/// optional parts of a cell are absent on purpose; every *other* way of failing
/// to open the array is still an error.
fn read_simbox_float_path(
    store: &ReadableWritableListableStorage,
    path: &str,
) -> Result<Option<Vec<F>>, MolRsError> {
    let arr = match Array::open(store.clone(), path) {
        Ok(arr) => arr,
        Err(zarrs::array::ArrayCreateError::MissingMetadata) => return Ok(None),
        Err(e) => return Err(e.into()),
    };
    let subset = ArraySubset::new_with_shape(arr.shape().to_vec());
    let dt = arr.data_type();
    if dt.is::<Float64DataType>() {
        let data: Vec<f64> = arr.retrieve_array_subset(&subset)?;
        Ok(Some(data))
    } else {
        Err(MolRsError::zarr(format!(
            "simbox array expected float64, got {dt:?}: the record has one float, `F = f64`"
        )))
    }
}

// ---------------------------------------------------------------------------
/// The one reserved child name in a frame group: the cell.
///
/// Rust cannot spell it `box` -- that is a reserved keyword -- but nothing
/// outside Rust source has that problem, so the stored name is `box`.
pub(crate) const BOX_GROUP: &str = "box";

/// The one reserved child name of a *block* group: the subgroup holding the
/// validity masks of that block's nullable columns.
///
/// One leading underscore, not two: Zarr V3 reserves the `__` prefix for node
/// names and `zarrs` enforces it, so `__validity__` is a name no store can
/// carry. A single underscore is legal everywhere and still says, to a reader
/// scanning a block, that this is not a chemistry column.
pub(crate) const VALIDITY_GROUP: &str = "_validity";

// Frame (system) write / read — writes all blocks under `{prefix}/`
// ---------------------------------------------------------------------------

#[cfg(feature = "zarr")]
/// Write one [`Frame`] as a Zarr group of blocks.
///
/// The group carries **no** schema-version attribute: `meta/molrec_version`
/// at the record root is the sole version key of the MolRec contract, and a
/// parallel per-frame version is forbidden by it.
///
/// # A nullable column carries its mask beside its values
///
/// A block column may carry a [validity
/// mask](crate::store::block::Block::validity), and the mask is data: without
/// it a row that holds *nothing* reads back as the default filled under it.
/// Each masked column of a block therefore writes one `bool` array, one flag
/// per row, at `<block>/`[`_validity`](VALIDITY_GROUP)`/<column>` — a
/// reserved **subgroup** of the block group, holding the masks of that block
/// and nothing else.
///
/// The subgroup is the reason the layout stays backward compatible in both
/// directions. [`read_frame_group`] skips every non-Array child of a block,
/// so a reader that predates masks — molrs <= 0.15, molrec, molvis — walks
/// past the group instead of taking it for a column; and a block no column
/// of which is masked writes no subgroup at all, so its store is
/// byte-identical to one written before masks existed. A store written then
/// reads now as fully valid, which is exactly what it was.
///
/// `_validity` is reserved among a block group's children the way `box` is
/// among a frame group's: a block carrying a column of that name is refused
/// here rather than silently merged with the masks.
pub(crate) fn write_frame_group(
    store: &ReadableWritableListableStorage,
    prefix: &str,
    frame: &Frame,
) -> Result<(), MolRsError> {
    // Erase before writing: this group is the target node, so whatever it held
    // before is not part of the frame being written.
    store.erase_prefix(&node_prefix(prefix)?)?;

    // The frame's meta document is this group's attribute map, not a child
    // group: the contract binds document sections as attributes, and a `meta`
    // child would also steal a name from the block namespace.
    let mut meta_attrs = serde_json::Map::new();
    for (k, v) in &frame.meta {
        meta_attrs.insert(k.clone(), v.to_attr_value());
    }
    GroupBuilder::new()
        .attributes(meta_attrs)
        .build(store.clone(), prefix)?
        .store_metadata()?;

    // The cell. `box` is the one reserved name among a frame group's children.
    if let Some(ref simbox) = frame.simbox {
        if frame.iter().any(|(name, _)| name == BOX_GROUP) {
            return Err(MolRsError::zarr(format!(
                "{BOX_GROUP:?} names the cell in a frame group; a block cannot take it"
            )));
        }
        write_simbox(store, &format!("{}/{}", prefix, BOX_GROUP), simbox)?;
    }

    // Blocks (atoms, bonds, angles, …). A block with no rows is still a block
    // and still has a count, so it gets a group and an attribute rather than
    // being dropped -- silently losing it would be data loss.
    for (block_name, block) in frame.iter() {
        if block.contains_key(VALIDITY_GROUP) {
            return Err(MolRsError::zarr(format!(
                "{VALIDITY_GROUP:?} names the validity masks of a block group; a column cannot \
                 take it"
            )));
        }
        let group_path = format!("{}/{}", prefix, block_name);
        let mut block_attrs = serde_json::Map::new();
        block_attrs.insert(
            "count".to_string(),
            serde_json::Value::from(block.nrows().unwrap_or(0)),
        );
        if let Some(shape) = block.structural_shape() {
            block_attrs.insert(
                "structural_shape".to_string(),
                serde_json::Value::Array(
                    shape
                        .iter()
                        .map(|n| serde_json::Value::from(*n as u64))
                        .collect(),
                ),
            );
        }
        GroupBuilder::new()
            .attributes(block_attrs)
            .build(store.clone(), &group_path)?
            .store_metadata()?;

        for (col_name, col) in block.iter() {
            let arr_path = format!("{}/{}/{}", prefix, block_name, col_name);
            write_column(store, &arr_path, col)?;
        }
        write_validity_group(store, &group_path, block)?;
    }

    Ok(())
}

/// Write the validity masks of `block` into its reserved `_validity`
/// subgroup, or write nothing at all when no column of it is masked.
///
/// One `bool` array per masked column, named after that column and carrying
/// one flag per row — the same leading row axis as the values it qualifies,
/// and no trailing axes, because a mask marks a *row* null whatever shape the
/// row has.
#[cfg(feature = "zarr")]
fn write_validity_group(
    store: &ReadableWritableListableStorage,
    block_path: &str,
    block: &Block,
) -> Result<(), MolRsError> {
    let masked: Vec<(&str, &[bool])> = block
        .iter()
        .filter_map(|(column, _)| block.validity(column).map(|mask| (column, mask)))
        .collect();
    if masked.is_empty() {
        return Ok(());
    }
    let group_path = format!("{}/{}", block_path, VALIDITY_GROUP);
    GroupBuilder::new()
        .build(store.clone(), &group_path)?
        .store_metadata()?;
    for (column, mask) in masked {
        let flags = ArrayD::from_shape_vec(vec![mask.len()], mask.to_vec()).map_err(shape_err)?;
        write_column(
            store,
            &format!("{}/{}", group_path, column),
            &Column::from_bool(flags),
        )?;
    }
    Ok(())
}

/// Read one [`Frame`] back from a Zarr group written by [`write_frame_group`].
pub(crate) fn read_frame_group(
    store: &ReadableWritableListableStorage,
    prefix: &str,
) -> Result<Frame, MolRsError> {
    let mut frame = Frame::new();

    // Meta lives in this group's own attributes.
    if let Ok(frame_group) = zarrs::group::Group::open(store.clone(), prefix) {
        for (k, v) in frame_group.attributes() {
            frame.meta.insert(k.clone(), MetaValue::from_attr_value(v));
        }
    }

    // The cell.
    let box_path = format!("{}/{}", prefix, BOX_GROUP);
    if zarrs::group::Group::open(store.clone(), &box_path).is_ok() {
        frame.simbox = Some(read_simbox(store, &box_path)?);
    }

    // Blocks
    let frame_node = Node::open(store, prefix)?;
    for child in frame_node.children() {
        let child_name = child.path().as_str().rsplit('/').next().unwrap_or("");
        if child_name == BOX_GROUP || child_name.is_empty() {
            continue;
        }
        if !matches!(child.metadata(), NodeMetadata::Group(_)) {
            continue;
        }
        let mut block = Block::new();
        let block_node = Node::open(store, child.path().as_str())?;
        for col_child in block_node.children() {
            if !matches!(col_child.metadata(), NodeMetadata::Array(_)) {
                continue;
            }
            let col_path = col_child.path().as_str();
            let col_name = col_path.rsplit('/').next().unwrap_or("");
            // A frame group's column is read whole: the array *is* the column.
            let whole =
                ArraySubset::new_with_shape(Array::open(store.clone(), col_path)?.shape().to_vec());
            let col = read_column(store, col_path, &whole)?;
            insert_column_into_block(&mut block, col_name, col)?;
        }
        if let Ok(group) = zarrs::group::Group::open(store.clone(), child.path().as_str()) {
            let attrs = group.attributes();
            if let Some(count) = attrs.get("count").and_then(|v| v.as_u64()) {
                let count = count as usize;
                if block.is_empty() {
                    block.resize(count).map_err(|e| {
                        MolRsError::zarr(format!("block {child_name:?} count={count}: {e}"))
                    })?;
                } else if block.nrows() != Some(count) {
                    return Err(MolRsError::zarr(format!(
                        "row_count_mismatch: block {child_name:?} count={count}, columns have {}",
                        block.nrows().unwrap_or(0)
                    )));
                }
            }
            if let Some(shape) = attrs.get("structural_shape").and_then(|v| v.as_array()) {
                let shape: Vec<usize> = shape
                    .iter()
                    .filter_map(|v| v.as_u64().map(|n| n as usize))
                    .collect();
                if !shape.is_empty() {
                    block.set_shape(&shape).map_err(|e| {
                        MolRsError::zarr(format!(
                            "block {child_name:?} structural_shape {shape:?}: {e}"
                        ))
                    })?;
                }
            }
        }
        read_validity_group(store, child.path().as_str(), child_name, &mut block)?;
        frame.insert(child_name, block);
    }

    Ok(frame)
}

/// Restore the validity masks [`write_validity_group`] wrote for `block`.
///
/// A block group with no `_validity` child is fully valid, which is what
/// every store written before masks were persisted is.
///
/// # Errors
///
/// A [`MolRsError::Zarr`] naming block and column when the mask is not a
/// `bool` array, when it names no column of the block, or when it does not
/// carry exactly one flag per row. A mask that disagrees with its column is a
/// corrupt store: padding or truncating it would invent the very answer the
/// mask exists to give.
fn read_validity_group(
    store: &ReadableWritableListableStorage,
    block_path: &str,
    block_name: &str,
    block: &mut Block,
) -> Result<(), MolRsError> {
    let group_path = format!("{}/{}", block_path, VALIDITY_GROUP);
    if zarrs::group::Group::open(store.clone(), &group_path).is_err() {
        return Ok(());
    }
    let rows = block.nrows().unwrap_or(0);
    for child in Node::open(store, &group_path)?.children() {
        if !matches!(child.metadata(), NodeMetadata::Array(_)) {
            continue;
        }
        let path = child.path().as_str();
        let column = path.rsplit('/').next().unwrap_or("");
        let array = Array::open(store.clone(), path)?;
        let dtype = array.data_type();
        if !dtype.is::<BoolDataType>() {
            return Err(MolRsError::zarr(format!(
                "validity mask of column {column:?} of block {block_name:?} is stored as \
                 {dtype:?}, and a mask is one bool per row"
            )));
        }
        let subset = ArraySubset::new_with_shape(array.shape().to_vec());
        let mask: Vec<bool> = array.retrieve_array_subset(&subset)?;
        if mask.len() != rows {
            return Err(MolRsError::zarr(format!(
                "validity mask of column {column:?} of block {block_name:?} carries {} flags for \
                 {rows} rows",
                mask.len()
            )));
        }
        block.set_validity(column, mask).map_err(|e| {
            MolRsError::zarr(format!(
                "validity mask of column {column:?} of block {block_name:?}: {e}"
            ))
        })?;
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Primitive array helpers
// ---------------------------------------------------------------------------

/// Write a flat `f64` payload of `shape` — the form the cell's `vectors` and
/// `origin` arrive in, already row-major.
#[cfg(feature = "zarr")]
pub(crate) fn write_f64_array(
    store: &ReadableWritableListableStorage,
    path: &str,
    shape: &[u64],
    data: &[F],
) -> Result<(), MolRsError> {
    let dims: Vec<usize> = shape.iter().map(|&s| s as usize).collect();
    write_typed_array(
        store,
        path,
        ArrayViewD::from_shape(dims, data).map_err(shape_err)?,
        data_type::float64(),
        0.0f64,
        plan(shape, DType::Float.itemsize()),
    )
}

fn shape_err(e: impl std::fmt::Display) -> MolRsError {
    MolRsError::zarr(format!("shape error: {}", e))
}

/// The store prefix that holds a node's own metadata key and every descendant,
/// for a node addressed by an absolute Zarr path such as `/frame/atoms`.
///
/// Erasing this prefix is how a writer clears its target before writing: Zarr
/// writes are key-by-key, so a rewrite that only stores the new keys inherits
/// every child of whatever was there before.
#[cfg(feature = "zarr")]
pub(crate) fn node_prefix(path: &str) -> Result<StorePrefix, MolRsError> {
    let trimmed = path.trim_matches('/');
    if trimmed.is_empty() {
        return Ok(StorePrefix::root());
    }
    StorePrefix::new(format!("{trimmed}/"))
        .map_err(|e| MolRsError::zarr(format!("invalid store prefix for {path:?}: {e}")))
}

/// Build a child path from a prefix, avoiding double slashes.
pub(crate) fn join_path(prefix: &str, child: &str) -> String {
    if prefix == "/" {
        format!("/{}", child)
    } else {
        format!("{}/{}", prefix.trim_end_matches('/'), child)
    }
}

#[cfg(all(test, feature = "filesystem"))]
mod tests {
    use super::*;
    use molrs::store::block::DType;
    use ndarray::array;
    use num_complex::Complex;
    use std::ffi::OsStr;
    use std::num::NonZeroU64;
    use std::path::{Path, PathBuf};
    use std::sync::Arc;
    use tempfile::TempDir;
    use zarrs::filesystem::FilesystemStore;

    /// The frame group every round-trip writes into.
    const FRAME: &str = "/frame";
    /// The one block in that frame.
    const BLOCK: &str = "atoms";
    /// A column name outside the schema vocabulary, so no canonical dtype
    /// promotion can mask the arrival width under test.
    const COLUMN: &str = "probe";

    fn store_in(dir: &TempDir) -> ReadableWritableListableStorage {
        Arc::new(FilesystemStore::new(dir.path()).unwrap())
    }

    /// Write one column as the sole column of a one-block frame and read the
    /// whole frame back, returning the column as it arrived from the store.
    fn round_trip_column(col: Column) -> Column {
        let dir = TempDir::new().unwrap();
        let store = store_in(&dir);
        let mut block = Block::new();
        block.insert_column(COLUMN, col).unwrap();
        let mut frame = Frame::new();
        frame.insert(BLOCK, block);
        write_frame_group(&store, FRAME, &frame).unwrap();
        read_frame_group(&store, FRAME)
            .unwrap()
            .get(BLOCK)
            .unwrap()
            .get(COLUMN)
            .unwrap()
            .clone()
    }

    // -- the 13-dtype matrix: one test per Column variant -------------------

    #[test]
    fn f64_column_round_trips_at_arrival_width() {
        let values = vec![1.0f64, -2.5, 1.0e-300, f64::MAX];
        let back = round_trip_column(Column::from_float(
            ArrayD::from_shape_vec(vec![4], values.clone()).unwrap(),
        ));
        assert_eq!(back.dtype(), DType::Float);
        assert_eq!(
            *back.as_float().unwrap(),
            ArrayD::from_shape_vec(vec![4], values).unwrap()
        );
    }

    #[test]
    fn i8_column_round_trips_at_arrival_width() {
        let values = vec![i8::MIN, -1, 0, i8::MAX];
        let back = round_trip_column(Column::from_i8(
            ArrayD::from_shape_vec(vec![4], values.clone()).unwrap(),
        ));
        assert_eq!(back.dtype(), DType::Int8);
        assert_eq!(
            *back.as_i8().unwrap(),
            ArrayD::from_shape_vec(vec![4], values).unwrap()
        );
    }

    /// A canonical identifier stored narrower than `u64` is refused on read,
    /// not widened — the in-memory insert widens, the store read does not.
    #[test]
    fn a_narrow_canonical_identifier_is_refused_on_read() {
        for (key, column) in [
            (
                "atomi",
                Column::from_u32(ArrayD::from_shape_vec(vec![2], vec![0u32, 1]).unwrap()),
            ),
            (
                "type_id",
                Column::from_u8(ArrayD::from_shape_vec(vec![2], vec![0u8, 1]).unwrap()),
            ),
            (
                "bond_type",
                Column::from_u16(ArrayD::from_shape_vec(vec![2], vec![1u16, 4]).unwrap()),
            ),
        ] {
            let dir = TempDir::new().unwrap();
            let store = store_in(&dir);
            for group in ["/f", "/f/b"] {
                GroupBuilder::new()
                    .build(store.clone(), group)
                    .unwrap()
                    .store_metadata()
                    .unwrap();
            }
            write_column(&store, &format!("/f/b/{key}"), &column).unwrap();
            let err = read_frame_group(&store, "/f").unwrap_err().to_string();
            assert!(
                err.contains(key) && err.contains(column.dtype().name()),
                "{err}"
            );

            // The same array under a non-canonical name reads at its width.
            write_column(&store, "/f/b/label", &column).unwrap();
            std::fs::remove_dir_all(dir.path().join("f/b").join(key)).unwrap();
            let back = read_frame_group(&store, "/f").unwrap();
            assert_eq!(
                back.get("b").unwrap().get("label").unwrap().dtype(),
                column.dtype()
            );
        }

        // In memory the same insert widens to u64.
        let mut block = Block::new();
        block
            .insert_column(
                "atomi",
                Column::from_u32(ArrayD::from_shape_vec(vec![1], vec![7u32]).unwrap()),
            )
            .unwrap();
        assert_eq!(block.get("atomi").unwrap().dtype(), DType::UInt);
    }

    #[test]
    fn i16_column_round_trips_at_arrival_width() {
        let values = vec![i16::MIN, -1, 0, i16::MAX];
        let back = round_trip_column(Column::from_i16(
            ArrayD::from_shape_vec(vec![4], values.clone()).unwrap(),
        ));
        assert_eq!(back.dtype(), DType::Int16);
        assert_eq!(
            *back.as_i16().unwrap(),
            ArrayD::from_shape_vec(vec![4], values).unwrap()
        );
    }

    #[test]
    fn i32_column_round_trips_at_arrival_width() {
        let values = vec![i32::MIN, -1, 0, i32::MAX];
        let back = round_trip_column(Column::from_int(
            ArrayD::from_shape_vec(vec![4], values.clone()).unwrap(),
        ));
        assert_eq!(back.dtype(), DType::Int);
        assert_eq!(
            *back.as_int().unwrap(),
            ArrayD::from_shape_vec(vec![4], values).unwrap()
        );
    }

    #[test]
    fn i64_column_round_trips_at_arrival_width() {
        let values = vec![i64::MIN, -1, 0, i64::MAX];
        let back = round_trip_column(Column::from_i64(
            ArrayD::from_shape_vec(vec![4], values.clone()).unwrap(),
        ));
        assert_eq!(back.dtype(), DType::Int64);
        assert_eq!(
            *back.as_i64().unwrap(),
            ArrayD::from_shape_vec(vec![4], values).unwrap()
        );
    }

    #[test]
    fn u8_column_round_trips_at_arrival_width() {
        let values = vec![0u8, 1, 128, u8::MAX];
        let back = round_trip_column(Column::from_u8(
            ArrayD::from_shape_vec(vec![4], values.clone()).unwrap(),
        ));
        assert_eq!(back.dtype(), DType::U8);
        assert_eq!(
            *back.as_u8().unwrap(),
            ArrayD::from_shape_vec(vec![4], values).unwrap()
        );
    }

    #[test]
    fn u16_column_round_trips_at_arrival_width() {
        let values = vec![0u16, 1, 40000, u16::MAX];
        let back = round_trip_column(Column::from_u16(
            ArrayD::from_shape_vec(vec![4], values.clone()).unwrap(),
        ));
        assert_eq!(back.dtype(), DType::UInt16);
        assert_eq!(
            *back.as_u16().unwrap(),
            ArrayD::from_shape_vec(vec![4], values).unwrap()
        );
    }

    #[test]
    fn u32_column_round_trips_at_arrival_width() {
        let values = vec![0u32, 1, 3_000_000_000, u32::MAX];
        let back = round_trip_column(Column::from_u32(
            ArrayD::from_shape_vec(vec![4], values.clone()).unwrap(),
        ));
        assert_eq!(back.dtype(), DType::UInt32);
        assert_eq!(
            *back.as_u32().unwrap(),
            ArrayD::from_shape_vec(vec![4], values).unwrap()
        );
    }

    #[test]
    fn u64_column_round_trips_at_arrival_width() {
        let values = vec![0u64, 1, 9_007_199_254_740_993, u64::MAX];
        let back = round_trip_column(Column::from_uint(
            ArrayD::from_shape_vec(vec![4], values.clone()).unwrap(),
        ));
        assert_eq!(back.dtype(), DType::UInt);
        assert_eq!(
            *back.as_uint().unwrap(),
            ArrayD::from_shape_vec(vec![4], values).unwrap()
        );
    }

    #[test]
    fn bool_column_round_trips_at_arrival_width() {
        let values = vec![true, false, true, true];
        let back = round_trip_column(Column::from_bool(
            ArrayD::from_shape_vec(vec![4], values.clone()).unwrap(),
        ));
        assert_eq!(back.dtype(), DType::Bool);
        assert_eq!(
            *back.as_bool().unwrap(),
            ArrayD::from_shape_vec(vec![4], values).unwrap()
        );
    }

    #[test]
    fn string_column_round_trips_at_arrival_width() {
        let values = vec![
            "atom".to_string(),
            String::new(),
            "Ω".to_string(),
            "x y".to_string(),
        ];
        let back = round_trip_column(Column::from_string(
            ArrayD::from_shape_vec(vec![4], values.clone()).unwrap(),
        ));
        assert_eq!(back.dtype(), DType::String);
        assert_eq!(
            *back.as_string().unwrap(),
            ArrayD::from_shape_vec(vec![4], values).unwrap()
        );
    }

    #[test]
    fn c64_column_round_trips_at_arrival_width() {
        let values = vec![
            Complex::new(1.5f32, -2.5f32),
            Complex::new(0.0f32, 0.0f32),
            Complex::new(-0.125f32, 65536.0f32),
            Complex::new(3.4028235e38f32, 1.1754944e-38f32),
        ];
        let back = round_trip_column(Column::from_c64(
            ArrayD::from_shape_vec(vec![4], values.clone()).unwrap(),
        ));
        assert_eq!(back.dtype(), DType::Complex64);
        assert_eq!(
            *back.as_c64().unwrap(),
            ArrayD::from_shape_vec(vec![4], values).unwrap()
        );
    }

    #[test]
    fn c128_column_round_trips_at_arrival_width() {
        let values = vec![
            Complex::new(1.5f64, -2.5f64),
            Complex::new(0.0f64, 0.0f64),
            Complex::new(-0.125f64, 65536.0f64),
            Complex::new(f64::MAX, 1.0e-300f64),
        ];
        let back = round_trip_column(Column::from_c128(
            ArrayD::from_shape_vec(vec![4], values.clone()).unwrap(),
        ));
        assert_eq!(back.dtype(), DType::Complex128);
        assert_eq!(
            *back.as_c128().unwrap(),
            ArrayD::from_shape_vec(vec![4], values).unwrap()
        );
    }

    // -- block-level shape and count ----------------------------------------

    #[test]
    fn structural_shape_round_trips_with_the_block() {
        let dir = TempDir::new().unwrap();
        let store = store_in(&dir);
        let mut block = Block::new();
        block
            .insert_column(
                COLUMN,
                Column::from_float(
                    ArrayD::from_shape_vec(vec![6], vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0]).unwrap(),
                ),
            )
            .unwrap();
        block.set_shape(&[3, 2]).unwrap();
        let mut frame = Frame::new();
        frame.insert(BLOCK, block);
        write_frame_group(&store, FRAME, &frame).unwrap();

        let back = read_frame_group(&store, FRAME).unwrap();
        assert_eq!(
            back.get(BLOCK).unwrap().structural_shape(),
            Some(&[3usize, 2usize][..])
        );
    }

    #[test]
    fn structural_shape_is_written_as_a_group_attribute() {
        let dir = TempDir::new().unwrap();
        let store = store_in(&dir);
        let mut block = Block::new();
        block
            .insert_column(
                COLUMN,
                Column::from_float(
                    ArrayD::from_shape_vec(vec![6], vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0]).unwrap(),
                ),
            )
            .unwrap();
        block.set_shape(&[3, 2]).unwrap();
        let mut frame = Frame::new();
        frame.insert(BLOCK, block);
        write_frame_group(&store, FRAME, &frame).unwrap();

        let group = zarrs::group::Group::open(store.clone(), &format!("{FRAME}/{BLOCK}")).unwrap();
        assert_eq!(
            group.attributes().get("structural_shape"),
            Some(&serde_json::json!([3, 2]))
        );
    }

    #[test]
    fn column_less_block_keeps_its_row_count() {
        let dir = TempDir::new().unwrap();
        let store = store_in(&dir);
        let mut block = Block::new();
        block.resize(7).unwrap();
        let mut frame = Frame::new();
        frame.insert("ghost", block);
        write_frame_group(&store, FRAME, &frame).unwrap();

        let back = read_frame_group(&store, FRAME).unwrap();
        assert_eq!(back.get("ghost").unwrap().nrows(), Some(7));
    }

    // -- bool is native, not a tagged uint8 (ac-005) -------------------------

    #[test]
    fn bool_column_is_written_as_the_native_zarr_bool_dtype() {
        let dir = TempDir::new().unwrap();
        let store = store_in(&dir);
        let mut block = Block::new();
        block
            .insert_column(
                COLUMN,
                Column::from_bool(
                    ArrayD::from_shape_vec(vec![3], vec![true, false, true]).unwrap(),
                ),
            )
            .unwrap();
        let mut frame = Frame::new();
        frame.insert(BLOCK, block);
        write_frame_group(&store, FRAME, &frame).unwrap();

        let arr = Array::open(store.clone(), &format!("{FRAME}/{BLOCK}/{COLUMN}")).unwrap();
        assert!(arr.data_type().is::<BoolDataType>());
    }

    #[test]
    fn bool_column_carries_no_molrs_dtype_attribute() {
        let dir = TempDir::new().unwrap();
        let store = store_in(&dir);
        let mut block = Block::new();
        block
            .insert_column(
                COLUMN,
                Column::from_bool(
                    ArrayD::from_shape_vec(vec![3], vec![true, false, true]).unwrap(),
                ),
            )
            .unwrap();
        let mut frame = Frame::new();
        frame.insert(BLOCK, block);
        write_frame_group(&store, FRAME, &frame).unwrap();

        let arr = Array::open(store.clone(), &format!("{FRAME}/{BLOCK}/{COLUMN}")).unwrap();
        assert_eq!(arr.attributes().get("molrs_dtype"), None);
    }

    // -- the cell: boundary / origin defaults (ac-006) -----------------------

    /// The one cell shape every box test writes: an orthogonal 10/11/12 cell.
    fn cell_vectors() -> [F; 9] {
        [10.0, 0.0, 0.0, 0.0, 11.0, 0.0, 0.0, 0.0, 12.0]
    }

    #[test]
    fn absent_boundary_reads_all_periodic() {
        let dir = TempDir::new().unwrap();
        let store = store_in(&dir);
        // Hand-built: the writer always emits `boundary`, but molrec's own
        // codec omits it for the default case, and that store must read as
        // fully periodic rather than as vacuum.
        GroupBuilder::new()
            .build(store.clone(), "/box")
            .unwrap()
            .store_metadata()
            .unwrap();
        write_f64_array(&store, "/box/vectors", &[3, 3], &cell_vectors()).unwrap();
        write_f64_array(&store, "/box/origin", &[3], &[0.0, 0.0, 0.0]).unwrap();

        let simbox = read_simbox(&store, "/box").unwrap();
        assert_eq!(simbox.pbc_view().to_vec(), vec![true, true, true]);
    }

    #[test]
    fn explicit_boundary_round_trips_unchanged() {
        let dir = TempDir::new().unwrap();
        let store = store_in(&dir);
        let simbox = SimBox::new(
            array![[10.0, 0.0, 0.0], [0.0, 11.0, 0.0], [0.0, 0.0, 12.0]],
            array![0.0, 0.0, 0.0],
            [true, false, true],
        )
        .unwrap();
        write_simbox(&store, "/box", &simbox).unwrap();

        let back = read_simbox(&store, "/box").unwrap();
        assert_eq!(back.pbc_view().to_vec(), vec![true, false, true]);
    }

    #[test]
    fn defined_cell_stays_defined_across_the_round_trip() {
        let dir = TempDir::new().unwrap();
        let store = store_in(&dir);
        let simbox = SimBox::new(
            array![[10.0, 0.0, 0.0], [0.0, 11.0, 0.0], [0.0, 0.0, 12.0]],
            array![0.0, 0.0, 0.0],
            [true, false, true],
        )
        .unwrap();
        write_simbox(&store, "/box", &simbox).unwrap();

        assert!(read_simbox(&store, "/box").unwrap().is_cell_defined());
    }

    #[test]
    fn undefined_cell_round_trips_as_false() {
        let dir = TempDir::new().unwrap();
        let store = store_in(&dir);
        // A no-cell box carries the identity matrix, per `SimBox::new_cell`:
        // geometry ops degrade to no-ops and only the flag says "undefined".
        let simbox = SimBox::new_cell(
            array![[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            array![0.0, 0.0, 0.0],
            [true, false, true],
            false,
        )
        .unwrap();
        write_simbox(&store, "/box", &simbox).unwrap();

        // `cell_defined` is emitted in this direction only, so it has to be
        // literally present and literally `false` -- an absent attribute is
        // read back as a *defined* cell, which is the silent loss this pins.
        let group = zarrs::group::Group::open(store.clone(), "/box").unwrap();
        assert_eq!(
            group.attributes().get("cell_defined"),
            Some(&serde_json::json!(false))
        );

        let back = read_simbox(&store, "/box").unwrap();
        assert!(!back.is_cell_defined());
    }

    /// An undefined cell's `vectors` are ignored on read: a zero matrix (or
    /// any other) is accepted, never inverted, and comes back as the identity
    /// — which is also what the writer emits for it.
    #[test]
    fn undefined_cell_accepts_any_vectors_and_writes_the_identity() {
        let dir = TempDir::new().unwrap();
        let store = store_in(&dir);
        let mut attrs = serde_json::Map::new();
        attrs.insert("cell_defined".into(), false.into());
        GroupBuilder::new()
            .attributes(attrs)
            .build(store.clone(), "/box")
            .unwrap()
            .store_metadata()
            .unwrap();
        write_f64_array(&store, "/box/vectors", &[3, 3], &[0.0; 9]).unwrap();

        let back = read_simbox(&store, "/box").unwrap();
        assert!(!back.is_cell_defined());
        assert_eq!(back.h_view(), ndarray::Array2::<F>::eye(3));

        write_simbox(&store, "/rewritten", &back).unwrap();
        let written = read_simbox_float_path(&store, "/rewritten/vectors")
            .unwrap()
            .unwrap();
        assert_eq!(written, vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]);
    }

    /// A *defined* cell with a singular matrix is still refused.
    #[test]
    fn a_defined_singular_cell_is_refused() {
        let dir = TempDir::new().unwrap();
        let store = store_in(&dir);
        GroupBuilder::new()
            .build(store.clone(), "/box")
            .unwrap()
            .store_metadata()
            .unwrap();
        write_f64_array(&store, "/box/vectors", &[3, 3], &[0.0; 9]).unwrap();
        assert!(read_simbox(&store, "/box").is_err());
    }

    #[test]
    fn absent_origin_reads_zero_origin() {
        let dir = TempDir::new().unwrap();
        let store = store_in(&dir);
        let mut attrs = serde_json::Map::new();
        attrs.insert(
            "boundary".to_string(),
            serde_json::json!([true, true, true]),
        );
        GroupBuilder::new()
            .attributes(attrs)
            .build(store.clone(), "/box")
            .unwrap()
            .store_metadata()
            .unwrap();
        write_f64_array(&store, "/box/vectors", &[3, 3], &cell_vectors()).unwrap();

        let simbox = read_simbox(&store, "/box").unwrap();
        assert_eq!(simbox.origin_view().to_vec(), vec![0.0, 0.0, 0.0]);
    }

    // -- sharding: many inner chunks, one file (ac-010's on-disk half) -------

    /// Every file under `dir`, recursively, that is not a `zarr.json` — the
    /// data files an array actually costs the filesystem.
    fn data_files(dir: &Path) -> Vec<PathBuf> {
        let mut out = Vec::new();
        let mut stack = vec![dir.to_path_buf()];
        while let Some(current) = stack.pop() {
            for entry in std::fs::read_dir(&current).unwrap().flatten() {
                let path = entry.path();
                if path.is_dir() {
                    stack.push(path);
                } else if path.file_name() != Some(OsStr::new("zarr.json")) {
                    out.push(path);
                }
            }
        }
        out.sort();
        out
    }

    /// A fixed-size column past `SHARD_ABOVE` is laid out as **one** shard file
    /// holding all seven inner chunks, and every value survives bit exact.
    ///
    /// This is the on-disk half of the plan goldens: `chunking::plan` deciding
    /// `shards = Some(..)` is only worth anything if `write_typed_array` then
    /// turns that decision into a single file rather than seven.
    #[test]
    fn a_five_chunk_column_shards_and_round_trips() {
        // chunking::plan(&[400_000], Some(8)), hard-coded from its goldens:
        // an 8 B row gives 512 KiB / 8 = 65 536 rows per inner chunk, and
        // ceil(400 000 / 65 536) = 7 chunks clears SHARD_ABOVE = 4, so the
        // shard spans 7 * 65 536 = 458 752 rows.
        const ROWS: usize = 400_000;
        const INNER_CHUNK_ROWS: u64 = 65_536;
        const SHARD_ROWS: u64 = 458_752;

        let dir = TempDir::new().unwrap();
        let store = store_in(&dir);
        // Index-derived and non-repeating: a constant column would read back
        // correctly even if the shard handed out the wrong chunk.
        let values: Vec<F> = (0..ROWS).map(|i| i as F * 0.5 - 1.0e-3).collect();
        let mut block = Block::new();
        block
            .insert_column(
                COLUMN,
                Column::from_float(ArrayD::from_shape_vec(vec![ROWS], values.clone()).unwrap()),
            )
            .unwrap();
        let mut frame = Frame::new();
        frame.insert(BLOCK, block);
        write_frame_group(&store, FRAME, &frame).unwrap();

        // Under sharding the array's own chunk extent *is* the shard, and the
        // planned chunk is the subchunk inside it.
        let arr = Array::open(store.clone(), &format!("{FRAME}/{BLOCK}/{COLUMN}")).unwrap();
        assert_eq!(
            arr.chunk_shape(&[0]).unwrap(),
            vec![NonZeroU64::new(SHARD_ROWS).unwrap()],
            "the array chunk extent must be the planned shard"
        );
        assert_eq!(arr.chunk_grid_shape(), &[1], "one shard covers the array");
        let metadata = serde_json::to_value(arr.metadata()).unwrap();
        let sharding = metadata["codecs"]
            .as_array()
            .unwrap()
            .iter()
            .find(|codec| codec["name"] == "sharding_indexed")
            .expect("a seven-chunk array must carry the sharding codec");
        assert_eq!(
            sharding["configuration"]["chunk_shape"],
            serde_json::json!([INNER_CHUNK_ROWS]),
            "the inner chunk must be the planned chunk"
        );

        // The whole point of a shard: seven chunks, one file.
        let files = data_files(&dir.path().join("frame").join(BLOCK).join(COLUMN));
        assert_eq!(
            files.len(),
            1,
            "a sharded column is one data file, found {files:?}"
        );

        let back = read_frame_group(&store, FRAME).unwrap();
        assert_eq!(
            *back
                .get(BLOCK)
                .unwrap()
                .get(COLUMN)
                .unwrap()
                .as_float()
                .unwrap(),
            ArrayD::from_shape_vec(vec![ROWS], values).unwrap(),
            "every row must come back bit exact through the shard"
        );
    }

    // -- nullable columns: the mask is data, not a view --------------------

    /// The reserved child name of a block group that holds its masks.
    ///
    /// Spelled out rather than imported from `super`: this is a name on disk
    /// that molrec and molvis read, so a rename of the Rust constant must
    /// break these tests rather than travel silently into them — the same
    /// reading `sequence`'s `SCHEMA_ATTRIBUTE` takes on its own name.
    ///
    /// **Blocked as spelled.** Zarr V3 reserves the `__` prefix for node
    /// names, and `zarrs` enforces it (`NodeName::validate`): every `Array` /
    /// `Group` / `Node` door refuses `/frame/atoms/__validity__` with a
    /// `NodePathError`, so the masks cannot be written under this name at all.
    /// The reserved name has to be one Zarr admits — `validity` or
    /// `_validity` — and this constant is the single place to change.
    const VALIDITY_GROUP: &str = "_validity";

    /// The one mask every nullable fixture carries: three rows, the first
    /// valid and the last two null. Asymmetric on purpose — a reversed or
    /// all-false mask cannot match it by accident.
    const MASK: [bool; 3] = [true, false, false];

    /// Write `block` as the sole block of a frame and read that frame back.
    fn round_trip_block(block: Block) -> Block {
        let dir = TempDir::new().unwrap();
        let store = store_in(&dir);
        let mut frame = Frame::new();
        frame.insert(BLOCK, block);
        write_frame_group(&store, FRAME, &frame).unwrap();
        read_frame_group(&store, FRAME)
            .unwrap()
            .get(BLOCK)
            .expect("the frame carries its block")
            .clone()
    }

    /// The names of a block group's children, enumerated the way
    /// [`read_frame_group`] enumerates them.
    fn block_child_names(store: &ReadableWritableListableStorage, block: &str) -> Vec<String> {
        Node::open(store, &format!("{FRAME}/{block}"))
            .expect("the block group exists")
            .children()
            .iter()
            .map(|child| {
                child
                    .path()
                    .as_str()
                    .rsplit('/')
                    .next()
                    .unwrap_or("")
                    .to_string()
            })
            .collect()
    }

    /// A three-row `Int` column whose last two rows hold nothing.
    fn masked_int_block() -> Block {
        let mut block = Block::new();
        block
            .insert_nullable(
                COLUMN,
                ArrayD::from_shape_vec(vec![3], vec![7i32, 0, 0]).unwrap(),
                MASK.to_vec(),
            )
            .unwrap();
        block
    }

    /// A three-row `F64` column whose last two rows hold nothing.
    fn masked_f64_block() -> Block {
        let mut block = Block::new();
        block
            .insert_nullable(
                COLUMN,
                ArrayD::from_shape_vec(vec![3], vec![0.5f64, 0.0, 0.0]).unwrap(),
                MASK.to_vec(),
            )
            .unwrap();
        block
    }

    /// A three-row `Str` column whose last two rows hold nothing.
    fn masked_string_block() -> Block {
        let mut block = Block::new();
        block
            .insert_nullable(
                COLUMN,
                ArrayD::from_shape_vec(
                    vec![3],
                    vec!["C".to_string(), String::new(), String::new()],
                )
                .unwrap(),
                MASK.to_vec(),
            )
            .unwrap();
        block
    }

    #[test]
    fn int_column_keeps_its_validity_mask_across_the_frame_round_trip() {
        let back = round_trip_block(masked_int_block());
        assert_eq!(back.validity(COLUMN), Some(&MASK[..]));
    }

    #[test]
    fn int_column_keeps_its_values_beside_its_mask() {
        let back = round_trip_block(masked_int_block());
        assert_eq!(
            *back.get(COLUMN).unwrap().as_int().unwrap(),
            ArrayD::from_shape_vec(vec![3], vec![7i32, 0, 0]).unwrap()
        );
    }

    #[test]
    fn f64_column_keeps_its_validity_mask_across_the_frame_round_trip() {
        let back = round_trip_block(masked_f64_block());
        assert_eq!(back.validity(COLUMN), Some(&MASK[..]));
    }

    #[test]
    fn f64_column_keeps_its_values_beside_its_mask() {
        let back = round_trip_block(masked_f64_block());
        assert_eq!(
            *back.get(COLUMN).unwrap().as_float().unwrap(),
            ArrayD::from_shape_vec(vec![3], vec![0.5f64, 0.0, 0.0]).unwrap()
        );
    }

    #[test]
    fn string_column_keeps_its_validity_mask_across_the_frame_round_trip() {
        let back = round_trip_block(masked_string_block());
        assert_eq!(back.validity(COLUMN), Some(&MASK[..]));
    }

    #[test]
    fn string_column_keeps_its_values_beside_its_mask() {
        let back = round_trip_block(masked_string_block());
        assert_eq!(
            *back.get(COLUMN).unwrap().as_string().unwrap(),
            ArrayD::from_shape_vec(vec![3], vec!["C".to_string(), String::new(), String::new()])
                .unwrap()
        );
    }

    /// A column nobody masked stays unmasked: `validity` says `Some` **iff**
    /// a row is null, so a round trip that invented an all-true mask would be
    /// a different block than the one written.
    #[test]
    fn a_fully_valid_column_reads_back_with_no_mask() {
        let mut block = Block::new();
        block
            .insert_column(
                COLUMN,
                Column::from_float(ArrayD::from_shape_vec(vec![3], vec![1.0, 2.0, 3.0]).unwrap()),
            )
            .unwrap();
        let back = round_trip_block(block);
        assert_eq!(back.validity(COLUMN), None);
    }

    /// No mask, no subgroup: a store written by a molrs that never masked a
    /// column is byte-identical to one written now, so it keeps reading.
    #[test]
    fn a_frame_with_no_masked_column_writes_no_validity_child() {
        let dir = TempDir::new().unwrap();
        let store = store_in(&dir);
        let mut block = Block::new();
        block
            .insert_column(
                COLUMN,
                Column::from_float(ArrayD::from_shape_vec(vec![3], vec![1.0, 2.0, 3.0]).unwrap()),
            )
            .unwrap();
        let mut frame = Frame::new();
        frame.insert(BLOCK, block);
        write_frame_group(&store, FRAME, &frame).unwrap();

        assert_eq!(block_child_names(&store, BLOCK), vec![COLUMN.to_string()]);
    }

    /// The mask lives in a reserved **subgroup**, not in a sibling array: a
    /// reader that predates masks skips non-Array children of a block group,
    /// so it ignores the subgroup instead of taking it for a column.
    #[test]
    fn a_masked_column_writes_its_mask_into_a_validity_subgroup() {
        let dir = TempDir::new().unwrap();
        let store = store_in(&dir);
        let mut frame = Frame::new();
        frame.insert(BLOCK, masked_int_block());
        write_frame_group(&store, FRAME, &frame).unwrap();

        let child = Node::open(&store, &format!("{FRAME}/{BLOCK}/{VALIDITY_GROUP}"))
            .expect("the mask subgroup exists");
        assert!(
            matches!(child.metadata(), NodeMetadata::Group(_)),
            "{VALIDITY_GROUP} must be a group, so an older reader skips it"
        );
    }

    /// Inside that subgroup the mask is a boolean array under the column's
    /// own name — one mask per masked column, addressable without an index.
    #[test]
    fn a_mask_is_stored_as_a_bool_array_named_after_its_column() {
        let dir = TempDir::new().unwrap();
        let store = store_in(&dir);
        let mut frame = Frame::new();
        frame.insert(BLOCK, masked_int_block());
        write_frame_group(&store, FRAME, &frame).unwrap();

        let path = format!("{FRAME}/{BLOCK}/{VALIDITY_GROUP}/{COLUMN}");
        let arr = Array::open(store.clone(), &path).expect("the mask array exists");
        assert!(arr.data_type().is::<BoolDataType>());
        let subset = ArraySubset::new_with_shape(arr.shape().to_vec());
        let stored: Vec<bool> = arr.retrieve_array_subset(&subset).unwrap();
        assert_eq!(stored, MASK.to_vec());
    }

    /// The mask subgroup's name is reserved among a block group's children
    /// exactly as `box` is among a frame group's: a column of that name would
    /// collide with the masks, so the write is refused rather than silently
    /// reshaped.
    ///
    /// Passes today only because Zarr refuses the `__` prefix outright (see
    /// [`VALIDITY_GROUP`]) — it becomes a real assertion the moment the
    /// reserved name is one Zarr admits.
    #[test]
    fn a_block_column_named_validity_is_refused_on_write() {
        let dir = TempDir::new().unwrap();
        let store = store_in(&dir);
        let mut block = Block::new();
        block
            .insert_column(
                VALIDITY_GROUP,
                Column::from_float(ArrayD::from_shape_vec(vec![3], vec![1.0, 2.0, 3.0]).unwrap()),
            )
            .unwrap();
        let mut frame = Frame::new();
        frame.insert(BLOCK, block);

        let err = write_frame_group(&store, FRAME, &frame)
            .expect_err("a column named __validity__ collides with the mask subgroup")
            .to_string();
        assert!(err.contains(VALIDITY_GROUP), "{err}");
    }

    /// A mask that does not cover its column's rows is a corrupt store, not a
    /// mask to pad or truncate — and the error has to say which column of
    /// which block, because that is all the operator can act on.
    #[test]
    fn a_mask_of_the_wrong_length_is_a_read_error_naming_block_and_column() {
        let dir = TempDir::new().unwrap();
        let store = store_in(&dir);
        let mut frame = Frame::new();
        frame.insert(BLOCK, masked_int_block());
        write_frame_group(&store, FRAME, &frame).unwrap();

        // Two flags for a three-row column: the one damage no reader can
        // resolve on its own.
        let path = format!("{FRAME}/{BLOCK}/{VALIDITY_GROUP}/{COLUMN}");
        store.erase_prefix(&node_prefix(&path).unwrap()).unwrap();
        write_column(
            &store,
            &path,
            &Column::from_bool(ArrayD::from_shape_vec(vec![2], vec![true, false]).unwrap()),
        )
        .unwrap();

        let err = read_frame_group(&store, FRAME)
            .expect_err("a mask that does not cover its column must not read")
            .to_string();
        assert!(err.contains(BLOCK) && err.contains(COLUMN), "{err}");
    }

    #[test]
    fn frame_group_attributes_round_trip_meta_keys_in_insertion_order() {
        let dir = TempDir::new().unwrap();
        let store = store_in(&dir);
        let mut frame = Frame::new();
        frame.meta.insert("z", "Z");
        frame.meta.insert("a", "A");
        frame.meta.insert("m", "M");
        write_frame_group(&store, FRAME, &frame).unwrap();

        let group = zarrs::group::Group::open(store.clone(), FRAME).unwrap();
        assert_eq!(
            group
                .attributes()
                .keys()
                .map(String::as_str)
                .collect::<Vec<_>>(),
            vec!["z", "a", "m"]
        );

        let back = read_frame_group(&store, FRAME).unwrap();
        assert_eq!(
            back.meta.keys().map(String::as_str).collect::<Vec<_>>(),
            vec!["z", "a", "m"]
        );
    }
}
