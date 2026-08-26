//! Low-level Zarr ↔ Block/Column/SimBox/Frame helpers.
//!
//! These functions are shared by the frame/trajectory Zarr backend. They convert between
//! molrs in-memory types and
//! Zarr V3 arrays/groups, always relative to a caller-supplied path prefix.

use zarrs::array::data_type::{
    BoolDataType, Complex64DataType, Complex128DataType, Float16DataType, Float32DataType,
    Float64DataType, Int8DataType, Int16DataType, Int32DataType, Int64DataType, StringDataType,
    UInt8DataType, UInt16DataType, UInt32DataType, UInt64DataType,
};
use zarrs::array::{Array, ArraySubset};
#[cfg(feature = "filesystem")]
use zarrs::array::{ArrayBuilder, data_type};
#[cfg(feature = "filesystem")]
use zarrs::group::GroupBuilder;
use zarrs::node::{Node, NodeMetadata};
use zarrs::storage::ReadableWritableListableStorage;

use ndarray::ArrayD;

use molrs::error::MolRsError;
use molrs::spatial::simbox::SimBox;
use molrs::store::block::{Block, Column};
use molrs::store::frame::Frame;
use molrs::store::meta::MetaValue;
use molrs::types::F;

// ---------------------------------------------------------------------------
// Column write
// ---------------------------------------------------------------------------

#[cfg(feature = "filesystem")]
pub(crate) fn write_column(
    store: &ReadableWritableListableStorage,
    path: &str,
    col: &Column,
) -> Result<(), MolRsError> {
    match col {
        Column::Float16(a) => {
            write_typed_array(store, path, a, data_type::float16(), half::f16::ZERO)
        }
        Column::Float32(a) => write_typed_array(store, path, a, data_type::float32(), 0.0f32),
        Column::Float(a) => write_float_array(store, path, a),
        Column::Int8(a) => write_typed_array(store, path, a, data_type::int8(), 0i8),
        Column::Int16(a) => write_typed_array(store, path, a, data_type::int16(), 0i16),
        Column::Int(a) => write_typed_array(store, path, a, data_type::int32(), 0i32),
        Column::Int64(a) => write_typed_array(store, path, a, data_type::int64(), 0i64),
        Column::UInt(a) => write_typed_array(store, path, a, data_type::uint64(), 0u64),
        Column::U8(a) => write_typed_array(store, path, a, data_type::uint8(), 0u8),
        Column::UInt16(a) => write_typed_array(store, path, a, data_type::uint16(), 0u16),
        Column::UInt32(a) => write_typed_array(store, path, a, data_type::uint32(), 0u32),
        Column::Bool(a) => write_typed_array(
            store,
            path,
            a,
            data_type::bool(),
            zarrs::array::FillValue::new(vec![0u8]),
        ),
        Column::String(a) => {
            let strings: Vec<String> = a.as_standard_layout().iter().cloned().collect();
            let shape: Vec<u64> = a.shape().iter().map(|&s| s as u64).collect();
            let chunk = shape.clone();
            let arr = ArrayBuilder::new(shape.clone(), chunk, data_type::string(), "")
                .build(store.clone(), path)?;
            arr.store_metadata()?;
            arr.store_array_subset(&ArraySubset::new_with_shape(shape), &strings)?;
            Ok(())
        }
        Column::Complex64(a) => write_typed_array(
            store,
            path,
            a,
            data_type::complex64(),
            zarrs::array::FillValue::new(vec![0u8; 8]),
        ),
        Column::Complex128(a) => write_typed_array(
            store,
            path,
            a,
            data_type::complex128(),
            zarrs::array::FillValue::new(vec![0u8; 16]),
        ),
    }
}

#[cfg(feature = "filesystem")]
fn write_float_array(
    store: &ReadableWritableListableStorage,
    path: &str,
    a: &ArrayD<F>,
) -> Result<(), MolRsError> {
    write_typed_array(store, path, a, data_type::float64(), 0.0)
}

/// Generic helper for f32/f64/i64/u32/u8 columns.
#[cfg(feature = "filesystem")]
fn write_typed_array<T>(
    store: &ReadableWritableListableStorage,
    path: &str,
    a: &ArrayD<T>,
    dt: zarrs::array::DataType,
    fill: impl Into<zarrs::array::builder::ArrayBuilderFillValue>,
) -> Result<(), MolRsError>
where
    T: zarrs::array::Element + Clone,
{
    let data = a.as_standard_layout();
    let shape: Vec<u64> = data.shape().iter().map(|&s| s as u64).collect();
    let chunk = shape.clone();
    let arr = ArrayBuilder::new(shape.clone(), chunk, dt, fill).build(store.clone(), path)?;
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

pub(crate) fn read_column(
    store: &ReadableWritableListableStorage,
    path: &str,
) -> Result<Column, MolRsError> {
    let arr = Array::open(store.clone(), path)?;
    let shape: Vec<usize> = arr.shape().iter().map(|&s| s as usize).collect();
    let subset = ArraySubset::new_with_shape(arr.shape().to_vec());

    let is_bool = arr.attributes().get("molrs_dtype").and_then(|v| v.as_str()) == Some("bool");
    let dt = arr.data_type();

    if dt.is::<Float16DataType>() {
        let data: Vec<half::f16> = arr.retrieve_array_subset(&subset)?;
        Ok(Column::from_f16(
            ArrayD::from_shape_vec(shape, data).map_err(shape_err)?,
        ))
    } else if dt.is::<Float32DataType>() {
        let data: Vec<f32> = arr.retrieve_array_subset(&subset)?;
        Ok(Column::from_f32(
            ArrayD::from_shape_vec(shape, data).map_err(shape_err)?,
        ))
    } else if dt.is::<Float64DataType>() {
        let data: Vec<f64> = arr.retrieve_array_subset(&subset)?;
        Ok(Column::from_float(
            ArrayD::from_shape_vec(shape, data).map_err(shape_err)?,
        ))
    } else if dt.is::<Int8DataType>() {
        let data: Vec<i8> = arr.retrieve_array_subset(&subset)?;
        Ok(Column::from_i8(
            ArrayD::from_shape_vec(shape, data).map_err(shape_err)?,
        ))
    } else if dt.is::<Int16DataType>() {
        let data: Vec<i16> = arr.retrieve_array_subset(&subset)?;
        Ok(Column::from_i16(
            ArrayD::from_shape_vec(shape, data).map_err(shape_err)?,
        ))
    } else if dt.is::<Int32DataType>() {
        let data: Vec<i32> = arr.retrieve_array_subset(&subset)?;
        Ok(Column::from_int(
            ArrayD::from_shape_vec(shape, data).map_err(shape_err)?,
        ))
    } else if dt.is::<Int64DataType>() {
        let data: Vec<i64> = arr.retrieve_array_subset(&subset)?;
        Ok(Column::from_i64(
            ArrayD::from_shape_vec(shape, data).map_err(shape_err)?,
        ))
    } else if dt.is::<UInt16DataType>() {
        let data: Vec<u16> = arr.retrieve_array_subset(&subset)?;
        Ok(Column::from_u16(
            ArrayD::from_shape_vec(shape, data).map_err(shape_err)?,
        ))
    } else if dt.is::<UInt32DataType>() {
        let data: Vec<u32> = arr.retrieve_array_subset(&subset)?;
        Ok(Column::from_u32(
            ArrayD::from_shape_vec(shape, data).map_err(shape_err)?,
        ))
    } else if dt.is::<UInt64DataType>() {
        let data: Vec<u64> = arr.retrieve_array_subset(&subset)?;
        Ok(Column::from_uint(
            ArrayD::from_shape_vec(shape, data).map_err(shape_err)?,
        ))
    } else if dt.is::<BoolDataType>() {
        let data: Vec<bool> = arr.retrieve_array_subset(&subset)?;
        Ok(Column::from_bool(
            ArrayD::from_shape_vec(shape, data).map_err(shape_err)?,
        ))
    } else if dt.is::<UInt8DataType>() && is_bool {
        let data: Vec<u8> = arr.retrieve_array_subset(&subset)?;
        let bools: Vec<bool> = data.into_iter().map(|v| v != 0).collect();
        Ok(Column::from_bool(
            ArrayD::from_shape_vec(shape, bools).map_err(shape_err)?,
        ))
    } else if dt.is::<UInt8DataType>() {
        let data: Vec<u8> = arr.retrieve_array_subset(&subset)?;
        Ok(Column::from_u8(
            ArrayD::from_shape_vec(shape, data).map_err(shape_err)?,
        ))
    } else if dt.is::<StringDataType>() {
        let data: Vec<String> = arr.retrieve_array_subset(&subset)?;
        Ok(Column::from_string(
            ArrayD::from_shape_vec(shape, data).map_err(shape_err)?,
        ))
    } else if dt.is::<Complex64DataType>() {
        let data: Vec<num_complex::Complex<f32>> = arr.retrieve_array_subset(&subset)?;
        Ok(Column::from_c64(
            ArrayD::from_shape_vec(shape, data).map_err(shape_err)?,
        ))
    } else if dt.is::<Complex128DataType>() {
        let data: Vec<num_complex::Complex<f64>> = arr.retrieve_array_subset(&subset)?;
        Ok(Column::from_c128(
            ArrayD::from_shape_vec(shape, data).map_err(shape_err)?,
        ))
    } else {
        Err(MolRsError::zarr(format!("unsupported dtype: {:?}", dt)))
    }
}

pub(crate) fn insert_column_into_block(
    block: &mut Block,
    name: &str,
    col: Column,
) -> Result<(), MolRsError> {
    // Zero-copy insert: hand the Arc-backed Column directly to the Block.
    block.insert_column(name, col).map_err(MolRsError::Block)
}

// ---------------------------------------------------------------------------
// SimBox write / read
// ---------------------------------------------------------------------------

#[cfg(feature = "filesystem")]
pub(crate) fn write_simbox(
    store: &ReadableWritableListableStorage,
    prefix: &str,
    simbox: &SimBox,
) -> Result<(), MolRsError> {
    // Per-axis periodicity is three booleans. JSON represents those exactly and
    // an attribute costs no files, where an array would cost two per cell.
    let mut attrs = serde_json::Map::new();
    attrs.insert(
        "boundary".to_string(),
        serde_json::Value::Array(
            simbox
                .pbc_view()
                .iter()
                .map(|&b| serde_json::Value::Bool(b))
                .collect(),
        ),
    );
    GroupBuilder::new()
        .attributes(attrs)
        .build(store.clone(), prefix)?
        .store_metadata()?;

    // vectors: [3,3] Float64 — lattice vectors are the COLUMNS, per contract.
    // Geometry stays f64 to match the in-memory science contract.
    let h_view = simbox.h_view();
    let h_data: Vec<F> = h_view.iter().copied().collect();
    write_f64_array(store, &format!("{}/vectors", prefix), &[3, 3], &h_data)?;

    // origin: [3] Float64
    let origin_view = simbox.origin_view();
    let origin_data: Vec<F> = origin_view.iter().copied().collect();
    write_f64_array(store, &format!("{}/origin", prefix), &[3], &origin_data)?;

    Ok(())
}

pub(crate) fn read_simbox(
    store: &ReadableWritableListableStorage,
    prefix: &str,
) -> Result<SimBox, MolRsError> {
    use ndarray::{Array2, array};

    // Prefer f64 (0.12+); accept legacy f32 stores and promote once.
    let h_data = read_simbox_float_path(store, &format!("{}/vectors", prefix))?;
    if h_data.len() != 9 {
        return Err(MolRsError::zarr(format!(
            "box vectors expected 9 values, got {}",
            h_data.len()
        )));
    }
    let h = Array2::from_shape_vec((3, 3), h_data).map_err(shape_err)?;

    let o_data = read_simbox_float_path(store, &format!("{}/origin", prefix))?;
    if o_data.len() != 3 {
        return Err(MolRsError::zarr(format!(
            "box origin expected 3 values, got {}",
            o_data.len()
        )));
    }
    let origin = array![o_data[0], o_data[1], o_data[2]];

    // Boundary flags ride as a group attribute; a cell without them is
    // non-periodic rather than malformed.
    let group = zarrs::group::Group::open(store.clone(), prefix)?;
    let pbc = match group
        .attributes()
        .get("boundary")
        .and_then(|v| v.as_array())
    {
        Some(flags) if flags.len() == 3 => [
            flags[0].as_bool().unwrap_or(false),
            flags[1].as_bool().unwrap_or(false),
            flags[2].as_bool().unwrap_or(false),
        ],
        _ => [false, false, false],
    };

    SimBox::new(h, origin, pbc).map_err(|e| MolRsError::zarr(format!("invalid box: {:?}", e)))
}

/// Read a simbox float array as `Vec<F>` (Float64 preferred; legacy Float32 promoted).
fn read_simbox_float_path(
    store: &ReadableWritableListableStorage,
    path: &str,
) -> Result<Vec<F>, MolRsError> {
    let arr = Array::open(store.clone(), path)?;
    let subset = ArraySubset::new_with_shape(arr.shape().to_vec());
    let dt = arr.data_type();
    if dt.is::<Float64DataType>() {
        let data: Vec<f64> = arr.retrieve_array_subset(&subset)?;
        Ok(data)
    } else if dt.is::<Float32DataType>() {
        let data: Vec<f32> = arr.retrieve_array_subset(&subset)?;
        Ok(data.into_iter().map(|v| v as F).collect())
    } else {
        Err(MolRsError::zarr(format!(
            "simbox array expected float32/float64, got {dt:?}"
        )))
    }
}

// ---------------------------------------------------------------------------
/// The one reserved child name in a frame group: the cell.
///
/// Rust cannot spell it `box` -- that is a reserved keyword -- but nothing
/// outside Rust source has that problem, so the stored name is `box`.
pub(crate) const BOX_GROUP: &str = "box";

// Frame (system) write / read — writes all blocks under `{prefix}/`
// ---------------------------------------------------------------------------

#[cfg(feature = "filesystem")]
/// Write one [`Frame`] as a Zarr group of blocks.
///
/// The group carries **no** schema-version attribute: `meta/record_schema_version`
/// at the record root is the sole version key of the MolRec contract, and a
/// parallel per-frame version is forbidden by it.
pub(crate) fn write_frame_group(
    store: &ReadableWritableListableStorage,
    prefix: &str,
    frame: &Frame,
) -> Result<(), MolRsError> {
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
            let col_name = col_child.path().as_str().rsplit('/').next().unwrap_or("");
            let col = read_column(store, col_child.path().as_str())?;
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
        frame.insert(child_name, block);
    }

    Ok(frame)
}

// ---------------------------------------------------------------------------
// Primitive array helpers
// ---------------------------------------------------------------------------

#[cfg(feature = "filesystem")]
pub(crate) fn write_f64_array(
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

/// Legacy f32 writer (kept for non-simbox callers / future dual-write tools).
#[cfg(feature = "filesystem")]
#[allow(dead_code)]
pub(crate) fn write_f32_array(
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
#[allow(dead_code)]
pub(crate) fn write_u8_array(
    store: &ReadableWritableListableStorage,
    path: &str,
    shape: &[u64],
    data: &[u8],
) -> Result<(), MolRsError> {
    let arr = ArrayBuilder::new(shape.to_vec(), shape.to_vec(), data_type::uint8(), 0u8)
        .build(store.clone(), path)?;
    arr.store_metadata()?;
    arr.store_array_subset(&ArraySubset::new_with_shape(shape.to_vec()), data)?;
    Ok(())
}

fn shape_err(e: impl std::fmt::Display) -> MolRsError {
    MolRsError::zarr(format!("shape error: {}", e))
}

/// Build a child path from a prefix, avoiding double slashes.
pub(crate) fn join_path(prefix: &str, child: &str) -> String {
    if prefix == "/" {
        format!("/{}", child)
    } else {
        format!("{}/{}", prefix.trim_end_matches('/'), child)
    }
}
