//! Low-level Zarr ↔ Block/Column/SimBox/Frame helpers.
//!
//! These functions are shared by MolRec Zarr backends. They convert between
//! molrs in-memory types and
//! Zarr V3 arrays/groups, always relative to a caller-supplied path prefix.

use zarrs::array::data_type::{
    Float32DataType, Float64DataType, Int32DataType, Int64DataType, StringDataType, UInt8DataType,
    UInt32DataType, UInt64DataType,
};
use zarrs::array::{Array, ArraySubset};
#[cfg(feature = "filesystem")]
use zarrs::array::{ArrayBuilder, data_type};
#[cfg(feature = "filesystem")]
use zarrs::group::GroupBuilder;
use zarrs::node::{Node, NodeMetadata};
use zarrs::storage::ReadableWritableListableStorage;

use ndarray::ArrayD;

use crate::block::{Block, Column};
use crate::error::MolRsError;
use crate::frame::Frame;
use crate::grid::Grid;
use crate::region::simbox::SimBox;
use crate::types::{F, I, U};

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
        Column::Float(a) => write_float_array(store, path, a),
        Column::Int(a) => {
            #[cfg(feature = "i64")]
            {
                write_typed_array(store, path, a, data_type::int64(), 0i64)
            }
            #[cfg(not(feature = "i64"))]
            {
                write_typed_array(store, path, a, data_type::int32(), 0i32)
            }
        }
        Column::UInt(a) => {
            #[cfg(feature = "u64")]
            {
                write_typed_array(store, path, a, data_type::uint64(), 0u64)
            }
            #[cfg(not(feature = "u64"))]
            {
                write_typed_array(store, path, a, data_type::uint32(), 0u32)
            }
        }
        Column::U8(a) => write_typed_array(store, path, a, data_type::uint8(), 0u8),
        Column::Bool(a) => {
            let u8_data: Vec<u8> = a.as_standard_layout().iter().map(|&b| b as u8).collect();
            let shape: Vec<u64> = a.shape().iter().map(|&s| s as u64).collect();
            let chunk = shape.clone();
            let mut attrs = serde_json::Map::new();
            attrs.insert("molrs_dtype".into(), "bool".into());
            let arr = ArrayBuilder::new(shape.clone(), chunk, data_type::uint8(), 0u8)
                .attributes(attrs)
                .build(store.clone(), path)?;
            arr.store_metadata()?;
            arr.store_array_subset(&ArraySubset::new_with_shape(shape), &u8_data)?;
            Ok(())
        }
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
    }
}

#[cfg(all(feature = "filesystem", not(feature = "f64")))]
fn write_float_array(
    store: &ReadableWritableListableStorage,
    path: &str,
    a: &ArrayD<F>,
) -> Result<(), MolRsError> {
    write_typed_array(store, path, a, data_type::float32(), 0.0)
}

#[cfg(all(feature = "filesystem", feature = "f64"))]
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
    fill: T,
) -> Result<(), MolRsError>
where
    T: zarrs::array::Element + Clone,
    zarrs::array::builder::ArrayBuilderFillValue: From<T>,
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

    if dt.is::<Float32DataType>() {
        let data: Vec<f32> = arr.retrieve_array_subset(&subset)?;
        Ok(Column::Float(
            ArrayD::from_shape_vec(shape, data.into_iter().map(|v| v as F).collect())
                .map_err(shape_err)?,
        ))
    } else if dt.is::<Float64DataType>() {
        let data: Vec<f64> = arr.retrieve_array_subset(&subset)?;
        Ok(Column::Float(
            ArrayD::from_shape_vec(shape, data.into_iter().map(|v| v as F).collect())
                .map_err(shape_err)?,
        ))
    } else if dt.is::<Int32DataType>() {
        let data: Vec<i32> = arr.retrieve_array_subset(&subset)?;
        Ok(Column::Int(
            ArrayD::from_shape_vec(shape, data.into_iter().map(|v| v as I).collect())
                .map_err(shape_err)?,
        ))
    } else if dt.is::<Int64DataType>() {
        let data: Vec<i64> = arr.retrieve_array_subset(&subset)?;
        Ok(Column::Int(
            ArrayD::from_shape_vec(shape, data.into_iter().map(|v| v as I).collect())
                .map_err(shape_err)?,
        ))
    } else if dt.is::<UInt32DataType>() {
        let data: Vec<u32> = arr.retrieve_array_subset(&subset)?;
        Ok(Column::UInt(
            ArrayD::from_shape_vec(shape, data.into_iter().map(|v| v as U).collect())
                .map_err(shape_err)?,
        ))
    } else if dt.is::<UInt64DataType>() {
        let data: Vec<u64> = arr.retrieve_array_subset(&subset)?;
        Ok(Column::UInt(
            ArrayD::from_shape_vec(shape, data.into_iter().map(|v| v as U).collect())
                .map_err(shape_err)?,
        ))
    } else if dt.is::<UInt8DataType>() && is_bool {
        let data: Vec<u8> = arr.retrieve_array_subset(&subset)?;
        let bools: Vec<bool> = data.into_iter().map(|v| v != 0).collect();
        Ok(Column::Bool(
            ArrayD::from_shape_vec(shape, bools).map_err(shape_err)?,
        ))
    } else if dt.is::<UInt8DataType>() {
        let data: Vec<u8> = arr.retrieve_array_subset(&subset)?;
        Ok(Column::U8(
            ArrayD::from_shape_vec(shape, data).map_err(shape_err)?,
        ))
    } else if dt.is::<StringDataType>() {
        let data: Vec<String> = arr.retrieve_array_subset(&subset)?;
        Ok(Column::String(
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
    match col {
        Column::Float(a) => block.insert(name, a).map_err(MolRsError::Block),
        Column::Int(a) => block.insert(name, a).map_err(MolRsError::Block),
        Column::UInt(a) => block.insert(name, a).map_err(MolRsError::Block),
        Column::U8(a) => block.insert(name, a).map_err(MolRsError::Block),
        Column::Bool(a) => block.insert(name, a).map_err(MolRsError::Block),
        Column::String(a) => block.insert(name, a).map_err(MolRsError::Block),
    }
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
    GroupBuilder::new()
        .build(store.clone(), prefix)?
        .store_metadata()?;

    // h: [3,3] Float32
    let h_view = simbox.h_view();
    #[allow(clippy::unnecessary_cast)]
    let h_data: Vec<f32> = h_view.iter().map(|&v| v as f32).collect();
    write_f32_array(store, &format!("{}/h", prefix), &[3, 3], &h_data)?;

    // origin: [3] Float32
    let origin_view = simbox.origin_view();
    #[allow(clippy::unnecessary_cast)]
    let origin_data: Vec<f32> = origin_view.iter().map(|&v| v as f32).collect();
    write_f32_array(store, &format!("{}/origin", prefix), &[3], &origin_data)?;

    // PBC: [3] UInt8
    let pbc_view = simbox.pbc_view();
    let pbc_data: Vec<u8> = pbc_view.iter().map(|&b| b as u8).collect();
    write_u8_array(store, &format!("{}/pbc", prefix), &[3], &pbc_data)?;

    Ok(())
}

pub(crate) fn read_simbox(
    store: &ReadableWritableListableStorage,
    prefix: &str,
) -> Result<SimBox, MolRsError> {
    use ndarray::{Array2, array};

    let h_arr = Array::open(store.clone(), &format!("{}/h", prefix))?;
    let h_data: Vec<f32> = h_arr.retrieve_array_subset(&ArraySubset::new_with_shape(vec![3, 3]))?;
    let h = Array2::from_shape_vec((3, 3), h_data.into_iter().map(|v| v as F).collect())
        .map_err(shape_err)?;

    let o_arr = Array::open(store.clone(), &format!("{}/origin", prefix))?;
    let o_data: Vec<f32> = o_arr.retrieve_array_subset(&ArraySubset::new_with_shape(vec![3]))?;
    let origin = array![o_data[0] as F, o_data[1] as F, o_data[2] as F];

    let p_arr = Array::open(store.clone(), &format!("{}/pbc", prefix))?;
    let p_data: Vec<u8> = p_arr.retrieve_array_subset(&ArraySubset::new_with_shape(vec![3]))?;
    let pbc = [p_data[0] != 0, p_data[1] != 0, p_data[2] != 0];

    SimBox::new(h, origin, pbc).map_err(|e| MolRsError::zarr(format!("invalid simbox: {:?}", e)))
}

// ---------------------------------------------------------------------------
// Frame (system) write / read — writes all blocks under `{prefix}/`
// ---------------------------------------------------------------------------

#[cfg(feature = "filesystem")]
pub(crate) fn write_system(
    store: &ReadableWritableListableStorage,
    prefix: &str,
    frame: &Frame,
) -> Result<(), MolRsError> {
    // System group
    GroupBuilder::new()
        .build(store.clone(), prefix)?
        .store_metadata()?;

    // Meta
    if !frame.meta.is_empty() {
        let mut meta_attrs = serde_json::Map::new();
        for (k, v) in &frame.meta {
            meta_attrs.insert(k.clone(), serde_json::Value::String(v.clone()));
        }
        GroupBuilder::new()
            .attributes(meta_attrs)
            .build(store.clone(), &format!("{}/meta", prefix))?
            .store_metadata()?;
    }

    // SimBox
    if let Some(ref simbox) = frame.simbox {
        write_simbox(store, &format!("{}/simbox", prefix), simbox)?;
    }

    // Blocks (atoms, bonds, angles, …) — skip empty blocks (zarrs requires nrows > 0)
    for (block_name, block) in frame.iter() {
        if block.nrows().map_or(true, |n| n == 0) {
            continue;
        }
        let group_path = format!("{}/{}", prefix, block_name);
        GroupBuilder::new()
            .build(store.clone(), &group_path)?
            .store_metadata()?;

        for (col_name, col) in block.iter() {
            let arr_path = format!("{}/{}/{}", prefix, block_name, col_name);
            write_column(store, &arr_path, col)?;
        }
    }

    if frame.grid_keys().next().is_some() {
        let grids_prefix = format!("{}/grids", prefix);
        GroupBuilder::new()
            .build(store.clone(), &grids_prefix)?
            .store_metadata()?;

        for (name, grid) in frame.grids() {
            write_grid(store, &format!("{}/{}", grids_prefix, name), grid)?;
        }
    }

    Ok(())
}

pub(crate) fn read_system(
    store: &ReadableWritableListableStorage,
    prefix: &str,
) -> Result<Frame, MolRsError> {
    let mut frame = Frame::new();

    // Meta
    if let Ok(meta_group) = zarrs::group::Group::open(store.clone(), &format!("{}/meta", prefix)) {
        for (k, v) in meta_group.attributes() {
            if let Some(s) = v.as_str() {
                frame.meta.insert(k.clone(), s.to_string());
            } else {
                frame.meta.insert(k.clone(), v.to_string());
            }
        }
    }

    // SimBox
    let simbox_path = format!("{}/simbox", prefix);
    if zarrs::group::Group::open(store.clone(), &simbox_path).is_ok() {
        frame.simbox = Some(read_simbox(store, &simbox_path)?);
    }

    // Blocks
    let system_node = Node::open(store, prefix)?;
    for child in system_node.children() {
        let child_name = child.path().as_str().rsplit('/').next().unwrap_or("");
        if child_name == "meta"
            || child_name == "simbox"
            || child_name == "grids"
            || child_name.is_empty()
        {
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
        frame.insert(child_name, block);
    }

    let grids_path = format!("{}/grids", prefix);
    if let Ok(grids_node) = Node::open(store, &grids_path) {
        for child in grids_node.children() {
            if !matches!(child.metadata(), NodeMetadata::Group(_)) {
                continue;
            }
            let name = child.path().as_str().rsplit('/').next().unwrap_or("");
            if name.is_empty() {
                continue;
            }
            let grid = read_grid(store, child.path().as_str())?;
            frame.insert_grid(name.to_string(), grid);
        }
    }

    Ok(frame)
}

// ---------------------------------------------------------------------------
// Primitive array helpers
// ---------------------------------------------------------------------------

#[cfg(feature = "filesystem")]
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

/// Write a [`Grid`] to a zarr group at `path`.
///
/// Layout:
/// ```text
/// {path}/                   ← group, attrs: dim, origin, cell, pbc
///   {array_name}            ← zarr array shape (nx, ny, nz), dtype float32
///   ...
/// ```
#[cfg(feature = "filesystem")]
pub(crate) fn write_grid(
    store: &ReadableWritableListableStorage,
    path: &str,
    grid: &Grid,
) -> Result<(), MolRsError> {
    let dim_json: Vec<serde_json::Value> =
        grid.dim.iter().map(|&v| serde_json::Value::from(v as u64)).collect();
    let origin_json: Vec<serde_json::Value> =
        grid.origin.iter().map(|&v| serde_json::Value::from(v as f64)).collect();
    let cell_json: Vec<serde_json::Value> = grid
        .cell
        .iter()
        .map(|row| {
            serde_json::Value::Array(
                row.iter().map(|&v| serde_json::Value::from(v as f64)).collect(),
            )
        })
        .collect();
    let pbc_json: Vec<serde_json::Value> =
        grid.pbc.iter().map(|&v| serde_json::Value::Bool(v)).collect();

    let mut attrs = serde_json::Map::new();
    attrs.insert("dim".into(), serde_json::Value::Array(dim_json));
    attrs.insert("origin".into(), serde_json::Value::Array(origin_json));
    attrs.insert("cell".into(), serde_json::Value::Array(cell_json));
    attrs.insert("pbc".into(), serde_json::Value::Array(pbc_json));

    GroupBuilder::new()
        .attributes(attrs)
        .build(store.clone(), path)?
        .store_metadata()?;

    let [nx, ny, nz] = grid.dim;
    for (name, flat) in grid.raw_arrays() {
        let arr_path = format!("{}/{}", path, name);
        let data_f32: Vec<f32> = flat.iter().map(|&v| v as f32).collect();
        write_f32_array(store, &arr_path, &[nx as u64, ny as u64, nz as u64], &data_f32)?;
    }

    Ok(())
}

/// Read a [`Grid`] from a zarr group at `path`.
pub(crate) fn read_grid(
    store: &ReadableWritableListableStorage,
    path: &str,
) -> Result<Grid, MolRsError> {
    let group = zarrs::group::Group::open(store.clone(), path)?;
    let attrs = group.attributes();

    let dim = {
        let arr = attrs
            .get("dim")
            .and_then(|v| v.as_array())
            .ok_or_else(|| MolRsError::zarr("grid missing 'dim' attribute"))?;
        [
            arr[0].as_u64().unwrap_or(0) as usize,
            arr[1].as_u64().unwrap_or(0) as usize,
            arr[2].as_u64().unwrap_or(0) as usize,
        ]
    };

    let origin = {
        let arr = attrs
            .get("origin")
            .and_then(|v| v.as_array())
            .ok_or_else(|| MolRsError::zarr("grid missing 'origin' attribute"))?;
        [
            arr[0].as_f64().unwrap_or(0.0) as F,
            arr[1].as_f64().unwrap_or(0.0) as F,
            arr[2].as_f64().unwrap_or(0.0) as F,
        ]
    };

    let cell = {
        let arr = attrs
            .get("cell")
            .and_then(|v| v.as_array())
            .ok_or_else(|| MolRsError::zarr("grid missing 'cell' attribute"))?;
        let mut cell = [[0.0f32 as F; 3]; 3];
        for (i, row) in arr.iter().enumerate() {
            if let Some(cols) = row.as_array() {
                for (j, v) in cols.iter().enumerate() {
                    cell[i][j] = v.as_f64().unwrap_or(0.0) as F;
                }
            }
        }
        cell
    };

    let pbc = {
        let arr = attrs
            .get("pbc")
            .and_then(|v| v.as_array())
            .ok_or_else(|| MolRsError::zarr("grid missing 'pbc' attribute"))?;
        [
            arr[0].as_bool().unwrap_or(false),
            arr[1].as_bool().unwrap_or(false),
            arr[2].as_bool().unwrap_or(false),
        ]
    };

    let mut grid = Grid::new(dim, origin, cell, pbc);

    // Read each child array as a named scalar field.
    let grid_node = Node::open(store, path)?;
    for child in grid_node.children() {
        if !matches!(child.metadata(), NodeMetadata::Array(_)) {
            continue;
        }
        let arr_name = child.path().as_str().rsplit('/').next().unwrap_or("");
        if arr_name.is_empty() {
            continue;
        }
        let [nx, ny, nz] = dim;
        let arr = Array::open(store.clone(), child.path().as_str())?;
        let data: Vec<f32> =
            arr.retrieve_array_subset(&ArraySubset::new_with_shape(vec![
                nx as u64, ny as u64, nz as u64,
            ]))?;
        let flat: Vec<F> = data.into_iter().map(|v| v as F).collect();
        grid.insert(arr_name, flat)
            .map_err(|e| MolRsError::zarr(format!("grid array '{}': {}", arr_name, e)))?;
    }

    Ok(grid)
}
