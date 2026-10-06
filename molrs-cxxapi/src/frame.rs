//! The frame handle for the C++ engine — the CXX face of `molrs::store::Frame`
//! (through `molrs_ffi::FrameRef`): introspection, typed column readers and
//! create-or-update writers, exact-dtype metadata and the simulation box.

use molrs::spatial::SimBox;
use molrs::store::{Block, Frame, MetaValue};
use ndarray::{Array1, Array2, ArrayD};

use crate::bridge;

// ── Frame bridge ─────────────────────────────────────────────────────────────
//
// `molrs_ffi::FrameRef` is the handle type used by every molrs language
// binding (python, wasm). It pairs a `FrameId` with a shared `Store`
// (`Rc<RefCell<Store>>`) — cloning is two `Rc` bumps and keeps the `Store`
// alive. Atomiverse C++ reads/writes a `molrs.Frame` entirely through this
// handle, so the SCF / MD pipeline never copies a whole frame.
//
// cxx `extern "Rust"` opaque types are crate-local by convention, so the
// bridged type is a thin newtype around `molrs_ffi::FrameRef` rather than a
// re-export. All `frame_*` functions deref through `.0`.

/// CXX-bridged opaque handle to a `molrs.Frame`.
///
/// Newtype around [`molrs_ffi::FrameRef`]; the inner handle carries the
/// shared `Store`, so cloning this and round-tripping it through a
/// `Box`/raw-pointer (e.g. a Python `PyCapsule`) keeps the same underlying
/// frame data — mutations through one handle are visible through all clones.
///
/// `#[repr(transparent)]` guarantees this newtype has exactly the layout of
/// the wrapped `molrs_ffi::FrameRef`, so a `*const molrs_ffi::FrameRef` (e.g.
/// the pointer carried by molrs-python's `_ffi_frameref_capsule`) and a
/// `*const FrameRef` are layout-compatible. `frame_clone_from_addr` relies on
/// this only as a defensive guarantee — it still dereferences the pointer as
/// the *real* `molrs_ffi::FrameRef` type, never as a layout cast.
#[repr(transparent)]
pub struct FrameRef(pub molrs_ffi::FrameRef);

/// Create a fresh standalone frame (new `Store` + empty `Frame`).
///
/// @return boxed opaque handle owning a fresh shared store
pub(crate) fn frame_new() -> Box<FrameRef> {
    Box::new(FrameRef(molrs_ffi::FrameRef::new_standalone()))
}

/// Clone a `molrs_ffi::FrameRef` reachable through a `molrs.FrameRef`
/// PyCapsule address into a bridge-side [`FrameRef`] handle.
///
/// This is the cross-extension ingress point. molrs-python's
/// `Frame._ffi_frameref_capsule()` produces a `PyCapsule` named
/// `molrs_ffi::abi::frameref_capsule_name()` — `"molrs.FrameRef/<major.minor>"`,
/// carrying the ABI line (builds before 0.14 used the unversioned
/// `"molrs.FrameRef"`). PyO3 heap-boxes the capsule payload, and that payload
/// is a `#[repr(transparent)]` `FrameRefPtr` — itself a `*mut FrameRef`
/// (a clone of the Python frame's handle). The capsule's `void*` is
/// therefore `*mut *mut molrs_ffi::FrameRef`.
///
/// `addr` is `PyCapsule_GetPointer` cast to `usize`. This function resolves
/// it as `*const *const molrs_ffi::FrameRef`, dereferences once to reach the
/// cloned `*const FrameRef`, then `.clone()`s it — two cheap `Rc` bumps —
/// producing a handle onto the *same* shared `Store`. Reads and writes
/// through the returned handle are visible in the originating Python
/// `molrs.Frame`.
///
/// # Safety
///
/// `addr` must be the pointer returned by `PyCapsule_GetPointer` on a
/// capsule from `molrs.Frame._ffi_frameref_capsule()`, named with this
/// build's `molrs_ffi::abi::frameref_capsule_name()` (the caller must pass
/// that exact name to `PyCapsule_GetPointer` — query it via
/// `molrs._ffi_abi_token()`), valid for the duration of this call. The name
/// carries the molrs minor line, so a cross-minor producer fails the name
/// check instead of being dereferenced here. The molrs FFI store is
/// single-threaded and GIL-guarded; the caller must hold the GIL (or
/// otherwise guarantee exclusive access) while calling.
///
/// @param addr `molrs.FrameRef` capsule pointer (a `*mut *mut FrameRef`)
/// @return boxed bridge handle sharing the same store as the source frame
pub(crate) unsafe fn frame_clone_from_addr(addr: usize) -> Box<FrameRef> {
    let pp = addr as *const *const molrs_ffi::FrameRef;
    let p = unsafe { *pp };
    let cloned = unsafe { (*p).clone() };
    Box::new(FrameRef(cloned))
}

/// List the block keys present in the frame.
///
/// @param fref frame handle
/// @return block names; empty if the frame has no blocks
pub(crate) fn frame_block_names(fref: &FrameRef) -> Vec<String> {
    fref.0
        .with(|f| f.keys().map(|s| s.to_string()).collect())
        .unwrap_or_default()
}

/// Test whether the frame contains a block.
///
/// @param fref  frame handle
/// @param block block key
/// @return true if the block exists
pub(crate) fn frame_has_block(fref: &FrameRef, block: &str) -> bool {
    fref.0.has_block(block)
}

/// List the column keys of a block.
///
/// @param fref  frame handle
/// @param block block key
/// @return column names; empty if the block is absent
pub(crate) fn frame_block_columns(fref: &FrameRef, block: &str) -> Vec<String> {
    match fref.0.block(block) {
        Ok(blk) => blk.keys().unwrap_or_default(),
        Err(_) => Vec::new(),
    }
}

/// Row count of a block.
///
/// @param fref  frame handle
/// @param block block key
/// @return number of rows; 0 if the block is absent or empty
pub(crate) fn frame_block_nrows(fref: &FrameRef, block: &str) -> i64 {
    match fref.0.block(block) {
        Ok(blk) => blk.nrows().unwrap_or(0) as i64,
        Err(_) => 0,
    }
}

pub(crate) fn empty_meta_entry(
    key: String,
    dtype: bridge::ffi::MetaType,
) -> bridge::ffi::MetaEntry {
    bridge::ffi::MetaEntry {
        key,
        dtype,
        bool_value: false,
        i32_value: 0,
        i64_value: 0,
        u32_value: 0,
        u64_value: 0,
        f64_value: 0.0,
        string_value: String::new(),
        bool_values: Vec::new(),
        i32_values: Vec::new(),
        i64_values: Vec::new(),
        u32_values: Vec::new(),
        u64_values: Vec::new(),
        f64_values: Vec::new(),
    }
}

/// Return every metadata entry with its exact dtype and native payload.
pub(crate) fn frame_meta_entries(fref: &FrameRef) -> Vec<bridge::ffi::MetaEntry> {
    fref.0
        .with(|frame| {
            frame
                .meta
                .iter()
                .map(|(key, value)| {
                    use bridge::ffi::MetaType;
                    let dtype = match value {
                        MetaValue::Bool(_) => MetaType::Bool,
                        MetaValue::I32(_) => MetaType::I32,
                        MetaValue::I64(_) => MetaType::I64,
                        MetaValue::U32(_) => MetaType::U32,
                        MetaValue::U64(_) => MetaType::U64,
                        MetaValue::F64(_) => MetaType::F64,
                        MetaValue::String(_) => MetaType::String,
                        MetaValue::Bool3(_) => MetaType::Bool3,
                        MetaValue::I32x3(_) => MetaType::I32x3,
                        MetaValue::I64x3(_) => MetaType::I64x3,
                        MetaValue::U32x3(_) => MetaType::U32x3,
                        MetaValue::U64x3(_) => MetaType::U64x3,
                        MetaValue::F64x3(_) => MetaType::F64x3,
                        MetaValue::F64x6(_) => MetaType::F64x6,
                        MetaValue::F64x9(_) => MetaType::F64x9,
                        MetaValue::Json(_) => MetaType::String,
                    };
                    let mut entry = empty_meta_entry(key.clone(), dtype);
                    match value {
                        MetaValue::Bool(v) => entry.bool_value = *v,
                        MetaValue::I32(v) => entry.i32_value = *v,
                        MetaValue::I64(v) => entry.i64_value = *v,
                        MetaValue::U32(v) => entry.u32_value = *v,
                        MetaValue::U64(v) => entry.u64_value = *v,
                        MetaValue::F64(v) => entry.f64_value = *v,
                        MetaValue::String(v) => entry.string_value = v.clone(),
                        MetaValue::Bool3(v) => {
                            entry.bool_values = v.iter().map(|v| u8::from(*v)).collect()
                        }
                        MetaValue::I32x3(v) => entry.i32_values = v.to_vec(),
                        MetaValue::I64x3(v) => entry.i64_values = v.to_vec(),
                        MetaValue::U32x3(v) => entry.u32_values = v.to_vec(),
                        MetaValue::U64x3(v) => entry.u64_values = v.to_vec(),
                        MetaValue::F64x3(v) => entry.f64_values = v.to_vec(),
                        MetaValue::F64x6(v) => entry.f64_values = v.to_vec(),
                        MetaValue::F64x9(v) => entry.f64_values = v.to_vec(),
                        MetaValue::Json(v) => entry.string_value = v.to_string(),
                    }
                    entry
                })
                .collect()
        })
        .unwrap_or_default()
}

fn fixed<T, const N: usize>(values: Vec<T>, dtype: &str) -> Result<[T; N], String> {
    values.try_into().map_err(|values: Vec<T>| {
        format!("metadata {dtype} expects {N} values, got {}", values.len())
    })
}

pub(crate) fn meta_from_entry(
    entry: bridge::ffi::MetaEntry,
) -> Result<(String, MetaValue), String> {
    use bridge::ffi::MetaType;
    if entry.key.is_empty() {
        return Err("metadata key must not be empty".into());
    }
    let value = match entry.dtype {
        MetaType::Bool => MetaValue::Bool(entry.bool_value),
        MetaType::I32 => MetaValue::I32(entry.i32_value),
        MetaType::I64 => MetaValue::I64(entry.i64_value),
        MetaType::U32 => MetaValue::U32(entry.u32_value),
        MetaType::U64 => MetaValue::U64(entry.u64_value),
        MetaType::F64 => MetaValue::F64(entry.f64_value),
        MetaType::String => MetaValue::String(entry.string_value),
        MetaType::Bool3 => {
            if entry.bool_values.iter().any(|&v| v > 1) {
                return Err("metadata bool3 values must be 0 or 1".into());
            }
            MetaValue::Bool3(fixed::<_, 3>(
                entry.bool_values.into_iter().map(|v| v != 0).collect(),
                "bool3",
            )?)
        }
        MetaType::I32x3 => MetaValue::I32x3(fixed(entry.i32_values, "i32x3")?),
        MetaType::I64x3 => MetaValue::I64x3(fixed(entry.i64_values, "i64x3")?),
        MetaType::U32x3 => MetaValue::U32x3(fixed(entry.u32_values, "u32x3")?),
        MetaType::U64x3 => MetaValue::U64x3(fixed(entry.u64_values, "u64x3")?),
        MetaType::F64x3 => MetaValue::F64x3(fixed(entry.f64_values, "f64x3")?),
        MetaType::F64x6 => MetaValue::F64x6(fixed(entry.f64_values, "f64x6")?),
        MetaType::F64x9 => MetaValue::F64x9(fixed(entry.f64_values, "f64x9")?),
        _ => return Err("unknown metadata dtype".into()),
    };
    Ok((entry.key, value))
}

/// Insert or replace one exact-dtype metadata entry.
pub(crate) fn frame_set_meta_entry(
    fref: &mut FrameRef,
    entry: bridge::ffi::MetaEntry,
) -> Result<(), String> {
    let (key, value) = meta_from_entry(entry)?;
    fref.0
        .with_meta_mut(|meta| {
            meta.insert(key, value);
        })
        .map_err(|err| err.to_string())
}

/// Declare the precision of an existing `f64` column: an absolute tolerance
/// in the column's units. A `*.mrec` writer (`write_frame`, a trajectory
/// writer minted from this frame) stores the column rounded to the largest
/// power of two not above it, shuffled and compressed; the in-memory values
/// are untouched.
///
/// @param fref      frame handle
/// @param block     block key
/// @param col       column key (must exist and be `f64`)
/// @param precision finite, within `[2^-1000, 2^1000]`
pub(crate) fn frame_set_precision(
    fref: &mut FrameRef,
    block: &str,
    col: &str,
    precision: f64,
) -> Result<(), String> {
    fref.0
        .with_mut(|frame| {
            frame
                .get_mut(block)
                .ok_or_else(|| format!("frame_set_precision: no block {block:?}"))?
                .set_precision(col, precision)
                .map_err(|e| format!("frame_set_precision {col}: {e}"))
        })
        .map_err(|e| format!("frame_set_precision: {e}"))?
}

/// Copy an `f64` column out of a block.
///
/// @param fref  frame handle
/// @param block block key
/// @param col   column key
/// @return owned column data; empty if the block or column is absent
pub(crate) fn frame_column_f64(fref: &FrameRef, block: &str, col: &str) -> Vec<f64> {
    match fref.0.block(block) {
        Ok(blk) => match blk.copy_f(col) {
            Ok(Some((data, _shape))) => data,
            _ => Vec::new(),
        },
        Err(_) => Vec::new(),
    }
}

/// Copy an `i32` column out of a block.
///
/// @param fref  frame handle
/// @param block block key
/// @param col   column key
/// @return owned column data; empty if the block or column is absent
pub(crate) fn frame_column_i32(fref: &FrameRef, block: &str, col: &str) -> Vec<i32> {
    match fref.0.block(block) {
        Ok(blk) => match blk.copy_i(col) {
            Ok(Some((data, _shape))) => data,
            _ => Vec::new(),
        },
        Err(_) => Vec::new(),
    }
}

/// Copy a domain-uint (`u64` / `Idx`) column out of a block.
///
/// @param fref  frame handle
/// @param block block key
/// @param col   column key
/// @return owned column data; empty if the block or column is absent
pub(crate) fn frame_column_u32(fref: &FrameRef, block: &str, col: &str) -> Vec<u64> {
    match fref.0.block(block) {
        Ok(blk) => match blk.copy_u(col) {
            Ok(Some((data, _shape))) => data,
            _ => Vec::new(),
        },
        Err(_) => Vec::new(),
    }
}

/// Copy a string column out of a block.
///
/// @param fref  frame handle
/// @param block block key
/// @param col   column key
/// @return owned column data; empty if the block or column is absent
pub(crate) fn frame_column_str(fref: &FrameRef, block: &str, col: &str) -> Vec<String> {
    match fref.0.block(block) {
        Ok(blk) => match blk.col_str(col) {
            Ok(Some(data)) => data,
            _ => Vec::new(),
        },
        Err(_) => Vec::new(),
    }
}

/// Read the simulation-cell matrix as 9 row-major `f64` (3x3 H).
///
/// @param fref frame handle
/// @return 9-element row-major H; empty if no simbox is set
pub(crate) fn frame_box(fref: &FrameRef) -> Vec<f64> {
    match fref.0.box_clone() {
        Ok(Some(sb)) => sb.h_view().iter().copied().collect(),
        _ => Vec::new(),
    }
}

/// Get-or-create a block by key, then run fallible `f` to populate it.
pub(crate) fn with_block_inserted_res<T, E>(
    frame: &mut Frame,
    block: &str,
    f: impl FnOnce(&mut Block) -> Result<T, E>,
) -> Result<T, E> {
    if let Some(blk) = frame.get_mut(block) {
        f(blk)
    } else {
        let mut blk = Block::new();
        let out = f(&mut blk)?;
        frame.insert(block, blk);
        Ok(out)
    }
}

/// Create or overwrite an `f64` column on a block.
///
/// @param fref  frame handle
/// @param block block key (created if absent)
/// @param col   column key (overwritten if present)
/// @param data  column values
pub(crate) fn frame_set_column_f64(
    fref: &mut FrameRef,
    block: &str,
    col: &str,
    data: &[f64],
) -> Result<(), String> {
    fref.0
        .with_mut(|frame| {
            with_block_inserted_res(frame, block, |blk| {
                blk.insert(col, Array1::from_vec(data.to_vec()).into_dyn())
                    .map_err(|e| format!("frame_set_column_f64 insert {col}: {e}"))
            })
        })
        .map_err(|e| format!("frame_set_column_f64: {e}"))?
}

/// Create or overwrite an `i32` column on a block.
///
/// @param fref  frame handle
/// @param block block key (created if absent)
/// @param col   column key (overwritten if present)
/// @param data  column values
pub(crate) fn frame_set_column_i32(
    fref: &mut FrameRef,
    block: &str,
    col: &str,
    data: &[i32],
) -> Result<(), String> {
    fref.0
        .with_mut(|frame| {
            with_block_inserted_res(frame, block, |blk| {
                blk.insert(col, Array1::from_vec(data.to_vec()).into_dyn())
                    .map_err(|e| format!("frame_set_column_i32 insert {col}: {e}"))
            })
        })
        .map_err(|e| format!("frame_set_column_i32: {e}"))?
}

/// Create or overwrite a domain-uint (`u64` / `Idx`) column on a block.
///
/// @param fref  frame handle
/// @param block block key (created if absent)
/// @param col   column key (overwritten if present)
/// @param data  column values
pub(crate) fn frame_set_column_u32(
    fref: &mut FrameRef,
    block: &str,
    col: &str,
    data: &[u64],
) -> Result<(), String> {
    fref.0
        .with_mut(|frame| {
            with_block_inserted_res(frame, block, |blk| {
                blk.insert(col, Array1::from_vec(data.to_vec()).into_dyn())
                    .map_err(|e| format!("frame_set_column_u32 insert {col}: {e}"))
            })
        })
        .map_err(|e| format!("frame_set_column_u32: {e}"))?
}

/// Create or overwrite a string column on a block.
///
/// @param fref  frame handle
/// @param block block key (created if absent)
/// @param col   column key (overwritten if present)
/// @param data  column values
pub(crate) fn frame_set_column_str(
    fref: &mut FrameRef,
    block: &str,
    col: &str,
    data: &[String],
) -> Result<(), String> {
    fref.0
        .with_mut(|frame| {
            with_block_inserted_res(frame, block, |blk| {
                blk.insert(
                    col,
                    Array1::from_vec(data.to_vec()).into_dyn() as ArrayD<String>,
                )
                .map_err(|e| format!("frame_set_column_str insert {col}: {e}"))
            })
        })
        .map_err(|e| format!("frame_set_column_str: {e}"))?
}

/// Set the simulation-cell matrix from 9 row-major `f64` (3x3 H).
///
/// The box origin is set to zero and PBC is enabled on all three axes,
/// matching how `take_frame`/`write_view` construct a [`SimBox`].
///
/// @param fref frame handle
/// @param h    9-element row-major 3x3 cell matrix
pub(crate) fn frame_set_box(fref: &mut FrameRef, h: &[f64]) -> Result<(), String> {
    if h.len() < 9 {
        return Err("frame_set_box: H must have 9 elements".into());
    }
    let mat = Array2::from_shape_vec((3, 3), h[..9].to_vec())
        .map_err(|e| format!("frame_set_box reshape: {e}"))?;
    let simbox = SimBox::new(mat, Array1::zeros(3), [true, true, true])
        .map_err(|e| format!("frame_set_box: singular cell matrix: {e:?}"))?;
    fref.0
        .set_box(Some(simbox))
        .map_err(|e| format!("frame_set_box: {e}"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::rc::Rc;

    /// Round-trip: write every column dtype + simbox through the bridge,
    /// then read them back and assert element-wise equality. Also exercises
    /// the introspection surface (`names`, `has`, `columns`, `nrows`).
    #[test]
    fn frame_roundtrip_all_dtypes() {
        let mut fref = frame_new();

        // One column per dtype, each on a key whose *schema* dtype matches it.
        // The vocabulary binds a key's dtype wherever it appears, so writing an
        // i32 into `id` (a UInt) is refused — which is the schema working, not
        // the bridge failing.
        let xs: Vec<f64> = vec![0.0, 1.5, -2.25];
        let spins: Vec<i32> = vec![-1, 0, 1];
        let ids: Vec<u64> = vec![10, 20, 30];
        let elems: Vec<String> = vec!["H".into(), "H".into(), "O".into()];

        frame_set_column_f64(&mut fref, "atoms", "x", &xs).unwrap();
        frame_set_column_i32(&mut fref, "atoms", "spin", &spins).unwrap();
        frame_set_column_u32(&mut fref, "atoms", "id", &ids).unwrap();
        frame_set_column_str(&mut fref, "atoms", "element", &elems).unwrap();

        // 9-elem row-major 3x3 H matrix.
        let h: Vec<f64> = vec![12.0, 0.0, 0.0, 0.0, 13.0, 0.0, 0.0, 0.0, 14.0];
        frame_set_box(&mut fref, &h).unwrap();

        // ── Introspection ──
        let names = frame_block_names(&fref);
        assert_eq!(names, vec!["atoms".to_string()]);
        assert!(frame_has_block(&fref, "atoms"));
        assert!(!frame_has_block(&fref, "bonds"));

        let mut cols = frame_block_columns(&fref, "atoms");
        cols.sort();
        assert_eq!(cols, vec!["element", "id", "spin", "x"]);
        assert_eq!(frame_block_nrows(&fref, "atoms"), 3);
        assert_eq!(frame_block_nrows(&fref, "missing"), 0);

        // ── Readers ──
        assert_eq!(frame_column_f64(&fref, "atoms", "x"), xs);
        assert_eq!(frame_column_i32(&fref, "atoms", "spin"), spins);
        assert_eq!(frame_column_u32(&fref, "atoms", "id"), ids);
        assert_eq!(frame_column_str(&fref, "atoms", "element"), elems);
        assert_eq!(frame_box(&fref), h);

        // ── Absent block / column → empty Vec, never a panic ──
        assert!(frame_column_f64(&fref, "atoms", "nope").is_empty());
        assert!(frame_column_f64(&fref, "missing", "x").is_empty());
        assert!(frame_column_i32(&fref, "missing", "spin").is_empty());
        assert!(frame_column_u32(&fref, "missing", "id").is_empty());
        assert!(frame_column_str(&fref, "missing", "element").is_empty());
        assert!(frame_block_columns(&fref, "missing").is_empty());

        // ── Update path: overwrite an existing column ──
        let xs2: Vec<f64> = vec![9.0, 8.0, 7.0];
        frame_set_column_f64(&mut fref, "atoms", "x", &xs2).unwrap();
        assert_eq!(frame_column_f64(&fref, "atoms", "x"), xs2);
        // other columns survive the update
        assert_eq!(frame_column_u32(&fref, "atoms", "id"), ids);
    }

    #[test]
    fn frame_metadata_roundtrip_preserves_exact_native_types() {
        use bridge::ffi::MetaType;
        let mut fref = frame_new();

        let mut tag = empty_meta_entry("tag".into(), MetaType::I64);
        tag.i64_value = 9_007_199_254_740_993;
        frame_set_meta_entry(&mut fref, tag).unwrap();

        let mut stress = empty_meta_entry("stress".into(), MetaType::F64x6);
        stress.f64_values = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        frame_set_meta_entry(&mut fref, stress).unwrap();

        let entries = frame_meta_entries(&fref);
        assert_eq!(entries.len(), 2);
        let tag = entries.iter().find(|entry| entry.key == "tag").unwrap();
        assert!(tag.dtype == MetaType::I64);
        assert_eq!(tag.i64_value, 9_007_199_254_740_993);
        let stress = entries.iter().find(|entry| entry.key == "stress").unwrap();
        assert!(stress.dtype == MetaType::F64x6);
        assert_eq!(stress.f64_values, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

        let mut malformed = empty_meta_entry("bad".into(), MetaType::F64x6);
        malformed.f64_values = vec![1.0; 5];
        assert!(frame_set_meta_entry(&mut fref, malformed).is_err());
    }

    #[test]
    fn frame_meta_entries_follow_insertion_order() {
        use bridge::ffi::MetaType;
        let mut fref = frame_new();

        // Not alphabetical: a reintroduced sort must fail this case.
        let mut tag = empty_meta_entry("tag".into(), MetaType::I64);
        tag.i64_value = 1;
        frame_set_meta_entry(&mut fref, tag).unwrap();

        let mut stress = empty_meta_entry("stress".into(), MetaType::F64);
        stress.f64_value = 2.0;
        frame_set_meta_entry(&mut fref, stress).unwrap();

        let mut run = empty_meta_entry("run".into(), MetaType::String);
        run.string_value = "third".into();
        frame_set_meta_entry(&mut fref, run).unwrap();

        let keys: Vec<String> = frame_meta_entries(&fref)
            .into_iter()
            .map(|entry| entry.key)
            .collect();
        assert_eq!(
            keys,
            vec!["tag".to_string(), "stress".to_string(), "run".to_string()]
        );
    }

    /// Capsule / shared-Store semantics: a cloned `FrameRef` shares the
    /// same `Store`. Round-tripping through `Box::into_raw`/`from_raw`
    /// (the PyCapsule pattern) must preserve `Rc::ptr_eq`, and a mutation
    /// through one handle must be visible through the other.
    #[test]
    fn cloned_frameref_shares_store() {
        let mut original = frame_new();
        frame_set_column_f64(&mut original, "atoms", "x", &[1.0, 2.0, 3.0]).unwrap();

        // Clone + simulate the PyCapsule carry: Box::into_raw → Box::from_raw.
        let cloned: FrameRef = FrameRef(original.0.clone());
        let raw: *mut FrameRef = Box::into_raw(Box::new(cloned));
        let mut recovered: Box<FrameRef> = unsafe { Box::from_raw(raw) };

        // Same Store behind both handles.
        assert!(Rc::ptr_eq(&original.0.store, &recovered.0.store));

        // Mutate through the recovered handle, observe through the original.
        frame_set_column_f64(&mut recovered, "atoms", "x", &[7.0, 8.0, 9.0]).unwrap();
        assert_eq!(
            frame_column_f64(&original, "atoms", "x"),
            vec![7.0, 8.0, 9.0]
        );

        // ...and vice versa.
        frame_set_column_u32(&mut original, "atoms", "id", &[1, 2, 3]).unwrap();
        assert_eq!(frame_column_u32(&recovered, "atoms", "id"), vec![1, 2, 3]);
    }
}
