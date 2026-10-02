//! `extern "C"` functions for Block column access.
//!
//! A **Block** is a heterogeneous column store: each column is a
//! contiguous, typed ndarray keyed by an interned string.  This module
//! One read returns the column's bytes. The dtype is an out-parameter,
//! not part of the function name.
//!
//! | Pattern   | Function | Semantics |
//! |-----------|----------|-----------|
//! | **Pointer (read)**  | `molrs_block_get` | Zero-copy bytes; `out_dtype` is the stored variant |
//! | **Pointer (write)** | `molrs_block_get_mut` | Zero-copy mutable bytes; version bumped |
//! | **Copy**  | `molrs_block_copy` | Copies the column's bytes into a caller buffer |
//! | **Insert**| `molrs_block_set_F/I/U` | Copies caller data into a new column |
//!
//! `get`, `get_mut`, and `copy` succeed for every fixed-width dtype the
//! store holds. A string column has no flat scalar buffer and returns
//! `TypeMismatch`, which is not `KeyNotFound`. Shape and dtype stay on
//! [`molrs_block_col_shape`] and [`molrs_block_col_dtype`].
//!
//! `F` is `f64`, `I` is `i32`, `Idx` is `u64`. `out_len` is an element
//! count. `molrs_block_copy`'s `buf_bytes` is a byte capacity.

use molrs::store::block::{Column, DType};
use molrs::types::{F, I, Idx};
use ndarray::ArrayD;

use crate::error::{self, MolrsDType, MolrsStatus, ffi_err_to_status};
use crate::handle::{MolrsBlockHandle, c_to_block_handle};
use crate::store::lock_store;
use crate::{ffi_try, null_check};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Resolve a C block handle to a Rust `BlockHandle`.
macro_rules! resolve_block {
    ($store:expr, $c_handle:expr) => {
        match c_to_block_handle($c_handle, &$store.interned_keys) {
            Some(bh) => bh,
            None => {
                error::set_last_error("invalid block handle or unknown key_id");
                return MolrsStatus::InvalidBlockHandle;
            }
        }
    };
}

/// Look up a column key string from an interned key_id.
macro_rules! resolve_col_key {
    ($store:expr, $col_key_id:expr) => {
        match $store.key_str($col_key_id) {
            Some(s) => s.to_owned(),
            None => {
                error::set_last_error(format!("unknown col_key_id {}", $col_key_id));
                return MolrsStatus::KeyNotFound;
            }
        }
    };
}

// ---------------------------------------------------------------------------
// Info
// ---------------------------------------------------------------------------

/// Get the number of rows in a block.
///
/// If the block has no columns yet, `*out` is set to 0.
///
/// # C signature
///
/// ```c
/// MolrsStatus molrs_block_nrows(MolrsBlockHandle block, size_t* out);
/// ```
///
/// # Arguments
///
/// * `block` -- Block handle.
/// * `out` -- On success, receives the row count.
///
/// # Returns
///
/// * `MolrsStatus::Ok` on success.
/// * `MolrsStatus::NullPointer` if `out` is null.
/// * `MolrsStatus::InvalidBlockHandle` if `block` is stale or invalid.
///
/// # Safety
///
/// * `block` must be a live block handle.
/// * `out` must point to a writable `size_t`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn molrs_block_nrows(
    block: MolrsBlockHandle,
    out: *mut usize,
) -> MolrsStatus {
    ffi_try!({
        null_check!(out);
        let store = lock_store();
        let bh = resolve_block!(store, &block);
        match store.inner.with_block(&bh, |b| b.nrows()) {
            Ok(Some(n)) => {
                unsafe { *out = n };
                MolrsStatus::Ok
            }
            Ok(None) => {
                unsafe { *out = 0 };
                MolrsStatus::Ok
            }
            Err(e) => ffi_err_to_status(&e),
        }
    })
}

/// Get the number of columns in a block.
///
/// # C signature
///
/// ```c
/// MolrsStatus molrs_block_ncols(MolrsBlockHandle block, size_t* out);
/// ```
///
/// # Arguments
///
/// * `block` -- Block handle.
/// * `out` -- On success, receives the column count.
///
/// # Returns
///
/// * `MolrsStatus::Ok` on success.
/// * `MolrsStatus::NullPointer` if `out` is null.
/// * `MolrsStatus::InvalidBlockHandle` if `block` is stale or invalid.
///
/// # Safety
///
/// * `block` must be a live block handle.
/// * `out` must point to a writable `size_t`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn molrs_block_ncols(
    block: MolrsBlockHandle,
    out: *mut usize,
) -> MolrsStatus {
    ffi_try!({
        null_check!(out);
        let store = lock_store();
        let bh = resolve_block!(store, &block);
        match store.inner.with_block(&bh, |b| b.len()) {
            Ok(n) => {
                unsafe { *out = n };
                MolrsStatus::Ok
            }
            Err(e) => ffi_err_to_status(&e),
        }
    })
}

/// Query the data type of a column.
///
/// The returned [`MolrsDType`] is the stored variant. An `i64` column is
/// `Int64`, not `Int`. Pair it with [`molrs_block_get`](crate::molrs_block_get).
///
/// # C signature
///
/// ```c
/// MolrsStatus molrs_block_col_dtype(MolrsBlockHandle block,
///                                    uint32_t col_key_id,
///                                    MolrsDType* out);
/// ```
///
/// # Arguments
///
/// * `block` -- Block handle.
/// * `col_key_id` -- Interned column name.
/// * `out` -- On success, receives the data type discriminant.
///
/// # Returns
///
/// * `MolrsStatus::Ok` on success.
/// * `MolrsStatus::NullPointer` if `out` is null.
/// * `MolrsStatus::KeyNotFound` if `col_key_id` was not interned or
///   the column does not exist.
/// * `MolrsStatus::InvalidBlockHandle` if `block` is stale.
///
/// # Safety
///
/// * `block` must be a live block handle.
/// * `out` must point to a writable `MolrsDType`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn molrs_block_col_dtype(
    block: MolrsBlockHandle,
    col_key_id: u32,
    out: *mut MolrsDType,
) -> MolrsStatus {
    ffi_try!({
        null_check!(out);
        let store = lock_store();
        let bh = resolve_block!(store, &block);
        let col_key = resolve_col_key!(store, col_key_id);
        let result = store.inner.with_block(&bh, |b| {
            b.get(&col_key).map(|col| MolrsDType::from(col.dtype()))
        });
        match result {
            Ok(Some(dt)) => {
                unsafe { *out = dt };
                MolrsStatus::Ok
            }
            Ok(None) => {
                error::set_last_error(format!("column '{}' not found", col_key));
                MolrsStatus::KeyNotFound
            }
            Err(e) => ffi_err_to_status(&e),
        }
    })
}

/// Query the shape (dimensionality) of a column.
///
/// On entry, `*inout_ndim` must hold the capacity of the `out_shape`
/// buffer.  On return, `*inout_ndim` is set to the actual number of
/// dimensions, and the first `min(capacity, ndim)` elements of
/// `out_shape` are filled.
///
/// # C signature
///
/// ```c
/// MolrsStatus molrs_block_col_shape(MolrsBlockHandle block,
///                                    uint32_t col_key_id,
///                                    size_t*  out_shape,
///                                    size_t*  inout_ndim);
/// ```
///
/// # Arguments
///
/// * `block` -- Block handle.
/// * `col_key_id` -- Interned column name.
/// * `out_shape` -- Buffer of at least `*inout_ndim` elements.
/// * `inout_ndim` -- In: buffer capacity.  Out: actual number of dimensions.
///
/// # Returns
///
/// * `MolrsStatus::Ok` on success.
/// * `MolrsStatus::NullPointer` if `out_shape` or `inout_ndim` is null.
/// * `MolrsStatus::KeyNotFound` if the column does not exist.
/// * `MolrsStatus::InvalidBlockHandle` if `block` is stale.
///
/// # Safety
///
/// * `block` must be a live block handle.
/// * `out_shape` must point to a buffer of at least `*inout_ndim`
///   writable `size_t` elements.
/// * `inout_ndim` must point to a writable `size_t`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn molrs_block_col_shape(
    block: MolrsBlockHandle,
    col_key_id: u32,
    out_shape: *mut usize,
    inout_ndim: *mut usize,
) -> MolrsStatus {
    ffi_try!({
        null_check!(out_shape);
        null_check!(inout_ndim);
        let store = lock_store();
        let bh = resolve_block!(store, &block);
        let col_key = resolve_col_key!(store, col_key_id);
        let result = store
            .inner
            .with_block(&bh, |b| b.get(&col_key).map(|col| col.shape().to_vec()));
        match result {
            Ok(Some(shape)) => {
                let max_ndim = unsafe { *inout_ndim };
                let actual_ndim = shape.len();
                unsafe { *inout_ndim = actual_ndim };
                let copy_len = actual_ndim.min(max_ndim);
                let out_slice = unsafe { std::slice::from_raw_parts_mut(out_shape, copy_len) };
                out_slice.copy_from_slice(&shape[..copy_len]);
                MolrsStatus::Ok
            }
            Ok(None) => {
                error::set_last_error(format!("column '{}' not found", col_key));
                MolrsStatus::KeyNotFound
            }
            Err(e) => ffi_err_to_status(&e),
        }
    })
}

// ---------------------------------------------------------------------------
// Zero-copy read, mutable read, and copy
// ---------------------------------------------------------------------------

#[derive(Debug)]
enum ReadFail {
    Missing,
    String,
    NonContiguous,
    TooSmall,
}

fn fail_status(fail: ReadFail, key: &str) -> MolrsStatus {
    let (status, msg) = match fail {
        ReadFail::Missing => (
            MolrsStatus::KeyNotFound,
            format!("column '{key}' not found"),
        ),
        ReadFail::String => (
            MolrsStatus::TypeMismatch,
            format!("column '{key}' is string; a string column has no flat scalar buffer"),
        ),
        ReadFail::NonContiguous => (
            MolrsStatus::NonContiguous,
            format!("column '{key}' is not contiguous"),
        ),
        ReadFail::TooSmall => (MolrsStatus::InvalidArgument, "buffer too small".to_string()),
    };
    error::set_last_error(msg);
    status
}

fn scalar_view(col: &Column) -> Result<(*const u8, usize, MolrsDType), ReadFail> {
    if col.dtype() == DType::String {
        return Err(ReadFail::String);
    }
    let dtype = MolrsDType::from(col.dtype());
    macro_rules! arm {
        ($method:ident) => {
            if let Some(arr) = col.$method() {
                let slice = arr.as_slice_memory_order().ok_or(ReadFail::NonContiguous)?;
                return Ok((slice.as_ptr() as *const u8, slice.len(), dtype));
            }
        };
    }
    arm!(as_float);
    arm!(as_i8);
    arm!(as_i16);
    arm!(as_int);
    arm!(as_i64);
    arm!(as_bool);
    arm!(as_u8);
    arm!(as_u16);
    arm!(as_u32);
    arm!(as_uint);
    arm!(as_c64);
    arm!(as_c128);
    Err(ReadFail::String)
}

fn scalar_view_mut(col: &mut Column) -> Result<(*mut u8, usize, MolrsDType), ReadFail> {
    if col.dtype() == DType::String {
        return Err(ReadFail::String);
    }
    let dtype = MolrsDType::from(col.dtype());
    macro_rules! arm {
        ($method:ident) => {
            if let Some(arr) = col.$method() {
                let slice = arr
                    .as_slice_memory_order_mut()
                    .ok_or(ReadFail::NonContiguous)?;
                return Ok((slice.as_mut_ptr() as *mut u8, slice.len(), dtype));
            }
        };
    }
    arm!(as_float_mut);
    arm!(as_i8_mut);
    arm!(as_i16_mut);
    arm!(as_int_mut);
    arm!(as_i64_mut);
    arm!(as_bool_mut);
    arm!(as_u8_mut);
    arm!(as_u16_mut);
    arm!(as_u32_mut);
    arm!(as_uint_mut);
    arm!(as_c64_mut);
    arm!(as_c128_mut);
    Err(ReadFail::String)
}

fn scalar_bytes(col: &Column) -> Result<Vec<u8>, ReadFail> {
    let (ptr, len, _) = scalar_view(col)?;
    let width = col.dtype().itemsize().ok_or(ReadFail::String)?;
    let nbytes = len.saturating_mul(width);
    // `ptr` addresses `len` elements of `width` bytes still owned by `col`.
    let bytes = unsafe { std::slice::from_raw_parts(ptr, nbytes) }.to_vec();
    Ok(bytes)
}

/// Byte pointer and element count for one column.
///
/// `*out_len` is the number of elements. The byte length is `*out_len` times
/// the dtype's item size (`Bool` is 1, `Complex64` is 8, `Complex128` is 16).
/// `*out_dtype` is the stored variant.
///
/// A missing column is `KeyNotFound`. A string column is `TypeMismatch`:
/// a string has no flat scalar buffer. Those are different statuses.
///
/// # Safety
///
/// * `block` must be a live block handle.
/// * `out_ptr`, `out_len`, and `out_dtype` must be non-null and writable.
/// * The returned pointer is valid until the block is mutated or the frame
///   is dropped.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn molrs_block_get(
    block: MolrsBlockHandle,
    col_key_id: u32,
    out_ptr: *mut *const u8,
    out_len: *mut usize,
    out_dtype: *mut MolrsDType,
) -> MolrsStatus {
    ffi_try!({
        null_check!(out_ptr);
        null_check!(out_len);
        null_check!(out_dtype);
        let store = lock_store();
        let bh = resolve_block!(store, &block);
        let col_key = resolve_col_key!(store, col_key_id);
        let result = store.inner.with_block(&bh, |b| match b.get(&col_key) {
            None => Err(ReadFail::Missing),
            Some(col) => scalar_view(col),
        });
        match result {
            Ok(Ok((ptr, len, dtype))) => {
                unsafe {
                    *out_ptr = ptr;
                    *out_len = len;
                    *out_dtype = dtype;
                }
                MolrsStatus::Ok
            }
            Ok(Err(fail)) => fail_status(fail, &col_key),
            Err(e) => ffi_err_to_status(&e),
        }
    })
}

/// Mutable byte pointer for one column. Bumps `block->block_version`.
///
/// Same dtype and status rules as [`molrs_block_get`]. Call
/// [`molrs_block_col_commit`] after writing. The version is already bumped;
/// commit is currently a no-op.
///
/// # Safety
///
/// * `block` must point to a live, writable `MolrsBlockHandle`.
/// * `out_ptr`, `out_len`, and `out_dtype` must be non-null and writable.
/// * The returned pointer is valid until another mutating call on the same
///   block, or until the frame is dropped.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn molrs_block_get_mut(
    block: *mut MolrsBlockHandle,
    col_key_id: u32,
    out_ptr: *mut *mut u8,
    out_len: *mut usize,
    out_dtype: *mut MolrsDType,
) -> MolrsStatus {
    ffi_try!({
        null_check!(block);
        null_check!(out_ptr);
        null_check!(out_len);
        null_check!(out_dtype);
        let c_handle = unsafe { &*block };
        let mut store = lock_store();
        let col_key = resolve_col_key!(store, col_key_id);
        let mut bh = resolve_block!(store, c_handle);
        let result = store
            .inner
            .with_block_mut(&mut bh, |b| match b.get_mut(&col_key) {
                None => Err(ReadFail::Missing),
                Some(col) => scalar_view_mut(col),
            });
        match result {
            Ok(Ok((ptr, len, dtype))) => {
                let c_block = unsafe { &mut *block };
                c_block.block_version = bh.version();
                unsafe {
                    *out_ptr = ptr;
                    *out_len = len;
                    *out_dtype = dtype;
                }
                MolrsStatus::Ok
            }
            Ok(Err(fail)) => fail_status(fail, &col_key),
            Err(e) => ffi_err_to_status(&e),
        }
    })
}

/// Copy one column's bytes into `out_buf`.
///
/// `buf_bytes` is the buffer's capacity in bytes, not elements. A buffer
/// shorter than `n_elements * itemsize` returns `InvalidArgument`.
/// A missing column is `KeyNotFound`. A string column is `TypeMismatch`.
///
/// # Safety
///
/// * `block` must be a live block handle.
/// * `out_buf` must point at `buf_bytes` writable bytes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn molrs_block_copy(
    block: MolrsBlockHandle,
    col_key_id: u32,
    out_buf: *mut u8,
    buf_bytes: usize,
) -> MolrsStatus {
    ffi_try!({
        null_check!(out_buf);
        let store = lock_store();
        let bh = resolve_block!(store, &block);
        let col_key = resolve_col_key!(store, col_key_id);
        let result = store.inner.with_block(&bh, |b| {
            let Some(col) = b.get(&col_key) else {
                return Err(ReadFail::Missing);
            };
            let bytes = scalar_bytes(col)?;
            if buf_bytes < bytes.len() {
                return Err(ReadFail::TooSmall);
            }
            Ok(bytes)
        });
        match result {
            Ok(Ok(data)) => {
                let out_slice = unsafe { std::slice::from_raw_parts_mut(out_buf, data.len()) };
                out_slice.copy_from_slice(&data);
                MolrsStatus::Ok
            }
            Ok(Err(fail)) => fail_status(fail, &col_key),
            Err(e) => ffi_err_to_status(&e),
        }
    })
}

/// No-op kept for callers that write through [`molrs_block_get_mut`].
///
/// The mutable read already bumps `block->block_version`. This call does
/// not bump it again.
///
/// # Safety
///
/// `block` must point at a live `MolrsBlockHandle`, or be null (which
/// returns `NullPointer`).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn molrs_block_col_commit(block: *mut MolrsBlockHandle) -> MolrsStatus {
    ffi_try!({
        null_check!(block);
        MolrsStatus::Ok
    })
}

#[cfg(test)]
mod scalar_view_tests {
    use super::*;
    use ndarray::Array1;

    #[test]
    fn i64_view_reports_int64_and_the_elements() {
        let col = Column::from_i64(Array1::from_vec(vec![7_i64, 8]).into_dyn());
        let (ptr, len, dtype) = scalar_view(&col).unwrap();
        assert_eq!(len, 2);
        assert_eq!(dtype, MolrsDType::Int64);
        let vals = unsafe { std::slice::from_raw_parts(ptr as *const i64, len) };
        assert_eq!(vals, &[7, 8]);
    }

    #[test]
    fn string_is_not_a_missing_column() {
        let col = Column::from_string(Array1::from_vec(vec!["a".to_string()]).into_dyn());
        assert!(matches!(scalar_view(&col), Err(ReadFail::String)));
        assert!(matches!(scalar_bytes(&col), Err(ReadFail::String)));
    }
}

// ---------------------------------------------------------------------------
// Insert columns
// ---------------------------------------------------------------------------

/// Macro to generate typed column insertion functions.
///
/// Each generated function copies caller-provided data into a new
/// column in the block.  If a column with the same key already exists,
/// it is replaced.
///
/// # Generated C signatures
///
/// ```c
/// MolrsStatus molrs_block_set_F(MolrsBlockHandle* block, uint32_t col_key_id,
///                                const molrs_float_t* data,
///                                const size_t* shape, size_t ndim);
/// MolrsStatus molrs_block_set_I(MolrsBlockHandle* block, uint32_t col_key_id,
///                                const molrs_int_t* data,
///                                const size_t* shape, size_t ndim);
/// MolrsStatus molrs_block_set_U(MolrsBlockHandle* block, uint32_t col_key_id,
///                                const molrs_uint_t* data,
///                                const size_t* shape, size_t ndim);
/// ```
///
/// # Safety (all generated functions)
///
/// * `block` must point to a live, writable `MolrsBlockHandle`.
/// * `data` must point to at least `product(shape[0..ndim])` elements.
/// * `shape` must point to `ndim` elements.
/// * `ndim` must be >= 1.
macro_rules! impl_block_insert {
    ($fn_name:ident, $ty:ty, $dtype_name:expr) => {
        #[doc = concat!("Insert (or replace) a `", $dtype_name, "` column, copying the caller's data.")]
        ///
        /// On success, the `block_version` in `*block` is updated.
        ///
        /// # Arguments
        ///
        /// * `block` -- Pointer to the block handle (updated in place).
        /// * `col_key_id` -- Interned column name.
        /// * `data` -- Pointer to the source data (row-major).
        /// * `shape` -- Pointer to `ndim` dimension sizes.
        /// * `ndim` -- Number of dimensions (must be >= 1).
        ///
        /// # Returns
        ///
        /// * `MolrsStatus::Ok` on success.
        /// * `MolrsStatus::NullPointer` if any pointer is null.
        /// * `MolrsStatus::InvalidArgument` if `ndim == 0`, total
        ///   element count is 0, or shape/data are inconsistent.
        /// * `MolrsStatus::KeyNotFound` if `col_key_id` was not interned.
        /// * `MolrsStatus::InvalidBlockHandle` if the block handle is stale.
        #[unsafe(no_mangle)]
        pub unsafe extern "C" fn $fn_name(
            block: *mut MolrsBlockHandle,
            col_key_id: u32,
            data: *const $ty,
            shape: *const usize,
            ndim: usize,
        ) -> MolrsStatus {
            ffi_try!({
                null_check!(block);
                null_check!(data);
                null_check!(shape);
                if ndim == 0 {
                    error::set_last_error("ndim must be > 0");
                    return MolrsStatus::InvalidArgument;
                }
                let shape_slice = unsafe { std::slice::from_raw_parts(shape, ndim) };
                let total_len: usize = shape_slice.iter().product();
                if total_len == 0 {
                    error::set_last_error("total element count is 0");
                    return MolrsStatus::InvalidArgument;
                }
                let data_slice = unsafe { std::slice::from_raw_parts(data, total_len) };
                let arr = match ArrayD::<$ty>::from_shape_vec(
                    shape_slice.to_vec(),
                    data_slice.to_vec(),
                ) {
                    Ok(a) => a,
                    Err(e) => {
                        error::set_last_error(format!("invalid shape/data: {e}"));
                        return MolrsStatus::InvalidArgument;
                    }
                };

                let c_handle = unsafe { &*block };
                let mut store = lock_store();
                let col_key = resolve_col_key!(store, col_key_id);
                let mut bh = resolve_block!(store, c_handle);

                let result = store
                    .inner
                    .with_block_mut(&mut bh, |b| b.insert(col_key.clone(), arr));
                match result {
                    Ok(Ok(_)) => {
                        let c_block = unsafe { &mut *block };
                        c_block.block_version = bh.version();
                        MolrsStatus::Ok
                    }
                    Ok(Err(block_err)) => {
                        error::set_last_error(format!("block insert error: {block_err}"));
                        MolrsStatus::InvalidArgument
                    }
                    Err(e) => ffi_err_to_status(&e),
                }
            })
        }
    };
}

impl_block_insert!(molrs_block_set_F, F, "float");
impl_block_insert!(molrs_block_set_I, I, "int");
impl_block_insert!(molrs_block_set_U, Idx, "uint");

// ---------------------------------------------------------------------------
// Runtime width queries
// ---------------------------------------------------------------------------

/// Return the byte size of the primary float type (`molrs_float_t`).
///
/// Returns `4` by default (`f32`) or `8` with the `f64` feature.
/// Use this at runtime to verify that the calling C code was compiled
/// with matching type widths.
///
/// # C signature
///
/// ```c
/// size_t molrs_sizeof_F(void);
/// ```
#[unsafe(no_mangle)]
pub extern "C" fn molrs_sizeof_F() -> usize {
    std::mem::size_of::<F>()
}

/// Return the byte size of the primary signed integer type (`molrs_int_t`).
///
/// Returns `4` by default (`i32`) or `8` with the `i64` feature.
///
/// # C signature
///
/// ```c
/// size_t molrs_sizeof_I(void);
/// ```
#[unsafe(no_mangle)]
pub extern "C" fn molrs_sizeof_I() -> usize {
    std::mem::size_of::<I>()
}

/// Return the byte size of the primary unsigned integer type (`molrs_uint_t`).
///
/// Returns `8` (`Idx` = `u64`).
///
/// # C signature
///
/// ```c
/// size_t molrs_sizeof_U(void);
/// ```
#[unsafe(no_mangle)]
pub extern "C" fn molrs_sizeof_U() -> usize {
    std::mem::size_of::<Idx>()
}
