//! `extern "C"` functions for Block column access.
//!
//! A **Block** is a heterogeneous column store: each column is a
//! contiguous, typed ndarray keyed by an interned string.  This module
//! provides three access patterns for every supported type (`F`, `I`, `U`):
//!
//! | Pattern   | Functions           | Semantics |
//! |-----------|---------------------|-----------|
//! | **Pointer (read)**  | `molrs_block_get_F/I/U` | Zero-copy read; pointer valid while store lock held |
//! | **Pointer (write)** | `molrs_block_get_F/I/U_mut` | Zero-copy mutable access; version bumped |
//! | **Copy**  | `molrs_block_copy_F/I/U` | Copies data into a caller-provided buffer |
//! | **Insert**| `molrs_block_set_F/I/U` | Copies caller data into a new column |
//!
//! Additional query functions: [`molrs_block_nrows`], [`molrs_block_ncols`],
//! [`molrs_block_col_dtype`], [`molrs_block_col_shape`].
//!
//! # Type mapping
//!
//! | Rust type | C typedef (default) | C typedef (wide features) |
//! |-----------|---------------------|---------------------------|
//! | `F`       | `float`             | `double`                  |
//! | `I`       | `int32_t`           | `int64_t`                 |
//! | `U`       | `uint32_t`          | `uint64_t`                |

use molrs::types::{F, I, U};
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
/// The returned [`MolrsDType`] tells the caller which accessor family
/// to use (e.g. `molrs_block_get_F` for `Float`).
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
// Zero-copy READ
// ---------------------------------------------------------------------------

/// Macro to generate typed zero-copy read functions.
///
/// Each generated function returns a read-only pointer into the column's
/// contiguous storage.
///
/// # Generated C signatures
///
/// ```c
/// MolrsStatus molrs_block_get_F(MolrsBlockHandle block, uint32_t col_key_id,
///                                const molrs_float_t** out_ptr, size_t* out_len);
/// MolrsStatus molrs_block_get_I(MolrsBlockHandle block, uint32_t col_key_id,
///                                const molrs_int_t** out_ptr, size_t* out_len);
/// MolrsStatus molrs_block_get_U(MolrsBlockHandle block, uint32_t col_key_id,
///                                const molrs_uint_t** out_ptr, size_t* out_len);
/// ```
///
/// # Safety (all generated functions)
///
/// * `block` must be a live block handle.
/// * `out_ptr` and `out_len` must be valid, non-null, writable pointers.
/// * The returned data pointer is valid only while the global store lock
///   is held.  In practice, because the lock is released before the
///   function returns, the pointer is valid until the block is mutated
///   or the frame is dropped.  The caller must not use the pointer after
///   calling any mutating `molrs_*` function on the same block/frame.
macro_rules! impl_col_ptr {
    ($fn_name:ident, $ty:ty, $getter:ident, $dtype_name:expr) => {
        #[doc = concat!("Get a read-only pointer to a contiguous `", $dtype_name, "` column.")]
        ///
        /// On success, `*out_ptr` points to the first element and
        /// `*out_len` is set to the total number of elements (product of
        /// all shape dimensions).
        ///
        /// The pointer is valid until the block is mutated or the frame
        /// is dropped.
        ///
        /// # Returns
        ///
        /// * `MolrsStatus::Ok` on success.
        /// * `MolrsStatus::NullPointer` if `out_ptr` or `out_len` is null.
        /// * `MolrsStatus::KeyNotFound` if the column does not exist.
        /// * `MolrsStatus::NonContiguous` if the column is not contiguous.
        /// * `MolrsStatus::InvalidBlockHandle` if `block` is stale.
        #[unsafe(no_mangle)]
        pub unsafe extern "C" fn $fn_name(
            block: MolrsBlockHandle,
            col_key_id: u32,
            out_ptr: *mut *const $ty,
            out_len: *mut usize,
        ) -> MolrsStatus {
            ffi_try!({
                null_check!(out_ptr);
                null_check!(out_len);
                let store = lock_store();
                let bh = resolve_block!(store, &block);
                let col_key = resolve_col_key!(store, col_key_id);
                let result = store.inner.with_block(&bh, |b| {
                    let arr = b.$getter(&col_key).ok_or(MolrsStatus::KeyNotFound)?;
                    let slice = arr
                        .as_slice_memory_order()
                        .ok_or(MolrsStatus::NonContiguous)?;
                    Ok((slice.as_ptr(), slice.len()))
                });
                match result {
                    Ok(Ok((ptr, len))) => {
                        unsafe {
                            *out_ptr = ptr;
                            *out_len = len;
                        }
                        MolrsStatus::Ok
                    }
                    Ok(Err(status)) => {
                        error::set_last_error(format!(
                            "column '{}': {}",
                            col_key,
                            if status == MolrsStatus::KeyNotFound {
                                "not found"
                            } else {
                                "not contiguous"
                            }
                        ));
                        status
                    }
                    Err(e) => ffi_err_to_status(&e),
                }
            })
        }
    };
}

impl_col_ptr!(molrs_block_get_F, F, get_float, "float");
impl_col_ptr!(molrs_block_get_I, I, get_int, "int");
impl_col_ptr!(molrs_block_get_U, U, get_uint, "uint");

// ---------------------------------------------------------------------------
// Zero-copy WRITE
// ---------------------------------------------------------------------------

/// Macro to generate typed zero-copy mutable pointer functions.
///
/// Each generated function returns a mutable pointer into the column's
/// contiguous storage and bumps the block version.
///
/// # Generated C signatures
///
/// ```c
/// MolrsStatus molrs_block_get_F_mut(MolrsBlockHandle* block, uint32_t col_key_id,
///                                    molrs_float_t** out_ptr, size_t* out_len);
/// MolrsStatus molrs_block_get_I_mut(MolrsBlockHandle* block, uint32_t col_key_id,
///                                    molrs_int_t** out_ptr, size_t* out_len);
/// MolrsStatus molrs_block_get_U_mut(MolrsBlockHandle* block, uint32_t col_key_id,
///                                    molrs_uint_t** out_ptr, size_t* out_len);
/// ```
///
/// # Safety (all generated functions)
///
/// * `block` must point to a live, writable `MolrsBlockHandle`.  Its
///   `block_version` field is updated in place after the call.
/// * `out_ptr` and `out_len` must be valid, non-null, writable pointers.
/// * The returned data pointer is valid until another mutating call is
///   made to the same block or the frame is dropped.
/// * The caller should call [`molrs_block_col_commit`] after writing
///   (currently a no-op, but reserved for future validation).
macro_rules! impl_col_ptr_mut {
    ($fn_name:ident, $ty:ty, $getter_mut:ident, $dtype_name:expr) => {
        #[doc = concat!("Get a mutable pointer to a contiguous `", $dtype_name, "` column.")]
        ///
        /// On success, `*out_ptr` points to the first element, `*out_len`
        /// is set to the total element count, and the `block_version`
        /// inside `*block` is updated.
        ///
        /// After writing through the pointer, call
        /// [`molrs_block_col_commit`] to finalize changes (currently a
        /// no-op but reserved for future use).
        ///
        /// # Returns
        ///
        /// * `MolrsStatus::Ok` on success.
        /// * `MolrsStatus::NullPointer` if any pointer argument is null.
        /// * `MolrsStatus::KeyNotFound` if the column does not exist.
        /// * `MolrsStatus::NonContiguous` if the column is not contiguous.
        /// * `MolrsStatus::InvalidBlockHandle` if the block handle is stale.
        #[unsafe(no_mangle)]
        pub unsafe extern "C" fn $fn_name(
            block: *mut MolrsBlockHandle,
            col_key_id: u32,
            out_ptr: *mut *mut $ty,
            out_len: *mut usize,
        ) -> MolrsStatus {
            ffi_try!({
                null_check!(block);
                null_check!(out_ptr);
                null_check!(out_len);
                let c_handle = unsafe { &*block };
                let mut store = lock_store();
                let col_key = resolve_col_key!(store, col_key_id);
                let mut bh = resolve_block!(store, c_handle);
                let result = store.inner.with_block_mut(&mut bh, |b| {
                    let arr = b.$getter_mut(&col_key).ok_or(MolrsStatus::KeyNotFound)?;
                    let slice = arr
                        .as_slice_memory_order_mut()
                        .ok_or(MolrsStatus::NonContiguous)?;
                    Ok((slice.as_mut_ptr(), slice.len()))
                });
                match result {
                    Ok(Ok((ptr, len))) => {
                        let c_block = unsafe { &mut *block };
                        c_block.block_version = bh.version();
                        unsafe {
                            *out_ptr = ptr;
                            *out_len = len;
                        }
                        MolrsStatus::Ok
                    }
                    Ok(Err(status)) => {
                        let msg = match status {
                            MolrsStatus::KeyNotFound => "column not found",
                            MolrsStatus::NonContiguous => "column not contiguous",
                            _ => "unknown error",
                        };
                        error::set_last_error(msg);
                        status
                    }
                    Err(e) => ffi_err_to_status(&e),
                }
            })
        }
    };
}

impl_col_ptr_mut!(molrs_block_get_F_mut, F, get_float_mut, "float");
impl_col_ptr_mut!(molrs_block_get_I_mut, I, get_int_mut, "int");
impl_col_ptr_mut!(molrs_block_get_U_mut, U, get_uint_mut, "uint");

/// Finalize changes after writing through a mutable column pointer.
///
/// In the current implementation this is a no-op because the block
/// version is already bumped by the `_mut` accessor.  It exists for
/// API completeness and for future validation hooks.
///
/// # C signature
///
/// ```c
/// MolrsStatus molrs_block_col_commit(MolrsBlockHandle* block);
/// ```
///
/// # Arguments
///
/// * `block` -- Pointer to the block handle that was used with a
///   `_mut` accessor.
///
/// # Returns
///
/// * `MolrsStatus::Ok` always (unless `block` is null).
/// * `MolrsStatus::NullPointer` if `block` is null.
///
/// # Safety
///
/// `block` must point to a valid `MolrsBlockHandle` (the same one
/// passed to the corresponding `_mut` call).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn molrs_block_col_commit(block: *mut MolrsBlockHandle) -> MolrsStatus {
    ffi_try!({
        null_check!(block);
        // Version was already bumped in _mut. Nothing to do.
        MolrsStatus::Ok
    })
}

// ---------------------------------------------------------------------------
// Copy path
// ---------------------------------------------------------------------------

/// Macro to generate typed copy functions.
///
/// Each generated function copies column data into a caller-owned buffer.
/// This is the safest access pattern because the caller controls the
/// buffer lifetime.
///
/// # Generated C signatures
///
/// ```c
/// MolrsStatus molrs_block_copy_F(MolrsBlockHandle block, uint32_t col_key_id,
///                                 molrs_float_t* out_buf, size_t buf_len);
/// MolrsStatus molrs_block_copy_I(MolrsBlockHandle block, uint32_t col_key_id,
///                                 molrs_int_t* out_buf, size_t buf_len);
/// MolrsStatus molrs_block_copy_U(MolrsBlockHandle block, uint32_t col_key_id,
///                                 molrs_uint_t* out_buf, size_t buf_len);
/// ```
///
/// # Safety (all generated functions)
///
/// * `block` must be a live block handle.
/// * `out_buf` must point to a buffer of at least `buf_len` elements of
///   the corresponding type.
macro_rules! impl_col_copy {
    ($fn_name:ident, $ty:ty, $getter:ident, $dtype_name:expr) => {
        #[doc = concat!("Copy a `", $dtype_name, "` column into a caller-provided buffer.")]
        ///
        /// # Returns
        ///
        /// * `MolrsStatus::Ok` on success.
        /// * `MolrsStatus::NullPointer` if `out_buf` is null.
        /// * `MolrsStatus::KeyNotFound` if the column does not exist.
        /// * `MolrsStatus::NonContiguous` if the column is not contiguous.
        /// * `MolrsStatus::InvalidArgument` if `buf_len` is smaller than
        ///   the column length.
        /// * `MolrsStatus::InvalidBlockHandle` if `block` is stale.
        #[unsafe(no_mangle)]
        pub unsafe extern "C" fn $fn_name(
            block: MolrsBlockHandle,
            col_key_id: u32,
            out_buf: *mut $ty,
            buf_len: usize,
        ) -> MolrsStatus {
            ffi_try!({
                null_check!(out_buf);
                let store = lock_store();
                let bh = resolve_block!(store, &block);
                let col_key = resolve_col_key!(store, col_key_id);
                let result = store.inner.with_block(&bh, |b| {
                    let arr = b.$getter(&col_key).ok_or(MolrsStatus::KeyNotFound)?;
                    let slice = arr
                        .as_slice_memory_order()
                        .ok_or(MolrsStatus::NonContiguous)?;
                    if buf_len < slice.len() {
                        return Err(MolrsStatus::InvalidArgument);
                    }
                    Ok(slice.to_vec())
                });
                match result {
                    Ok(Ok(data)) => {
                        let out_slice =
                            unsafe { std::slice::from_raw_parts_mut(out_buf, data.len()) };
                        out_slice.copy_from_slice(&data);
                        MolrsStatus::Ok
                    }
                    Ok(Err(status)) => {
                        let msg = match status {
                            MolrsStatus::KeyNotFound => "column not found",
                            MolrsStatus::NonContiguous => "column not contiguous",
                            MolrsStatus::InvalidArgument => "buffer too small",
                            _ => "unknown error",
                        };
                        error::set_last_error(msg);
                        status
                    }
                    Err(e) => ffi_err_to_status(&e),
                }
            })
        }
    };
}

impl_col_copy!(molrs_block_copy_F, F, get_float, "float");
impl_col_copy!(molrs_block_copy_I, I, get_int, "int");
impl_col_copy!(molrs_block_copy_U, U, get_uint, "uint");

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
impl_block_insert!(molrs_block_set_U, U, "uint");

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
/// Returns `4` by default (`u32`) or `8` with the `u64` feature.
///
/// # C signature
///
/// ```c
/// size_t molrs_sizeof_U(void);
/// ```
#[unsafe(no_mangle)]
pub extern "C" fn molrs_sizeof_U() -> usize {
    std::mem::size_of::<U>()
}
