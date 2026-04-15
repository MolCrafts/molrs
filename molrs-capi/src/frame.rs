//! `extern "C"` functions for Frame lifecycle and access.
//!
//! A **Frame** is the top-level data container in molrs.  It maps string
//! keys (e.g. `"atoms"`, `"bonds"`, `"angles"`) to [`Block`]s, carries
//! an optional [`SimBox`](molrs::region::simbox::SimBox) for periodic
//! boundary conditions, and stores arbitrary key-value metadata.
//!
//! # Typical column layout
//!
//! | Block key  | Column  | C type           | Description                     |
//! |------------|---------|------------------|---------------------------------|
//! | `"atoms"`  | `"symbol"` | string        | Element symbol ("C", "N", ...)  |
//! | `"atoms"`  | `"x"`   | `molrs_float_t` | Cartesian x coordinate in Angstrom     |
//! | `"atoms"`  | `"y"`   | `molrs_float_t` | Cartesian y coordinate in Angstrom     |
//! | `"atoms"`  | `"z"`   | `molrs_float_t` | Cartesian z coordinate in Angstrom     |
//! | `"atoms"`  | `"mass"`| `molrs_float_t` | Atomic mass in amu              |
//! | `"bonds"`  | `"i"`   | `molrs_uint_t`  | First atom index (0-based)      |
//! | `"bonds"`  | `"j"`   | `molrs_uint_t`  | Second atom index (0-based)     |
//! | `"bonds"`  | `"order"`| `molrs_float_t`| Bond order (1.0, 1.5, 2.0, ...) |

use std::ffi::{CStr, CString, c_char};

use molrs::block::Block;

use crate::error::{self, MolrsStatus, ffi_err_to_status};
use crate::handle::{
    MolrsBlockHandle, MolrsFrameHandle, MolrsSimBoxHandle, block_handle_to_c, frame_id_to_handle,
    handle_to_frame_id, handle_to_simbox_key, simbox_key_to_handle,
};
use crate::store::lock_store;
use crate::{ffi_try, null_check};

/// Parse a SMILES string and create a frame containing atoms and bonds.
///
/// The frame will contain an `"atoms"` block with `"symbol"`, `"x"`,
/// `"y"`, `"z"` columns and a `"bonds"` block with `"i"`, `"j"`,
/// `"order"` columns.  Initial coordinates are 2D layout coordinates
/// (not optimised 3D); use `embed` for 3D embedding.
///
/// # C signature
///
/// ```c
/// MolrsStatus molrs_frame_from_smiles(const char* smiles,
///                                      MolrsFrameHandle* out);
/// ```
///
/// # Arguments
///
/// * `smiles` -- Null-terminated SMILES string (e.g. `"CCO"`).
/// * `out` -- On success, receives the new frame handle.
///
/// # Returns
///
/// * `MolrsStatus::Ok` on success.
/// * `MolrsStatus::NullPointer` if either pointer is null.
/// * `MolrsStatus::Utf8Error` if `smiles` is not valid UTF-8.
/// * `MolrsStatus::ParseError` if the SMILES string is malformed.
///
/// # Safety
///
/// * `smiles` must be a valid, null-terminated UTF-8 C string.
/// * `out` must point to a writable `MolrsFrameHandle`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn molrs_frame_from_smiles(
    smiles: *const c_char,
    out: *mut MolrsFrameHandle,
) -> MolrsStatus {
    ffi_try!({
        null_check!(smiles);
        null_check!(out);
        let c_str = unsafe { CStr::from_ptr(smiles) };
        let smiles_str = match c_str.to_str() {
            Ok(s) => s,
            Err(_) => {
                error::set_last_error("SMILES string is not valid UTF-8");
                return MolrsStatus::Utf8Error;
            }
        };

        let ir = match molrs_smiles::parse_smiles(smiles_str) {
            Ok(ir) => ir,
            Err(e) => {
                error::set_last_error(format!("{e}"));
                return MolrsStatus::ParseError;
            }
        };
        let mol = match molrs_smiles::to_atomistic(&ir) {
            Ok(m) => m,
            Err(e) => {
                error::set_last_error(format!("{e}"));
                return MolrsStatus::ParseError;
            }
        };
        let frame = mol.to_frame();

        let mut store = lock_store();
        let id = store.inner.frame_new();
        if let Err(e) = store.inner.set_frame(id, frame) {
            return ffi_err_to_status(&e);
        }
        unsafe { *out = frame_id_to_handle(id) };
        MolrsStatus::Ok
    })
}

/// Create a new, empty frame with no blocks, no SimBox, and no metadata.
///
/// # C signature
///
/// ```c
/// MolrsStatus molrs_frame_new(MolrsFrameHandle* out);
/// ```
///
/// # Arguments
///
/// * `out` -- On success, receives the new frame handle.
///
/// # Returns
///
/// * `MolrsStatus::Ok` on success.
/// * `MolrsStatus::NullPointer` if `out` is null.
///
/// # Safety
///
/// `out` must point to a writable `MolrsFrameHandle`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn molrs_frame_new(out: *mut MolrsFrameHandle) -> MolrsStatus {
    ffi_try!({
        null_check!(out);
        let mut store = lock_store();
        let id = store.inner.frame_new();
        unsafe { *out = frame_id_to_handle(id) };
        MolrsStatus::Ok
    })
}

/// Drop a frame and invalidate all handles that reference it.
///
/// Any [`MolrsBlockHandle`] derived from this frame becomes invalid
/// after this call.
///
/// # C signature
///
/// ```c
/// MolrsStatus molrs_frame_drop(MolrsFrameHandle handle);
/// ```
///
/// # Arguments
///
/// * `handle` -- The frame to destroy.
///
/// # Returns
///
/// * `MolrsStatus::Ok` on success.
/// * `MolrsStatus::InvalidFrameHandle` if `handle` is stale or unknown.
///
/// # Safety
///
/// The caller must not use `handle` or any block handle derived from it
/// after this call.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn molrs_frame_drop(handle: MolrsFrameHandle) -> MolrsStatus {
    ffi_try!({
        let mut store = lock_store();
        let id = handle_to_frame_id(handle);
        match store.inner.frame_drop(id) {
            Ok(()) => MolrsStatus::Ok,
            Err(e) => ffi_err_to_status(&e),
        }
    })
}

/// Deep-clone a frame, returning a new independent handle.
///
/// The cloned frame is a complete copy of all blocks, columns, SimBox,
/// and metadata.  Modifications to the clone do not affect the original.
///
/// # C signature
///
/// ```c
/// MolrsStatus molrs_frame_clone(MolrsFrameHandle src,
///                                MolrsFrameHandle* out);
/// ```
///
/// # Arguments
///
/// * `src` -- The source frame to clone.
/// * `out` -- On success, receives the handle to the new frame.
///
/// # Returns
///
/// * `MolrsStatus::Ok` on success.
/// * `MolrsStatus::NullPointer` if `out` is null.
/// * `MolrsStatus::InvalidFrameHandle` if `src` is stale or unknown.
///
/// # Safety
///
/// `out` must point to a writable `MolrsFrameHandle`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn molrs_frame_clone(
    src: MolrsFrameHandle,
    out: *mut MolrsFrameHandle,
) -> MolrsStatus {
    ffi_try!({
        null_check!(out);
        let mut store = lock_store();
        let src_id = handle_to_frame_id(src);
        let cloned = match store.inner.clone_frame(src_id) {
            Ok(f) => f,
            Err(e) => return ffi_err_to_status(&e),
        };
        let new_id = store.inner.frame_new();
        if let Err(e) = store.inner.set_frame(new_id, cloned) {
            return ffi_err_to_status(&e);
        }
        unsafe { *out = frame_id_to_handle(new_id) };
        MolrsStatus::Ok
    })
}

/// Insert an empty block into a frame under the given interned key.
///
/// If a block with the same key already exists it is replaced.
///
/// # C signature
///
/// ```c
/// MolrsStatus molrs_frame_set_block(MolrsFrameHandle frame,
///                                    uint32_t key_id,
///                                    size_t   nrows);
/// ```
///
/// # Arguments
///
/// * `frame` -- Target frame.
/// * `key_id` -- Interned key (see [`molrs_intern_key`](crate::molrs_intern_key)).
///   Conventional keys: `"atoms"`, `"bonds"`, `"angles"`, `"dihedrals"`.
/// * `_nrows` -- Reserved for future use; currently ignored.
///
/// # Returns
///
/// * `MolrsStatus::Ok` on success.
/// * `MolrsStatus::KeyNotFound` if `key_id` has not been interned.
/// * `MolrsStatus::InvalidFrameHandle` if `frame` is stale.
///
/// # Safety
///
/// `frame` must be a live frame handle.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn molrs_frame_set_block(
    frame: MolrsFrameHandle,
    key_id: u32,
    _nrows: usize,
) -> MolrsStatus {
    ffi_try!({
        let mut store = lock_store();
        let frame_id = handle_to_frame_id(frame);
        let key_str = match store.key_str(key_id) {
            Some(s) => s.to_owned(),
            None => {
                error::set_last_error(format!("unknown key_id {key_id}"));
                return MolrsStatus::KeyNotFound;
            }
        };
        let block = Block::new();
        match store.inner.set_block(frame_id, &key_str, block) {
            Ok(()) => MolrsStatus::Ok,
            Err(e) => ffi_err_to_status(&e),
        }
    })
}

/// Remove a block from a frame by its interned key.
///
/// All [`MolrsBlockHandle`]s that reference this block become invalid.
///
/// # C signature
///
/// ```c
/// MolrsStatus molrs_frame_remove_block(MolrsFrameHandle frame,
///                                       uint32_t key_id);
/// ```
///
/// # Arguments
///
/// * `frame` -- Target frame.
/// * `key_id` -- Interned key of the block to remove.
///
/// # Returns
///
/// * `MolrsStatus::Ok` on success.
/// * `MolrsStatus::KeyNotFound` if `key_id` was not interned or the
///   block does not exist.
/// * `MolrsStatus::InvalidFrameHandle` if `frame` is stale.
///
/// # Safety
///
/// `frame` must be a live frame handle.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn molrs_frame_remove_block(
    frame: MolrsFrameHandle,
    key_id: u32,
) -> MolrsStatus {
    ffi_try!({
        let mut store = lock_store();
        let frame_id = handle_to_frame_id(frame);
        let key_str = match store.key_str(key_id) {
            Some(s) => s.to_owned(),
            None => {
                error::set_last_error(format!("unknown key_id {key_id}"));
                return MolrsStatus::KeyNotFound;
            }
        };
        match store.inner.remove_block(frame_id, &key_str) {
            Ok(()) => MolrsStatus::Ok,
            Err(e) => ffi_err_to_status(&e),
        }
    })
}

/// Obtain a block handle for a named block inside a frame.
///
/// The returned handle can be used with `molrs_block_*` functions to
/// inspect and modify individual columns.
///
/// # C signature
///
/// ```c
/// MolrsStatus molrs_frame_get_block(MolrsFrameHandle frame,
///                                    uint32_t key_id,
///                                    MolrsBlockHandle* out);
/// ```
///
/// # Arguments
///
/// * `frame` -- The owning frame.
/// * `key_id` -- Interned key of the desired block.
/// * `out` -- On success, receives the block handle.
///
/// # Returns
///
/// * `MolrsStatus::Ok` on success.
/// * `MolrsStatus::NullPointer` if `out` is null.
/// * `MolrsStatus::KeyNotFound` if `key_id` was not interned or no
///   block with that key exists in the frame.
/// * `MolrsStatus::InvalidFrameHandle` if `frame` is stale.
///
/// # Safety
///
/// * `frame` must be a live frame handle.
/// * `out` must point to a writable `MolrsBlockHandle`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn molrs_frame_get_block(
    frame: MolrsFrameHandle,
    key_id: u32,
    out: *mut MolrsBlockHandle,
) -> MolrsStatus {
    ffi_try!({
        null_check!(out);
        let store = lock_store();
        let frame_id = handle_to_frame_id(frame);
        let key_str = match store.key_str(key_id) {
            Some(s) => s.to_owned(),
            None => {
                error::set_last_error(format!("unknown key_id {key_id}"));
                return MolrsStatus::KeyNotFound;
            }
        };
        let bh = match store.inner.get_block(frame_id, &key_str) {
            Ok(h) => h,
            Err(e) => return ffi_err_to_status(&e),
        };
        let c_handle = match block_handle_to_c(&bh, &store.key_to_id) {
            Some(h) => h,
            None => {
                error::set_last_error("failed to intern block key");
                return MolrsStatus::InternalError;
            }
        };
        unsafe { *out = c_handle };
        MolrsStatus::Ok
    })
}

/// Associate a SimBox with a frame.
///
/// The SimBox is cloned from the global SimBox store into the frame.
/// Changes to the original SimBox handle after this call do not affect
/// the frame's copy.
///
/// # C signature
///
/// ```c
/// MolrsStatus molrs_frame_set_simbox(MolrsFrameHandle frame,
///                                     MolrsSimBoxHandle simbox);
/// ```
///
/// # Arguments
///
/// * `frame` -- Target frame.
/// * `simbox` -- A live SimBox handle to clone into the frame.
///
/// # Returns
///
/// * `MolrsStatus::Ok` on success.
/// * `MolrsStatus::InvalidFrameHandle` if `frame` is stale.
/// * `MolrsStatus::InvalidSimBoxHandle` if `simbox` is stale.
///
/// # Safety
///
/// Both `frame` and `simbox` must be live handles.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn molrs_frame_set_simbox(
    frame: MolrsFrameHandle,
    simbox: MolrsSimBoxHandle,
) -> MolrsStatus {
    ffi_try!({
        let mut store = lock_store();
        let frame_id = handle_to_frame_id(frame);
        let sb_key = handle_to_simbox_key(simbox);
        let sb = match store.simboxes.get(sb_key) {
            Some(sb) => sb.clone(),
            None => {
                error::set_last_error("invalid simbox handle");
                return MolrsStatus::InvalidSimBoxHandle;
            }
        };
        match store.inner.set_frame_simbox(frame_id, Some(sb)) {
            Ok(()) => MolrsStatus::Ok,
            Err(e) => ffi_err_to_status(&e),
        }
    })
}

/// Remove the SimBox from a frame, leaving it with no periodic cell.
///
/// # C signature
///
/// ```c
/// MolrsStatus molrs_frame_clear_simbox(MolrsFrameHandle frame);
/// ```
///
/// # Arguments
///
/// * `frame` -- Target frame.
///
/// # Returns
///
/// * `MolrsStatus::Ok` on success (including if the frame had no SimBox).
/// * `MolrsStatus::InvalidFrameHandle` if `frame` is stale.
///
/// # Safety
///
/// `frame` must be a live frame handle.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn molrs_frame_clear_simbox(frame: MolrsFrameHandle) -> MolrsStatus {
    ffi_try!({
        let mut store = lock_store();
        let frame_id = handle_to_frame_id(frame);
        match store.inner.set_frame_simbox(frame_id, None) {
            Ok(()) => MolrsStatus::Ok,
            Err(e) => ffi_err_to_status(&e),
        }
    })
}

/// Extract the SimBox from a frame, cloning it into the SimBox store.
///
/// A new SimBox handle is created each time this function is called.
/// The caller is responsible for freeing it with
/// [`molrs_simbox_drop`](crate::simbox::molrs_simbox_drop).
///
/// # C signature
///
/// ```c
/// MolrsStatus molrs_frame_get_simbox(MolrsFrameHandle frame,
///                                     MolrsSimBoxHandle* out);
/// ```
///
/// # Arguments
///
/// * `frame` -- Source frame.
/// * `out` -- On success, receives a new SimBox handle.
///
/// # Returns
///
/// * `MolrsStatus::Ok` on success.
/// * `MolrsStatus::NullPointer` if `out` is null.
/// * `MolrsStatus::KeyNotFound` if the frame has no SimBox.
/// * `MolrsStatus::InvalidFrameHandle` if `frame` is stale.
///
/// # Safety
///
/// * `frame` must be a live frame handle.
/// * `out` must point to a writable `MolrsSimBoxHandle`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn molrs_frame_get_simbox(
    frame: MolrsFrameHandle,
    out: *mut MolrsSimBoxHandle,
) -> MolrsStatus {
    ffi_try!({
        null_check!(out);
        let mut store = lock_store();
        let frame_id = handle_to_frame_id(frame);
        let sb_clone = match store.inner.with_frame_simbox(frame_id, |opt| opt.cloned()) {
            Ok(Some(sb)) => sb,
            Ok(None) => {
                error::set_last_error("frame has no simbox");
                return MolrsStatus::KeyNotFound;
            }
            Err(e) => return ffi_err_to_status(&e),
        };
        let key = store.simboxes.insert(sb_clone);
        unsafe { *out = simbox_key_to_handle(key) };
        MolrsStatus::Ok
    })
}

/// Set a metadata key-value pair on a frame.
///
/// Both key and value are stored as UTF-8 strings.  If the key already
/// exists, its value is overwritten.
///
/// # C signature
///
/// ```c
/// MolrsStatus molrs_frame_set_meta(MolrsFrameHandle frame,
///                                   const char* key,
///                                   const char* value);
/// ```
///
/// # Arguments
///
/// * `frame` -- Target frame.
/// * `key` -- Null-terminated UTF-8 metadata key.
/// * `value` -- Null-terminated UTF-8 metadata value.
///
/// # Returns
///
/// * `MolrsStatus::Ok` on success.
/// * `MolrsStatus::NullPointer` if `key` or `value` is null.
/// * `MolrsStatus::Utf8Error` if `key` or `value` is not valid UTF-8.
/// * `MolrsStatus::InvalidFrameHandle` if `frame` is stale.
///
/// # Safety
///
/// * `frame` must be a live frame handle.
/// * `key` and `value` must be valid, null-terminated C strings.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn molrs_frame_set_meta(
    frame: MolrsFrameHandle,
    key: *const c_char,
    value: *const c_char,
) -> MolrsStatus {
    ffi_try!({
        null_check!(key);
        null_check!(value);
        let key_str = match unsafe { CStr::from_ptr(key) }.to_str() {
            Ok(s) => s.to_owned(),
            Err(_) => {
                error::set_last_error("meta key is not valid UTF-8");
                return MolrsStatus::Utf8Error;
            }
        };
        let val_str = match unsafe { CStr::from_ptr(value) }.to_str() {
            Ok(s) => s.to_owned(),
            Err(_) => {
                error::set_last_error("meta value is not valid UTF-8");
                return MolrsStatus::Utf8Error;
            }
        };
        let mut store = lock_store();
        let frame_id = handle_to_frame_id(frame);
        match store.inner.with_frame_mut(frame_id, |f| {
            f.meta.insert(key_str, val_str);
        }) {
            Ok(()) => MolrsStatus::Ok,
            Err(e) => ffi_err_to_status(&e),
        }
    })
}

/// Retrieve a metadata value from a frame by key.
///
/// The returned string is heap-allocated and must be freed by the
/// caller with [`molrs_free_string`](crate::molrs_free_string).
///
/// # C signature
///
/// ```c
/// MolrsStatus molrs_frame_get_meta(MolrsFrameHandle frame,
///                                   const char* key,
///                                   char** out);
/// ```
///
/// # Arguments
///
/// * `frame` -- Source frame.
/// * `key` -- Null-terminated UTF-8 metadata key.
/// * `out` -- On success, receives a pointer to a heap-allocated,
///   null-terminated copy of the value.
///
/// # Returns
///
/// * `MolrsStatus::Ok` on success.
/// * `MolrsStatus::NullPointer` if `key` or `out` is null.
/// * `MolrsStatus::Utf8Error` if `key` is not valid UTF-8.
/// * `MolrsStatus::KeyNotFound` if the metadata key does not exist.
/// * `MolrsStatus::InvalidFrameHandle` if `frame` is stale.
///
/// # Safety
///
/// * `frame` must be a live frame handle.
/// * `key` must be a valid, null-terminated C string.
/// * `out` must point to a writable `char*`.
/// * The caller owns the returned string and must free it with
///   [`molrs_free_string`](crate::molrs_free_string).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn molrs_frame_get_meta(
    frame: MolrsFrameHandle,
    key: *const c_char,
    out: *mut *mut c_char,
) -> MolrsStatus {
    ffi_try!({
        null_check!(key);
        null_check!(out);
        let key_str = match unsafe { CStr::from_ptr(key) }.to_str() {
            Ok(s) => s,
            Err(_) => {
                error::set_last_error("meta key is not valid UTF-8");
                return MolrsStatus::Utf8Error;
            }
        };
        let store = lock_store();
        let frame_id = handle_to_frame_id(frame);
        let result = store.inner.with_frame_simbox(frame_id, |_| {
            // We need frame access, not simbox. Use clone_frame instead.
        });
        // Actually we need to read metadata. Use clone_frame to get the frame.
        let frame_clone = match store.inner.clone_frame(frame_id) {
            Ok(f) => f,
            Err(e) => return ffi_err_to_status(&e),
        };
        drop(result);
        match frame_clone.meta.get(key_str) {
            Some(val) => {
                let c_val = CString::new(val.as_str()).unwrap_or_default();
                unsafe { *out = c_val.into_raw() };
                MolrsStatus::Ok
            }
            None => {
                error::set_last_error(format!("meta key '{}' not found", key_str));
                MolrsStatus::KeyNotFound
            }
        }
    })
}
