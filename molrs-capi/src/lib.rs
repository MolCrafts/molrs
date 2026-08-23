#![allow(clippy::missing_safety_doc)]
//! C API for molrs -- stable ABI for C and C++ integration.
//!
//! This crate exposes a flat, handle-based C API for creating and
//! manipulating molecular simulation data (frames, blocks, simulation
//! boxes, and force fields).
//!
//! # Design
//!
//! * **Handle-based** -- all objects live in a global, mutex-protected
//!   store.  C callers receive opaque `MolrsFrameHandle`,
//!   `MolrsBlockHandle`, `MolrsBoxHandle`, or `MolrsForceFieldHandle`
//!   values (plain `repr(C)` structs that fit in two machine words).
//! * **Status codes** -- every function returns [`MolrsStatus`].  On
//!   failure the last error message is stored in a thread-local buffer
//!   and retrieved via [`molrs_last_error`].
//! * **Panic safety** -- every `extern "C"` body is wrapped in
//!   `catch_unwind` so Rust panics never unwind across the FFI boundary.
//!
//! # Float / integer precision
//!
//! | C typedef        | Default  | Wide (`f64` / `i64` / `u64` features) |
//! |------------------|----------|---------------------------------------|
//! | `molrs_float_t`  | `float`  | `double`                              |
//! | `molrs_int_t`    | `int32_t`| `int64_t`                             |
//! | `molrs_uint_t`   | `uint32_t`| `uint64_t`                           |
//!
//! # Typical C usage
//!
//! ```c
//! #include "molrs_capi.h"
//!
//! molrs_init();
//!
//! MolrsFrameHandle frame;
//! molrs_frame_new(&frame);
//!
//! // ... populate frame with blocks, columns, simbox ...
//!
//! molrs_frame_drop(frame);
//! molrs_shutdown();
//! ```
//!
//! # Safety
//!
//! All `extern "C"` functions are unsafe because they accept raw pointers
//! from C callers.  See per-function documentation for pointer validity
//! and lifetime requirements.

// All public unsafe functions now have `# Safety` sections in their rustdoc.

pub mod block;
pub mod error;
pub mod forcefield;
pub mod frame;
pub mod handle;
pub mod schema;
pub mod simbox;
mod store;

use std::ffi::{CStr, CString, c_char};

pub use error::{MolrsDType, MolrsStatus};
pub use frame::{MolrsMetaType, MolrsMetaValue};
pub use handle::{MolrsBlockHandle, MolrsBoxHandle, MolrsForceFieldHandle, MolrsFrameHandle};

use store::lock_store;

/// Primary floating-point scalar (`molrs_float_t` in the C header) — always `f64`.
pub type F = f64;

// ---------------------------------------------------------------------------
// Internal helper: catch panics at the FFI boundary
// ---------------------------------------------------------------------------

macro_rules! ffi_try {
    ($body:expr) => {{
        match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| $body)) {
            Ok(status) => status,
            Err(_) => {
                error::set_last_error("internal panic in FFI call");
                MolrsStatus::InternalError
            }
        }
    }};
}
pub(crate) use ffi_try;

/// Null-pointer check helper. Returns `NullPointer` status if null.
macro_rules! null_check {
    ($ptr:expr) => {
        if $ptr.is_null() {
            error::set_last_error(concat!("null pointer: ", stringify!($ptr)));
            return MolrsStatus::NullPointer;
        }
    };
}
pub(crate) use null_check;

// ---------------------------------------------------------------------------
// Lifecycle & Utilities
// ---------------------------------------------------------------------------

/// Initialize the global object store.
///
/// Safe to call multiple times -- the second and subsequent calls are
/// no-ops.  Must be called before any other `molrs_*` function.
///
/// # C signature
///
/// ```c
/// void molrs_init(void);
/// ```
///
/// # Safety
///
/// No pointer arguments.  This function only initialises the internal
/// mutex-protected singleton and cannot violate memory safety.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn molrs_init() {
    // Force lazy initialization of the global store.
    drop(lock_store());
}

/// C API version of this library — the handshake constant.
///
/// Incremented on any breaking change to a function signature, a handle's
/// semantics, or a `repr(C)` type in this header (mirrors molrs-cxxapi's
/// `CXX_API_VERSION`). A dlopen consumer compares
/// [`molrs_c_api_version`]`()` against the `MOLRS_C_API_VERSION` its header
/// was compiled with before calling anything else.
pub const MOLRS_C_API_VERSION: u32 = 1;

/// Report the C API version compiled into this library.
///
/// # C signature
///
/// ```c
/// uint32_t molrs_c_api_version(void);
/// ```
///
/// # Safety
///
/// No pointer arguments; returns a constant.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn molrs_c_api_version() -> u32 {
    MOLRS_C_API_VERSION
}

/// Report the `molcrafts-molrs` core version compiled into this library.
///
/// Returns a pointer to a static null-terminated UTF-8 string, e.g.
/// `"0.14.0"`. Informational — the compatibility gate is
/// [`molrs_c_api_version`]; this identifies the exact molrs release for
/// diagnostics.
///
/// # C signature
///
/// ```c
/// const char* molrs_version(void);
/// ```
///
/// # Safety
///
/// The caller must not write through the returned pointer. The pointer is
/// valid for the lifetime of the process.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn molrs_version() -> *const c_char {
    static VERSION: std::sync::OnceLock<CString> = std::sync::OnceLock::new();
    VERSION
        .get_or_init(|| CString::new(molrs::VERSION).expect("version has no NUL"))
        .as_ptr()
}

/// Destroy all objects and reset the global store.
///
/// Every handle obtained before this call becomes invalid.
/// It is safe (but unnecessary) to call [`molrs_init`] again afterwards.
///
/// # C signature
///
/// ```c
/// void molrs_shutdown(void);
/// ```
///
/// # Safety
///
/// After this call, any previously obtained handle (frame, block,
/// simbox, forcefield) is dangling.  Using a stale handle will return
/// `MolrsStatus::Invalid*Handle`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn molrs_shutdown() {
    let mut store = lock_store();
    store.clear();
}

/// Retrieve the last error message for the calling thread.
///
/// Returns a pointer to a null-terminated UTF-8 string describing the
/// most recent error.  If no error has occurred, an empty string (`""`)
/// is returned (never a null pointer).
///
/// # C signature
///
/// ```c
/// const char* molrs_last_error(void);
/// ```
///
/// # Lifetime
///
/// The returned pointer is valid until the **next** `molrs_*` call that
/// sets an error **on the same thread**.  Copy the string immediately if
/// you need to keep it.
///
/// # Safety
///
/// The caller must not write through the returned pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn molrs_last_error() -> *const c_char {
    error::last_error_ptr()
}

/// Intern a key string, returning a compact integer identifier.
///
/// Interning avoids repeated string-to-key lookups when the same block
/// or column name is used across many calls.  The same input string
/// always returns the same `key_id` (idempotent).
///
/// # C signature
///
/// ```c
/// MolrsStatus molrs_intern_key(const char* key, uint32_t* out_key_id);
/// ```
///
/// # Arguments
///
/// * `key` -- Null-terminated UTF-8 string (e.g. `"atoms"`, `"x"`).
/// * `out_key_id` -- On success, receives the interned key identifier.
///
/// # Returns
///
/// * `MolrsStatus::Ok` on success.
/// * `MolrsStatus::NullPointer` if `key` or `out_key_id` is null.
/// * `MolrsStatus::Utf8Error` if `key` is not valid UTF-8.
///
/// # Safety
///
/// * `key` must be a valid, null-terminated C string.
/// * `out_key_id` must point to a writable `uint32_t`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn molrs_intern_key(key: *const c_char, out_key_id: *mut u32) -> MolrsStatus {
    ffi_try!({
        null_check!(key);
        null_check!(out_key_id);
        let c_str = unsafe { CStr::from_ptr(key) };
        let key_str = match c_str.to_str() {
            Ok(s) => s,
            Err(_) => {
                error::set_last_error("key is not valid UTF-8");
                return MolrsStatus::Utf8Error;
            }
        };
        let mut store = lock_store();
        let id = store.intern(key_str);
        unsafe { *out_key_id = id };
        MolrsStatus::Ok
    })
}

/// Look up the string name for a previously interned `key_id`.
///
/// # C signature
///
/// ```c
/// const char* molrs_key_name(uint32_t key_id);
/// ```
///
/// # Arguments
///
/// * `key_id` -- An identifier obtained from [`molrs_intern_key`].
///
/// # Returns
///
/// A pointer to a null-terminated UTF-8 string, or `NULL` if `key_id`
/// is unknown.
///
/// # Lifetime
///
/// The returned pointer is valid until [`molrs_shutdown`] is called.
///
/// # Safety
///
/// The caller must not write through or free the returned pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn molrs_key_name(key_id: u32) -> *const c_char {
    let store = lock_store();
    match store.interned_keys.get(key_id as usize) {
        Some(cstr) => cstr.as_ptr(),
        None => std::ptr::null(),
    }
}

/// Free a string that was allocated by the C API.
///
/// Several functions (e.g. [`molrs_ff_to_json`](crate::forcefield::molrs_ff_to_json),
/// [`molrs_frame_read_meta`](crate::frame::molrs_frame_read_meta))
/// return heap-allocated C strings that the caller owns.  Pass those
/// pointers to this function when they are no longer needed.
///
/// Passing `NULL` is a safe no-op.
///
/// # C signature
///
/// ```c
/// void molrs_free_string(char* s);
/// ```
///
/// # Arguments
///
/// * `s` -- A string pointer previously returned by a `molrs_*` function,
///   or `NULL`.
///
/// # Safety
///
/// * `s` must have been allocated by this library (via `CString::into_raw`).
/// * `s` must not be used after this call.
/// * Double-free is undefined behaviour.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn molrs_free_string(s: *mut c_char) {
    if !s.is_null() {
        unsafe {
            drop(CString::from_raw(s));
        }
    }
}
