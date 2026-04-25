//! `extern "C"` functions for ForceField operations.
//!
//! A **ForceField** is a collection of interaction styles (atom, bond,
//! angle, dihedral, improper, pair, kspace) and per-type parameter sets.
//! This module exposes functions to build a force field programmatically
//! from C, serialize/deserialize it as JSON, and query its contents.
//!
//! # Typical workflow
//!
//! ```c
//! MolrsForceFieldHandle ff;
//! molrs_ff_new("my_ff", &ff);
//!
//! // Define a pair style with a global cutoff parameter
//! const char* pk[] = {"cutoff"};
//! double      pv[] = {12.0};
//! molrs_ff_def_pairstyle(ff, "lj/cut", pk, pv, 1);
//!
//! // Define a type under that style
//! const char* tk[] = {"epsilon", "sigma"};
//! double      tv[] = {0.1553, 3.166};
//! molrs_ff_def_type(ff, "pair", "lj/cut", "OW-OW", tk, tv, 2);
//!
//! // Serialize to JSON for storage
//! char*  json;
//! size_t json_len;
//! molrs_ff_to_json(ff, &json, &json_len);
//! // ... write json to file ...
//! molrs_free_string(json);
//!
//! molrs_ff_drop(ff);
//! ```
//!
//! # Parameter values
//!
//! All parameter values are `double` (`f64`) regardless of the float
//! precision feature, because force field parameters require full
//! precision for numerical stability.

use std::ffi::{CStr, CString, c_char};

use molrs_ff::ForceField;

use crate::error::{self, MolrsStatus};
use crate::handle::{MolrsForceFieldHandle, ff_key_to_handle, handle_to_ff_key};
use crate::store::lock_store;
use crate::{ffi_try, null_check};

// ---------------------------------------------------------------------------
// Lifecycle
// ---------------------------------------------------------------------------

/// Create a new, empty ForceField with the given name.
///
/// # C signature
///
/// ```c
/// MolrsStatus molrs_ff_new(const char* name,
///                           MolrsForceFieldHandle* out);
/// ```
///
/// # Arguments
///
/// * `name` -- Null-terminated UTF-8 force field name (e.g. `"OPLS"`,
///   `"MMFF94"`).
/// * `out` -- On success, receives the new ForceField handle.
///
/// # Returns
///
/// * `MolrsStatus::Ok` on success.
/// * `MolrsStatus::NullPointer` if `name` or `out` is null.
/// * `MolrsStatus::Utf8Error` if `name` is not valid UTF-8.
///
/// # Safety
///
/// * `name` must be a valid, null-terminated C string.
/// * `out` must point to a writable `MolrsForceFieldHandle`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn molrs_ff_new(
    name: *const c_char,
    out: *mut MolrsForceFieldHandle,
) -> MolrsStatus {
    ffi_try!({
        null_check!(name);
        null_check!(out);
        let name_str = match unsafe { CStr::from_ptr(name) }.to_str() {
            Ok(s) => s,
            Err(_) => {
                error::set_last_error("name is not valid UTF-8");
                return MolrsStatus::Utf8Error;
            }
        };
        let ff = ForceField::new(name_str);
        let mut store = lock_store();
        let key = store.forcefields.insert(ff);
        unsafe { *out = ff_key_to_handle(key) };
        MolrsStatus::Ok
    })
}

/// Destroy a ForceField and invalidate its handle.
///
/// # C signature
///
/// ```c
/// MolrsStatus molrs_ff_drop(MolrsForceFieldHandle handle);
/// ```
///
/// # Arguments
///
/// * `handle` -- The ForceField to destroy.
///
/// # Returns
///
/// * `MolrsStatus::Ok` on success.
/// * `MolrsStatus::InvalidForceFieldHandle` if `handle` is stale.
///
/// # Safety
///
/// The caller must not use `handle` after this call.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn molrs_ff_drop(handle: MolrsForceFieldHandle) -> MolrsStatus {
    ffi_try!({
        let mut store = lock_store();
        let key = handle_to_ff_key(handle);
        match store.forcefields.remove(key) {
            Some(_) => MolrsStatus::Ok,
            None => {
                error::set_last_error("invalid forcefield handle");
                MolrsStatus::InvalidForceFieldHandle
            }
        }
    })
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Extract a ForceField ref from the store.
macro_rules! get_ff {
    ($store:expr, $handle:expr) => {
        match $store.forcefields.get(handle_to_ff_key($handle)) {
            Some(ff) => ff,
            None => {
                error::set_last_error("invalid forcefield handle");
                return MolrsStatus::InvalidForceFieldHandle;
            }
        }
    };
}

macro_rules! get_ff_mut {
    ($store:expr, $handle:expr) => {
        match $store.forcefields.get_mut(handle_to_ff_key($handle)) {
            Some(ff) => ff,
            None => {
                error::set_last_error("invalid forcefield handle");
                return MolrsStatus::InvalidForceFieldHandle;
            }
        }
    };
}

/// Parse C string arrays into `Vec<(&str, f64)>` params.
unsafe fn parse_params<'a>(
    param_keys: *const *const c_char,
    param_values: *const f64,
    n_params: usize,
) -> Result<Vec<(&'a str, f64)>, MolrsStatus> {
    if n_params == 0 {
        return Ok(Vec::new());
    }
    if param_keys.is_null() || param_values.is_null() {
        error::set_last_error("null param_keys or param_values");
        return Err(MolrsStatus::NullPointer);
    }
    let keys_slice = unsafe { std::slice::from_raw_parts(param_keys, n_params) };
    let vals_slice = unsafe { std::slice::from_raw_parts(param_values, n_params) };
    let mut params = Vec::with_capacity(n_params);
    for i in 0..n_params {
        if keys_slice[i].is_null() {
            error::set_last_error(format!("param_keys[{i}] is null"));
            return Err(MolrsStatus::NullPointer);
        }
        let key = match unsafe { CStr::from_ptr(keys_slice[i]) }.to_str() {
            Ok(s) => s,
            Err(_) => {
                error::set_last_error(format!("param_keys[{i}] is not valid UTF-8"));
                return Err(MolrsStatus::Utf8Error);
            }
        };
        params.push((key, vals_slice[i]));
    }
    Ok(params)
}

// ---------------------------------------------------------------------------
// Style definition
// ---------------------------------------------------------------------------

/// Define (or get) an atom style on a ForceField.
///
/// If the style already exists, this is a no-op and the existing style
/// is returned internally.
///
/// # C signature
///
/// ```c
/// MolrsStatus molrs_ff_def_atomstyle(MolrsForceFieldHandle ff,
///                                     const char* name);
/// ```
///
/// # Arguments
///
/// * `ff` -- ForceField handle.
/// * `name` -- Style name (e.g. `"full"`, `"charge"`).
///
/// # Returns
///
/// * `MolrsStatus::Ok` on success.
/// * `MolrsStatus::NullPointer` if `name` is null.
/// * `MolrsStatus::Utf8Error` if `name` is not valid UTF-8.
/// * `MolrsStatus::InvalidForceFieldHandle` if `ff` is stale.
///
/// # Safety
///
/// * `ff` must be a live ForceField handle.
/// * `name` must be a valid, null-terminated C string.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn molrs_ff_def_atomstyle(
    ff: MolrsForceFieldHandle,
    name: *const c_char,
) -> MolrsStatus {
    ffi_try!({
        null_check!(name);
        let name_str = match unsafe { CStr::from_ptr(name) }.to_str() {
            Ok(s) => s,
            Err(_) => {
                error::set_last_error("style name is not valid UTF-8");
                return MolrsStatus::Utf8Error;
            }
        };
        let mut store = lock_store();
        let ff = get_ff_mut!(store, ff);
        ff.def_atomstyle(name_str);
        MolrsStatus::Ok
    })
}

/// Define (or get) a bond style on a ForceField.
///
/// # C signature
///
/// ```c
/// MolrsStatus molrs_ff_def_bondstyle(MolrsForceFieldHandle ff,
///                                     const char* name);
/// ```
///
/// # Arguments
///
/// * `ff` -- ForceField handle.
/// * `name` -- Style name (e.g. `"harmonic"`, `"morse"`).
///
/// # Returns
///
/// * `MolrsStatus::Ok` on success.
/// * `MolrsStatus::NullPointer` if `name` is null.
/// * `MolrsStatus::Utf8Error` if `name` is not valid UTF-8.
/// * `MolrsStatus::InvalidForceFieldHandle` if `ff` is stale.
///
/// # Safety
///
/// * `ff` must be a live ForceField handle.
/// * `name` must be a valid, null-terminated C string.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn molrs_ff_def_bondstyle(
    ff: MolrsForceFieldHandle,
    name: *const c_char,
) -> MolrsStatus {
    ffi_try!({
        null_check!(name);
        let name_str = match unsafe { CStr::from_ptr(name) }.to_str() {
            Ok(s) => s,
            Err(_) => {
                error::set_last_error("style name is not valid UTF-8");
                return MolrsStatus::Utf8Error;
            }
        };
        let mut store = lock_store();
        let ff = get_ff_mut!(store, ff);
        ff.def_bondstyle(name_str);
        MolrsStatus::Ok
    })
}

/// Define (or get) an angle style on a ForceField.
///
/// # C signature
///
/// ```c
/// MolrsStatus molrs_ff_def_anglestyle(MolrsForceFieldHandle ff,
///                                      const char* name);
/// ```
///
/// # Arguments
///
/// * `ff` -- ForceField handle.
/// * `name` -- Style name (e.g. `"harmonic"`, `"cosine"`).
///
/// # Returns
///
/// * `MolrsStatus::Ok` on success.
/// * `MolrsStatus::NullPointer` if `name` is null.
/// * `MolrsStatus::Utf8Error` if `name` is not valid UTF-8.
/// * `MolrsStatus::InvalidForceFieldHandle` if `ff` is stale.
///
/// # Safety
///
/// * `ff` must be a live ForceField handle.
/// * `name` must be a valid, null-terminated C string.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn molrs_ff_def_anglestyle(
    ff: MolrsForceFieldHandle,
    name: *const c_char,
) -> MolrsStatus {
    ffi_try!({
        null_check!(name);
        let name_str = match unsafe { CStr::from_ptr(name) }.to_str() {
            Ok(s) => s,
            Err(_) => {
                error::set_last_error("style name is not valid UTF-8");
                return MolrsStatus::Utf8Error;
            }
        };
        let mut store = lock_store();
        let ff = get_ff_mut!(store, ff);
        ff.def_anglestyle(name_str);
        MolrsStatus::Ok
    })
}

/// Define (or get) a pair style on a ForceField, with optional
/// style-level parameters (e.g. global cutoff).
///
/// # C signature
///
/// ```c
/// MolrsStatus molrs_ff_def_pairstyle(MolrsForceFieldHandle ff,
///                                     const char* name,
///                                     const char** param_keys,
///                                     const double* param_values,
///                                     size_t n_params);
/// ```
///
/// # Arguments
///
/// * `ff` -- ForceField handle.
/// * `name` -- Style name (e.g. `"lj/cut"`, `"buck"`).
/// * `param_keys` -- Array of `n_params` null-terminated parameter name
///   strings (e.g. `"cutoff"`).  May be `NULL` if `n_params == 0`.
/// * `param_values` -- Array of `n_params` `double` values.
///   May be `NULL` if `n_params == 0`.
/// * `n_params` -- Number of style-level parameters.
///
/// # Returns
///
/// * `MolrsStatus::Ok` on success.
/// * `MolrsStatus::NullPointer` if `name` is null, or if `n_params > 0`
///   and `param_keys`/`param_values` is null.
/// * `MolrsStatus::Utf8Error` if any string is not valid UTF-8.
/// * `MolrsStatus::InvalidForceFieldHandle` if `ff` is stale.
///
/// # Safety
///
/// * `ff` must be a live ForceField handle.
/// * `name` must be a valid, null-terminated C string.
/// * `param_keys` (if non-null) must point to `n_params` valid C string
///   pointers.
/// * `param_values` (if non-null) must point to `n_params` `double` values.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn molrs_ff_def_pairstyle(
    ff: MolrsForceFieldHandle,
    name: *const c_char,
    param_keys: *const *const c_char,
    param_values: *const f64,
    n_params: usize,
) -> MolrsStatus {
    ffi_try!({
        null_check!(name);
        let name_str = match unsafe { CStr::from_ptr(name) }.to_str() {
            Ok(s) => s,
            Err(_) => {
                error::set_last_error("style name is not valid UTF-8");
                return MolrsStatus::Utf8Error;
            }
        };
        let params = match unsafe { parse_params(param_keys, param_values, n_params) } {
            Ok(p) => p,
            Err(status) => return status,
        };
        let mut store = lock_store();
        let ff = get_ff_mut!(store, ff);
        ff.def_pairstyle(name_str, &params);
        MolrsStatus::Ok
    })
}

// ---------------------------------------------------------------------------
// Type definition (unified)
// ---------------------------------------------------------------------------

/// Define a type on a style using the unified name format.
///
/// This is the primary function for populating a force field with
/// per-type parameters.
///
/// # C signature
///
/// ```c
/// MolrsStatus molrs_ff_def_type(MolrsForceFieldHandle ff,
///                                const char* style_category,
///                                const char* style_name,
///                                const char* type_name,
///                                const char** param_keys,
///                                const double* param_values,
///                                size_t n_params);
/// ```
///
/// # Arguments
///
/// * `ff` -- ForceField handle.
/// * `style_category` -- One of `"atom"`, `"bond"`, `"angle"`,
///   `"dihedral"`, `"improper"`, `"pair"`.
/// * `style_name` -- Style name (e.g. `"harmonic"`, `"lj/cut"`).
///   If the style does not yet exist it is created.
/// * `type_name` -- Unified type name format:
///   - Atom: `"A"` (single type name)
///   - Bond: `"A-B"` (dash-separated pair)
///   - Angle: `"A-B-C"` (three types)
///   - Dihedral: `"A-B-C-D"` (four types)
///   - Pair: `"A-B"` (pair types, e.g. `"OW-OW"`)
/// * `param_keys` -- Array of `n_params` parameter name strings.
///   May be `NULL` if `n_params == 0`.
/// * `param_values` -- Array of `n_params` `double` values.
///   May be `NULL` if `n_params == 0`.
/// * `n_params` -- Number of parameters for this type.
///
/// # Returns
///
/// * `MolrsStatus::Ok` on success.
/// * `MolrsStatus::NullPointer` if any required pointer is null.
/// * `MolrsStatus::Utf8Error` if any string is not valid UTF-8.
/// * `MolrsStatus::InvalidArgument` if `style_category` is not one of
///   the recognised values.
/// * `MolrsStatus::InvalidForceFieldHandle` if `ff` is stale.
///
/// # Safety
///
/// * `ff` must be a live ForceField handle.
/// * All string arguments must be valid, null-terminated C strings.
/// * `param_keys` (if non-null) must point to `n_params` valid C string
///   pointers.
/// * `param_values` (if non-null) must point to `n_params` doubles.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn molrs_ff_def_type(
    ff: MolrsForceFieldHandle,
    style_category: *const c_char,
    style_name: *const c_char,
    type_name: *const c_char,
    param_keys: *const *const c_char,
    param_values: *const f64,
    n_params: usize,
) -> MolrsStatus {
    ffi_try!({
        null_check!(style_category);
        null_check!(style_name);
        null_check!(type_name);

        let cat_str = match unsafe { CStr::from_ptr(style_category) }.to_str() {
            Ok(s) => s,
            Err(_) => {
                error::set_last_error("style_category is not valid UTF-8");
                return MolrsStatus::Utf8Error;
            }
        };
        let name_str = match unsafe { CStr::from_ptr(style_name) }.to_str() {
            Ok(s) => s,
            Err(_) => {
                error::set_last_error("style_name is not valid UTF-8");
                return MolrsStatus::Utf8Error;
            }
        };
        let type_str = match unsafe { CStr::from_ptr(type_name) }.to_str() {
            Ok(s) => s,
            Err(_) => {
                error::set_last_error("type_name is not valid UTF-8");
                return MolrsStatus::Utf8Error;
            }
        };
        let params = match unsafe { parse_params(param_keys, param_values, n_params) } {
            Ok(p) => p,
            Err(status) => return status,
        };

        let mut store = lock_store();
        let ff = get_ff_mut!(store, ff);
        let style = match cat_str {
            "atom" => ff.def_atomstyle(name_str),
            "bond" => ff.def_bondstyle(name_str),
            "angle" => ff.def_anglestyle(name_str),
            "pair" => ff.def_pairstyle(name_str, &[]),
            "dihedral" => ff.def_dihedralstyle(name_str),
            "improper" => ff.def_improperstyle(name_str),
            _ => {
                error::set_last_error(format!("unknown style category: {cat_str}"));
                return MolrsStatus::InvalidArgument;
            }
        };
        style.def_type(type_str, &params);
        MolrsStatus::Ok
    })
}

// ---------------------------------------------------------------------------
// Queries
// ---------------------------------------------------------------------------

/// Get the number of interaction styles defined in a ForceField.
///
/// # C signature
///
/// ```c
/// MolrsStatus molrs_ff_style_count(MolrsForceFieldHandle ff,
///                                   size_t* out);
/// ```
///
/// # Arguments
///
/// * `ff` -- ForceField handle.
/// * `out` -- On success, receives the style count.
///
/// # Returns
///
/// * `MolrsStatus::Ok` on success.
/// * `MolrsStatus::NullPointer` if `out` is null.
/// * `MolrsStatus::InvalidForceFieldHandle` if `ff` is stale.
///
/// # Safety
///
/// * `ff` must be a live ForceField handle.
/// * `out` must point to a writable `size_t`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn molrs_ff_style_count(
    ff: MolrsForceFieldHandle,
    out: *mut usize,
) -> MolrsStatus {
    ffi_try!({
        null_check!(out);
        let store = lock_store();
        let ff = get_ff!(store, ff);
        unsafe { *out = ff.styles().len() };
        MolrsStatus::Ok
    })
}

/// Get the category and name of a style by positional index.
///
/// Use [`molrs_ff_style_count`] to determine the valid index range
/// `[0, count)`.
///
/// Both returned strings are heap-allocated and must be freed with
/// [`molrs_free_string`](crate::molrs_free_string).
///
/// # C signature
///
/// ```c
/// MolrsStatus molrs_ff_get_style_name(MolrsForceFieldHandle ff,
///                                      size_t index,
///                                      char** out_category,
///                                      char** out_name);
/// ```
///
/// # Arguments
///
/// * `ff` -- ForceField handle.
/// * `index` -- Zero-based style index.
/// * `out_category` -- Receives a heap-allocated category string
///   (e.g. `"pair"`, `"bond"`).
/// * `out_name` -- Receives a heap-allocated style name string
///   (e.g. `"lj/cut"`, `"harmonic"`).
///
/// # Returns
///
/// * `MolrsStatus::Ok` on success.
/// * `MolrsStatus::NullPointer` if `out_category` or `out_name` is null.
/// * `MolrsStatus::InvalidArgument` if `index >= style_count`.
/// * `MolrsStatus::InvalidForceFieldHandle` if `ff` is stale.
///
/// # Safety
///
/// * `ff` must be a live ForceField handle.
/// * `out_category` and `out_name` must each point to a writable `char*`.
/// * The caller owns both returned strings and must free them.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn molrs_ff_get_style_name(
    ff: MolrsForceFieldHandle,
    index: usize,
    out_category: *mut *mut c_char,
    out_name: *mut *mut c_char,
) -> MolrsStatus {
    ffi_try!({
        null_check!(out_category);
        null_check!(out_name);
        let store = lock_store();
        let ff = get_ff!(store, ff);
        let styles = ff.styles();
        if index >= styles.len() {
            error::set_last_error(format!(
                "style index {} out of range (count={})",
                index,
                styles.len()
            ));
            return MolrsStatus::InvalidArgument;
        }
        let style = &styles[index];
        let cat = CString::new(style.category()).unwrap_or_default();
        let name = CString::new(style.name.as_str()).unwrap_or_default();
        unsafe {
            *out_category = cat.into_raw();
            *out_name = name.into_raw();
        }
        MolrsStatus::Ok
    })
}

// ---------------------------------------------------------------------------
// JSON serialization
// ---------------------------------------------------------------------------

/// Serialize a ForceField to a JSON string.
///
/// The returned string is heap-allocated and must be freed with
/// [`molrs_free_string`](crate::molrs_free_string).
///
/// # C signature
///
/// ```c
/// MolrsStatus molrs_ff_to_json(MolrsForceFieldHandle ff,
///                               char** out_json,
///                               size_t* out_len);
/// ```
///
/// # Arguments
///
/// * `ff` -- ForceField handle.
/// * `out_json` -- Receives a heap-allocated, null-terminated JSON string.
/// * `out_len` -- Receives the byte length of the JSON string
///   (not counting the null terminator).
///
/// # Returns
///
/// * `MolrsStatus::Ok` on success.
/// * `MolrsStatus::NullPointer` if `out_json` or `out_len` is null.
/// * `MolrsStatus::InvalidForceFieldHandle` if `ff` is stale.
///
/// # Safety
///
/// * `ff` must be a live ForceField handle.
/// * `out_json` must point to a writable `char*`.
/// * `out_len` must point to a writable `size_t`.
/// * The caller owns the returned string and must free it.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn molrs_ff_to_json(
    ff: MolrsForceFieldHandle,
    out_json: *mut *mut c_char,
    out_len: *mut usize,
) -> MolrsStatus {
    ffi_try!({
        null_check!(out_json);
        null_check!(out_len);
        let store = lock_store();
        let ff = get_ff!(store, ff);

        let json = ff_to_json_string(ff);
        let len = json.len();
        let c_json = CString::new(json).unwrap_or_default();
        unsafe {
            *out_json = c_json.into_raw();
            *out_len = len;
        }
        MolrsStatus::Ok
    })
}

/// Deserialize a ForceField from a JSON string.
///
/// The JSON format matches the output of [`molrs_ff_to_json`]:
///
/// ```json
/// {
///   "name": "my_ff",
///   "styles": [
///     {
///       "category": "pair",
///       "name": "lj/cut",
///       "params": {"cutoff": 12.0},
///       "types": [
///         {"name": "OW-OW", "params": {"epsilon": 0.1553, "sigma": 3.166}}
///       ]
///     }
///   ]
/// }
/// ```
///
/// # C signature
///
/// ```c
/// MolrsStatus molrs_ff_from_json(const char* json,
///                                 MolrsForceFieldHandle* out);
/// ```
///
/// # Arguments
///
/// * `json` -- Null-terminated JSON string.
/// * `out` -- On success, receives the new ForceField handle.
///
/// # Returns
///
/// * `MolrsStatus::Ok` on success.
/// * `MolrsStatus::NullPointer` if `json` or `out` is null.
/// * `MolrsStatus::Utf8Error` if `json` is not valid UTF-8.
/// * `MolrsStatus::InvalidArgument` if the JSON is malformed or
///   contains unknown style categories.
///
/// # Safety
///
/// * `json` must be a valid, null-terminated C string.
/// * `out` must point to a writable `MolrsForceFieldHandle`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn molrs_ff_from_json(
    json: *const c_char,
    out: *mut MolrsForceFieldHandle,
) -> MolrsStatus {
    ffi_try!({
        null_check!(json);
        null_check!(out);
        let json_str = match unsafe { CStr::from_ptr(json) }.to_str() {
            Ok(s) => s,
            Err(_) => {
                error::set_last_error("JSON is not valid UTF-8");
                return MolrsStatus::Utf8Error;
            }
        };
        let ff = match ff_from_json_string(json_str) {
            Ok(ff) => ff,
            Err(msg) => {
                error::set_last_error(msg);
                return MolrsStatus::InvalidArgument;
            }
        };
        let mut store = lock_store();
        let key = store.forcefields.insert(ff);
        unsafe { *out = ff_key_to_handle(key) };
        MolrsStatus::Ok
    })
}

// ---------------------------------------------------------------------------
// JSON helpers (manual serialization, no serde derives needed)
// ---------------------------------------------------------------------------

fn ff_to_json_string(ff: &ForceField) -> String {
    use std::fmt::Write;
    let mut json = String::new();
    write!(json, "{{\"name\":{},\"styles\":[", json_escape(&ff.name)).unwrap();

    for (i, style) in ff.styles().iter().enumerate() {
        if i > 0 {
            json.push(',');
        }
        write!(
            json,
            "{{\"category\":{},\"name\":{},\"params\":{{",
            json_escape(style.category()),
            json_escape(&style.name),
        )
        .unwrap();
        write_params(&mut json, &style.params);
        json.push_str("},\"types\":[");

        let type_params = style.defs.collect_type_params();
        for (j, (tname, tparams)) in type_params.iter().enumerate() {
            if j > 0 {
                json.push(',');
            }
            write!(json, "{{\"name\":{},\"params\":{{", json_escape(tname)).unwrap();
            write_params(&mut json, tparams);
            json.push_str("}}");
        }
        json.push_str("]}");
    }

    json.push_str("]}");
    json
}

fn ff_from_json_string(json: &str) -> Result<ForceField, String> {
    let val: serde_json::Value =
        serde_json::from_str(json).map_err(|e| format!("JSON parse error: {e}"))?;
    let name = val["name"].as_str().ok_or("missing 'name' field")?;
    let mut ff = ForceField::new(name);

    let styles = val["styles"].as_array().ok_or("missing 'styles' array")?;
    for style_val in styles {
        let category = style_val["category"]
            .as_str()
            .ok_or("missing style 'category'")?;
        let style_name = style_val["name"].as_str().ok_or("missing style 'name'")?;

        let style_params = extract_params(&style_val["params"]);

        let style = match category {
            "atom" => ff.def_atomstyle(style_name),
            "bond" => ff.def_bondstyle(style_name),
            "angle" => ff.def_anglestyle(style_name),
            "dihedral" => ff.def_dihedralstyle(style_name),
            "improper" => ff.def_improperstyle(style_name),
            "pair" => ff.def_pairstyle(style_name, &style_params),
            "kspace" => ff.def_kspacestyle(style_name, &style_params),
            _ => return Err(format!("unknown category: {category}")),
        };

        if let Some(types) = style_val["types"].as_array() {
            for type_val in types {
                let type_name = type_val["name"].as_str().ok_or("missing type 'name'")?;
                let type_params = extract_params(&type_val["params"]);
                style.def_type(type_name, &type_params);
            }
        }
    }

    Ok(ff)
}

fn extract_params(val: &serde_json::Value) -> Vec<(&str, f64)> {
    let mut params = Vec::new();
    if let Some(obj) = val.as_object() {
        for (k, v) in obj {
            if let Some(f) = v.as_f64() {
                // We need 'static str but have &String. Use leak for simplicity.
                // This is called rarely (JSON import) so the leak is acceptable.
                // A better approach would be to collect owned strings separately.
                params.push((k.as_str(), f));
            }
        }
    }
    params
}

fn json_escape(s: &str) -> String {
    format!(
        "\"{}\"",
        s.replace('\\', "\\\\")
            .replace('"', "\\\"")
            .replace('\n', "\\n")
            .replace('\r', "\\r")
            .replace('\t', "\\t")
    )
}

fn write_params(json: &mut String, params: &molrs_ff::forcefield::Params) {
    use std::fmt::Write;
    for (i, (k, v)) in params.iter().enumerate() {
        if i > 0 {
            json.push(',');
        }
        write!(json, "{}:{}", json_escape(k), v).unwrap();
    }
}
