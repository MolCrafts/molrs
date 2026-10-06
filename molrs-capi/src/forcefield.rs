//! `extern "C"` functions for ForceField operations.
//!
//! A **ForceField** is a collection of interaction styles (atom, bond,
//! angle, dihedral, improper, pair, cmap) and per-type parameter sets.
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
//! molrs_ff_def_style(ff, "pair", "lj/cut", pk, pv, 1);
//!
//! // Define a type under that style: its name, then its endpoints
//! const char* ow[] = {"OW"};
//! const char* tk[] = {"epsilon", "sigma"};
//! double      tv[] = {0.1553, 3.166};
//! molrs_ff_def_type(ff, "pair", "lj/cut", "OW", ow, 1, tk, tv, 2);
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
//!
//! The `molrs_ff_def_*` calls take numeric parameters only. String
//! parameters (a pair style's `mixing`, an atom type's `element`) reach a
//! force field from C through [`molrs_ff_from_json`], whose document
//! carries both.

use std::ffi::{CStr, CString, c_char};

use molrs::ff::forcefield::{DefError, Params, Style};
use molrs::ff::{ForceField, SpecialBonds};
use serde_json::{Value, json};

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
// Definition: def_style / def_type
// ---------------------------------------------------------------------------

/// Read a required C string argument as UTF-8.
unsafe fn arg_str<'a>(ptr: *const c_char, what: &str) -> Result<&'a str, MolrsStatus> {
    if ptr.is_null() {
        error::set_last_error(format!("{what} is null"));
        return Err(MolrsStatus::NullPointer);
    }
    unsafe { CStr::from_ptr(ptr) }.to_str().map_err(|_| {
        error::set_last_error(format!("{what} is not valid UTF-8"));
        MolrsStatus::Utf8Error
    })
}

/// Parse an array of `n` C strings (type endpoints).
unsafe fn parse_strings<'a>(
    ptrs: *const *const c_char,
    n: usize,
    what: &str,
) -> Result<Vec<&'a str>, MolrsStatus> {
    if n == 0 {
        return Ok(Vec::new());
    }
    if ptrs.is_null() {
        error::set_last_error(format!("{what} is null"));
        return Err(MolrsStatus::NullPointer);
    }
    let slice = unsafe { std::slice::from_raw_parts(ptrs, n) };
    slice
        .iter()
        .enumerate()
        .map(|(i, &p)| unsafe { arg_str(p, &format!("{what}[{i}]")) })
        .collect()
}

/// Unwrap a `Result<T, MolrsStatus>`, returning the status from the calling
/// `extern "C"` function on error.
macro_rules! or_return {
    ($e:expr) => {
        match $e {
            Ok(v) => v,
            Err(status) => return status,
        }
    };
}

/// Map a definition error to `InvalidArgument`, recording its message.
fn invalid_argument(e: DefError) -> MolrsStatus {
    error::set_last_error(e.to_string());
    MolrsStatus::InvalidArgument
}

/// Define the `category` style named `name` on a ForceField.
///
/// A style is identified by `(category, name)`: re-defining it with equal
/// params keeps the existing style; with different params it is
/// `InvalidArgument` and the first definition is kept.
///
/// # C signature
///
/// ```c
/// MolrsStatus molrs_ff_def_style(MolrsForceFieldHandle ff,
///                                 const char* category,
///                                 const char* name,
///                                 const char** param_keys,
///                                 const double* param_values,
///                                 size_t n_params);
/// ```
///
/// # Arguments
///
/// * `ff` -- ForceField handle.
/// * `category` -- One of `"atom"`, `"bond"`, `"angle"`, `"dihedral"`,
///   `"improper"`, `"pair"`, `"cmap"`.
/// * `name` -- Style name (e.g. `"harmonic"`, `"lj/cut"`).
/// * `param_keys` -- Array of `n_params` style-level parameter names
///   (e.g. `"cutoff"`). May be `NULL` if `n_params == 0`.
/// * `param_values` -- Array of `n_params` `double` values. May be `NULL`
///   if `n_params == 0`.
/// * `n_params` -- Number of style-level parameters.
///
/// Parameters are numeric only; string params (e.g. `mixing`) reach a
/// force field from C through [`molrs_ff_from_json`].
///
/// # Returns
///
/// * `MolrsStatus::Ok` on success.
/// * `MolrsStatus::NullPointer` if a required pointer is null.
/// * `MolrsStatus::Utf8Error` if any string is not valid UTF-8.
/// * `MolrsStatus::InvalidArgument` if `category` is not one of the values
///   above, or the style is already defined with different params.
/// * `MolrsStatus::InvalidForceFieldHandle` if `ff` is stale.
///
/// # Safety
///
/// * `ff` must be a live ForceField handle.
/// * `category` and `name` must be valid, null-terminated C strings.
/// * `param_keys` (if non-null) must point to `n_params` valid C string
///   pointers; `param_values` (if non-null) to `n_params` doubles.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn molrs_ff_def_style(
    ff: MolrsForceFieldHandle,
    category: *const c_char,
    name: *const c_char,
    param_keys: *const *const c_char,
    param_values: *const f64,
    n_params: usize,
) -> MolrsStatus {
    ffi_try!({
        let category = or_return!(unsafe { arg_str(category, "category") });
        let name = or_return!(unsafe { arg_str(name, "name") });
        let params = or_return!(unsafe { parse_params(param_keys, param_values, n_params) });
        let mut store = lock_store();
        let ff = get_ff_mut!(store, ff);
        match ff.def_style(category, name, Params::from_pairs(&params)) {
            Ok(_) => MolrsStatus::Ok,
            Err(e) => invalid_argument(e),
        }
    })
}

/// Define a type on an existing style: its name and its endpoints.
///
/// The style `(style_category, style_name)` must already exist (see
/// [`molrs_ff_def_style`]); it is never created implicitly. The name is an
/// opaque identifier stored verbatim and never split into endpoints (`CT-OH`
/// and MMFF's `0_1_5` alike); the endpoints are the ones given.
///
/// # C signature
///
/// ```c
/// MolrsStatus molrs_ff_def_type(MolrsForceFieldHandle ff,
///                                const char* style_category,
///                                const char* style_name,
///                                const char* type_name,
///                                const char** endpoints,
///                                size_t n_endpoints,
///                                const char** param_keys,
///                                const double* param_values,
///                                size_t n_params);
/// ```
///
/// # Arguments
///
/// * `ff` -- ForceField handle.
/// * `style_category` / `style_name` -- An existing style.
/// * `type_name` -- The type name, stored as given.
/// * `endpoints` -- Array of `n_endpoints` atom-type names. The count must
///   match the category: atom 0, pair 1 (a self pair) or 2, bond 2, angle 3,
///   dihedral and improper 4. May be `NULL` if `n_endpoints == 0`.
/// * `param_keys` / `param_values` / `n_params` -- Numeric type parameters;
///   the arrays may be `NULL` if `n_params == 0`.
///
/// # Returns
///
/// * `MolrsStatus::Ok` on success.
/// * `MolrsStatus::NullPointer` if a required pointer is null.
/// * `MolrsStatus::Utf8Error` if any string is not valid UTF-8.
/// * `MolrsStatus::InvalidArgument` if the style does not exist, the
///   endpoint count does not match the category, or the style already
///   defines `type_name` with other endpoints or params (an identical
///   re-definition is `Ok` and changes nothing).
/// * `MolrsStatus::InvalidForceFieldHandle` if `ff` is stale.
///
/// # Safety
///
/// * `ff` must be a live ForceField handle.
/// * All string arguments must be valid, null-terminated C strings.
/// * `endpoints` (if non-null) must point to `n_endpoints` valid C string
///   pointers.
/// * `param_keys` (if non-null) must point to `n_params` valid C string
///   pointers; `param_values` (if non-null) to `n_params` doubles.
#[unsafe(no_mangle)]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn molrs_ff_def_type(
    ff: MolrsForceFieldHandle,
    style_category: *const c_char,
    style_name: *const c_char,
    type_name: *const c_char,
    endpoints: *const *const c_char,
    n_endpoints: usize,
    param_keys: *const *const c_char,
    param_values: *const f64,
    n_params: usize,
) -> MolrsStatus {
    ffi_try!({
        let category = or_return!(unsafe { arg_str(style_category, "style_category") });
        let style_name = or_return!(unsafe { arg_str(style_name, "style_name") });
        let type_name = or_return!(unsafe { arg_str(type_name, "type_name") });
        let endpoints = or_return!(unsafe { parse_strings(endpoints, n_endpoints, "endpoints") });
        let params = or_return!(unsafe { parse_params(param_keys, param_values, n_params) });
        let mut store = lock_store();
        let ff = get_ff_mut!(store, ff);
        let Some(style) = ff.get_style_mut(category, style_name) else {
            return invalid_argument(DefError::UnknownStyle {
                category: category.to_owned(),
                name: style_name.to_owned(),
            });
        };
        match style.def_type(type_name, &endpoints, Params::from_pairs(&params)) {
            Ok(_) => MolrsStatus::Ok,
            Err(e) => invalid_argument(e),
        }
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
        let name = CString::new(style.name()).unwrap_or_default();
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
/// The document carries every definition -- the declared `units` and
/// `special_bonds` (each written only when declared), numeric and string
/// params of every style and type, and every type's endpoints -- in the format
/// described at [`molrs_ff_from_json`].
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
///   "units": "real",
///   "special_bonds": {"lj": [0.0, 0.0, 0.5], "coul": [0.0, 0.0, 0.8333]},
///   "styles": [
///     {
///       "category": "pair",
///       "name": "lj/cut",
///       "params": {"cutoff": 12.0},
///       "str_params": {"mixing": "geometric"},
///       "types": [
///         {"name": "OW", "endpoints": ["OW", "OW"],
///          "params": {"epsilon": 0.1553, "sigma": 3.166}, "str_params": {}}
///       ]
///     },
///     {
///       "category": "cmap",
///       "name": "charmm",
///       "params": {},
///       "str_params": {},
///       "types": [
///         {"name": "C-N-CA-C-N", "endpoints": ["C", "N", "CA", "C", "N"],
///          "params": {}, "str_params": {},
///          "array_params": {"grid": [[0.0, 0.1], [0.2, 0.3]]}}
///       ]
///     }
///   ]
/// }
/// ```
///
/// `units` and `special_bonds` are optional: present means declared, absent
/// means undeclared, and the force field is rebuilt declaring exactly what the
/// document holds. `array_params` (on a style or a type) is optional and
/// written only when the definition holds an array param: each value is the
/// array as nested lists of numbers, one level per axis, every list of a
/// level the same length. Every other key shown is required and no other key
/// is accepted. The document is carried whole or refused: a missing or
/// unknown key, a non-string `units`, a non-number in `params`, a non-string
/// in `str_params` or `endpoints`, a ragged or non-numeric array in
/// `array_params`, a `special_bonds` array whose length is not 3, an unknown
/// category, or an endpoint count that does not match the category is
/// `InvalidArgument` -- nothing is skipped.
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
/// * `MolrsStatus::InvalidArgument` if the JSON is malformed or does not
///   match the document above.
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
// JSON document: carries every definition or refuses it
// ---------------------------------------------------------------------------

type JsonMap = serde_json::Map<String, Value>;

/// Serialize every piece of declared state: name, the declared `units` and
/// `special_bonds` (absent when undeclared), and per style its category, name,
/// numeric, string and array params, and per type its name, endpoints,
/// numeric, string and array params.
fn ff_to_json_string(ff: &ForceField) -> String {
    let styles: Vec<Value> = ff
        .styles()
        .iter()
        .map(|style| {
            let mut entry = json!({
                "category": style.category(),
                "name": style.name(),
                "params": numeric_params(style.params()),
                "str_params": string_params(style.params()),
                "types": type_rows(style),
            });
            put_array_params(&mut entry, style.params());
            entry
        })
        .collect();
    let mut doc = JsonMap::new();
    doc.insert("name".into(), json!(ff.name));
    if let Some(units) = ff.declared_units() {
        doc.insert("units".into(), json!(units));
    }
    if let Some(special) = ff.declared_special_bonds() {
        doc.insert(
            "special_bonds".into(),
            json!({"lj": special.lj, "coul": special.coul}),
        );
    }
    doc.insert("styles".into(), json!(styles));
    Value::Object(doc).to_string()
}

fn numeric_params(params: &Params) -> JsonMap {
    params
        .iter()
        .map(|(k, v)| (k.to_owned(), json!(v)))
        .collect()
}

fn string_params(params: &Params) -> JsonMap {
    params
        .iter_strings()
        .map(|(k, v)| (k.to_owned(), json!(v)))
        .collect()
}

/// `array_params` on `entry` when `params` holds an array param: each array
/// as nested lists, one level per axis.
fn put_array_params(entry: &mut Value, params: &Params) {
    fn nested(view: ndarray::ArrayViewD<'_, f64>) -> Value {
        if view.ndim() == 0 {
            return json!(view.first().copied().unwrap_or_default());
        }
        Value::Array(view.outer_iter().map(nested).collect())
    }
    let arrays: JsonMap = params
        .iter_arrays()
        .map(|(k, v)| (k.to_owned(), nested(v.view())))
        .collect();
    if !arrays.is_empty() {
        entry["array_params"] = Value::Object(arrays);
    }
}

/// Every type of a style as `{name, endpoints, params, str_params}` (and
/// `array_params` when it has any), in definition order.
fn type_rows(style: &Style) -> Vec<Value> {
    style
        .type_rows()
        .into_iter()
        .map(|(name, endpoints, params)| {
            let mut row = json!({
                "name": name,
                "endpoints": endpoints,
                "params": numeric_params(params),
                "str_params": string_params(params),
            });
            put_array_params(&mut row, params);
            row
        })
        .collect()
}

/// Rebuild a force field from the [`ff_to_json_string`] document.
///
/// Every style is defined through `def_style` with its full params, every type
/// through `def_type` with its endpoints. Nothing is skipped: a missing or
/// unknown key at any level, a non-string `units`, a non-number in `params`, a
/// non-string in `str_params` or `endpoints`, or a `special_bonds` array whose
/// length is not 3 is an error. `units` and `special_bonds` are declared
/// exactly when present.
fn ff_from_json_string(json: &str) -> Result<ForceField, String> {
    let val: Value = serde_json::from_str(json).map_err(|e| format!("JSON parse error: {e}"))?;
    let doc = JsonObject::new(
        &val,
        "document",
        &["name", "styles"],
        &["units", "special_bonds"],
    )?;
    let mut ff = ForceField::new(doc.str("name")?);

    if doc.has("units") {
        ff.set_units(doc.str("units")?);
    }
    if doc.has("special_bonds") {
        let special = JsonObject::new(
            doc.get("special_bonds")?,
            "special_bonds",
            &["lj", "coul"],
            &[],
        )?;
        ff.set_special_bonds(SpecialBonds {
            lj: special.weights("lj")?,
            coul: special.weights("coul")?,
        });
    }

    for (i, style_val) in doc.array("styles")?.iter().enumerate() {
        let at = format!("styles[{i}]");
        let style_obj = JsonObject::new(
            style_val,
            &at,
            &["category", "name", "params", "str_params", "types"],
            &["array_params"],
        )?;
        let style = ff
            .def_style(
                style_obj.str("category")?,
                style_obj.str("name")?,
                style_obj.params()?,
            )
            .map_err(|e| format!("{at}: {e}"))?;
        for (j, type_val) in style_obj.array("types")?.iter().enumerate() {
            let at = format!("{at}.types[{j}]");
            let type_obj = JsonObject::new(
                type_val,
                &at,
                &["name", "endpoints", "params", "str_params"],
                &["array_params"],
            )?;
            let endpoints = type_obj
                .array("endpoints")?
                .iter()
                .enumerate()
                .map(|(k, v)| {
                    v.as_str()
                        .ok_or_else(|| format!("{at}.endpoints[{k}] is not a string"))
                })
                .collect::<Result<Vec<&str>, String>>()?;
            style
                .def_type(type_obj.str("name")?, &endpoints, type_obj.params()?)
                .map_err(|e| format!("{at}: {e}"))?;
        }
    }

    Ok(ff)
}

/// The array nested lists `value` spell: one level per axis, every list of a
/// level the same length, numbers at the leaves (a bare number is 0-d).
fn json_array(value: &Value, at: &str) -> Result<ndarray::ArrayD<f64>, String> {
    let mut shape = Vec::new();
    let mut level = value;
    while let Value::Array(items) = level {
        shape.push(items.len());
        match items.first() {
            Some(first) => level = first,
            None => break,
        }
    }
    fn flatten(value: &Value, shape: &[usize], out: &mut Vec<f64>, at: &str) -> Result<(), String> {
        match (value, shape) {
            (Value::Array(items), [n, rest @ ..]) if items.len() == *n => items
                .iter()
                .try_for_each(|item| flatten(item, rest, out, at)),
            (Value::Number(n), []) => {
                out.push(n.as_f64().expect("a JSON number reads as f64"));
                Ok(())
            }
            _ => Err(format!(
                "{at} is not a rectangular array of numbers (shape {shape:?})"
            )),
        }
    }
    let mut values = Vec::new();
    flatten(value, &shape, &mut values, at)?;
    ndarray::ArrayD::from_shape_vec(shape, values).map_err(|e| format!("{at}: {e}"))
}

/// A JSON object holding every `required` key, any of the `optional` keys and
/// nothing else, with typed field readers that name the offending path on
/// error.
struct JsonObject<'a> {
    at: &'a str,
    map: &'a JsonMap,
}

impl<'a> JsonObject<'a> {
    fn new(
        val: &'a Value,
        at: &'a str,
        required: &[&str],
        optional: &[&str],
    ) -> Result<Self, String> {
        let map = val
            .as_object()
            .ok_or_else(|| format!("{at} is not an object"))?;
        if let Some(unknown) = map
            .keys()
            .find(|k| !required.contains(&k.as_str()) && !optional.contains(&k.as_str()))
        {
            return Err(format!("{at}: unknown key '{unknown}'"));
        }
        if let Some(missing) = required.iter().find(|k| !map.contains_key(**k)) {
            return Err(format!("{at}: missing key '{missing}'"));
        }
        Ok(Self { at, map })
    }

    fn has(&self, key: &str) -> bool {
        self.map.contains_key(key)
    }

    fn get(&self, key: &str) -> Result<&'a Value, String> {
        self.map
            .get(key)
            .ok_or_else(|| format!("{}: missing key '{key}'", self.at))
    }

    fn str(&self, key: &str) -> Result<&'a str, String> {
        self.get(key)?
            .as_str()
            .ok_or_else(|| format!("{}.{key} is not a string", self.at))
    }

    fn array(&self, key: &str) -> Result<&'a Vec<Value>, String> {
        self.get(key)?
            .as_array()
            .ok_or_else(|| format!("{}.{key} is not an array", self.at))
    }

    fn object(&self, key: &str) -> Result<&'a JsonMap, String> {
        self.get(key)?
            .as_object()
            .ok_or_else(|| format!("{}.{key} is not an object", self.at))
    }

    /// `params` (numbers), `str_params` (strings) and the optional
    /// `array_params` (nested lists of numbers) as one [`Params`].
    fn params(&self) -> Result<Params, String> {
        let mut params = Params::new();
        for (k, v) in self.object("params")? {
            let v = v
                .as_f64()
                .ok_or_else(|| format!("{}.params.{k} is not a number", self.at))?;
            params.set(k, v);
        }
        for (k, v) in self.object("str_params")? {
            let v = v
                .as_str()
                .ok_or_else(|| format!("{}.str_params.{k} is not a string", self.at))?;
            params.set_str(k, v);
        }
        if self.has("array_params") {
            for (k, v) in self.object("array_params")? {
                let at = format!("{}.array_params.{k}", self.at);
                params.set_array(k, json_array(v, &at)?);
            }
        }
        Ok(params)
    }

    /// A `[1-2, 1-3, 1-4]` weight triple: exactly three numbers.
    fn weights(&self, key: &str) -> Result<[f64; 3], String> {
        let values = self
            .array(key)?
            .iter()
            .map(Value::as_f64)
            .collect::<Option<Vec<f64>>>()
            .ok_or_else(|| format!("{}.{key} holds a non-number", self.at))?;
        values.try_into().map_err(|v: Vec<f64>| {
            format!(
                "{}.{key} must hold 3 weights [1-2, 1-3, 1-4], got {}",
                self.at,
                v.len()
            )
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use molrs::ff::SpecialBonds;
    use molrs::ff::forcefield::Params;

    fn round_trip(ff: &ForceField) -> ForceField {
        ff_from_json_string(&ff_to_json_string(ff)).unwrap()
    }

    #[test]
    fn json_round_trip_keeps_a_string_style_param() {
        let mut params = Params::from_pairs(&[("cutoff", 10.0)]);
        params.set_str("mixing", "geometric");
        let mut ff = ForceField::new("rt");
        ff.def_style("pair", "lj/cut", params).unwrap();

        let back = round_trip(&ff);
        let style = back.get_style("pair", "lj/cut").unwrap();
        assert_eq!(style.params().get_str("mixing"), Some("geometric"));
    }

    #[test]
    fn json_round_trip_keeps_bond_style_params() {
        let mut ff = ForceField::new("rt");
        ff.def_style("bond", "harmonic", Params::from_pairs(&[("scale", 0.5)]))
            .unwrap();

        let back = round_trip(&ff);
        let style = back.get_style("bond", "harmonic").unwrap();
        assert_eq!(style.params().get("scale"), Some(0.5));
    }

    /// A type's endpoints are data, not its name: MMFF's `0_1_5` comes back
    /// on `1`, `5` only because the document carries them.
    #[test]
    fn json_round_trip_keeps_the_given_endpoints() {
        let mut ff = ForceField::new("rt");
        ff.def_style("bond", "mmff", Params::new())
            .unwrap()
            .def_type("0_1_5", &["1", "5"], Params::from_pairs(&[("kb", 4.258)]))
            .unwrap();

        let back = round_trip(&ff);
        let style = back.get_style("bond", "mmff").unwrap();
        assert_eq!(
            style.type_endpoints("0_1_5"),
            Some(vec!["1".to_string(), "5".to_string()])
        );
    }

    /// An array param travels as nested lists and comes back at its shape,
    /// on a cmap type with five endpoints.
    #[test]
    fn json_round_trip_keeps_a_cmap_grid() {
        let grid =
            ndarray::ArrayD::from_shape_fn(vec![3, 3], |ix| (ix[0] * 3 + ix[1]) as f64 / 7.0);
        let mut params = Params::new();
        params.set_array("grid", grid.clone());
        let mut ff = ForceField::new("rt");
        ff.def_style("cmap", "charmm", Params::new())
            .unwrap()
            .def_type("c", &["C", "N", "CA", "C", "N"], params)
            .unwrap();

        let json = ff_to_json_string(&ff);
        assert!(json.contains("\"array_params\""), "{json}");
        let back = ff_from_json_string(&json).unwrap();
        let cmap = back.get_cmaptypes()[0];
        assert_eq!(cmap.params.get_array("grid"), Some(&grid));
        assert_eq!(cmap.mtom, "N");
    }

    #[test]
    fn a_ragged_array_param_is_refused() {
        let doc = r#"{"name": "x", "styles": [{"category": "cmap", "name": "charmm",
            "params": {}, "str_params": {}, "types": [{"name": "c",
            "endpoints": ["A", "B", "C", "D", "E"], "params": {}, "str_params": {},
            "array_params": {"grid": [[1.0, 2.0], [3.0]]}}]}]}"#;
        let err = ff_from_json_string(doc).unwrap_err();
        assert!(err.contains("array_params.grid"), "{err}");
        let doc = doc.replace("[3.0]", r#"[3.0, "x"]"#);
        assert!(ff_from_json_string(&doc).is_err());
    }

    #[test]
    fn json_round_trip_keeps_special_bonds() {
        let special = SpecialBonds {
            lj: [0.0, 0.0, 0.5],
            coul: [0.0, 0.0, 0.8333],
        };
        let mut ff = ForceField::new("rt");
        ff.set_special_bonds(special);

        let back = round_trip(&ff);
        assert_eq!(*back.special_bonds(), special);
    }

    #[test]
    fn json_round_trip_keeps_declared_units() {
        let mut ff = ForceField::new("rt");
        ff.set_units("lj");

        let back = round_trip(&ff);
        assert_eq!(back.declared_units(), Some("lj"));
    }

    /// Undeclared state is not written as a default and read back as a
    /// declaration.
    #[test]
    fn json_round_trip_keeps_an_undeclared_forcefield_undeclared() {
        let ff = ForceField::new("rt");

        let back = round_trip(&ff);
        assert_eq!(back.declared_units(), None);
        assert_eq!(back.declared_special_bonds(), None);
    }

    #[test]
    fn json_from_string_refuses_a_numeric_units() {
        let doc = r#"{
            "name": "doc",
            "units": 1,
            "special_bonds": {"lj": [0.0, 0.0, 0.5], "coul": [0.0, 0.0, 0.8333]},
            "styles": [{
                "category": "bond", "name": "harmonic", "params": {}, "str_params": {},
                "types": [{"name": "CT-OH", "endpoints": ["CT", "OH"],
                           "params": {"k": 300.0, "r0": 1.4}, "str_params": {}}]
            }]
        }"#;
        assert!(ff_from_json_string(doc).is_err());
    }

    /// The control for the refusals below: each differs from this document in
    /// exactly one place.
    const VALID: &str = r#"{
        "name": "doc",
        "special_bonds": {"lj": [0.0, 0.0, 0.5], "coul": [0.0, 0.0, 0.8333]},
        "styles": [{
            "category": "bond", "name": "harmonic", "params": {}, "str_params": {},
            "types": [{"name": "CT-OH", "endpoints": ["CT", "OH"],
                       "params": {"k": 300.0, "r0": 1.4}, "str_params": {}}]
        }]
    }"#;

    #[test]
    fn json_from_string_accepts_a_complete_document() {
        assert!(ff_from_json_string(VALID).is_ok());
    }

    #[test]
    fn json_from_string_refuses_a_bool_param_value() {
        let doc = r#"{
            "name": "doc",
            "special_bonds": {"lj": [0.0, 0.0, 0.5], "coul": [0.0, 0.0, 0.8333]},
            "styles": [{
                "category": "bond", "name": "harmonic", "params": {}, "str_params": {},
                "types": [{"name": "CT-OH", "endpoints": ["CT", "OH"],
                           "params": {"k": true, "r0": 1.4}, "str_params": {}}]
            }]
        }"#;
        assert!(ff_from_json_string(doc).is_err());
    }

    #[test]
    fn json_from_string_refuses_an_unknown_key() {
        let doc = r#"{
            "name": "doc",
            "bogus": 1,
            "special_bonds": {"lj": [0.0, 0.0, 0.5], "coul": [0.0, 0.0, 0.8333]},
            "styles": [{
                "category": "bond", "name": "harmonic", "params": {}, "str_params": {},
                "types": [{"name": "CT-OH", "endpoints": ["CT", "OH"],
                           "params": {"k": 300.0, "r0": 1.4}, "str_params": {}}]
            }]
        }"#;
        assert!(ff_from_json_string(doc).is_err());
    }

    /// Seam only: the Rust conflict rule is unit-tested in molrs; here a
    /// conflicting `molrs_ff_def_type` must surface as `InvalidArgument`.
    #[test]
    fn def_type_conflicting_redefinition_is_invalid_argument() {
        let name = CString::new("conflict").unwrap();
        let bond = CString::new("bond").unwrap();
        let harmonic = CString::new("harmonic").unwrap();
        let ct_oh = CString::new("CT-OH").unwrap();
        let ct = CString::new("CT").unwrap();
        let oh = CString::new("OH").unwrap();
        let ends = [ct.as_ptr(), oh.as_ptr()];
        let k = CString::new("k").unwrap();
        let r0 = CString::new("r0").unwrap();
        let keys = [k.as_ptr(), r0.as_ptr()];
        let first = [300.0, 1.4];
        let second = [310.0, 1.4];
        let mut ff = MolrsForceFieldHandle { idx: 0, version: 0 };

        // SAFETY: every pointer is a live CString / array of the stated length,
        // and `ff` is written by `molrs_ff_new` before use.
        unsafe {
            assert_eq!(molrs_ff_new(name.as_ptr(), &mut ff), MolrsStatus::Ok);
            assert_eq!(
                molrs_ff_def_style(
                    ff,
                    bond.as_ptr(),
                    harmonic.as_ptr(),
                    std::ptr::null(),
                    std::ptr::null(),
                    0
                ),
                MolrsStatus::Ok
            );
            assert_eq!(
                molrs_ff_def_type(
                    ff,
                    bond.as_ptr(),
                    harmonic.as_ptr(),
                    ct_oh.as_ptr(),
                    ends.as_ptr(),
                    2,
                    keys.as_ptr(),
                    first.as_ptr(),
                    2
                ),
                MolrsStatus::Ok
            );
            let status = molrs_ff_def_type(
                ff,
                bond.as_ptr(),
                harmonic.as_ptr(),
                ct_oh.as_ptr(),
                ends.as_ptr(),
                2,
                keys.as_ptr(),
                second.as_ptr(),
                2,
            );
            assert_eq!(molrs_ff_drop(ff), MolrsStatus::Ok);
            assert_eq!(status, MolrsStatus::InvalidArgument);
        }
    }

    #[test]
    fn json_from_string_refuses_a_two_element_special_bonds_array() {
        let doc = r#"{
            "name": "doc",
            "special_bonds": {"lj": [0.0, 0.5], "coul": [0.0, 0.0, 0.8333]},
            "styles": [{
                "category": "bond", "name": "harmonic", "params": {}, "str_params": {},
                "types": [{"name": "CT-OH", "endpoints": ["CT", "OH"],
                           "params": {"k": 300.0, "r0": 1.4}, "str_params": {}}]
            }]
        }"#;
        assert!(ff_from_json_string(doc).is_err());
    }
}
