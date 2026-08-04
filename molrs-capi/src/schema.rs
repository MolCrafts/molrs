//! C API for inspecting the Frame schema.
//!
//! The vocabulary crosses the boundary as **JSON**, not as a mirrored struct
//! layout. A `MolrsColumnSpec` struct would have to be kept in sync by hand in
//! the C header, which is the very duplication this schema exists to remove —
//! and it would break its ABI every time a field is added. One JSON document
//! costs the caller a parse and can never disagree with the Rust tables.
//!
//! Per-key lookups are provided as scalars for callers that only need to check
//! one column and do not want to parse anything.

use std::ffi::{CStr, CString, c_char};

use molrs::store::schema;

/// The whole Frame vocabulary as a JSON document.
///
/// # C signature
///
/// ```c
/// char* molrs_schema_json(void);
/// ```
///
/// # Returns
///
/// A heap-allocated, NUL-terminated JSON string the caller owns. Free it with
/// [`molrs_free_string`](crate::molrs_free_string). Never `NULL`.
///
/// The bytes are stable across runs, so two releases of the library can be
/// diffed to see exactly what changed about the data contract.
///
/// # Safety
///
/// The returned pointer must be freed exactly once with `molrs_free_string`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn molrs_schema_json() -> *mut c_char {
    CString::new(schema::document().to_json())
        .expect("schema JSON contains no interior NUL")
        .into_raw()
}

/// Vocabulary version — what the names and dtypes *mean*.
///
/// Distinct from the serialization envelope version. A caller that persists
/// frames should record this alongside the data.
///
/// # C signature
///
/// ```c
/// uint32_t molrs_schema_vocab_version(void);
/// ```
#[unsafe(no_mangle)]
pub extern "C" fn molrs_schema_vocab_version() -> u32 {
    schema::FRAME_VOCAB_VERSION
}

/// Number of canonical columns in the vocabulary.
///
/// # C signature
///
/// ```c
/// size_t molrs_schema_column_count(void);
/// ```
#[unsafe(no_mangle)]
pub extern "C" fn molrs_schema_column_count() -> usize {
    schema::SCHEMA_COLUMNS.len()
}

/// Number of canonical blocks in the vocabulary.
///
/// # C signature
///
/// ```c
/// size_t molrs_schema_block_count(void);
/// ```
#[unsafe(no_mangle)]
pub extern "C" fn molrs_schema_block_count() -> usize {
    schema::SCHEMA_BLOCKS.len()
}

/// Declared dtype of a canonical column, as a static string.
///
/// # C signature
///
/// ```c
/// const char* molrs_schema_column_dtype(const char* key);
/// ```
///
/// # Returns
///
/// One of `"float"`, `"int"`, `"uint"`, `"bool"`, `"u8"`, `"string"`, or
/// `NULL` when `key` is not in the vocabulary. A `NULL` return means the key
/// is **unconstrained**, not that it is invalid — unspecified keys are the
/// documented extension point.
///
/// The returned pointer is static; do **not** free it.
///
/// # Safety
///
/// `key` must be a valid NUL-terminated UTF-8 string.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn molrs_schema_column_dtype(key: *const c_char) -> *const c_char {
    if key.is_null() {
        return std::ptr::null();
    }
    let Ok(key) = (unsafe { CStr::from_ptr(key) }).to_str() else {
        return std::ptr::null();
    };
    match schema::column(key) {
        // Every DType name is a `&'static str` with no interior NUL, and the
        // table is `'static`, so a static C string can be handed out without
        // an allocation the caller would have to free.
        Some(spec) => match spec.dtype.name() {
            "float" => c"float".as_ptr(),
            "int" => c"int".as_ptr(),
            "uint" => c"uint".as_ptr(),
            "bool" => c"bool".as_ptr(),
            "u8" => c"u8".as_ptr(),
            _ => c"string".as_ptr(),
        },
        None => std::ptr::null(),
    }
}

/// Whether a block name is part of the canonical vocabulary.
///
/// # C signature
///
/// ```c
/// bool molrs_schema_has_block(const char* name);
/// ```
///
/// A `false` return does not mean the block is illegal: the block set is open,
/// and a frame may carry blocks the vocabulary does not name.
///
/// # Safety
///
/// `name` must be a valid NUL-terminated UTF-8 string.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn molrs_schema_has_block(name: *const c_char) -> bool {
    if name.is_null() {
        return false;
    }
    let Ok(name) = (unsafe { CStr::from_ptr(name) }).to_str() else {
        return false;
    };
    schema::block(name).is_some()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn json_is_valid_and_non_empty() {
        let p = unsafe { molrs_schema_json() };
        assert!(!p.is_null());
        let s = unsafe { CStr::from_ptr(p) }.to_str().unwrap().to_string();
        unsafe { crate::molrs_free_string(p) };
        let v: serde_json::Value = serde_json::from_str(&s).expect("valid JSON");
        assert_eq!(v["vocabVersion"], schema::FRAME_VOCAB_VERSION);
    }

    #[test]
    fn counts_match_the_tables() {
        assert_eq!(molrs_schema_column_count(), schema::SCHEMA_COLUMNS.len());
        assert_eq!(molrs_schema_block_count(), schema::SCHEMA_BLOCKS.len());
    }

    #[test]
    fn dtype_lookup_matches_the_table_for_every_key() {
        // Scans the table rather than spot-checking, so a key whose C-side
        // mapping is wrong cannot hide behind the ones that are right.
        for spec in schema::SCHEMA_COLUMNS {
            let k = CString::new(spec.key).unwrap();
            let p = unsafe { molrs_schema_column_dtype(k.as_ptr()) };
            assert!(!p.is_null(), "{} not found", spec.key);
            let got = unsafe { CStr::from_ptr(p) }.to_str().unwrap();
            assert_eq!(got, spec.dtype.name(), "{}", spec.key);
        }
    }

    #[test]
    fn unknown_key_is_null_not_a_crash() {
        let k = CString::new("definitely_not_canonical").unwrap();
        assert!(unsafe { molrs_schema_column_dtype(k.as_ptr()) }.is_null());
        assert!(unsafe { molrs_schema_column_dtype(std::ptr::null()) }.is_null());
    }

    #[test]
    fn block_membership_matches_the_table() {
        for spec in schema::SCHEMA_BLOCKS {
            let n = CString::new(spec.name).unwrap();
            assert!(unsafe { molrs_schema_has_block(n.as_ptr()) }, "{}", spec.name);
        }
        let n = CString::new("not_a_block").unwrap();
        assert!(!unsafe { molrs_schema_has_block(n.as_ptr()) });
    }
}
