//! Runtime validation of the mrec record schema.
//!
//! The language-neutral JSON Schema is published by molrec
//! (`schema/core/record.schema.json`). This module is the executable form
//! molrs runs on every `*.mrec` door: path suffix, the `meta` version key,
//! and the reserved names. Writers always stamp `molrec_version`; readers
//! validate it only when it is present. Re-exported as [`crate::io::mrec::schema`].

use serde_json::{Map as JsonMap, Value as JsonValue};

use molrs::MolRsError;
use molrs::store::frame::Frame;
pub use molrs::store::record::{MOLREC_VERSION, RESERVED_META_KEYS};

/// Refuse the retired scientific path brand `.zarr` / `.zarr.zip`.
///
/// Other names are accepted; the conventional suffix is `.mrec`.
#[cfg(feature = "filesystem")]
pub fn validate_path(path: &std::path::Path) -> Result<(), MolRsError> {
    let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
    if name.ends_with(".zarr") || name.ends_with(".zarr.zip") {
        return Err(MolRsError::zarr(format!(
            "{} uses the retired .zarr path; scientific record path is *.mrec",
            path.display()
        )));
    }
    Ok(())
}

/// Validate the `meta` version key against the mrec contract.
///
/// `molrec_version` is **optional on read**: an absent key performs no
/// version check, so a foreign store — or one written before molrs stamped the
/// key — opens. A present key must be an integer in `1..=`[`MOLREC_VERSION`];
/// anything else (`0`, a newer version, a string, `null`, a float) is refused.
/// Identity of a record is the `*.mrec/` path suffix plus a Zarr root, not this
/// key; the key says which contract wrote it.
pub fn validate_meta(attrs: &JsonMap<String, JsonValue>) -> Result<(), MolRsError> {
    let Some(value) = attrs.get("molrec_version") else {
        return Ok(());
    };
    let version = value.as_u64().ok_or_else(|| {
        MolRsError::zarr(format!(
            "molrec_version must be a positive integer, found {value}"
        ))
    })?;
    // Accept any version this reader is new enough to understand (`1..=N`) and
    // reject only a *newer* one; a hard `!= N` gate would turn every future
    // version bump into a mutual hard fork with already-written stores.
    if version == 0 || version > MOLREC_VERSION {
        return Err(MolRsError::zarr(format!(
            "unsupported molrec_version {version}; this reader supports 1..={MOLREC_VERSION}"
        )));
    }
    Ok(())
}

/// The producer's `meta` with `molrec_version` stamped in, validated.
///
/// Every record molrs writes carries the version it was written at, so a
/// reader that checks it never has to guess. A producer that set it keeps its
/// value, which is how a writer for an older contract stays expressible.
///
/// Shared by the whole-record writer and the streaming one.
pub(crate) fn stamped_meta(
    meta: &JsonMap<String, JsonValue>,
) -> Result<JsonMap<String, JsonValue>, MolRsError> {
    let mut stamped = meta.clone();
    stamped
        .entry("molrec_version".to_string())
        .or_insert_with(|| JsonValue::from(MOLREC_VERSION));
    validate_meta(&stamped)?;
    Ok(stamped)
}

/// Judge a snapshot or system-definition frame against the Frame vocabulary.
pub fn validate_frame(frame: &Frame) -> Result<(), MolRsError> {
    frame.validate()
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn meta(version: u64) -> JsonMap<String, JsonValue> {
        json!({ "molrec_version": version })
            .as_object()
            .cloned()
            .unwrap()
    }

    #[test]
    fn molrec_version_one_passes() {
        validate_meta(&meta(1)).unwrap();
    }

    /// An absent version is no version check: a foreign store, or one written
    /// before molrs stamped the key, opens.
    #[test]
    fn missing_molrec_version_is_accepted() {
        let attrs = json!({ "producer": "test" }).as_object().cloned().unwrap();
        validate_meta(&attrs).unwrap();
        validate_meta(&JsonMap::new()).unwrap();
    }

    /// The retired `format_name`/`record_schema_version` keys are neither
    /// checked nor honoured: they do not stand in for `molrec_version`, and a
    /// bogus value under them does not fail the read.
    #[test]
    fn retired_brand_keys_are_neither_checked_nor_honoured() {
        let attrs = json!({
            "format_name": "mrec",
            "record_schema_version": 99,
        })
        .as_object()
        .cloned()
        .unwrap();
        validate_meta(&attrs).unwrap();
    }

    /// `null` is not a version: present means validated.
    #[test]
    fn a_null_molrec_version_is_refused() {
        let attrs = json!({ "molrec_version": null })
            .as_object()
            .cloned()
            .unwrap();
        let err = validate_meta(&attrs).unwrap_err().to_string();
        assert!(err.contains("molrec_version"), "{err}");
    }

    #[test]
    fn a_float_molrec_version_is_refused() {
        let attrs = json!({ "molrec_version": 1.0 })
            .as_object()
            .cloned()
            .unwrap();
        let err = validate_meta(&attrs).unwrap_err().to_string();
        assert!(err.contains("molrec_version"), "{err}");
    }

    /// The writer stamps the current version, and keeps a producer's own.
    #[test]
    fn stamped_meta_adds_the_version_and_keeps_a_producer_one() {
        let stamped = stamped_meta(&JsonMap::new()).unwrap();
        assert_eq!(stamped["molrec_version"].as_u64(), Some(MOLREC_VERSION));
        let mut own = meta(1);
        own.insert("producer".into(), "test".into());
        assert_eq!(stamped_meta(&own).unwrap(), own);
        assert!(stamped_meta(&meta(2)).is_err());
    }

    #[test]
    fn a_non_integer_molrec_version_is_refused() {
        let attrs = json!({ "molrec_version": "1" })
            .as_object()
            .cloned()
            .unwrap();
        let err = validate_meta(&attrs).unwrap_err().to_string();
        assert!(err.contains("molrec_version"), "{err}");
    }

    #[test]
    fn molrec_version_zero_is_refused() {
        let err = validate_meta(&meta(0)).unwrap_err().to_string();
        assert!(err.contains("molrec_version"), "{err}");
    }

    #[test]
    fn newer_molrec_version_is_refused() {
        let err = validate_meta(&meta(2)).unwrap_err().to_string();
        assert!(err.contains("molrec_version"), "{err}");
    }

    #[cfg(feature = "filesystem")]
    #[test]
    fn retired_zarr_suffix_is_refused() {
        let err = validate_path(std::path::Path::new("water.zarr"))
            .unwrap_err()
            .to_string();
        assert!(err.contains(".mrec"), "{err}");
    }

    #[test]
    fn empty_frame_passes_vocabulary() {
        validate_frame(&Frame::new()).unwrap();
    }
}
