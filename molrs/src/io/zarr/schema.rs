//! Runtime validation of the mrec record schema.
//!
//! The language-neutral JSON Schema is published by molrec
//! (`schema/core/record.schema.json`). This module is the executable form
//! molrs runs on every `*.mrec` door: path suffix, the `meta` version key,
//! and the reserved names. Writers always stamp the current `molrec_version`;
//! readers validate it when present and read a version-1 store (or one
//! without the key) through the reader's version-1 conversion. Re-exported as
//! [`crate::io::mrec::schema`].

use serde_json::{Map as JsonMap, Value as JsonValue};

use molrs::core::Frame;
use molrs::core::MolRsError;
use molrs::io::mrec::MOLREC_VERSION;

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
/// `molrec_version` is **optional on read**: an absent key is a store written
/// before version 1, which opens ([`read_version`] says how it is read). A
/// present key must be an integer in `1..=`[`MOLREC_VERSION`]; anything else
/// (`0`, a newer version, a string, `null`, a float) is refused. Identity of a
/// record is the `*.mrec/` path suffix plus a Zarr root, not this key; the key
/// says which contract wrote it.
pub fn validate_meta(attrs: &JsonMap<String, JsonValue>) -> Result<(), MolRsError> {
    read_version(attrs).map(|_| ())
}

/// The contract version a store's sections are read under: its validated
/// `molrec_version`, or `1` when the key is absent — a store from before
/// version 1 is read by version 1's rules (best effort), never as the current
/// version.
///
/// # Errors
///
/// A [`MolRsError::Zarr`] for a key [`validate_meta`] refuses.
pub fn read_version(attrs: &JsonMap<String, JsonValue>) -> Result<u64, MolRsError> {
    let Some(value) = attrs.get("molrec_version") else {
        return Ok(1);
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
    Ok(version)
}

/// The producer's `meta` with the current `molrec_version` stamped in.
///
/// Every record molrs writes is written in the current contract, so it
/// carries [`MOLREC_VERSION`] whatever the producer's map says: the key is
/// reserved, and a record read from a version-1 store is converted on read,
/// so writing it back writes the current version.
///
/// Shared by the whole-record writer and the streaming one.
pub(crate) fn stamped_meta(meta: &JsonMap<String, JsonValue>) -> JsonMap<String, JsonValue> {
    let mut stamped = meta.clone();
    stamped.insert(
        "molrec_version".to_string(),
        JsonValue::from(MOLREC_VERSION),
    );
    stamped
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
    fn every_supported_molrec_version_passes() {
        for version in 1..=MOLREC_VERSION {
            assert_eq!(read_version(&meta(version)).unwrap(), version);
        }
    }

    /// An absent version opens, and is read under version 1's rules: a
    /// foreign store, or one written before molrs stamped the key.
    #[test]
    fn missing_molrec_version_is_accepted_and_read_as_version_one() {
        let attrs = json!({ "producer": "test" }).as_object().cloned().unwrap();
        validate_meta(&attrs).unwrap();
        assert_eq!(read_version(&attrs).unwrap(), 1);
        assert_eq!(read_version(&JsonMap::new()).unwrap(), 1);
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

    /// The writer stamps the current version over whatever the producer's
    /// map says, and keeps every other key.
    #[test]
    fn stamped_meta_writes_the_current_version() {
        let stamped = stamped_meta(&JsonMap::new());
        assert_eq!(stamped["molrec_version"].as_u64(), Some(MOLREC_VERSION));
        for claimed in [1, MOLREC_VERSION + 1] {
            let mut own = meta(claimed);
            own.insert("producer".into(), "test".into());
            let stamped = stamped_meta(&own);
            assert_eq!(stamped["molrec_version"].as_u64(), Some(MOLREC_VERSION));
            assert_eq!(stamped["producer"], "test");
        }
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
        let err = validate_meta(&meta(MOLREC_VERSION + 1))
            .unwrap_err()
            .to_string();
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
