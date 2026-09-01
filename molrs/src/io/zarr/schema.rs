//! Runtime validation of the mrec record schema.
//!
//! The language-neutral JSON Schema is published by molrec
//! (`schema/core/record.schema.json`). This module is the executable form
//! molrs runs on every `*.mrec` door: path suffix, the `meta` version key,
//! and the reserved names. Re-exported as [`crate::io::mrec::schema`].

use serde_json::{Map as JsonMap, Value as JsonValue};

use molrs::MolRsError;
use molrs::store::frame::Frame;
use molrs::store::record::MOLREC_VERSION as RECORD_MOLREC_VERSION;
use molrs::store::trajectory::Trajectory;

/// Sole version key of a record, stamped into `meta.molrec_version`.
pub const MOLREC_VERSION: u64 = RECORD_MOLREC_VERSION;

/// Reserved `meta` keys owned by the contract, not the producer.
pub const RESERVED_META_KEYS: [&str; 1] = ["molrec_version"];

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

/// Validate the mandatory `meta` version key against the mrec contract.
///
/// `molrec_version` must be present and in `1..=`[`MOLREC_VERSION`]. It is
/// the sole version key of a record; identity is this key plus the `*.mrec/`
/// path suffix.
pub fn validate_meta(attrs: &JsonMap<String, JsonValue>) -> Result<(), MolRsError> {
    let version = attrs
        .get("molrec_version")
        .and_then(JsonValue::as_u64)
        .ok_or_else(|| MolRsError::zarr("meta is missing 'molrec_version'"))?;
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

/// Judge a snapshot or system-definition frame against the Frame vocabulary.
pub fn validate_frame(frame: &Frame) -> Result<(), MolRsError> {
    frame.validate()
}

/// Judge a trajectory's axes, then each frame against the Frame vocabulary.
pub fn validate_trajectory(trajectory: &Trajectory) -> Result<(), MolRsError> {
    trajectory.validate()?;
    for frame in &trajectory.frames {
        frame.validate()?;
    }
    Ok(())
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

    #[test]
    fn missing_molrec_version_is_refused() {
        let mut attrs = meta(1);
        attrs.remove("molrec_version");
        let err = validate_meta(&attrs).unwrap_err().to_string();
        assert!(err.contains("molrec_version"), "{err}");
    }

    /// A store from the retired `format_name`/`record_schema_version` era
    /// carries neither key the contract now names, and is refused for the
    /// missing `molrec_version`, not for the keys it does carry.
    #[test]
    fn retired_brand_keys_do_not_identify_a_record() {
        let attrs = json!({
            "format_name": "mrec",
            "record_schema_version": 1,
        })
        .as_object()
        .cloned()
        .unwrap();
        let err = validate_meta(&attrs).unwrap_err().to_string();
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
