//! Extensions to `MolRec` that require force-field knowledge.
//!
//! These are free functions rather than methods because `MolRec` is defined
//! in `molrs-core` and cannot take a `ForceField` dep without creating a
//! circular dependency.

use molrs::molrec::MolRec;
use molrs::frame::Frame;
use serde_json::{Value as JsonValue, json};

use crate::forcefield::ForceField;

/// Build a `MolRec` whose method metadata is populated from a force-field definition.
pub fn molrec_from_forcefield(frame: Frame, forcefield: &ForceField) -> MolRec {
    let mut rec = MolRec::new(frame);
    set_forcefield_metadata(&mut rec, forcefield);
    rec
}

/// Populate method metadata on an existing `MolRec` from a force-field definition.
pub fn set_forcefield_metadata(rec: &mut MolRec, ff: &ForceField) {
    let styles: Vec<JsonValue> = ff
        .styles()
        .iter()
        .map(|style| {
            json!({
                "category": style.category(),
                "name": style.name,
            })
        })
        .collect();
    rec.method = json!({
        "type": "classical",
        "description": "Force-field-derived molecular record",
        "classical": {
            "force_field": {
                "name": ff.name,
                "styles": styles,
            }
        }
    });
}
