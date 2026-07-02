//! Force-field method metadata as a plain JSON blob.
//!
//! A pure function (not a method) because it needs a `ForceField` dep, which
//! core types must not take. Callers attach the returned JSON wherever they
//! keep provenance (e.g. a `Frame`'s metadata) — there is no record aggregate.

use serde_json::{Value as JsonValue, json};

use crate::ff::forcefield::ForceField;

/// Build the `method` metadata JSON for a force-field definition.
pub fn forcefield_method_json(ff: &ForceField) -> JsonValue {
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
    json!({
        "type": "classical",
        "description": "Force-field-derived molecular record",
        "classical": {
            "force_field": {
                "name": ff.name,
                "styles": styles,
            }
        }
    })
}
