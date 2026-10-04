//! Canonical molecular field-name constants.
//!
//! Column keys are a re-export of [`crate::store::schema::consts`]. The schema
//! is the source of truth: a key is declared once, in the column table, and
//! the constant is emitted from that same declaration.
//!
//! A key's dtype is [`crate::store::schema::column()`]'s answer, never a second
//! table here.
//!
//! Frame meta keys name `frame.meta` entries, not block columns, so they are
//! declared once below and are not part of the column schema.
//!
//! # Examples
//!
//! ```
//! use molrs::store::keys;
//!
//! assert_eq!(keys::X, "x");
//! assert_eq!(keys::COORDS, [keys::X, keys::Y, keys::Z]);
//! assert_eq!(keys::UNITS, "units");
//! ```

pub use crate::store::schema::consts::*;

use crate::store::schema::document::{KeysDocument, NamedGroup, NamedValue};
use crate::store::schema::{self, NamedConst};

macro_rules! meta_keys {
    ($( $(#[$meta:meta])* pub const $name:ident: &str = $value:literal; )*) => {
        $( $(#[$meta])* pub const $name: &str = $value; )*

        /// Frame-meta keys. Not columns — [`SCHEMA_COLUMNS`](crate::store::schema::SCHEMA_COLUMNS)
        /// does not list them, and the bindings project this slice rather than
        /// spelling the strings again.
        pub static META_KEYS: &[NamedConst] = &[
            $(
                NamedConst { const_name: stringify!($name), value: $name },
            )*
        ];
    };
}

meta_keys! {
    /// Frame meta key: the atom-type inventory, packed as `"1:C,2:H"`.
    ///
    /// Declares types no row needs to use; read by
    /// [`TypeLabels`](crate::core::store::type_labels::TypeLabels).
    pub const ATOM_TYPE_LABELS: &str = "atom_type_labels";
    /// Frame meta key: the bond-type inventory, packed as `"1:c3-h1,2:c3-c3"`.
    pub const BOND_TYPE_LABELS: &str = "bond_type_labels";
    /// Frame meta key: the angle-type inventory, packed as `"id:label,…"`.
    pub const ANGLE_TYPE_LABELS: &str = "angle_type_labels";
    /// Frame meta key: the dihedral-type inventory, packed as `"id:label,…"`.
    pub const DIHEDRAL_TYPE_LABELS: &str = "dihedral_type_labels";
    /// Frame meta key: the improper-type inventory, packed as `"id:label,…"`.
    pub const IMPROPER_TYPE_LABELS: &str = "improper_type_labels";
    /// Frame meta key: the unit system the frame's numbers are in, as the
    /// force-field `units` object — `{"preset": "real"}`, or quantities such
    /// as `{"length": "nm", "energy": "kJ/mol"}` (molrec `conventions.md`,
    /// "Units on a frame"). A bare string reads as `{"preset": <string>}`
    /// ([`units_preset`]). Absent means the file or
    /// caller stated none.
    ///
    /// The LAMMPS molecule-JSON reader writes it and its writer emits it back
    /// (`io::data::lammps_molecule`); molrs converts no frame between presets.
    pub const UNITS: &str = "units";
}

/// The preset a frame's [`UNITS`] meta value names: the
/// `preset` of a units object, or a bare preset string (the form molrs wrote
/// before the object). `None` for a value that names no preset.
pub fn units_preset(value: &crate::store::meta::MetaValue) -> Option<&str> {
    use crate::store::meta::MetaValue;
    match value {
        MetaValue::String(preset) => Some(preset),
        MetaValue::Json(serde_json::Value::Object(object)) => {
            object.get("preset").and_then(serde_json::Value::as_str)
        }
        _ => None,
    }
}

/// Column keys, groups, block names, and frame-meta keys, from the tables.
pub fn keys_document() -> KeysDocument {
    KeysDocument {
        columns: schema::SCHEMA_COLUMNS
            .iter()
            .map(|c| NamedValue {
                const_name: c.const_name.to_string(),
                value: c.key.to_string(),
            })
            .collect(),
        groups: schema::KEY_GROUPS
            .iter()
            .map(|g| NamedGroup {
                const_name: g.const_name.to_string(),
                values: g.keys.iter().map(|k| (*k).to_string()).collect(),
            })
            .collect(),
        blocks: schema::BLOCK_NAMES
            .iter()
            .map(|spec| NamedValue {
                const_name: spec.const_name.to_string(),
                value: spec.value.to_string(),
            })
            .collect(),
        block_groups: schema::BLOCK_GROUPS
            .iter()
            .map(|g| NamedGroup {
                const_name: g.const_name.to_string(),
                values: g.keys.iter().map(|k| (*k).to_string()).collect(),
            })
            .collect(),
        meta: META_KEYS
            .iter()
            .map(|spec| NamedValue {
                const_name: spec.const_name.to_string(),
                value: spec.value.to_string(),
            })
            .collect(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn keys_document_covers_every_declared_name() {
        let doc = keys_document();
        assert_eq!(doc.columns.len(), schema::SCHEMA_COLUMNS.len());
        for (got, spec) in doc.columns.iter().zip(schema::SCHEMA_COLUMNS) {
            assert_eq!(got.const_name, spec.const_name);
            assert_eq!(got.value, spec.key);
        }
        assert_eq!(doc.groups.len(), schema::KEY_GROUPS.len());
        for (got, group) in doc.groups.iter().zip(schema::KEY_GROUPS) {
            assert_eq!(got.const_name, group.const_name);
            assert_eq!(
                got.values.iter().map(String::as_str).collect::<Vec<_>>(),
                group.keys
            );
        }
        assert_eq!(doc.blocks.len(), schema::BLOCK_NAMES.len());
        for (got, spec) in doc.blocks.iter().zip(schema::BLOCK_NAMES) {
            assert_eq!(got.const_name, spec.const_name);
            assert_eq!(got.value, spec.value);
        }
        assert_eq!(doc.block_groups.len(), schema::BLOCK_GROUPS.len());
        for (got, group) in doc.block_groups.iter().zip(schema::BLOCK_GROUPS) {
            assert_eq!(got.const_name, group.const_name);
            assert_eq!(
                got.values.iter().map(String::as_str).collect::<Vec<_>>(),
                group.keys
            );
        }
        assert_eq!(doc.meta.len(), META_KEYS.len());
        for (got, spec) in doc.meta.iter().zip(META_KEYS) {
            assert_eq!(got.const_name, spec.const_name);
            assert_eq!(got.value, spec.value);
            assert_eq!(got.value, spec.const_name.to_ascii_lowercase());
        }
        let columns: std::collections::HashSet<_> =
            doc.columns.iter().map(|c| c.value.as_str()).collect();
        for spec in &doc.meta {
            assert!(
                !columns.contains(spec.value.as_str()),
                "{} is a column, not frame meta",
                spec.value
            );
        }
    }
}
