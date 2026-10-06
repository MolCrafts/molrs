//! OPLS-AA typing metadata: per-type SMARTS definition, overrides, explicit
//! priority and overlay layer.
//!
//! This is the typing-metadata half of an OPLS-AA force field, kept separate
//! from the potential parameters (mirroring
//! [`MMFFParams`](crate::ff::typifier::mmff::MMFFParams) versus the
//! [`ForceField`](crate::ff::forcefield::ForceField)). The shipped table is
//! joined from the molrs-owned rules of
//! [`crate::ff::params::oplsaa_typing`]; for a caller's own OPLS / CL&P XML, the
//! potential reader
//! ([`OplsXmlReader`](crate::ff::forcefield::readers::opls::OplsXmlReader))
//! drops the `def` / `overrides` / `priority` / `layer` attributes and
//! `read_typing_xml_str`
//! reads them into the [`OplsTypingMeta`] table here.
//!
//! # How the fields rank candidates
//!
//! The table carries the inputs; the
//! [`LayeredTypingEngine`](super::layered::LayeredTypingEngine) ranks with
//! them. `layer` and `overrides` define a pairwise *dominance*: a type on a
//! higher layer, or on the same layer and overriding another (directly or
//! transitively), always wins over it. `priority` (absent = 0) only orders
//! candidates that nothing dominates. No field is folded into a single score.
//!
//! # Scope
//!
//! Only types carrying a SMARTS `def` participate in automatic SMARTS typing.
//! Rows with no `def` (the united-atom `opls_001`–`opls_134` block, for one)
//! can only be assigned by hand or read back from a LAMMPS data file.

use std::collections::HashMap;

use crate::ff::forcefield::xml::{attr_str, forcefield_root};

/// One `<Type>` row of an OPLS-AA `<AtomTypes>` section, holding the typing
/// metadata (not the potential parameters).
///
/// This is the **runtime** typing record. Its static, compile-time counterpart
/// is [`OplsRuleRow`](crate::ff::params::OplsRuleRow): the shipped OPLS-AA
/// typifier builds one `OplsTypeRow` from each `OplsRuleRow`, taking `class`
/// from the matching [`OplsAtomRow`](crate::ff::params::OplsAtomRow) and
/// `layer` 0, while
/// `read_typing_xml_str`
/// builds them from XML attributes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OplsTypeRow {
    /// Chemical class (the `class` attribute, e.g. `"CT"`). Bonded forces key on
    /// this class vocabulary, distinct from the `opls_NNN` type vocabulary.
    pub class: String,
    /// SMARTS definition (the `def` attribute), or `None` for legacy rows that
    /// carry no `def` and therefore cannot be matched automatically.
    pub def: Option<String>,
    /// Type names this row overrides (parsed from a comma-separated `overrides`
    /// attribute); empty when absent.
    pub overrides: Vec<String>,
    /// Explicit `priority` attribute, if present (absent reads as 0). It orders
    /// only candidates that no other candidate dominates.
    pub priority: Option<i64>,
    /// Overlay layer (the `layer` attribute); `0` (base force field) when absent.
    pub layer: u32,
}

/// Parsed OPLS-AA typing metadata: [`OplsTypeRow`]s keyed by `opls_NNN` type
/// name.
///
/// Read from the same XML as the potential [`ForceField`](crate::ff::forcefield::ForceField)
/// but kept separate — this table drives SMARTS atom typing, the `ForceField`
/// drives energy evaluation.
#[derive(Debug, Clone, Default)]
pub struct OplsTypingMeta {
    rows: HashMap<String, OplsTypeRow>,
}

impl OplsTypingMeta {
    /// Create an empty metadata table.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create from pre-parsed rows keyed by `opls_NNN`.
    pub fn from_rows(rows: HashMap<String, OplsTypeRow>) -> Self {
        Self { rows }
    }

    /// Insert or replace a row.
    pub fn insert(&mut self, name: impl Into<String>, row: OplsTypeRow) {
        self.rows.insert(name.into(), row);
    }

    /// Look up a row by `opls_NNN` name.
    pub fn get(&self, name: &str) -> Option<&OplsTypeRow> {
        self.rows.get(name)
    }

    /// Number of rows.
    pub fn len(&self) -> usize {
        self.rows.len()
    }

    /// Whether the table is empty.
    pub fn is_empty(&self) -> bool {
        self.rows.is_empty()
    }

    /// Iterate `(name, row)` pairs.
    pub fn iter(&self) -> impl Iterator<Item = (&String, &OplsTypeRow)> {
        self.rows.iter()
    }
}

/// Parse OPLS-AA typing metadata ([`OplsTypingMeta`]) from an XML string.
///
/// Reads each `<Type>` of the `<AtomTypes>` section, transcribing the typing
/// attributes the potential reader
/// ([`OplsXmlReader`](crate::ff::forcefield::readers::opls::OplsXmlReader))
/// drops: `class`, `def` (SMARTS), `overrides` (comma list), `priority`, and
/// `layer`. This is purely additive — it never touches the potential
/// `ForceField`; the two are read from the same XML but kept separate (as MMFF's
/// typing metadata is).
///
/// Rows with no `def` are still recorded (with `def = None`); they are legacy
/// types excluded from automatic SMARTS typing.
///
/// # Errors
///
/// Returns `Err` if the root element is not `<ForceField>`, a `<Type>` lacks the
/// required `name`/`class`, or `priority`/`layer` is present but non-integer.
pub(crate) fn read_typing_xml_str(xml: &str) -> Result<OplsTypingMeta, String> {
    let doc = roxmltree::Document::parse(xml).map_err(|e| format!("XML parse error: {}", e))?;

    let root = forcefield_root(&doc)?;

    let mut meta = OplsTypingMeta::new();

    for child in root.children().filter(|n| n.is_element()) {
        if child.tag_name().name() != "AtomTypes" {
            continue;
        }
        for t in child
            .children()
            .filter(|n| n.is_element() && n.tag_name().name() == "Type")
        {
            let name = attr_str(&t, "name")?.to_owned();
            let class = attr_str(&t, "class")?.to_owned();
            let def = t.attribute("def").map(str::to_owned);
            let overrides = t
                .attribute("overrides")
                .map(parse_overrides)
                .unwrap_or_default();
            let priority = match t.attribute("priority") {
                None => None,
                Some(s) => Some(s.parse::<i64>().map_err(|_| {
                    format!(
                        "<Type name={:?}> attribute 'priority' is not an integer: {:?}",
                        name, s
                    )
                })?),
            };
            let layer = match t.attribute("layer") {
                None => 0,
                Some(s) => s.parse::<u32>().map_err(|_| {
                    format!(
                        "<Type name={:?}> attribute 'layer' is not an integer: {:?}",
                        name, s
                    )
                })?,
            };
            meta.insert(
                name,
                OplsTypeRow {
                    class,
                    def,
                    overrides,
                    priority,
                    layer,
                },
            );
        }
    }

    Ok(meta)
}

/// Split a comma-separated `overrides` attribute into trimmed, non-empty names.
fn parse_overrides(raw: &str) -> Vec<String> {
    raw.split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(str::to_owned)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- OPLS typing metadata reader (the inline-fixture parse is
    //     `test_opls_typing_overrides_and_layer_defaults`; the rest are edge
    //     cases) ------------------------------------------------------------

    #[test]
    fn test_opls_typing_overrides_and_layer_defaults() {
        // A modern row with overrides + a legacy row with no `def`. (Inline
        // edge fixture: exercises overrides splitting + def=None + default layer.)
        let xml = r#"<ForceField name="OPLS-AA">
          <AtomTypes>
            <Type name="opls_135" class="CT" element="C" mass="12.011" def="[C;X4](C)(H)(H)H"/>
            <Type name="opls_146" class="HA" element="H" mass="1.008" def="[H][c]" overrides="opls_144, opls_140"/>
            <Type name="opls_001" class="opls_001" element="C" mass="12.011"/>
          </AtomTypes>
        </ForceField>"#;
        let meta = read_typing_xml_str(xml).unwrap();

        let r135 = meta.get("opls_135").unwrap();
        assert_eq!(r135.class, "CT");
        assert_eq!(r135.def.as_deref(), Some("[C;X4](C)(H)(H)H"));
        assert!(r135.overrides.is_empty());
        assert_eq!(r135.layer, 0);
        assert_eq!(r135.priority, None);

        let r146 = meta.get("opls_146").unwrap();
        assert_eq!(
            r146.overrides,
            vec!["opls_144".to_string(), "opls_140".to_string()]
        );

        // Legacy row: no def.
        let r001 = meta.get("opls_001").unwrap();
        assert_eq!(r001.def, None);
    }

    #[test]
    fn test_opls_typing_explicit_priority_and_layer() {
        let xml = r#"<ForceField name="OPLS-AA">
          <AtomTypes>
            <Type name="opls_x" class="CT" def="[C]" priority="7" layer="2"/>
          </AtomTypes>
        </ForceField>"#;
        let meta = read_typing_xml_str(xml).unwrap();
        let r = meta.get("opls_x").unwrap();
        assert_eq!(r.priority, Some(7));
        assert_eq!(r.layer, 2);
    }

    #[test]
    fn test_opls_typing_missing_class_errors() {
        let xml = r#"<ForceField name="OPLS-AA">
          <AtomTypes><Type name="opls_135" def="[C]"/></AtomTypes>
        </ForceField>"#;
        let err = read_typing_xml_str(xml).unwrap_err();
        assert!(err.contains("class"), "err: {err}");
    }

    #[test]
    fn test_opls_typing_bad_priority_errors() {
        let xml = r#"<ForceField name="OPLS-AA">
          <AtomTypes><Type name="opls_135" class="CT" priority="high"/></AtomTypes>
        </ForceField>"#;
        let err = read_typing_xml_str(xml).unwrap_err();
        assert!(err.contains("priority"), "err: {err}");
    }

    #[test]
    fn test_opls_typing_wrong_root_errors() {
        let err = read_typing_xml_str(r#"<System name="x"/>"#).unwrap_err();
        assert!(err.contains("ForceField"), "err: {err}");
    }
}
