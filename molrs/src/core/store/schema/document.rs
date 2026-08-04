//! The vocabulary as an inspectable, serializable document.
//!
//! The compile-time tables are the source of truth; this is an owned projection
//! of them. It exists so the schema is something a user can *look at* — print
//! it, publish it, diff two releases of it — from every binding, rather than a
//! rule that only exists inside the Rust type system.

use super::{FRAME_VOCAB_VERSION, SCHEMA_BLOCKS, SCHEMA_COLUMNS};
use serde::{Deserialize, Serialize};

/// A canonical column, as data.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ColumnDoc {
    /// Canonical key.
    pub key: String,
    /// Constant name (`ATOMI`) exported by the language bindings.
    pub const_name: String,
    /// Storage dtype: `float` | `int` | `uint` | `bool` | `u8` | `string`.
    pub dtype: String,
    /// `scalar`, or `vec(n)`.
    pub shape: String,
    /// Unit symbol, or empty for dimensionless / unit-free.
    pub unit: String,
    /// One-line meaning.
    pub doc: String,
}

/// A canonical block, as data.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct BlockDoc {
    /// Canonical block name.
    pub name: String,
    /// `node`, `relation(k)`, or `grid`.
    pub row_kind: String,
    /// Block the endpoints index into, for relation blocks.
    pub endpoint_target: Option<String>,
    /// Endpoint column keys, in position order.
    pub endpoint_columns: Vec<String>,
    /// Columns that must be present.
    pub required: Vec<String>,
    /// Conventional but optional columns.
    pub optional: Vec<String>,
    /// Whether columns outside the vocabulary are admissible here.
    pub open: bool,
    /// One-line meaning.
    pub doc: String,
}

/// The whole vocabulary, owned and serializable.
///
/// Two runs produce byte-identical JSON — the tables are sorted and the
/// document preserves that order — so `diff`ing the artifact across releases
/// shows exactly what changed about the contract.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SchemaDocument {
    /// Stable identity of this schema.
    pub id: String,
    /// [`FRAME_VOCAB_VERSION`] — what the names and dtypes mean.
    pub vocab_version: u32,
    /// Every canonical column.
    pub columns: Vec<ColumnDoc>,
    /// Every canonical block.
    pub blocks: Vec<BlockDoc>,
}

/// Borrow the compile-time tables into an owned document.
pub fn document() -> SchemaDocument {
    SchemaDocument {
        id: "https://molcrafts.org/schema/frame/v1".to_string(),
        vocab_version: FRAME_VOCAB_VERSION,
        columns: SCHEMA_COLUMNS
            .iter()
            .map(|c| ColumnDoc {
                key: c.key.to_string(),
                const_name: c.const_name.to_string(),
                dtype: c.dtype.name().to_string(),
                shape: c.shape.to_string(),
                unit: c.unit.to_string(),
                doc: c.doc.to_string(),
            })
            .collect(),
        blocks: SCHEMA_BLOCKS
            .iter()
            .map(|b| BlockDoc {
                name: b.name.to_string(),
                row_kind: b.row_kind.to_string(),
                endpoint_target: b.endpoints.map(|e| e.target.to_string()),
                endpoint_columns: b.endpoint_columns().iter().map(|s| s.to_string()).collect(),
                required: b.required.iter().map(|s| s.to_string()).collect(),
                optional: b.optional.iter().map(|s| s.to_string()).collect(),
                open: b.open,
                doc: b.doc.to_string(),
            })
            .collect(),
    }
}

impl SchemaDocument {
    /// Canonical JSON — stable across runs, so two releases can be diffed.
    pub fn to_json(&self) -> String {
        serde_json::to_string_pretty(self).expect("schema document is always serializable")
    }

    /// The published Markdown tables.
    pub fn to_markdown(&self) -> String {
        let mut out = String::new();
        out.push_str(&format!(
            "# Frame schema (vocabulary v{})\n\n`{}`\n\n## Columns\n\n",
            self.vocab_version, self.id
        ));
        out.push_str("| key | dtype | shape | unit | meaning |\n|---|---|---|---|---|\n");
        for c in &self.columns {
            let unit = if c.unit.is_empty() { "—" } else { &c.unit };
            out.push_str(&format!(
                "| `{}` | {} | {} | {} | {} |\n",
                c.key, c.dtype, c.shape, unit, c.doc
            ));
        }
        out.push_str("\n## Blocks\n\n");
        out.push_str(
            "| block | rows | endpoints → | required | meaning |\n|---|---|---|---|---|\n",
        );
        for b in &self.blocks {
            let ep = match &b.endpoint_target {
                Some(t) => format!("`{}` → `{}`", b.endpoint_columns.join("`, `"), t),
                None => "—".to_string(),
            };
            let req = if b.required.is_empty() {
                "—".to_string()
            } else {
                format!("`{}`", b.required.join("`, `"))
            };
            out.push_str(&format!(
                "| `{}` | {} | {} | {} | {} |\n",
                b.name, b.row_kind, ep, req, b.doc
            ));
        }
        out
    }
}

impl std::fmt::Display for SchemaDocument {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.to_markdown())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn document_covers_every_table_entry() {
        let d = document();
        assert_eq!(d.columns.len(), SCHEMA_COLUMNS.len());
        assert_eq!(d.blocks.len(), SCHEMA_BLOCKS.len());
        assert_eq!(d.vocab_version, FRAME_VOCAB_VERSION);
    }

    #[test]
    fn json_round_trips_losslessly() {
        // The published artifact is what downstream tooling and the other
        // bindings read; if it cannot come back unchanged it is not a contract.
        let d = document();
        let back: SchemaDocument = serde_json::from_str(&d.to_json()).expect("valid json");
        assert_eq!(d, back);
    }

    #[test]
    fn json_is_stable_across_runs() {
        assert_eq!(document().to_json(), document().to_json());
    }

    #[test]
    fn markdown_names_every_column_and_block() {
        let md = document().to_markdown();
        for c in SCHEMA_COLUMNS {
            assert!(md.contains(&format!("`{}`", c.key)), "{} missing", c.key);
        }
        for b in SCHEMA_BLOCKS {
            assert!(md.contains(&format!("`{}`", b.name)), "{} missing", b.name);
        }
    }
}
