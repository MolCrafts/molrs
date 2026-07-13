//! chem-perceive-10 ac-002 — the last runtime text parse on the force-field path.
//!
//! `ff/typifier/estimate/tables.rs` still `include_str!`s two JSON assets and
//! `serde_json::from_str`s them at runtime — `gaff_equiv.json` (6159 lines: the
//! parmchk2 equivalence / substitution table, its penalty weights and its defaults)
//! and `gaff_empirical.json` (87 lines: the Badger bond-`k` and GAFF angle `Z`/`C`
//! constants). Every other parameter table in molrs is compiled Rust data
//! (`ff::params`); these two are the last text molrs parses to build a force field,
//! and ac-002 takes them to zero.
//!
//! Three gates, because there are three separate ways the parse survives:
//!
//! 1. **No JSON deserializer under `ff/`** — the parse itself. Scoped to the whole
//!    force-field tree, not just the typifier: the criterion says *last*, and a
//!    parse that moved one directory up would satisfy a narrower gate while leaving
//!    the property false.
//! 2. **No embedded table text in `estimate/tables.rs`** — the *asset*. A module
//!    that still carries the JSON as a `&'static str` has not been converted; it has
//!    at best moved the parse behind a helper the first gate cannot see.
//! 3. **No `.expect()` in `estimate/tables.rs`** — the fail-fast violation the
//!    parse dragged in with it. `EquivTable::load()` currently *panics* on malformed
//!    embedded data, on a data path, at runtime. Once the table is Rust data the
//!    question cannot arise — a malformed table is a compile error — so the absence
//!    of the panic is the proof the conversion actually happened, rather than a
//!    `from_str` being swapped for a `from_slice`.
//!
//! ## `serde_json` is not banned outright — a *parse* is
//!
//! `ff/forcefield_meta.rs` builds a provenance blob with `json!` and hands it back
//! as a `Value`. That is a JSON **writer**, on nobody's parameter path, and it is
//! not what ac-002 is about. Gate 1 therefore bans the deserializer entry points
//! (`from_str` / `from_slice` / `from_reader` / `from_value`), which is the property
//! that actually matters, and gate 2 bans `serde_json` outright in the one module
//! that must not have it at all.

use super::source_gate::{ff_source, ff_sources, lines_containing_in};

/// The module ac-002 names. `include_str!` + `serde_json::from_str` live here.
const TABLES: &str = "typifier/estimate/tables.rs";

/// Every way serde_json turns text into data.
///
/// Bans the *deserializer*, not the crate: emitting JSON (`json!`, `to_string`) is
/// not a runtime table parse, and `ff/forcefield_meta.rs` legitimately does it.
const DESERIALIZERS: [&str; 4] = [
    "serde_json::from_str",
    "serde_json::from_slice",
    "serde_json::from_reader",
    "serde_json::from_value",
];

/// Nothing under `src/ff/` parses JSON at runtime.
#[test]
fn the_force_field_tree_never_deserializes_json_at_runtime() {
    let sources = ff_sources();
    let hits: Vec<String> = DESERIALIZERS
        .iter()
        .flat_map(|needle| lines_containing_in(&sources, needle))
        .collect();

    assert!(
        hits.is_empty(),
        "the force-field path still parses JSON at runtime. Every parameter table in \
         molrs is compiled Rust data (`ff::params`); the estimator's two are the last \
         that are not, and a malformed one is supposed to be a COMPILE error, not a \
         runtime panic:\n{}",
        hits.join("\n")
    );
}

/// The estimator's tables are Rust data, not embedded text.
///
/// Separate from the gate above because it is a separate way to keep the bug: a
/// module that still holds `gaff_equiv.json` as a `&'static str` has not been
/// converted, whoever ends up doing the parsing.
#[test]
fn the_estimate_tables_embed_no_table_text() {
    let tables = [ff_source(TABLES)];
    let hits: Vec<String> = ["include_str!", "serde_json", "Deserialize"]
        .iter()
        .flat_map(|needle| lines_containing_in(&tables, needle))
        .collect();

    assert!(
        hits.is_empty(),
        "`{TABLES}` still carries its tables as embedded JSON text. They belong in \
         `ff::params::generated` as typed `&'static` Rust, emitted by \
         `scripts/gen_param_tables.py` and guarded by MANIFEST.sha256, exactly like \
         `gaff.dat` and the seven `ATOMTYPE_*.DEF`:\n{}",
        hits.join("\n")
    );
}

/// The estimator's tables cannot panic at load: there is no load.
///
/// `EquivTable::load()` / `EmpiricalTable::load()` today `.expect()` on the embedded
/// asset — a panic on a data path, defended by nothing but the fact that the asset
/// happens to be well-formed today. molrs's rule is fail-fast at the boundary, and
/// for a compiled table the boundary is `rustc`. So the `.expect()` going away is
/// not cosmetic: it is what proves the table stopped being data-at-runtime.
///
/// Scoped to the module's production code ([`ff_source`] cuts the inline
/// `#[cfg(test)]` module): a unit test's `.expect("os present")` is not a runtime
/// panic on embedded data, and failing the gate on it would only teach the next
/// reader to loosen it.
#[test]
fn the_estimate_tables_never_expect_on_embedded_data() {
    let tables = [ff_source(TABLES)];
    let hits = lines_containing_in(&tables, ".expect(");

    assert!(
        hits.is_empty(),
        "`{TABLES}` still panics on its own embedded data. A compiled table cannot be \
         malformed at runtime — that is the point of compiling it — so there is \
         nothing left to `.expect()`:\n{}",
        hits.join("\n")
    );
}
