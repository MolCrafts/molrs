//! ac-004 — no constructor may yield an empty parameter table.
//!
//! Today four constructors do exactly that:
//!
//! ```text
//! BCCCorrectionTable::new()   -> empty HashMaps
//! BCCCorrector::default()     -> wraps the empty table
//! AM1BCCTypifier::new(b)      -> wraps the empty corrector
//! AM1BCCTypifier::default()   -> ditto, with the unavailable AM1 backend
//! ```
//!
//! Each hands back an object that constructs, type-checks, and then fails on the
//! FIRST bond it sees with `missing BCC correction for bond ..`. That is a
//! constructible-but-non-functional object, which molrs's no-fallback-values rule
//! forbids: a model with no parameter set must be a compile error or an `Err`,
//! never a silent empty table that only reveals itself at charge time.
//!
//! The criterion allows either cure, so the tests must survive both — and that is
//! why they read the source instead of calling the constructors. A test that
//! *named* `BCCCorrectionTable::new()` would stop compiling the moment that
//! constructor was deleted, i.e. it would break on the very fix it demands, and
//! whoever applied the fix would have to delete the test to land it.
//!
//! Paired with a behavioural half that no source scan can give: whatever
//! constructors survive, the tables they build must be populated.

use molrs::ff::typifier::am1bcc::BCCCorrectionTable;

use super::source_gate::{
    default_impls_of_parameterized_models, lines_containing, parameterless_constructors,
};

/// The scanner must be able to see the source, or every gate below is vacuous.
///
/// A source gate has one failure mode that matters: reading nothing (wrong path,
/// over-eager comment stripping) and reporting "no violations found". Pin a
/// string that must be there.
#[test]
fn the_source_gate_can_read_the_typifier_tree() {
    let hits = lines_containing("BCCCorrectionTable");
    assert!(
        !hits.is_empty(),
        "the source gate found no mention of `BCCCorrectionTable` under \
         molrs/src/ff/typifier/ — it is reading the wrong tree, and every gate \
         in this file is passing vacuously"
    );
}

/// No parameterized model implements `Default`.
#[test]
fn no_parameterized_model_implements_default() {
    let offenders = default_impls_of_parameterized_models();
    assert!(
        offenders.is_empty(),
        "these models can be built with no parameter set at all, by `Default`. \
         An empty correction table is not a default — it is a table that fails on \
         every bond:\n{}\n\nDrop the `Default`; make the parameter set an argument.",
        offenders.join("\n")
    );
}

/// Nor does any `new` build one without a parameter set.
///
/// The gate is scoped to the impl blocks of the parameter-owning models, so a
/// legitimate `fn new(table: BCCCorrectionTable) -> Self` (the table came from a
/// parameter set) and a fallible `fn new(..) -> Result<Self, _>` both pass.
#[test]
fn no_constructor_builds_a_parameterized_model_without_a_parameter_set() {
    let offenders = parameterless_constructors();
    assert!(
        offenders.is_empty(),
        "these constructors hand back a parameter-table-owning model with no \
         parameter set, no table, and no way to refuse — the object exists but \
         fails on every bond:\n{}\n\nTake the parameter set (or a table built from \
         one) as an argument, or return `Result`.",
        offenders.join("\n")
    );
}

/// The behavioural half: the constructors that DO survive build populated tables.
///
/// Deleting every constructor would satisfy the source gates above and leave the
/// module useless. This is the other side of that: whatever remains must produce
/// a table with corrections in it.
#[test]
fn every_surviving_constructor_builds_a_populated_table() {
    for (name, table) in [
        ("bcc", BCCCorrectionTable::bcc()),
        ("abcg2", BCCCorrectionTable::abcg2()),
    ] {
        let table = table.unwrap_or_else(|e| panic!("{name}: build the table: {e}"));
        assert!(
            !table.is_empty(),
            "{name}: the parameter set built an EMPTY correction table — the very \
             footgun ac-004 removes, reintroduced through the front door"
        );
    }
}
