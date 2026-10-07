//! The columns of a style's table: one row per force-field type, named by
//! `name`, keyed by its endpoint columns, annotated by string columns, and
//! carrying one column per parameter (molrec `docs/spec/forcefield.md`).
//!
//! The record's `forcefield` section ([`crate::io::mrec::ForceFieldSection`])
//! validates its tables against this vocabulary; the force-field model reads
//! its rows by it.

/// The endpoint columns of a style table, in position order.
pub const ENDPOINT_COLUMNS: [&str; 5] = ["itom", "jtom", "ktom", "ltom", "mtom"];

/// The annotation columns of a style table: `string`, nullable.
pub const ANNOTATION_COLUMNS: [&str; 7] = [
    "class",
    "element",
    "smarts",
    "smirks",
    "overrides",
    "desc",
    "doi",
];

/// The array parameter whose shape the chapter pins: a `cmap` row's
/// correction table, `f64[T, N, N]` — an `N × N` grid over the two
/// dihedrals, every row of the table sharing `N`. Any other parameter may be
/// an array too (`f64[T, S…]`).
pub const CMAP_GRID: &str = "grid";

/// Whether a style-table column is a parameter: not `name`, an endpoint or an
/// annotation column. Rows restating one pair are compared on these alone.
pub fn is_parameter_column(column: &str) -> bool {
    column != "name" && !ENDPOINT_COLUMNS.contains(&column) && !ANNOTATION_COLUMNS.contains(&column)
}
