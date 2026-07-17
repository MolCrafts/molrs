//! Integration tests for the `perceive` module — the `Perceive` builder.
//!
//! The module tree mirrors `src/perceive/`: one test module per perception
//! (`rings`, `aromaticity`, `bond_type`, `hydrogens`, `stereo`, `rotatable`) plus three
//! cross-cutting modules for the properties the builder exists to provide —
//! `immutability` (every `find_*` clones; the input is never touched),
//! `composition` (graph-out means the finders chain), and `edge_cases`.
//!
//! The free functions wrapped here (`find_rings`, `perceive_aromaticity`,
//! `add_hydrogens`, `assign_stereo_from_3d`, `detect_rotatable_bonds`) have four
//! different shapes — side table, mutating + count, graph-out, map. The builder
//! normalises all of them to graph-in / graph-out, so the expectations below are
//! a verbatim re-projection of what those functions already produce; no new
//! domain numbers are introduced. Fixtures are built in code (no file I/O).

#[path = "perceive/aromaticity.rs"]
mod aromaticity;
#[path = "perceive/bond_type.rs"]
mod bond_type;
/// The compat alias `chem-perceive-01` left behind (chem-perceive-13 ac-004) —
/// a source gate, because an alias is invisible to a behavioural test.
#[path = "perceive/chem_alias.rs"]
mod chem_alias;
#[path = "perceive/composition.rs"]
mod composition;
#[path = "perceive/edge_cases.rs"]
mod edge_cases;
#[path = "perceive/equivalence.rs"]
mod equivalence;
#[path = "perceive/fixtures.rs"]
mod fixtures;
#[path = "perceive/hydrogens.rs"]
mod hydrogens;
#[path = "perceive/immutability.rs"]
mod immutability;
#[path = "perceive/rings.rs"]
mod rings;
#[path = "perceive/rotatable.rs"]
mod rotatable;
#[path = "perceive/stereo.rs"]
mod stereo;
