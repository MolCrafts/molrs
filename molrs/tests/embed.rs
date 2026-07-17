//! Integration tests for `molrs-embed`.
//!
//! The module tree below mirrors the crate's `src/` layout. The public surface
//! is the `generate_3d` pipeline plus the `options`/`report` types. `pipeline`
//! builds molecules in code; `distgeom` / `etkdg` / `torsions` load V2000 SDF /
//! JSON fixtures from `tests/embed/fixtures/` against RDKit reference data.

#[path = "embed/pipeline.rs"]
mod pipeline;

#[path = "embed/distgeom.rs"]
mod distgeom;

#[path = "embed/etkdg.rs"]
mod etkdg;

#[path = "embed/torsions.rs"]
mod torsions;
