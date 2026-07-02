//! Integration tests for `molrs-ff` (package `molcrafts-molrs-ff`).
//!
//! The module tree below MIRRORS the crate's `src/` layout: one test module per
//! source module (or subdir `mod.rs`). Tests exercise the PUBLIC API end-to-end
//! with inputs BUILT IN CODE — no fabricated file-format strings. The only file
//! "input" used is the embedded `molrs::data::MMFF94_XML` parameter set, which is
//! the crate's own intended entry point (`MMFFTypifier::mmff94()`), not a
//! hand-crafted fixture.
//!
//! Inline `#[cfg(test)]` modules in `src/**` remain the pure-function unit layer;
//! this target is the end-to-end integration layer the crate previously lacked.

#[path = "ff/helpers.rs"]
mod helpers;

#[path = "ff/forcefield.rs"]
mod forcefield;
#[path = "ff/forcefield_meta.rs"]
mod forcefield_meta;
#[path = "ff/readers/lammps.rs"]
mod readers_lammps;
#[path = "ff/readers/opls.rs"]
mod readers_opls;

#[path = "ff/potential/mod.rs"]
mod potential;

#[path = "ff/typifier/mod.rs"]
mod typifier;

#[path = "ff/mmff/mod.rs"]
mod mmff;
