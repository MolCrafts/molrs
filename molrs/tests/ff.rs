//! Integration tests for `molrs-ff` (package `molcrafts-molrs-ff`).
//!
//! The module tree below MIRRORS the crate's `src/` layout: one test module per
//! source module (or subdir `mod.rs`). Tests exercise the PUBLIC API end-to-end.
//! Parser/writer tests use short built-in format fixtures; typifier and potential
//! tests use graphs built in code or committed embedded data such as
//! `molrs::data::MMFF94_XML`.
//!
//! Inline `#[cfg(test)]` modules in `src/**` remain the pure-function unit layer;
//! this target is the end-to-end integration layer the crate previously lacked.

#[path = "ff/helpers.rs"]
mod helpers;

#[path = "ff/forcefield.rs"]
mod forcefield;
#[path = "ff/forcefield_meta.rs"]
mod forcefield_meta;
#[path = "ff/frcmod.rs"]
mod frcmod;
#[path = "ff/params.rs"]
mod params;
#[path = "ff/readers/lammps.rs"]
mod readers_lammps;
#[path = "ff/readers/opls.rs"]
mod readers_opls;

#[path = "ff/potential/mod.rs"]
mod potential;

#[path = "ff/typifier/mod.rs"]
mod typifier;

/// The charge models (`ff::charge`). Declared after `typifier` because it consumes
/// that module's AmberTools oracle and molecule builder — one fixture for the target,
/// so no test can build its molecule from antechamber's answers instead of a user's
/// input.
#[path = "ff/charge/mod.rs"]
mod charge;

#[path = "ff/mmff/mod.rs"]
mod mmff;
