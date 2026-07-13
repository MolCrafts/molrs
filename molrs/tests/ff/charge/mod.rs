//! Integration tests for the `ff::charge` module tree. Mirrors `src/ff/charge/`.
//!
//! The charge models and the trait that generalizes them. The AmberTools oracle
//! they are measured against is declared once for the whole `ff` test target, next
//! to the typifier tests that were its first consumer, and is reached from here as
//! `crate::typifier::antechamber_oracle` — one fixture, one builder, so a charge
//! test cannot quietly build its molecule from antechamber's ANSWERS (perceived
//! bond types, BCC atom types) instead of a user's INPUT.

#[path = "model.rs"]
mod model;

#[path = "bcc.rs"]
mod bcc;

#[path = "antechamber.rs"]
mod antechamber;

/// The zero-QM corner of the 2×2 (`chem-perceive-08`): Gasteiger/PEOE against
/// `antechamber -c gas`, reached through the same trait with `assign(&mol, None)`.
#[path = "gasteiger.rs"]
mod gasteiger;
/// ac-002 / ac-004's structural halves: the `d` column is never a cubic coefficient,
/// and the plumbing never branches on the concrete model.
#[path = "gasteiger_source.rs"]
mod gasteiger_source;

#[path = "pull_trait_deleted.rs"]
mod pull_trait_deleted;
