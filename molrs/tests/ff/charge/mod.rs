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

#[path = "pull_trait_deleted.rs"]
mod pull_trait_deleted;
