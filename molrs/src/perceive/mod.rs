//! Chemical perception algorithms operating on molecular graphs:
//! aromaticity, bond-type perception, hydrogen handling, ring detection,
//! stereochemistry, rotatable bonds, and SMARTS matching.
//!
//! Gasteiger charges used to live here. They are a *charge model*, not a
//! perception, and they now sit with the other charge models in
//! [`ff::charge`](crate::ff::charge) — one implementation, reached through the
//! [`ChargeModel`](crate::ff::charge::ChargeModel) trait.
//!
//! The layer's public face is the [`Perceive`] builder, which gives every
//! perception one shape — graph in / graph out, non-mutating:
//! `Perceive::new().find_rings(&mol) -> Atomistic`. The free functions it wraps
//! remain available (and re-exported at the crate root) for callers that want
//! the raw side table / map.

pub mod aromaticity;
pub mod bond_type;
pub mod builder;
pub mod equivalence;
pub mod hydrogens;
pub mod rings;
pub mod rotatable;
pub mod smarts;
pub mod stereo;

pub use builder::Perceive;
