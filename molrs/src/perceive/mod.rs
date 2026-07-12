//! Chemical perception algorithms operating on molecular graphs:
//! aromaticity, Gasteiger charges, hydrogen handling, ring detection,
//! stereochemistry, rotatable bonds, and SMARTS matching.
//!
//! The layer's public face is the [`Perceive`] builder, which gives every
//! perception one shape — graph in / graph out, non-mutating:
//! `Perceive::new().find_rings(&mol) -> Atomistic`. The free functions it wraps
//! remain available (and re-exported at the crate root) for callers that want
//! the raw side table / map.

pub mod aromaticity;
pub mod bond_type;
pub mod builder;
pub mod gasteiger;
pub mod hydrogens;
pub mod rings;
pub mod rotatable;
pub mod smarts;
pub mod stereo;

pub use builder::Perceive;
