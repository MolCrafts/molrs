//! AMBER's files: the prmtop topology (structure and force field), inpcrd /
//! restrt coordinates, antechamber `.ac`, prep residue templates, and frcmod
//! parameter modifications.
//!
//! The doors are functions of [`crate::io`]:
//! [`read_amber_prmtop`](crate::io::read_amber_prmtop) /
//! [`read_amber_prmtop_str`](crate::io::read_amber_prmtop_str) (structure),
//! [`read_amber_prmtop_forcefield`](crate::io::read_amber_prmtop_forcefield),
//! [`read_amber_inpcrd`](crate::io::read_amber_inpcrd) /
//! [`read_amber_inpcrd_str`](crate::io::read_amber_inpcrd_str),
//! [`read_amber_ac`](crate::io::read_amber_ac) /
//! [`read_amber_ac_str`](crate::io::read_amber_ac_str),
//! [`read_amber_prep`](crate::io::read_amber_prep) /
//! [`read_amber_prep_str`](crate::io::read_amber_prep_str) /
//! [`write_amber_prep`](crate::io::write_amber_prep) /
//! [`write_amber_prep_str`](crate::io::write_amber_prep_str), and
//! [`write_amber_frcmod`](crate::io::write_amber_frcmod) /
//! [`write_amber_frcmod_str`](crate::io::write_amber_frcmod_str). This module
//! holds the family's classes and records: [`AmberPrmtopForcefieldReader`],
//! [`AmberFrcmodWriter`], the prep records ([`PrepResidue`], [`PrepAtom`]),
//! and [`merge_inpcrd`], which lays an inpcrd's coordinates onto a prmtop's
//! frame.

pub(crate) mod ac;
#[cfg(feature = "ff")]
pub(crate) mod frcmod;
pub(crate) mod inpcrd;
pub(crate) mod prep;
pub(crate) mod prmtop;
#[cfg(all(test, feature = "ff"))]
mod prmtop_check;
#[cfg(feature = "ff")]
pub(crate) mod prmtop_forcefield;
pub(crate) mod prmtop_tables;

#[cfg(feature = "ff")]
pub use frcmod::AmberFrcmodWriter;
pub use inpcrd::merge_inpcrd;
pub use prep::{PrepAtom, PrepResidue};
#[cfg(feature = "ff")]
pub use prmtop_forcefield::AmberPrmtopForcefieldReader;
