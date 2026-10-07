//! MOL2, the Tripos section-delimited molecule format.
//!
//! The doors are functions of [`crate::io`]: [`read_mol2`](crate::io::read_mol2)
//! (the first molecule), [`read_mol2_trajectory`](crate::io::read_mol2_trajectory)
//! (every molecule) and [`write_mol2`](crate::io::write_mol2). This module holds
//! the format's classes, [`Mol2Reader`] and [`Mol2Writer`], over any byte
//! stream.

pub(crate) mod codec;

pub use codec::{Mol2Reader, Mol2Writer};
