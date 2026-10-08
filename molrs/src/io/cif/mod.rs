//! CIF, the IUCr Crystallographic Information File (small-molecule CIF and the
//! mmCIF `_atom_site` loop).
//!
//! The doors are functions of [`crate::io`]: [`read_cif`](crate::io::read_cif)
//! (the first `data_` block), [`read_cif_trajectory`](crate::io::read_cif_trajectory)
//! (every block) and [`write_cif`](crate::io::write_cif). This module holds the
//! format's classes, [`CifReader`] and [`CifWriter`], over any byte stream.

pub(crate) mod codec;

pub use codec::{CifReader, CifWriter};
