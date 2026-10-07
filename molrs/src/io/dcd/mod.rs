//! DCD, the CHARMM / NAMD binary trajectory format (also written by OpenMM and
//! LAMMPS).
//!
//! The doors are functions of [`crate::io`]:
//! [`read_dcd_trajectory`](crate::io::read_dcd_trajectory),
//! [`read_dcd_bytes`](crate::io::read_dcd_bytes) and
//! [`write_dcd_trajectory`](crate::io::write_dcd_trajectory). This module holds
//! the format's classes: [`DcdReader`] (O(1) random access;
//! [`DcdReader::open`] opens a path), [`DcdWriter`], and [`DcdIndexBuilder`],
//! the chunked frame indexer.

pub(crate) mod codec;

pub use codec::{DcdIndexBuilder, DcdReader, DcdWriter};
