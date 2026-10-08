//! XYZ and extended XYZ (the `Properties=` / `Lattice=` comment-line
//! convention of ASE and QUIP).
//!
//! The doors are functions of [`crate::io`]: [`read_xyz`](crate::io::read_xyz),
//! [`read_xyz_trajectory`](crate::io::read_xyz_trajectory),
//! [`read_xyz_bytes`](crate::io::read_xyz_bytes),
//! [`write_xyz`](crate::io::write_xyz) and
//! [`write_xyz_trajectory`](crate::io::write_xyz_trajectory). This module holds
//! the format's classes: [`XyzReader`] (random access over a seekable stream),
//! [`XyzWriter`], and [`XyzIndexBuilder`], the chunked frame indexer.

pub(crate) mod codec;

pub use codec::{XyzIndexBuilder, XyzReader, XyzWriter};
