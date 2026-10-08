//! TRR, the GROMACS full-precision trajectory format (XDR-encoded).
//!
//! The doors are functions of [`crate::io`]:
//! [`read_trr_trajectory`](crate::io::read_trr_trajectory),
//! [`read_trr_bytes`](crate::io::read_trr_bytes) and
//! [`write_trr_trajectory`](crate::io::write_trr_trajectory). This module holds
//! the format's classes: [`TrrReader`] ([`TrrReader::open`] opens a path),
//! [`TrrWriter`], and [`TrrIndexBuilder`], the chunked frame indexer.

pub(crate) mod codec;

pub use codec::{TrrIndexBuilder, TrrReader, TrrWriter};
