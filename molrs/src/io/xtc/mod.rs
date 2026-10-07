//! XTC, the GROMACS compressed trajectory format (XDR-encoded, lossy
//! coordinate compression).
//!
//! The doors are functions of [`crate::io`]:
//! [`read_xtc_trajectory`](crate::io::read_xtc_trajectory),
//! [`read_xtc_bytes`](crate::io::read_xtc_bytes) and
//! [`write_xtc_trajectory`](crate::io::write_xtc_trajectory). This module holds
//! the format's classes: [`XtcReader`] ([`XtcReader::open`] opens a path),
//! [`XtcWriter`], and [`XtcIndexBuilder`], the chunked frame indexer.

pub(crate) mod codec;

pub use codec::{XtcIndexBuilder, XtcReader, XtcWriter};
