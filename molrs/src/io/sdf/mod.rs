//! SDF, the MDL structure-data file (V2000 molfile records).
//!
//! The doors are functions of [`crate::io`]: [`read_sdf`](crate::io::read_sdf)
//! (the first record), [`read_sdf_trajectory`](crate::io::read_sdf_trajectory)
//! (every record) and [`read_sdf_bytes`](crate::io::read_sdf_bytes). This module
//! holds the format's classes: [`SdfReader`] over any byte stream and
//! [`SdfIndexBuilder`], the chunked record indexer.

pub(crate) mod codec;

pub use codec::{SdfIndexBuilder, SdfReader};
