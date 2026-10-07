//! GRO, the GROMACS fixed-column structure and trajectory format.
//!
//! The doors are functions of [`crate::io`]: [`read_gro`](crate::io::read_gro)
//! (the first frame), [`read_gro_trajectory`](crate::io::read_gro_trajectory)
//! (every frame), [`write_gro`](crate::io::write_gro) and
//! [`write_gro_trajectory`](crate::io::write_gro_trajectory). This module holds
//! the format's classes, [`GroReader`] and [`GroWriter`], over any byte
//! stream.

pub(crate) mod codec;

pub use codec::{GroReader, GroWriter};
