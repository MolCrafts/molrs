//! Shared LAMMPS I/O primitives used by both the data-file reader
//! ([`crate::io::data::lammps_data`]) and the dump trajectory reader
//! ([`crate::io::trajectory::lammps_dump`]).
//!
//! - [`common`] — tokenisation, type refs, array/block helpers, optional columns
//! - [`atom_style`] — `read_data` Atoms column layouts for every fixed atom style
//! - [`box_bounds`] — orthogonal / triclinic bounds → [`SimBox`]

pub(crate) mod atom_style;
pub(crate) mod box_bounds;
pub(crate) mod common;
