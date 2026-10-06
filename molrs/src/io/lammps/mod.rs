//! Shared LAMMPS I/O primitives used by both the data-file reader
//! ([`crate::io::data::lammps_data`]) and the dump trajectory reader
//! ([`crate::io::trajectory::lammps_dump`]).
//!
//! - [`fields`] — a line's fields: tokens, numbers, type references and label maps
//! - [`columns`] — the Frame columns built from them, and the dump attribute names
//! - [`atom_style`] — `read_data` Atoms column layouts for every fixed atom style
//! - [`box_bounds`] — orthogonal / triclinic bounds → [`SimBox`]

pub(crate) mod atom_style;
pub(crate) mod box_bounds;
pub(crate) mod columns;
pub(crate) mod fields;
