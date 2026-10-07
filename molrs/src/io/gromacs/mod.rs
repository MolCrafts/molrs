//! GROMACS topologies (`.top` / `.itp`): force-field directives and whole
//! systems.
//!
//! GROMACS's coordinate and trajectory formats have their own modules
//! ([`gro`](crate::io::gro), [`trr`](crate::io::trr), [`xtc`](crate::io::xtc)).
//! The topology's classes are this module's:
//! [`GromacsTopForcefieldReader`] (its `read` / `read_str` read the force-field
//! directives, `read_system` / `read_system_str` a whole system) and
//! [`GromacsTopForcefieldWriter`] (its inverse, `write_system_str` included).

pub(crate) mod top_reader;
pub(crate) mod top_writer;

pub use top_reader::GromacsTopForcefieldReader;
pub use top_writer::GromacsTopForcefieldWriter;
