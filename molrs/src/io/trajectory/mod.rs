//! Multi-frame trajectory formats: DCD, LAMMPS dump, and the GROMACS binary
//! formats TRR (full precision) and XTC (compressed).
//!
//! Single-structure formats that also support multi-frame streaming (XYZ,
//! PDB) live in [`crate::io::data`]; the chunk-based indexing infrastructure
//! shared by all of them is in [`crate::io::streaming`]. The GROMACS readers
//! share the minimal XDR primitives in [`xdr`].

pub mod cube_traj;
pub mod dcd;
pub mod lammps_dump;
pub mod trr;
pub mod xdr;
pub mod xtc;
