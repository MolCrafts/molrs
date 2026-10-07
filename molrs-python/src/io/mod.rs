//! Python bindings for `molrs::io` (`molrs.io`): every file reader and
//! writer — structure, trajectory and force-field files, SMILES / CGsmiles
//! text, `*.mrec` records, frame bytes. One file per Rust io namespace:
//!
//! | binding          | Rust owner                 | what                                   |
//! |------------------|----------------------------|----------------------------------------|
//! | [`data`]         | `io::data`                 | PDB, XYZ, GRO, LAMMPS data, MOL2, …    |
//! | [`trajectory`]   | `io::trajectory`           | LAMMPS dump, DCD, TRR, XTC, XYZ frames |
//! | [`mesh`]         | `io::mesh`                 | STL                                    |
//! | [`csv`]          | `io::csv`                  | Block CSV                              |
//! | [`format`]       | `io::{read,write}_frame`   | extension-dispatched doors             |
//! | [`forcefield`]   | `io::forcefield`           | force-field files                      |
//! | [`frame_bytes`]  | `stream` wire codec        | a frame as MessagePack / JSON bytes    |
//! | [`smiles`]       | `io::smiles`               | SMILES / SMARTS text                   |
//! | [`cgsmiles`]     | `io::smiles` (CGsmiles)    | coarse-grained line notation           |
//! | [`log`]          | `io::log`                  | LAMMPS log files                       |
//! | [`bond_react`]   | `io::data::bond_react`     | LAMMPS `fix bond/react` templates      |
//! | [`mrec`]         | `io::mrec`                 | `*.mrec` scientific records            |
//!
//! Every factory has one shape: a function flat on `molrs.io`
//! (`read_<fmt>[_<what>]` / `write_<fmt>[_<what>]`), or a class of the
//! format's own submodule (`molrs.io.<fmt>.<Fmt>Reader` / `<Fmt>Writer`).
//! A class that belongs to one format is that format's submodule's:
//! `molrs.io.smiles`, `molrs.io.log`, `molrs.io.lammps_bond_react`,
//! `molrs.io.mrec`, `molrs.io.trajectory`. Every reader emits the canonical
//! column names; nothing in Python renames a format's columns.

pub mod bond_react;
pub mod cgsmiles;
mod csv;
mod data;
mod forcefield;
mod format;
mod frame_bytes;
mod json;
// The log parser reads a path, and the record store is a filesystem store in
// the core crate: both are native-only, behind `fs`.
#[cfg(feature = "fs")]
mod log;
mod mesh;
#[cfg(feature = "fs")]
pub mod mrec;
pub(crate) mod smiles;
mod trajectory;

use pyo3::prelude::*;

/// Register every `molrs::io` binding on the native module.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    data::register(m)?;
    trajectory::register(m)?;
    mesh::register(m)?;
    csv::register(m)?;
    format::register(m)?;
    forcefield::register(m)?;
    frame_bytes::register(m)?;
    smiles::register(m)?;
    cgsmiles::register(m)?;
    bond_react::register(m)?;
    #[cfg(feature = "fs")]
    {
        log::register(m)?;
        mrec::register_doors(m)?;
        crate::add_submodule(m, "mrec", "molrs.io.mrec", mrec::register)?;
    }
    Ok(())
}
