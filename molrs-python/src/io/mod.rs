//! Python bindings for `molrs::io` (`molrs.io`): structure and trajectory
//! file formats, SMILES / CGsmiles text, and `*.mrec` records. One file per
//! Rust io namespace:
//!
//! | binding          | Rust owner                 | what                                   |
//! |------------------|----------------------------|----------------------------------------|
//! | [`data`]         | `io::data`                 | PDB, XYZ, GRO, LAMMPS data, MOL2, …    |
//! | [`trajectory`]   | `io::trajectory`           | LAMMPS dump, DCD, TRR, XTC, XYZ frames |
//! | [`mesh`]         | `io::mesh`                 | STL                                    |
//! | [`csv`]          | `io::csv`                  | Block CSV                              |
//! | [`format`]       | `io::{read,write}_frame`   | extension-dispatched doors             |
//! | [`smiles`]       | `io::smiles`               | SMILES / SMARTS text                   |
//! | [`cgsmiles`]     | `io::smiles` (CGsmiles)    | coarse-grained line notation           |
//! | [`log`]          | `io::log`                  | LAMMPS log files                       |
//! | [`bond_react`]   | `io::data::bond_react`     | LAMMPS `fix bond/react` templates      |
//! | [`mrec`]         | `io::mrec`                 | `*.mrec` scientific records            |
//!
//! Every reader emits the canonical column names; nothing in Python renames a
//! format's columns. Force-field file formats are `molrs.ff.forcefield`'s.

pub mod bond_react;
pub mod cgsmiles;
mod csv;
mod data;
mod format;
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
    smiles::register(m)?;
    cgsmiles::register(m)?;
    bond_react::register(m)?;
    #[cfg(feature = "fs")]
    {
        log::register(m)?;
        crate::add_submodule(m, "mrec", "molrs.io.mrec", mrec::register)?;
    }
    Ok(())
}
