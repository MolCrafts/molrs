//! Zarr V3 storage — the reference binding of the MolRec record contract for
//! array-shaped sections (frame / trajectory / system / large observables).
//!
//! Append-oriented **metrics** use a separate JSONL binding
//! (`metrics/metrics.jsonl`); see `record_io` module docs and molrec
//! `docs/spec/metrics.md`. This crate does not implement the JSONL writer.

mod error;
mod frame_io;
mod record_io;

pub use record_io::{count_frames_in_store, read_frame_from_store, read_record_store};
#[cfg(feature = "filesystem")]
pub use record_io::{
    read_record_file, read_trajectory_file, write_record_file, write_record_store,
    write_trajectory_file,
};

/// Unit system tag stored in root metadata.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub enum UnitSystem {
    /// kcal/mol, Angstrom, fs, g/mol, e (LAMMPS "real")
    #[default]
    Real,
    /// eV, Angstrom, fs, amu, e (LAMMPS "metal")
    Metal,
    /// kJ/mol, nm, ps, g/mol, e (GROMACS)
    Gromacs,
    /// Hartree, Bohr, amu_e, e (atomic units)
    Atomic,
}

impl UnitSystem {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Real => "real",
            Self::Metal => "metal",
            Self::Gromacs => "gromacs",
            Self::Atomic => "atomic",
        }
    }

    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "real" => Some(Self::Real),
            "metal" => Some(Self::Metal),
            "gromacs" => Some(Self::Gromacs),
            "atomic" => Some(Self::Atomic),
            _ => None,
        }
    }
}

/// Provenance metadata describing how a simulation was produced.
#[derive(Clone, Debug, Default)]
pub struct Provenance {
    pub program: Option<String>,
    pub version: Option<String>,
    pub method: Option<String>,
    pub seed: Option<u64>,
}
