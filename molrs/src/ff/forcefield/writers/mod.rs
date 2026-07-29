//! Writers that serialize a molrs [`ForceField`] into an *external* format.
//!
//! Symmetric to [`crate::ff::forcefield::readers`]: a writer owns the translation
//! from molrs units (Å, kcal/mol, radians, e) back to the foreign convention.
//! The inverse of each reader lands here so unit conversion stays at one
//! boundary pair and never leaks into kernels or call sites.
//!
//! Concrete writers: [`LammpsFfWriter`](lammps::LammpsFfWriter) (LAMMPS `*.ff`
//! include, AMBER/GAFF flavour — inverse of
//! [`LammpsFfReader`](super::readers::lammps::LammpsFfReader)).

pub mod lammps;

use crate::ff::forcefield::ForceField;

/// Serialize a molrs [`ForceField`] into an external format string / file.
///
/// Implementors own format-specific layout **and** the inverse unit conversion
/// of the matching reader.
pub trait ForceFieldWriter {
    /// Serialize to an in-memory string.
    fn write_str(&self, ff: &ForceField) -> Result<String, String>;

    /// Write to a file on disk. Defaults to [`write_str`](ForceFieldWriter::write_str)
    /// then `std::fs::write`.
    fn write(&self, ff: &ForceField, path: &str) -> Result<(), String> {
        let text = self.write_str(ff)?;
        std::fs::write(path, text).map_err(|e| format!("write {}: {}", path, e))
    }
}
