//! Writers that serialize a molrs [`ForceField`] into an *external* format.
//!
//! Symmetric to [`crate::ff::forcefield::readers`]: a writer owns the translation
//! from molrs's convention — LAMMPS's (molrs-python docs, "Force-field
//! conventions") — back to the foreign one.
//! The inverse of each reader lands here so unit conversion stays at one
//! boundary pair and never leaks into kernels or call sites.
//!
//! Concrete writers: [`LammpsFfWriter`](lammps::LammpsFfWriter),
//! [`AmberFrcmodFfWriter`](frcmod::AmberFrcmodFfWriter),
//! [`GromacsTopFfWriter`](gromacs::GromacsTopFfWriter),
//! [`XmlForceFieldWriter`](xml::XmlForceFieldWriter).

pub mod frcmod;
pub mod gromacs;
pub mod lammps;
pub mod xml;

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
