//! Readers that parse *external* force-field formats into a molrs
//! [`ForceField`].
//!
//! These differ from [`crate::ff::forcefield::xml`], which reads molrs's own native
//! schema. A reader here owns the translation from a foreign format — element
//! and attribute names, **and unit and factor normalization** — into the
//! force-field IR (LAMMPS standard): every style's energy expression, factors
//! (no hidden ½) and parameter units are the LAMMPS style's, angle-valued
//! parameters in degrees, in a LAMMPS unit preset (`real` for every reader but
//! the LAMMPS one, which keeps the file's `units`). The resulting `ForceField`
//! needs no downstream fixup.
//!
//! Concrete readers: [`OplsXmlReader`](opls::OplsXmlReader) (OPLS-AA / GROMACS
//! XML, nm/kJ-mol, Ryckaert–Bellemans torsions),
//! [`LammpsFfReader`](lammps::LammpsFfReader) (a LAMMPS `*.ff` include, AMBER/GAFF
//! flavour — inverse of
//! [`LammpsFfWriter`](super::writers::lammps::LammpsFfWriter)),
//! [`AmberPrmtopFfReader`](prmtop::AmberPrmtopFfReader) (AMBER and chamber
//! (CHARMM) prmtop parameter tables), and [`GromacsTopFfReader`](gromacs::GromacsTopFfReader) (GROMACS
//! `.top`/`.itp` section tables).

pub mod gromacs;
pub mod lammps;
pub mod opls;
pub mod prmtop;
#[cfg(test)]
mod prmtop_check;

use crate::ff::forcefield::ForceField;

/// Parse a force-field definition from an external format into a molrs
/// [`ForceField`], normalized to molrs's (LAMMPS's) convention.
///
/// Implementors own format-specific element/attribute mapping and unit
/// conversion. Reading is **total**: a malformed document or a missing required
/// attribute is an `Err`, never a silently-skipped parameter that would later
/// read as zero.
pub trait ForceFieldReader {
    /// Parse from an in-memory string.
    fn read_str(&self, text: &str) -> Result<ForceField, String>;

    /// Parse from a file on disk. Defaults to reading the file and delegating to
    /// [`read_str`](ForceFieldReader::read_str).
    fn read(&self, path: &str) -> Result<ForceField, String> {
        let text = std::fs::read_to_string(path).map_err(|e| format!("read {}: {}", path, e))?;
        self.read_str(&text)
    }
}
