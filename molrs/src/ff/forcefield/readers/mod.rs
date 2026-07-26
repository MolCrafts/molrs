//! Readers that parse *external* force-field formats into a molrs
//! [`ForceField`].
//!
//! These differ from [`crate::ff::forcefield::xml`], which reads molrs's own native
//! schema. A reader here owns the translation from a foreign format — element
//! and attribute names, **and unit normalization** — into molrs units
//! (Å, kcal/mol, radians, e). The resulting `ForceField` is pure molrs units;
//! there is no downstream unit fixup.
//!
//! Concrete readers: [`OplsXmlReader`](opls::OplsXmlReader) (OPLS-AA / GROMACS
//! XML, nm/kJ-mol, Ryckaert–Bellemans torsions) and
//! [`LammpsFfReader`](lammps::LammpsFfReader) (a LAMMPS `*.ff` include, AMBER/GAFF
//! flavour). Further formats (AMBER prmtop) land as additional implementors of
//! [`ForceFieldReader`].

pub mod lammps;
pub mod opls;

use crate::ff::forcefield::ForceField;

/// Parse a force-field definition from an external format into a molrs
/// [`ForceField`], normalized to molrs units (Å, kcal/mol, radians, e).
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
