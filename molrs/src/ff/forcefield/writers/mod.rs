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

use std::fmt;
use std::ops::Deref;

use crate::ff::forcefield::ForceField;
use crate::ff::ir::IrError;

/// Serialize a molrs [`ForceField`] into an external format string / file.
///
/// Implementors own format-specific layout **and** the inverse unit conversion
/// of the matching reader.
pub trait ForceFieldWriter {
    /// Serialize to an in-memory string.
    fn write_str(&self, ff: &ForceField) -> Result<String, WriteError>;

    /// Write to a file on disk. Defaults to [`write_str`](ForceFieldWriter::write_str)
    /// then `std::fs::write`.
    fn write(&self, ff: &ForceField, path: &str) -> Result<(), WriteError> {
        let text = self.write_str(ff)?;
        std::fs::write(path, text).map_err(|e| format!("write {}: {}", path, e).into())
    }
}

/// Why a writer wrote nothing.
///
/// An engine that cannot hold a style refuses it with a typed
/// [`IrError::NoEngineForm`] (`ff-ir-02-protocol` §8), kept here so a caller
/// matches on the refusal ([`ir`](Self::ir)) instead of reading a message —
/// a binding raises `molrs.ff.ir.NoEngineForm` from it. Anything else (a
/// parameter the format has no field for, a malformed row) is the message
/// alone. Either way the error reads as its message: it dereferences to
/// `str`.
///
/// ```
/// use molrs::ff::forcefield::{ForceField, Params};
/// use molrs::ff::forcefield::writers::ForceFieldWriter;
/// use molrs::ff::forcefield::writers::gromacs::GromacsTopFfWriter;
/// use molrs::ff::ir::IrError;
///
/// let mut ff = ForceField::new("t");
/// let mut style = Params::new();
/// style.set_str("expression", "k*(r-r0)^4");
/// ff.def_style("bond", "quartic", style).unwrap();
/// let err = GromacsTopFfWriter::new().write_str(&ff).unwrap_err();
/// assert!(matches!(
///     err.ir(),
///     Some(IrError::NoEngineForm { engine, style, .. }) if engine == "GROMACS" && style == "quartic"
/// ));
/// assert!(err.contains("GROMACS has no form for bond `quartic`"));
/// ```
#[derive(Clone, Debug, PartialEq)]
pub struct WriteError {
    message: String,
    refusal: Option<Box<IrError>>,
}

impl WriteError {
    /// The force-field IR's refusal, when this is one.
    pub fn ir(&self) -> Option<&IrError> {
        self.refusal.as_deref()
    }

    /// The message.
    pub fn message(&self) -> &str {
        &self.message
    }

    /// The same error, its message prefixed with `what` (`"bonds label
    /// `CT-HC`: …"`); a refusal stays typed.
    pub fn context(mut self, what: impl fmt::Display) -> Self {
        self.message = format!("{what}: {}", self.message);
        self
    }
}

impl Deref for WriteError {
    type Target = str;

    fn deref(&self) -> &str {
        &self.message
    }
}

impl fmt::Display for WriteError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for WriteError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        self.refusal
            .as_deref()
            .map(|e| e as &(dyn std::error::Error + 'static))
    }
}

impl From<IrError> for WriteError {
    fn from(e: IrError) -> Self {
        Self {
            message: e.to_string(),
            refusal: Some(Box::new(e)),
        }
    }
}

impl From<String> for WriteError {
    fn from(message: String) -> Self {
        Self {
            message,
            refusal: None,
        }
    }
}

impl From<&str> for WriteError {
    fn from(message: &str) -> Self {
        message.to_owned().into()
    }
}

impl From<WriteError> for String {
    fn from(e: WriteError) -> Self {
        e.message
    }
}
