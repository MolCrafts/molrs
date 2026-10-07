//! The writer side of the io contract: [`Writer`] / [`FrameWriter`], which
//! every per-format writer class implements; [`check_write_frame`], the Frame
//! schema check each one runs before writing; and [`ForceFieldWriter`] /
//! [`ForceFieldWriteError`], the force-field writers' contract (feature `ff`).

use std::io::Result;
use std::io::Write;

use crate::core::Frame;

/// Generic writer for data destinations.
pub trait Writer {
    /// Underlying writer type.
    type W: Write;
    /// Construct a new writer from the underlying writer.
    fn new(writer: Self::W) -> Self;
}

/// A writer that emits one logical frame at a time.
///
/// Mirrors [`FrameReader`](crate::io::reader::FrameReader).
pub trait FrameWriter: Writer {
    /// Write one frame.
    ///
    /// The frame is checked against the Frame schema first
    /// ([`check_write_frame`]). A non-conforming frame produces a file that
    /// looks fine and is wrong — the expensive kind of failure, found later by
    /// whatever reads it.
    fn write(&mut self, frame: &Frame) -> Result<()>;
}

/// Check a frame against the Frame schema before writing it — the write-side
/// mirror of [`check_read_frame`](crate::io::reader::check_read_frame).
///
/// Every [`FrameWriter::write`] calls this first.
pub fn check_write_frame<F: crate::core::FrameAccess>(frame: &F) -> Result<()> {
    crate::core::schema::Validator::canonical()
        .validate(frame)
        .map_err(|report| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("refusing to write a frame that violates the Frame schema:\n{report}"),
            )
        })
}

/// Serialize a molrs [`ForceField`](crate::ff::forcefield::ForceField) into a
/// force-field file format, string or file.
///
/// Implementors own format-specific layout **and** the inverse unit conversion
/// of the matching reader.
#[cfg(feature = "ff")]
pub trait ForceFieldWriter {
    /// Serialize to an in-memory string.
    fn write_str(
        &self,
        ff: &crate::ff::forcefield::ForceField,
    ) -> std::result::Result<String, ForceFieldWriteError>;

    /// Write to a file on disk. Defaults to [`write_str`](ForceFieldWriter::write_str)
    /// then `std::fs::write`.
    fn write(
        &self,
        ff: &crate::ff::forcefield::ForceField,
        path: &str,
    ) -> std::result::Result<(), ForceFieldWriteError> {
        let text = self.write_str(ff)?;
        std::fs::write(path, text).map_err(|e| format!("write {}: {}", path, e).into())
    }
}

/// Write a force-field writer's `text` to `path`, an error naming the file.
#[cfg(feature = "ff")]
pub(crate) fn write_forcefield_text(
    path: &std::path::Path,
    text: &str,
) -> std::result::Result<(), ForceFieldWriteError> {
    std::fs::write(path, text).map_err(|e| format!("write {}: {e}", path.display()).into())
}

/// Why a writer wrote nothing.
///
/// An engine that cannot hold a style refuses it with a typed
/// [`IrError::NoEngineForm`](crate::ff::ir::IrError::NoEngineForm) (`ff-ir-02-protocol` §8), kept here so a caller
/// matches on the refusal ([`ir`](Self::ir)) instead of reading a message —
/// a binding raises `molrs.ff.ir.NoEngineForm` from it. Anything else (a
/// parameter the format has no field for, a malformed row) is the message
/// alone. Either way the error reads as its message: it dereferences to
/// `str`.
///
/// ```
/// use molrs::ff::forcefield::{ForceField, Params};
/// use molrs::io::writer::ForceFieldWriter;
/// use molrs::io::gromacs::GromacsTopForcefieldWriter;
/// use molrs::ff::ir::IrError;
///
/// let mut ff = ForceField::new("t");
/// let mut style = Params::new();
/// style.set_str("expression", "k*(r-r0)^4");
/// ff.def_style("bond", "quartic", style).unwrap();
/// let err = GromacsTopForcefieldWriter::new().write_str(&ff).unwrap_err();
/// assert!(matches!(
///     err.ir(),
///     Some(IrError::NoEngineForm { engine, style, .. }) if engine == "GROMACS" && style == "quartic"
/// ));
/// assert!(err.contains("GROMACS has no form for bond `quartic`"));
/// ```
#[cfg(feature = "ff")]
#[derive(Clone, Debug, PartialEq)]
pub struct ForceFieldWriteError {
    message: String,
    refusal: Option<Box<crate::ff::ir::IrError>>,
}

#[cfg(feature = "ff")]
impl ForceFieldWriteError {
    /// The force-field IR's refusal, when this is one.
    pub fn ir(&self) -> Option<&crate::ff::ir::IrError> {
        self.refusal.as_deref()
    }

    /// The message.
    pub fn message(&self) -> &str {
        &self.message
    }

    /// The same error, its message prefixed with `what` (`"bonds label
    /// `CT-HC`: …"`); a refusal stays typed.
    pub fn context(mut self, what: impl std::fmt::Display) -> Self {
        self.message = format!("{what}: {}", self.message);
        self
    }
}

#[cfg(feature = "ff")]
impl std::ops::Deref for ForceFieldWriteError {
    type Target = str;

    fn deref(&self) -> &str {
        &self.message
    }
}

#[cfg(feature = "ff")]
impl std::fmt::Display for ForceFieldWriteError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.message)
    }
}

#[cfg(feature = "ff")]
impl std::error::Error for ForceFieldWriteError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        self.refusal
            .as_deref()
            .map(|e| e as &(dyn std::error::Error + 'static))
    }
}

#[cfg(feature = "ff")]
impl From<crate::ff::ir::IrError> for ForceFieldWriteError {
    fn from(e: crate::ff::ir::IrError) -> Self {
        Self {
            message: e.to_string(),
            refusal: Some(Box::new(e)),
        }
    }
}

#[cfg(feature = "ff")]
impl From<String> for ForceFieldWriteError {
    fn from(message: String) -> Self {
        Self {
            message,
            refusal: None,
        }
    }
}

#[cfg(feature = "ff")]
impl From<&str> for ForceFieldWriteError {
    fn from(message: &str) -> Self {
        message.to_owned().into()
    }
}

#[cfg(feature = "ff")]
impl From<ForceFieldWriteError> for String {
    fn from(e: ForceFieldWriteError) -> Self {
        e.message
    }
}
