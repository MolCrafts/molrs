//! [`CompileError`]: why a force field did not compile into kernels.

use std::fmt;

use crate::ff::ir::IrError;

/// Why [`PotentialCompiler`](crate::ff::potential::PotentialCompiler) (or a
/// kernel constructor it calls) built no kernels.
///
/// A refusal of the force-field IR stays typed ([`CompileError::Ir`]), so a
/// caller — a binding raising one exception class per [`IrError`] variant —
/// matches on it instead of reading a message.
///
/// ```
/// use molrs::ff::forcefield::{ForceField, Params};
/// use molrs::ff::ir::IrError;
/// use molrs::ff::potential::{CompileError, PotentialCompiler};
/// use molrs::Frame;
///
/// let mut ff = ForceField::new("t");
/// ff.def_style("bond", "nosuch", Params::new()).unwrap();
/// let mut frame = Frame::new();
/// let mut bonds = molrs::Block::new();
/// bonds.insert("atomi", ndarray::arr1(&[0u32]).into_dyn()).unwrap();
/// bonds.insert("atomj", ndarray::arr1(&[1u32]).into_dyn()).unwrap();
/// frame.insert("bonds", bonds);
/// let err = PotentialCompiler::new(&ff).compile(&frame).unwrap_err();
/// assert!(matches!(err, CompileError::Ir(IrError::NoKernel { .. })));
/// ```
#[derive(Clone, Debug, PartialEq)]
pub enum CompileError {
    /// The force-field IR refused a style, a parameter or a term.
    Ir(IrError),
    /// Anything else that stops the compile: a missing block or column, an
    /// unknown type label, special-bonds weights a compiled pair list cannot
    /// carry.
    Invalid(String),
}

impl CompileError {
    /// The IR refusal, when this is one.
    pub fn ir(&self) -> Option<&IrError> {
        match self {
            CompileError::Ir(e) => Some(e),
            CompileError::Invalid(_) => None,
        }
    }
}

impl fmt::Display for CompileError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CompileError::Ir(e) => e.fmt(f),
            CompileError::Invalid(message) => f.write_str(message),
        }
    }
}

impl std::error::Error for CompileError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            CompileError::Ir(e) => Some(e),
            CompileError::Invalid(_) => None,
        }
    }
}

impl From<IrError> for CompileError {
    fn from(e: IrError) -> Self {
        CompileError::Ir(e)
    }
}

impl From<String> for CompileError {
    fn from(message: String) -> Self {
        CompileError::Invalid(message)
    }
}

impl From<&str> for CompileError {
    fn from(message: &str) -> Self {
        CompileError::Invalid(message.to_owned())
    }
}
