//! Compiled SMARTS pattern — the unit of work handed to the matcher.
//!
//! A [`SmartsPattern`] is produced by compiling a SMARTS string once and then
//! re-using it across many target molecules. Compilation parses the SMARTS,
//! validates ring closures, and precomputes any per-pattern data the matcher
//! needs (atom-predicate AST, adjacency graph, most-specific priority).
//!
//! # Status
//!
//! The public API is stable; the implementation is pending the session-2
//! matcher rollout. Both [`SmartsPattern::compile`] and the matching entry
//! points currently return [`SmartsError::NotYetImplemented`] so downstream
//! crates can wire call sites without waiting for the implementation.
//!
//! # Example
//!
//! ```
//! # use molrs_smiles::smarts::{SmartsPattern, SmartsError};
//! let pat = SmartsPattern::compile("[C;X4]");
//! assert!(matches!(pat, Err(SmartsError::NotYetImplemented)));
//! ```

use std::fmt;

use crate::error::SmilesError;

/// A compiled SMARTS pattern ready to be matched against target molecules.
///
/// Construct via [`SmartsPattern::compile`]. Compiled patterns are immutable
/// and implement `Send + Sync` so they can be shared across rayon-parallel
/// pattern-matching passes (required by the ETKDG torsion-preference lookup,
/// which matches hundreds of patterns against each molecule).
#[derive(Debug, Clone)]
pub struct SmartsPattern {
    // Opaque placeholder. The real fields (parsed SmilesIR, precomputed
    // predicate tree, ring-membership cache) arrive with the session-2
    // matcher implementation.
    _private: (),
}

/// Errors produced while compiling or matching a SMARTS pattern.
#[derive(Debug, Clone, PartialEq)]
pub enum SmartsError {
    /// The SMARTS string failed to parse or validate.
    Parse(SmilesError),
    /// A requested feature is not yet implemented in this build.
    ///
    /// This variant is used while the matcher implementation is being staged
    /// in. It will be removed once the implementation is complete.
    NotYetImplemented,
}

impl fmt::Display for SmartsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SmartsError::Parse(e) => write!(f, "SMARTS parse error: {e}"),
            SmartsError::NotYetImplemented => {
                write!(f, "SMARTS feature not yet implemented")
            }
        }
    }
}

impl std::error::Error for SmartsError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            SmartsError::Parse(e) => Some(e),
            SmartsError::NotYetImplemented => None,
        }
    }
}

impl From<SmilesError> for SmartsError {
    fn from(e: SmilesError) -> Self {
        SmartsError::Parse(e)
    }
}

impl SmartsPattern {
    /// Parse and compile a SMARTS pattern string.
    ///
    /// Returns [`SmartsError::Parse`] if the input is syntactically invalid or
    /// fails ring-closure validation. Returns
    /// [`SmartsError::NotYetImplemented`] while the compiler is being staged in.
    ///
    /// # Example
    ///
    /// ```
    /// # use molrs_smiles::smarts::{SmartsPattern, SmartsError};
    /// assert!(matches!(
    ///     SmartsPattern::compile("[C;X4]"),
    ///     Err(SmartsError::NotYetImplemented)
    /// ));
    /// ```
    pub fn compile(_pattern: &str) -> Result<Self, SmartsError> {
        Err(SmartsError::NotYetImplemented)
    }
}

// Compile-time assertion that SmartsPattern is thread-safe, per the ETKDG
// multi-conformer generation contract (see spec §6, architect finding H1).
const _: fn() = || {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<SmartsPattern>();
};
