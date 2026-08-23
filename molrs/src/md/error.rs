//! Errors from the in-process MD engine.

use std::fmt;

use molrs::units::UnitsError;

/// Failures on the MD surface: construction, binding, and policy guards.
#[derive(Debug)]
pub enum MdError {
    /// A constructor or bind argument is out of domain.
    Invalid(String),
    /// A required binding (force provider, mass) is missing.
    Unbound(String),
    /// Neighbour-list completeness / unwrapped-position guard failed.
    Neighbor(String),
    /// Unit conversion through [`molrs::units`] failed.
    Units(UnitsError),
}

impl fmt::Display for MdError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Invalid(msg) | Self::Unbound(msg) | Self::Neighbor(msg) => f.write_str(msg),
            Self::Units(err) => write!(f, "MD unit conversion: {err}"),
        }
    }
}

impl std::error::Error for MdError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Units(err) => Some(err),
            _ => None,
        }
    }
}

impl From<UnitsError> for MdError {
    fn from(err: UnitsError) -> Self {
        Self::Units(err)
    }
}
