//! Error types for SMILES / SMARTS parsing.

use std::fmt;

use crate::chem::ast::Span;
use molrs::error::MolRsError;

/// Error produced by the SMILES / SMARTS parser or validator.
#[derive(Debug, Clone, PartialEq)]
pub struct SmilesError {
    pub kind: SmilesErrorKind,
    pub span: Span,
    /// The original input string (kept for diagnostic display).
    pub input: String,
}

/// Specific error variants.
#[derive(Debug, Clone, PartialEq)]
pub enum SmilesErrorKind {
    /// A character was encountered that is not valid in the current context.
    UnexpectedChar(char),
    /// Reached end-of-input while more tokens were expected.
    UnexpectedEnd,
    /// `[` was opened but never closed.
    UnclosedBracket,
    /// `(` was opened but never closed.
    UnclosedBranch,
    /// A ring-closure digit was opened but never paired.
    UnmatchedRingClosure(u16),
    /// The element symbol is not recognised.
    InvalidElement(String),
    /// A charge specification could not be parsed.
    InvalidCharge,
    /// An empty string was passed to the parser.
    EmptyInput,
    /// Characters remain after the molecule was fully parsed.
    TrailingCharacters,
    /// A SMARTS query primitive is not recognised.
    InvalidQueryPrimitive(String),
    /// `$(` was opened but the matching `)` was not found.
    UnclosedRecursive,
    /// Recursive SMARTS nesting exceeded the depth limit.
    RecursionLimit,
    /// Ring closure bond types are inconsistent between open and close.
    RingBondConflict { rnum: u16 },
}

impl SmilesError {
    /// Convenience constructor.
    pub fn new(kind: SmilesErrorKind, span: Span, input: &str) -> Self {
        Self {
            kind,
            span,
            input: input.to_owned(),
        }
    }
}

impl fmt::Display for SmilesError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let pos = self.span.start;
        let msg = match &self.kind {
            SmilesErrorKind::UnexpectedChar(c) => format!("unexpected character '{c}'"),
            SmilesErrorKind::UnexpectedEnd => "unexpected end of input".to_owned(),
            SmilesErrorKind::UnclosedBracket => "unclosed bracket '['".to_owned(),
            SmilesErrorKind::UnclosedBranch => "unclosed branch '('".to_owned(),
            SmilesErrorKind::UnmatchedRingClosure(n) => {
                format!("unmatched ring closure {n}")
            }
            SmilesErrorKind::InvalidElement(s) => format!("invalid element '{s}'"),
            SmilesErrorKind::InvalidCharge => "invalid charge specification".to_owned(),
            SmilesErrorKind::EmptyInput => "empty input".to_owned(),
            SmilesErrorKind::TrailingCharacters => "trailing characters after molecule".to_owned(),
            SmilesErrorKind::InvalidQueryPrimitive(s) => {
                format!("invalid SMARTS query primitive '{s}'")
            }
            SmilesErrorKind::UnclosedRecursive => "unclosed recursive SMARTS '$('".to_owned(),
            SmilesErrorKind::RecursionLimit => "recursive SMARTS nesting limit exceeded".to_owned(),
            SmilesErrorKind::RingBondConflict { rnum } => {
                format!("conflicting bond types on ring closure {rnum}")
            }
        };
        write!(f, "SMILES parse error at position {pos}: {msg}")?;

        // Caret-style context line (only when input is short enough to be useful).
        if self.input.len() <= 120 {
            write!(f, "\n  {}\n  ", self.input)?;
            for _ in 0..pos.min(self.input.len()) {
                write!(f, " ")?;
            }
            write!(f, "^")?;
        }
        Ok(())
    }
}

impl std::error::Error for SmilesError {}

impl From<SmilesError> for MolRsError {
    fn from(e: SmilesError) -> Self {
        MolRsError::Parse {
            line: None,
            message: e.to_string(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_display_with_caret() {
        let err = SmilesError::new(
            SmilesErrorKind::UnexpectedChar('X'),
            Span::new(3, 4),
            "CC(X)O",
        );
        let s = err.to_string();
        assert!(s.contains("position 3"));
        assert!(s.contains("unexpected character 'X'"));
        assert!(s.contains("CC(X)O"));
        assert!(s.contains("   ^"));
    }

    #[test]
    fn test_into_molrs_error() {
        let err = SmilesError::new(SmilesErrorKind::EmptyInput, Span::new(0, 0), "");
        let molrs: MolRsError = err.into();
        let msg = format!("{molrs}");
        assert!(msg.contains("empty input"));
    }
}
