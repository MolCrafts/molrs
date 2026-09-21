//! Error types for SMILES / SMARTS parsing.

use std::fmt;

use crate::io::smiles::chem::ast::{BondKind, Span};
use molrs::error::MolRsError;

/// Error produced by any stage of the line-notation pipeline: the parser, the
/// validator, the IR → graph walker, or the writer.
#[derive(Debug, Clone, PartialEq)]
pub struct SmilesError {
    /// Which rule was broken; the payload, if any, is the offending text.
    pub kind: SmilesErrorKind,
    /// Byte range of the offending text within `input`. Its start is the
    /// position reported by [`Display`](fmt::Display) and the column the caret
    /// points at.
    pub span: Span,
    /// The original input string (kept for diagnostic display). Empty when the
    /// error was raised away from the parser — by the IR → graph walker or the
    /// writer, which are handed an IR and never see the text it came from.
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
    /// Graph → IR / IR → string emit failure (message is the reason).
    Emit(String),
    /// A bonding descriptor (`[$]`, `[<]`, `[>]`, `[!]`) met a stage that
    /// speaks plain SMILES, which has no such notation: `parse_smiles` on a
    /// descriptor bracket, `validate_smiles` on a node carrying one, or
    /// `write_smiles` / `write_smarts` asked to emit one. This is a routing
    /// error, and the rendered message says where to go instead — it names
    /// `parse_fragment_smiles`, the entry point of the dialect that does
    /// accept descriptors.
    ///
    /// SMARTS *parsing* never raises it: there `[$(C)]` is recursive SMARTS
    /// and `[!C]` is a negated atom, so the bracket belongs to the query
    /// parser and is left to it.
    DescriptorInPlainSmiles,
    /// An IR carrying bonding descriptors was handed to `to_atomistic`, the
    /// plain IR → graph conversion, which has nowhere to put them and would
    /// silently drop them. The rendered message names `fragment_to_atomistic`,
    /// the conversion that returns them alongside the graph.
    DescriptorsUnconvertible,
    /// A bond symbol was written *inside* a descriptor bracket — the
    /// `BigSMILES` spelling `[<=1]`, or `[$-]`. `CGsmiles` writes the order
    /// outside the bracket (`CC=[$]`), and accepting both spellings would give
    /// one meaning two notations, so the in-bracket form is refused rather
    /// than translated.
    BondInsideDescriptor,
    /// A descriptor label is not ASCII alphanumeric, as in `[$a+]`; the
    /// payload is the rejected label text (here `a+`).
    InvalidDescriptorLabel(String),
    /// A bond order was written next to a descriptor that no formed bond could
    /// take — aromatic (`c:[$]`), directional (`/`, `\`), wildcard (`~`) or
    /// ring (`@`); the payload is the offending kind. Raised by the parser at
    /// the descriptor's construction site, and by `write_fragment_smiles` for
    /// a hand-built IR that no parser could have produced.
    InvalidDescriptorOrder(BondKind),
    /// A descriptor was written where no atom can anchor it: `[$]` alone, or
    /// `C.[$]`, whose second `.`-separated component contains no atom for the
    /// descriptor to bind to.
    DanglingDescriptor,
    /// A `CGsmiles` atom-level annotation — a weight `[C;0.5]`, a chirality
    /// `[C;1;S]`, or wildcard overloading `[*;s=C,0]` — was written. The
    /// fragment dialect does not support these, and this error names the
    /// feature rather than claiming a missing `]`. The payload is the
    /// annotation text between `;` and the closing `]`, or the end of input if
    /// the bracket is never closed. Raised in the fragment dialect
    /// only; `parse_smiles("[C;0.5]")` still reports
    /// [`SmilesErrorKind::UnclosedBracket`].
    AtomAnnotationUnsupported(String),
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

/// Renders as one message line, `SMILES parse error at position <n>: <reason>`,
/// optionally followed by two context lines: the input, and a caret under the
/// byte the span starts at.
///
/// The context lines are skipped when there is no input text to point into —
/// errors raised by the IR → graph walker and by the writer carry none — and
/// when the input is longer than 120 bytes, where a caret under a wrapped line
/// helps nobody.
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
            SmilesErrorKind::Emit(s) => format!("emit error: {s}"),
            SmilesErrorKind::DescriptorInPlainSmiles => {
                "bonding descriptor is not plain SMILES notation \
                 — parse a fragment body with parse_fragment_smiles"
                    .to_owned()
            }
            SmilesErrorKind::DescriptorsUnconvertible => "this atom carries bonding descriptors \
                 — convert a fragment body with fragment_to_atomistic"
                .to_owned(),
            SmilesErrorKind::BondInsideDescriptor => {
                "bond symbol inside a bonding descriptor bracket \
                 — the bond order is written outside it, as in CC=[$]"
                    .to_owned()
            }
            SmilesErrorKind::InvalidDescriptorLabel(s) => {
                format!("bonding descriptor label '{s}' is not alphanumeric")
            }
            SmilesErrorKind::InvalidDescriptorOrder(k) => {
                format!("bond order {k:?} cannot annotate a bonding descriptor")
            }
            SmilesErrorKind::DanglingDescriptor => {
                "bonding descriptor has no atom to bind to".to_owned()
            }
            SmilesErrorKind::AtomAnnotationUnsupported(s) => {
                format!("atom annotation ';{s}' is unsupported")
            }
        };
        write!(f, "SMILES parse error at position {pos}: {msg}")?;

        // Caret-style context line, when there is an input to point into and
        // it is short enough to be useful. Errors raised away from the scanner
        // (the IR → graph walker, the writer) carry no input text, and a caret
        // under an empty line points at nothing.
        if !self.input.is_empty() && self.input.len() <= 120 {
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

    // -- bonding-descriptor variants ----------------------------------------

    use crate::io::smiles::chem::ast::BondKind;

    /// The message body of a rendered error: the first line after the
    /// `"… position N: "` prefix that every variant shares.
    fn message(kind: SmilesErrorKind) -> String {
        let rendered = SmilesError::new(kind, Span::new(0, 3), "[$]").to_string();
        let first = rendered.lines().next().expect("rendered error is empty");
        match first.split_once(": ") {
            Some((_, body)) => body.to_owned(),
            None => first.to_owned(),
        }
    }

    #[test]
    fn test_display_descriptor_in_plain_smiles_names_fragment_parser() {
        let msg = message(SmilesErrorKind::DescriptorInPlainSmiles);
        assert!(msg.contains("parse_fragment_smiles"), "message was {msg:?}");
    }

    #[test]
    fn test_display_descriptors_unconvertible_names_fragment_converter() {
        let msg = message(SmilesErrorKind::DescriptorsUnconvertible);
        assert!(msg.contains("fragment_to_atomistic"), "message was {msg:?}");
    }

    #[test]
    fn test_display_bond_inside_descriptor_is_non_empty() {
        let msg = message(SmilesErrorKind::BondInsideDescriptor);
        assert!(!msg.is_empty());
        assert!(msg.contains("descriptor"), "message was {msg:?}");
    }

    #[test]
    fn test_display_invalid_descriptor_label_shows_the_label() {
        let msg = message(SmilesErrorKind::InvalidDescriptorLabel("a+".to_owned()));
        assert!(msg.contains("a+"), "message was {msg:?}");
    }

    #[test]
    fn test_display_invalid_descriptor_order_is_non_empty() {
        let msg = message(SmilesErrorKind::InvalidDescriptorOrder(BondKind::Aromatic));
        assert!(!msg.is_empty());
        assert!(msg.contains("descriptor"), "message was {msg:?}");
    }

    #[test]
    fn test_display_dangling_descriptor_is_non_empty() {
        let msg = message(SmilesErrorKind::DanglingDescriptor);
        assert!(!msg.is_empty());
        assert!(msg.contains("descriptor"), "message was {msg:?}");
    }

    #[test]
    fn test_display_atom_annotation_unsupported_shows_the_annotation() {
        let msg = message(SmilesErrorKind::AtomAnnotationUnsupported("0.5".to_owned()));
        assert!(msg.contains("0.5"), "message was {msg:?}");
        assert!(
            msg.to_lowercase().contains("unsupported"),
            "message was {msg:?}"
        );
    }

    #[test]
    fn test_display_atom_annotation_unsupported_does_not_claim_unclosed_bracket() {
        let msg = message(SmilesErrorKind::AtomAnnotationUnsupported("0.5".to_owned()));
        assert!(
            !msg.to_lowercase().contains("unclosed"),
            "message was {msg:?}"
        );
    }

    #[test]
    fn test_display_descriptor_messages_are_pairwise_distinct() {
        let messages = [
            message(SmilesErrorKind::DescriptorInPlainSmiles),
            message(SmilesErrorKind::DescriptorsUnconvertible),
            message(SmilesErrorKind::BondInsideDescriptor),
            message(SmilesErrorKind::InvalidDescriptorLabel("a+".to_owned())),
            message(SmilesErrorKind::InvalidDescriptorOrder(BondKind::Aromatic)),
            message(SmilesErrorKind::DanglingDescriptor),
            message(SmilesErrorKind::AtomAnnotationUnsupported("0.5".to_owned())),
        ];
        let mut sorted = messages.to_vec();
        sorted.sort();
        sorted.dedup();
        assert_eq!(sorted.len(), messages.len(), "messages were {messages:?}");
    }
}
