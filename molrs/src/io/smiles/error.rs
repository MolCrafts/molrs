//! Error types shared by the three line notations this module family reads
//! and writes: SMILES, SMARTS and `CGsmiles`.
//!
//! There is one error struct, [`SmilesError`], one list of reasons,
//! [`SmilesErrorKind`], and one field, [`Notation`], saying which language the
//! offending text was meant to be — because many reasons (`TrailingCharacters`
//! and friends) are raised by all three, and the entry point is the only place
//! that knows which was being parsed.

use std::fmt;

use crate::io::smiles::chem::ast::{BondKind, Span};
use molrs::error::MolRsError;

/// Which line notation was being read or written when an error was raised.
///
/// One error type serves all three notations, so the notation is a *fact owned
/// by the entry point* rather than something a reader could infer from the
/// variant name: `TrailingCharacters` is raised by `parse_smiles`,
/// `parse_smarts` and `parse_cgsmiles` alike, and only the entry point knows
/// which language the offending text was meant to be. Every construction site
/// therefore states it — see [`SmilesError::new`].
///
/// [`Display`](fmt::Display) renders the conventional spelling of each
/// notation (`SMILES`, `SMARTS`, `CGsmiles`), which is the prefix of every
/// rendered [`SmilesError`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Notation {
    /// SMILES, including the SMILES fragment body (the fragment dialect is
    /// SMILES widened with bonding descriptors, not a notation of its own).
    Smiles,
    /// SMARTS, the SMILES query language.
    Smarts,
    /// `CGsmiles`, the coarse-grained resolution notation.
    CGsmiles,
}

impl fmt::Display for Notation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let name = match self {
            Notation::Smiles => "SMILES",
            Notation::Smarts => "SMARTS",
            Notation::CGsmiles => "CGsmiles",
        };
        f.write_str(name)
    }
}

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
    /// Which notation was being parsed or written, as the rendered message
    /// names it. Stamped by the entry point, never derived from [`kind`]:
    /// several kinds are shared by all three notations.
    ///
    /// [`kind`]: SmilesError::kind
    pub notation: Notation,
}

/// Why a string was refused: the rule that was broken, with the offending
/// text as payload where naming it helps.
///
/// A variant says *what* went wrong, never *which notation* was being read —
/// most of the first group below is raised by all three — so a caller that
/// wants the language reads [`SmilesError::notation`] instead. The variants
/// fall into three groups: the shared grammar rules, the bonding-descriptor
/// rules of the SMILES fragment dialect (`DescriptorInPlainSmiles` onwards),
/// and the coarse-graph rules of `CGsmiles` (`Cg*`).
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

    // -- CGsmiles coarse-graph kinds ----------------------------------------
    /// A `CGsmiles` block contains nothing: `{}`. The reference implementation
    /// skips such a block silently; molrs names it, because a resolution level
    /// with no node is never what the writer meant.
    CgEmptyBlock,
    /// A node annotation could not be read as a `key=value` pair or as a
    /// positional field: `[#A;]`, `[#A;=1]`, `[#A;q=x]`, or a fourth
    /// positional slot (the table has three: the name, `q` and `w`). The
    /// payload is the offending annotation text.
    CgMalformedAnnotation(String),
    /// A node annotation was read but molrs does not model it yet: a
    /// non-default weight (`[#A;w=2]`, `[#A;0;0.5]`) or a chirality
    /// (`[#A;x=S]`). The payload names the key and the value that was written.
    CgUnsupportedAnnotation {
        /// The annotation key, after positional binding (`w`, `x`, …).
        key: String,
        /// The value as written.
        value: String,
    },
    /// An annotation was written on the wildcard node `[#*;…]`, which stands
    /// for "any fragment" and therefore has no properties of its own. The
    /// payload is the annotation text.
    CgAnnotationOnWildcard(String),
    /// A symbol in bond position is not one of `-`, `=`, `#`, `$`, or a second
    /// bond symbol was written before the first had anything to bond
    /// (`{[#A]=-[#B]}`). The refused symbols include `.`, the zero-order
    /// (virtual) edge, which the notation permits and molrs declines to model.
    CgInvalidBondOrder,
    /// The count after `|` is not a positive decimal integer: `{[#A]|0}`,
    /// `{[#A]|x}`. The payload is the text that was read as the count.
    CgInvalidRepeatCount(String),
    /// A ring closure would repeat an edge the graph already has, or would
    /// bond a node to itself. The coarse graph is a *simple* graph — at most
    /// one edge per pair of nodes, and no self-loops — so a second bond
    /// between the same two nodes is written with a bond-order symbol, not
    /// with a second ring marker. Checked once the block has been read, so
    /// `i` and `j` are final node indices.
    CgDuplicateEdge {
        /// Index of the node that opened the marker.
        i: usize,
        /// Index of the node that closed it.
        j: usize,
    },
    /// `|` was written where the unit to repeat is not a single well-formed
    /// subgraph: inside an open branch, or after a node carrying more than one
    /// branch.
    CgRepeatOnBranchedNode,
    /// A ring marker was opened or closed inside a repeated unit. A marker is
    /// a one-shot identity — [`SmilesErrorKind::CgDuplicateEdge`] forbids two
    /// closures of one marker — so replaying it is either a silent overwrite
    /// or a self-bond.
    CgRepeatOnRingMarker,
    /// A bond symbol has nothing on one side to bond to: written immediately
    /// before the closing `}` of a block (`{[#A]=}`), before the `)` that ends
    /// a branch (`{[#A]([#B]=)}`), or before the first node of the block
    /// (`{=[#A]}`). The input is not exhausted in any of these, so
    /// `UnexpectedEnd` would be a false report.
    CgDanglingBond,
    /// A ring marker is malformed: `%` not followed by exactly two digits, as
    /// in `{[#A]%1}`, or a marker number of zero. Zero is refused in both
    /// spellings — a bare `0` and `%00` — so one-digit markers run `1`..`9`
    /// and the two-digit form spans `%01`..`%99`.
    CgInvalidRingMarker,
}

impl SmilesError {
    /// Build an error at `span` within `input`, reported as an error of
    /// `notation`.
    ///
    /// `notation` has no default on purpose: several kinds are shared by all
    /// three notations, so a defaulted [`Notation::Smiles`] would silently
    /// mislabel every SMARTS and `CGsmiles` diagnostic that reuses one.
    pub fn new(kind: SmilesErrorKind, span: Span, input: &str, notation: Notation) -> Self {
        Self {
            kind,
            span,
            input: input.to_owned(),
            notation,
        }
    }
}

impl SmilesErrorKind {
    /// The one-line reason this kind reports, without the notation prefix or
    /// the caret context that `Display for SmilesError` wraps it in.
    fn message(&self) -> String {
        match self {
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
            SmilesErrorKind::CgEmptyBlock => {
                "empty coarse-graph block '{}' — a resolution level needs at least one node"
                    .to_owned()
            }
            SmilesErrorKind::CgMalformedAnnotation(s) => {
                format!("malformed node annotation '{s}'")
            }
            SmilesErrorKind::CgUnsupportedAnnotation { key, value } => {
                format!("node annotation '{key}={value}' is not supported")
            }
            SmilesErrorKind::CgAnnotationOnWildcard(s) => {
                format!("annotation '{s}' written on the wildcard node '[#*]'")
            }
            SmilesErrorKind::CgInvalidBondOrder => {
                "invalid coarse bond order — write one of '-', '=', '#', '$'".to_owned()
            }
            SmilesErrorKind::CgInvalidRepeatCount(s) => {
                format!("invalid repeat count '{s}' — '|' takes a positive integer")
            }
            SmilesErrorKind::CgDuplicateEdge { i, j } => format!(
                "nodes {i} and {j} are already joined — use bond order symbols instead of a \
                 second edge"
            ),
            SmilesErrorKind::CgRepeatOnBranchedNode => {
                "'|' cannot repeat a unit with an open branch or with more than one branch"
                    .to_owned()
            }
            SmilesErrorKind::CgRepeatOnRingMarker => {
                "'|' cannot repeat a unit that opens or closes a ring marker".to_owned()
            }
            SmilesErrorKind::CgDanglingBond => {
                "bond symbol with nothing to bond to before the end of the block".to_owned()
            }
            SmilesErrorKind::CgInvalidRingMarker => {
                "invalid ring marker — '%' takes exactly two digits".to_owned()
            }
        }
    }
}

/// Renders as one message line, `<notation> parse error at position <n>:
/// <reason>`, optionally followed by two context lines: the input, and a caret
/// under the byte the span starts at. The prefix is [`SmilesError::notation`]
/// as the entry point stamped it — `SMILES`, `SMARTS` or `CGsmiles`.
///
/// The context lines are skipped when there is no input text to point into —
/// errors raised by the IR → graph walker and by the writer carry none — and
/// when the input is longer than 120 bytes, where a caret under a wrapped line
/// helps nobody.
impl fmt::Display for SmilesError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let pos = self.span.start;
        let msg = self.kind.message();
        write!(f, "{} parse error at position {pos}: {msg}", self.notation)?;

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
            Notation::Smiles,
        );
        let s = err.to_string();
        assert!(s.contains("position 3"));
        assert!(s.contains("unexpected character 'X'"));
        assert!(s.contains("CC(X)O"));
        assert!(s.contains("   ^"));
    }

    #[test]
    fn test_into_molrs_error() {
        let err = SmilesError::new(
            SmilesErrorKind::EmptyInput,
            Span::new(0, 0),
            "",
            Notation::Smiles,
        );
        let molrs: MolRsError = err.into();
        let msg = format!("{molrs}");
        assert!(msg.contains("empty input"));
    }

    // -- bonding-descriptor variants ----------------------------------------

    use crate::io::smiles::chem::ast::BondKind;

    /// The message body of a rendered error: the first line after the
    /// `"… position N: "` prefix that every variant shares.
    fn message(kind: SmilesErrorKind) -> String {
        let rendered = SmilesError::new(kind, Span::new(0, 3), "[$]", Notation::Smiles).to_string();
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

    // -- notation prefix ----------------------------------------------------

    /// The full text of a `CGsmiles` error over `{[#PEO][#PEO]}[#X]`, whose
    /// trailing block starts at byte 14.
    fn cgsmiles_trailing_rendered() -> String {
        SmilesError::new(
            SmilesErrorKind::TrailingCharacters,
            Span::new(14, 15),
            "{[#PEO][#PEO]}[#X]",
            Notation::CGsmiles,
        )
        .to_string()
    }

    #[test]
    fn test_display_cgsmiles_reused_kind_names_the_cgsmiles_notation() {
        let rendered = cgsmiles_trailing_rendered();
        assert!(
            rendered.starts_with("CGsmiles parse error at position 14"),
            "rendered error was {rendered:?}"
        );
    }

    #[test]
    fn test_display_cgsmiles_caret_points_at_the_column_of_the_full_input() {
        let rendered = cgsmiles_trailing_rendered();
        let caret = rendered.lines().nth(2).expect("caret line is missing");
        assert_eq!(caret, format!("  {}^", " ".repeat(14)));
    }

    #[test]
    fn test_display_smiles_notation_renders_the_smiles_prefix() {
        let rendered = SmilesError::new(
            SmilesErrorKind::UnclosedBranch,
            Span::new(2, 3),
            "CC(",
            Notation::Smiles,
        )
        .to_string();
        assert!(
            rendered.starts_with("SMILES parse error"),
            "rendered error was {rendered:?}"
        );
    }

    #[test]
    fn test_display_smarts_notation_renders_the_smarts_prefix() {
        let rendered = SmilesError::new(
            SmilesErrorKind::InvalidQueryPrimitive("Q".to_owned()),
            Span::new(1, 2),
            "[Q]",
            Notation::Smarts,
        )
        .to_string();
        assert!(
            rendered.starts_with("SMARTS parse error"),
            "rendered error was {rendered:?}"
        );
    }

    // -- CGsmiles coarse-graph variants -------------------------------------

    /// The message body of a `CGsmiles` error, the counterpart of
    /// [`message`] for the coarse-graph kinds.
    fn cg_message(kind: SmilesErrorKind) -> String {
        let rendered =
            SmilesError::new(kind, Span::new(0, 6), "{[#A]}", Notation::CGsmiles).to_string();
        let first = rendered.lines().next().expect("rendered error is empty");
        match first.split_once(": ") {
            Some((_, body)) => body.to_owned(),
            None => first.to_owned(),
        }
    }

    #[test]
    fn test_display_cg_empty_block_is_non_empty() {
        let msg = cg_message(SmilesErrorKind::CgEmptyBlock);
        assert!(!msg.is_empty());
        assert!(msg.to_lowercase().contains("empty"), "message was {msg:?}");
    }

    #[test]
    fn test_display_cg_malformed_annotation_shows_the_annotation() {
        let msg = cg_message(SmilesErrorKind::CgMalformedAnnotation("q=x".to_owned()));
        assert!(msg.contains("q=x"), "message was {msg:?}");
    }

    #[test]
    fn test_display_cg_unsupported_annotation_shows_key_and_value() {
        let msg = cg_message(SmilesErrorKind::CgUnsupportedAnnotation {
            key: "w".to_owned(),
            value: "0.5".to_owned(),
        });
        assert!(msg.contains('w'), "message was {msg:?}");
        assert!(msg.contains("0.5"), "message was {msg:?}");
    }

    #[test]
    fn test_display_cg_annotation_on_wildcard_shows_the_annotation() {
        let msg = cg_message(SmilesErrorKind::CgAnnotationOnWildcard("q=1".to_owned()));
        assert!(msg.contains("q=1"), "message was {msg:?}");
    }

    #[test]
    fn test_display_cg_invalid_bond_order_is_non_empty() {
        let msg = cg_message(SmilesErrorKind::CgInvalidBondOrder);
        assert!(!msg.is_empty());
        assert!(msg.to_lowercase().contains("bond"), "message was {msg:?}");
    }

    #[test]
    fn test_display_cg_invalid_repeat_count_shows_the_count() {
        let msg = cg_message(SmilesErrorKind::CgInvalidRepeatCount("0".to_owned()));
        assert!(msg.contains('0'), "message was {msg:?}");
    }

    #[test]
    fn test_display_cg_duplicate_edge_names_both_nodes() {
        let msg = cg_message(SmilesErrorKind::CgDuplicateEdge { i: 0, j: 1 });
        assert!(msg.contains('0'), "message was {msg:?}");
        assert!(msg.contains('1'), "message was {msg:?}");
    }

    #[test]
    fn test_display_cg_repeat_on_branched_node_is_non_empty() {
        let msg = cg_message(SmilesErrorKind::CgRepeatOnBranchedNode);
        assert!(!msg.is_empty());
        assert!(msg.to_lowercase().contains("branch"), "message was {msg:?}");
    }

    #[test]
    fn test_display_cg_repeat_on_ring_marker_is_non_empty() {
        let msg = cg_message(SmilesErrorKind::CgRepeatOnRingMarker);
        assert!(!msg.is_empty());
        assert!(msg.to_lowercase().contains("ring"), "message was {msg:?}");
    }

    #[test]
    fn test_display_cg_dangling_bond_is_non_empty() {
        let msg = cg_message(SmilesErrorKind::CgDanglingBond);
        assert!(!msg.is_empty());
        assert!(msg.to_lowercase().contains("bond"), "message was {msg:?}");
    }

    #[test]
    fn test_display_cg_invalid_ring_marker_is_non_empty() {
        let msg = cg_message(SmilesErrorKind::CgInvalidRingMarker);
        assert!(!msg.is_empty());
        assert!(msg.to_lowercase().contains("ring"), "message was {msg:?}");
    }

    #[test]
    fn test_display_cg_messages_are_pairwise_distinct() {
        let messages = [
            cg_message(SmilesErrorKind::CgEmptyBlock),
            cg_message(SmilesErrorKind::CgMalformedAnnotation("q=x".to_owned())),
            cg_message(SmilesErrorKind::CgUnsupportedAnnotation {
                key: "w".to_owned(),
                value: "0.5".to_owned(),
            }),
            cg_message(SmilesErrorKind::CgAnnotationOnWildcard("q=1".to_owned())),
            cg_message(SmilesErrorKind::CgInvalidBondOrder),
            cg_message(SmilesErrorKind::CgInvalidRepeatCount("0".to_owned())),
            cg_message(SmilesErrorKind::CgDuplicateEdge { i: 0, j: 1 }),
            cg_message(SmilesErrorKind::CgRepeatOnBranchedNode),
            cg_message(SmilesErrorKind::CgRepeatOnRingMarker),
            cg_message(SmilesErrorKind::CgDanglingBond),
            cg_message(SmilesErrorKind::CgInvalidRingMarker),
        ];
        let mut sorted = messages.to_vec();
        sorted.sort();
        sorted.dedup();
        assert_eq!(sorted.len(), messages.len(), "messages were {messages:?}");
    }
}
