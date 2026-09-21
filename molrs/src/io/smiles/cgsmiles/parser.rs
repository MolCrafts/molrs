//! Recursive-descent parser for one `CGsmiles` coarse-graph block.
//!
//! The grammar of a block is small enough to read in one token loop:
//! [`CgParser::step`] reads the next token and folds it into the graph being
//! built, and every construct — a node, a bonding descriptor, a bond symbol, a
//! branch, a ring marker, a `|n` repeat — is one arm of it.
//!
//! Two decisions are worth stating before the code. The bond symbol is *not*
//! a different token for each position the notation allows it in: it is read
//! once and remembered as `pending`, and whatever comes next spends it — a
//! node as the chain bond (which is also how the branch bonds of
//! `{[#A](=[#B])[#C]}` and `{[#A]([#B])=[#C]}` arrive, since a node follows
//! the symbol in both), a ring marker as the ring bond, a descriptor bracket
//! as the order of the bond its pairing will create, and `|` as the bond that
//! chains the repeated copies. And the `|n` repeat re-reads the unit's bytes
//! by rewinding this same scanner over the full input
//! ([`Scanner::seek`](crate::io::smiles::chem::scanner::Scanner::seek)), so a
//! diagnostic raised inside a copy still points at a position in the string
//! the caller passed in.
//!
//! Comments below cite grammar rules as `R2.4`, `R2.20` and so on. Those are
//! the numbered rules of `.claude/specs/cgsmiles-01b-graph.md` § Domain basis,
//! each of which records where it came from — the `CGsmiles` documentation,
//! the reference implementation at a named commit, or an inference marked as
//! such. `F1.9` and the like name that spec's worked fixtures. `01c` and
//! `01d` are the next two specs in the same chain
//! (`.claude/specs/cgsmiles-01c-fragments.md`, which resolves a node name to
//! a fragment body, and `cgsmiles-01d-resolve.md`, which expands a level into
//! atoms); naming one marks work this file deliberately leaves undone.

use std::collections::{HashMap, HashSet};

use crate::core::types::F;
use crate::io::smiles::cgsmiles::ast::{CGBondOrder, CGEdge, CGGraph, CGNode, CGSmilesIR};
use crate::io::smiles::chem::ast::{BondKind, BondingDescriptor, DescriptorKind, Span};
use crate::io::smiles::chem::scanner::Scanner;
use crate::io::smiles::chem::validation::validate_descriptor;
use crate::io::smiles::error::{Notation, SmilesError, SmilesErrorKind};

/// The only mapping weight the parser accepts: `w = 1`, the notation's own
/// default (R2.14). `w` is the dimensionless mapping weight the notation
/// attaches to a node; molrs models no other value yet, so writing one is
/// refused rather than silently dropped.
const DEFAULT_WEIGHT: F = 1.0;

/// A node's annotations after the dialect's reserved keys have been read out
/// of them.
struct BoundAnnotations {
    /// The partial charge written as `q`, in elementary charge units `e`.
    charge: Option<F>,
    /// Every pair the table does not reserve, verbatim and in written order.
    rest: Vec<(String, String)>,
}

/// A ring marker that has been opened and is waiting for its closing twin.
struct OpenRing {
    /// The node the opening marker was written after.
    node: usize,
    /// The bond order written before the *opening* marker, which is the order
    /// the closure edge takes (R2.5).
    order: CGBondOrder,
    /// Byte range of the opening marker, for the unmatched-closure error.
    span: Span,
}

/// The stretch of input `|n` would repeat: the node most recently written at
/// branch depth 0, plus everything attached to it since.
#[derive(Clone)]
struct RepeatUnit {
    /// Byte offset the unit starts at (the node's `[`).
    start: usize,
    /// Byte offset one past the unit's last token. A bond symbol never moves
    /// it, which is what keeps the symbol before `|` out of the copied text.
    end: usize,
    /// Index of the node each copy is chained through (R2.20: the anchor is
    /// part of the unit).
    anchor: usize,
    /// How many branches were opened on the anchor. More than one is
    /// [`SmilesErrorKind::CgRepeatOnBranchedNode`] (R2.23).
    branches: usize,
    /// Whether a ring marker was opened or closed inside the unit, which
    /// [`SmilesErrorKind::CgRepeatOnRingMarker`] refuses.
    ring_touched: bool,
}

/// Parser state for one `CGsmiles` block.
///
/// Private by design: it has exactly one user-visible step, so the public
/// surface is the free function
/// [`parse_cgsmiles`](crate::io::smiles::parse_cgsmiles) that runs it, in the
/// shape `parse_smiles` and `parse_smarts` already use.
pub(super) struct CgParser<'a> {
    /// Cursor over the whole input, and the source of every diagnostic.
    scanner: Scanner<'a>,
    /// Nodes of the level being built, in parse order.
    nodes: Vec<CGNode>,
    /// Edges of the level being built, in parse order.
    edges: Vec<CGEdge>,
    /// Ring markers opened and not yet closed, by marker number.
    open_rings: HashMap<u16, OpenRing>,
    /// Anchor node of each open branch, innermost last.
    branches: Vec<usize>,
    /// The node the next chain element bonds to.
    current: Option<usize>,
    /// A bond symbol that has been read and not yet spent, with its offset.
    pending: Option<(CGBondOrder, usize)>,
    /// The stretch of input a `|n` written here would repeat — `None` before
    /// the first node of the block, and again after a `|n` consumes it.
    unit: Option<RepeatUnit>,
}

impl<'a> CgParser<'a> {
    /// A parser over the whole of `text`, reading it as
    /// [`Notation::CGsmiles`].
    pub(super) fn new(text: &'a str) -> Self {
        Self {
            scanner: Scanner::new(text, Notation::CGsmiles),
            nodes: Vec::new(),
            edges: Vec::new(),
            open_rings: HashMap::new(),
            branches: Vec::new(),
            current: None,
            pending: None,
            unit: None,
        }
    }

    /// Read the one block this version of the parser accepts and return its
    /// graph.
    ///
    /// # Errors
    ///
    /// Every malformed input named by [`SmilesErrorKind`]'s eleven `Cg*`
    /// variants, plus eight kinds shared with the atomistic notations —
    /// [`SmilesErrorKind::EmptyInput`], [`SmilesErrorKind::UnexpectedChar`],
    /// [`SmilesErrorKind::UnexpectedEnd`], [`SmilesErrorKind::UnclosedBracket`],
    /// [`SmilesErrorKind::UnclosedBranch`],
    /// [`SmilesErrorKind::UnmatchedRingClosure`],
    /// [`SmilesErrorKind::RingBondConflict`] and
    /// [`SmilesErrorKind::TrailingCharacters`] — and the three descriptor
    /// kinds [`SmilesErrorKind::BondInsideDescriptor`],
    /// [`SmilesErrorKind::InvalidDescriptorLabel`] and
    /// [`SmilesErrorKind::DanglingDescriptor`] that `validate_descriptor` and
    /// `anchor_descriptor` raise.
    /// [`SmilesErrorKind::InvalidDescriptorOrder`] cannot occur: a coarse bond
    /// order only ever maps to one of the four orders it allows.
    pub(super) fn parse(mut self) -> Result<CGSmilesIR, SmilesError> {
        if self.scanner.input().is_empty() {
            return Err(self.scanner.error(SmilesErrorKind::EmptyInput));
        }
        self.scanner.expect('{')?;
        while self.scanner.peek() != Some('}') {
            if self.scanner.is_done() {
                return Err(self.scanner.error(SmilesErrorKind::UnexpectedEnd));
            }
            self.step()?;
        }
        self.finish()
    }

    /// Close the block and hand back the level it describes.
    fn finish(mut self) -> Result<CGSmilesIR, SmilesError> {
        if let Some((_, at)) = self.pending {
            let span = Span::new(at, at + 1);
            return Err(self.scanner.error_at(SmilesErrorKind::CgDanglingBond, span));
        }
        if !self.branches.is_empty() {
            return Err(self.scanner.error(SmilesErrorKind::UnclosedBranch));
        }
        self.scanner.advance(); // '}'
        if self.nodes.is_empty() {
            let span = self.scanner.span_from(0);
            return Err(self.scanner.error_at(SmilesErrorKind::CgEmptyBlock, span));
        }
        if let Some((&rnum, open)) = self.open_rings.iter().next() {
            let kind = SmilesErrorKind::UnmatchedRingClosure(rnum);
            return Err(self.scanner.error_at(kind, open.span));
        }
        self.check_simple_graph()?;
        if !self.scanner.is_done() {
            return Err(self.scanner.error(SmilesErrorKind::TrailingCharacters));
        }
        let span = self.scanner.span_from(0);
        let level = CGGraph {
            nodes: self.nodes,
            edges: self.edges,
        };
        Ok(CGSmilesIR {
            levels: vec![level],
            span,
        })
    }

    /// Refuse a multigraph: a second edge between one pair of nodes, or an
    /// edge from a node to itself (R2.12 — "use bond order symbols instead").
    ///
    /// A pass over the finished edge list, in the shape of
    /// `chem::validation::validate_ring_closures` over a finished IR, and it
    /// has to be one: the duplicate only comes into being when the closing
    /// marker is read, while a structural error written after that closure —
    /// a `|` over a unit that closed a ring — is the earlier complaint about
    /// the same text and must be the one reported.
    fn check_simple_graph(&self) -> Result<(), SmilesError> {
        let mut seen: HashSet<(usize, usize)> = HashSet::new();
        for edge in &self.edges {
            let pair = (edge.i.min(edge.j), edge.i.max(edge.j));
            if edge.i == edge.j || !seen.insert(pair) {
                let kind = SmilesErrorKind::CgDuplicateEdge {
                    i: edge.i,
                    j: edge.j,
                };
                return Err(self.scanner.error_at(kind, edge.span));
            }
        }
        Ok(())
    }

    /// Read the next token of the block and fold it into the graph.
    ///
    /// Every token but one also extends the repeatable unit to the cursor,
    /// which is why [`CgParser::mark_unit_end`] is called here rather than in
    /// each arm. The exception is the bond symbol: it returns early, so the
    /// symbol written before a `|` stays out of the copied text (R2.20).
    fn step(&mut self) -> Result<(), SmilesError> {
        match self.scanner.peek() {
            Some('[') if self.at_descriptor() => self.read_descriptor()?,
            Some('[') => self.read_node()?,
            Some('(') => self.open_branch()?,
            Some(')') => self.close_branch()?,
            Some('|') => self.read_repeat()?,
            Some('%') => self.read_ring_marker()?,
            Some(c) if c.is_ascii_digit() => self.read_ring_marker()?,
            Some(c) => return self.read_bond(c),
            None => return Err(self.scanner.error(SmilesErrorKind::UnexpectedEnd)),
        }
        self.mark_unit_end();
        Ok(())
    }

    // -- vocabulary ---------------------------------------------------------

    /// True when the cursor sits on a bonding-descriptor bracket rather than
    /// on a node: `[` followed by one of `$ < > !`. `[#` starts a node, and
    /// the two are told apart by this one character of lookahead.
    fn at_descriptor(&self) -> bool {
        self.scanner.peek() == Some('[')
            && matches!(self.scanner.peek_next(), Some('$' | '<' | '>' | '!'))
    }

    /// The multiplicity a bond symbol writes, or `None` for a character that
    /// is not one. `.` (order 0) is deliberately not among them.
    fn bond_order(ch: char) -> Option<CGBondOrder> {
        match ch {
            '-' => Some(CGBondOrder::Single),
            '=' => Some(CGBondOrder::Double),
            '#' => Some(CGBondOrder::Triple),
            '$' => Some(CGBondOrder::Quadruple),
            _ => None,
        }
    }

    /// The atomistic bond kind a coarse multiplicity annotates a bonding
    /// descriptor with: a descriptor's order is stored as a `BondKind`,
    /// because the bond its pairing creates is an ordinary atomistic bond.
    fn bond_kind(order: CGBondOrder) -> BondKind {
        match order {
            CGBondOrder::Single => BondKind::Single,
            CGBondOrder::Double => BondKind::Double,
            CGBondOrder::Triple => BondKind::Triple,
            CGBondOrder::Quadruple => BondKind::Quadruple,
        }
    }

    /// Spend the pending bond symbol, defaulting to a single bond.
    fn take_pending(&mut self) -> CGBondOrder {
        self.pending
            .take()
            .map_or(CGBondOrder::Single, |(order, _)| order)
    }

    /// Extend the repeatable unit up to the cursor.
    fn mark_unit_end(&mut self) {
        let pos = self.scanner.pos();
        if let Some(unit) = self.unit.as_mut() {
            unit.end = pos;
        }
    }

    // -- bonds and branches -------------------------------------------------

    /// Remember a bond symbol for whatever token comes next.
    fn read_bond(&mut self, ch: char) -> Result<(), SmilesError> {
        let at = self.scanner.pos();
        let span = Span::new(at, at + 1);
        let Some(order) = Self::bond_order(ch) else {
            return Err(self
                .scanner
                .error_at(SmilesErrorKind::CgInvalidBondOrder, span));
        };
        if self.pending.is_some() {
            return Err(self
                .scanner
                .error_at(SmilesErrorKind::CgInvalidBondOrder, span));
        }
        self.scanner.advance();
        self.pending = Some((order, at));
        Ok(())
    }

    /// Open `(`: push the node written before it as the branch's anchor.
    fn open_branch(&mut self) -> Result<(), SmilesError> {
        let Some(anchor) = self.current else {
            return Err(self.scanner.error(SmilesErrorKind::UnexpectedChar('(')));
        };
        if self.branches.is_empty()
            && let Some(unit) = self.unit.as_mut()
        {
            unit.branches += 1;
        }
        self.scanner.advance();
        self.branches.push(anchor);
        Ok(())
    }

    /// Close `)`: the chain continues from the branch's anchor, not from the
    /// last node inside the branch.
    fn close_branch(&mut self) -> Result<(), SmilesError> {
        if let Some((_, at)) = self.pending {
            let span = Span::new(at, at + 1);
            return Err(self.scanner.error_at(SmilesErrorKind::CgDanglingBond, span));
        }
        let Some(anchor) = self.branches.pop() else {
            return Err(self.scanner.error(SmilesErrorKind::UnexpectedChar(')')));
        };
        self.scanner.advance();
        self.current = Some(anchor);
        Ok(())
    }

    // -- nodes --------------------------------------------------------------

    /// Read `[#NAME;annotations]` and append it to the level.
    fn read_node(&mut self) -> Result<(), SmilesError> {
        let start = self.scanner.pos();
        self.scanner.expect('[')?;
        self.scanner.expect('#')?;
        let name_start = self.scanner.pos();
        while matches!(self.scanner.peek(), Some(c) if c.is_ascii_alphanumeric() || c == '*') {
            self.scanner.advance();
        }
        let name = self.scanner.input()[name_start..self.scanner.pos()].to_owned();
        if name.is_empty() {
            return Err(match self.scanner.peek() {
                Some(c) => self.scanner.error(SmilesErrorKind::UnexpectedChar(c)),
                None => self.scanner.error(SmilesErrorKind::UnexpectedEnd),
            });
        }
        let fields = self.read_annotation_fields(start)?;
        self.scanner.expect(']')?;
        let span = self.scanner.span_from(start);
        let bound = self.bind_annotations(&name, &fields, span)?;
        self.push_node(
            CGNode {
                name,
                charge: bound.charge,
                annotations: bound.rest,
                descriptors: Vec::new(),
                span,
            },
            start,
        )
    }

    /// Collect the raw `;`-separated annotation fields of a node, unparsed.
    fn read_annotation_fields(
        &mut self,
        node_start: usize,
    ) -> Result<Vec<(String, Span)>, SmilesError> {
        let mut fields = Vec::new();
        while self.scanner.peek() == Some(';') {
            self.scanner.advance();
            let start = self.scanner.pos();
            while !matches!(self.scanner.peek(), Some(';' | ']') | None) {
                self.scanner.advance();
            }
            if self.scanner.is_done() {
                let span = self.scanner.span_from(node_start);
                return Err(self
                    .scanner
                    .error_at(SmilesErrorKind::UnclosedBracket, span));
            }
            let text = self.scanner.input()[start..self.scanner.pos()].to_owned();
            fields.push((text, self.scanner.span_from(start)));
        }
        Ok(fields)
    }

    /// Bind the raw annotation fields to the dialect's reserved table.
    ///
    /// The table is positional (R2.14): slot 1 is the `#NAME` itself, slot 2
    /// is `q`, slot 3 is `w`. A field spelled `key=value` names its own key
    /// and takes no slot; a bare field takes the next one, and a fourth is
    /// [`SmilesErrorKind::CgMalformedAnnotation`]. Everything the table does
    /// not reserve is kept verbatim (R2.16).
    fn bind_annotations(
        &self,
        name: &str,
        fields: &[(String, Span)],
        span: Span,
    ) -> Result<BoundAnnotations, SmilesError> {
        if name == "*" && !fields.is_empty() {
            let text = fields
                .iter()
                .map(|(field, _)| field.as_str())
                .collect::<Vec<_>>()
                .join(";");
            let kind = SmilesErrorKind::CgAnnotationOnWildcard(text);
            return Err(self.scanner.error_at(kind, span));
        }
        let mut slot = 2usize;
        let mut bound = BoundAnnotations {
            charge: None,
            rest: Vec::new(),
        };
        for (text, field_span) in fields {
            let (key, value) = self.split_annotation(text, *field_span, &mut slot)?;
            match key.as_str() {
                "q" => bound.charge = Some(self.read_charge(&value, text, *field_span)?),
                "w" => self.check_weight(&value, text, *field_span)?,
                "x" => {
                    let kind = SmilesErrorKind::CgUnsupportedAnnotation { key, value };
                    return Err(self.scanner.error_at(kind, *field_span));
                }
                _ => bound.rest.push((key, value)),
            }
        }
        Ok(bound)
    }

    /// Split one annotation field into the key it names and the value it
    /// carries, taking a positional slot when it names no key.
    fn split_annotation(
        &self,
        text: &str,
        span: Span,
        slot: &mut usize,
    ) -> Result<(String, String), SmilesError> {
        let malformed = || {
            let kind = SmilesErrorKind::CgMalformedAnnotation(text.to_owned());
            self.scanner.error_at(kind, span)
        };
        if let Some((key, value)) = text.split_once('=') {
            if key.is_empty() || value.is_empty() || value.contains('=') {
                return Err(malformed());
            }
            return Ok((key.to_owned(), value.to_owned()));
        }
        if text.is_empty() {
            return Err(malformed());
        }
        let key = match *slot {
            2 => "q",
            3 => "w",
            _ => return Err(malformed()),
        };
        *slot += 1;
        Ok((key.to_owned(), text.to_owned()))
    }

    /// Read `q` as a partial charge in `e`.
    fn read_charge(&self, value: &str, text: &str, span: Span) -> Result<F, SmilesError> {
        value.parse::<F>().map_err(|_| {
            let kind = SmilesErrorKind::CgMalformedAnnotation(text.to_owned());
            self.scanner.error_at(kind, span)
        })
    }

    /// Accept `w` only at its default value.
    ///
    /// A mapping weight is meaningful notation that molrs v1 does not model;
    /// refusing a non-default one is a stated choice, not a parse failure, so
    /// it names the key and the value rather than the character.
    fn check_weight(&self, value: &str, text: &str, span: Span) -> Result<(), SmilesError> {
        let Ok(weight) = value.parse::<F>() else {
            let kind = SmilesErrorKind::CgMalformedAnnotation(text.to_owned());
            return Err(self.scanner.error_at(kind, span));
        };
        if weight == DEFAULT_WEIGHT {
            return Ok(());
        }
        let kind = SmilesErrorKind::CgUnsupportedAnnotation {
            key: "w".to_owned(),
            value: value.to_owned(),
        };
        Err(self.scanner.error_at(kind, span))
    }

    /// Append a node, bond it to the chain, and start a new repeatable unit
    /// when it is written at branch depth 0.
    fn push_node(&mut self, node: CGNode, start: usize) -> Result<(), SmilesError> {
        let span = node.span;
        let index = self.nodes.len();
        match self.current {
            Some(previous) => {
                let order = self.take_pending();
                self.nodes.push(node);
                self.edges.push(CGEdge {
                    i: previous,
                    j: index,
                    order,
                    span,
                });
            }
            None => {
                if let Some((_, at)) = self.pending {
                    let at_span = Span::new(at, at + 1);
                    return Err(self
                        .scanner
                        .error_at(SmilesErrorKind::CgDanglingBond, at_span));
                }
                self.nodes.push(node);
            }
        }
        self.current = Some(index);
        if self.branches.is_empty() {
            self.unit = Some(RepeatUnit {
                start,
                end: self.scanner.pos(),
                anchor: index,
                branches: 0,
                ring_touched: false,
            });
        }
        Ok(())
    }

    // -- bonding descriptors ------------------------------------------------

    /// Read `[glyph label]` and anchor it on the node written before it.
    fn read_descriptor(&mut self) -> Result<(), SmilesError> {
        let start = self.scanner.pos();
        self.scanner.expect('[')?;
        let kind = match self.scanner.advance() {
            Some('$') => DescriptorKind::Symmetric,
            Some('<') => DescriptorKind::Left,
            Some('>') => DescriptorKind::Right,
            Some('!') => DescriptorKind::Shared,
            Some(c) => return Err(self.scanner.error(SmilesErrorKind::UnexpectedChar(c))),
            None => return Err(self.scanner.error(SmilesErrorKind::UnexpectedEnd)),
        };
        let label_start = self.scanner.pos();
        loop {
            match self.scanner.peek() {
                Some(']') => break,
                Some(c) if Self::bond_order(c).is_some() => {
                    return Err(self.scanner.error(SmilesErrorKind::BondInsideDescriptor));
                }
                Some(_) => {
                    self.scanner.advance();
                }
                None => {
                    let span = self.scanner.span_from(start);
                    return Err(self
                        .scanner
                        .error_at(SmilesErrorKind::UnclosedBracket, span));
                }
            }
        }
        let label = self.scanner.input()[label_start..self.scanner.pos()].to_owned();
        self.scanner.advance(); // ']'
        let order = self.pending.take().map(|(o, _)| Self::bond_kind(o));
        let descriptor = BondingDescriptor { kind, label, order };
        self.anchor_descriptor(descriptor, self.scanner.span_from(start))
    }

    /// Validate a descriptor and hang it on the current node.
    ///
    /// `validate_descriptor` is the shared grammar check; it is told which
    /// notation to stamp, so its error needs no re-stamping here.
    fn anchor_descriptor(
        &mut self,
        descriptor: BondingDescriptor,
        span: Span,
    ) -> Result<(), SmilesError> {
        validate_descriptor(&descriptor, span, self.scanner.input(), Notation::CGsmiles)?;
        let node = self
            .current
            .and_then(|index| self.nodes.get_mut(index))
            .ok_or_else(|| {
                self.scanner
                    .error_at(SmilesErrorKind::DanglingDescriptor, span)
            })?;
        node.descriptors.push(descriptor);
        Ok(())
    }

    // -- ring markers -------------------------------------------------------

    /// Read `1`..`9` or `%nn` and open or close the ring it names.
    fn read_ring_marker(&mut self) -> Result<(), SmilesError> {
        let start = self.scanner.pos();
        let Some(glyph) = self.scanner.peek() else {
            return Err(self.scanner.error(SmilesErrorKind::UnexpectedEnd));
        };
        let Some(node) = self.current else {
            return Err(self.scanner.error(SmilesErrorKind::UnexpectedChar(glyph)));
        };
        let rnum = self.read_marker_number()?;
        if let Some(unit) = self.unit.as_mut() {
            unit.ring_touched = true;
        }
        match self.open_rings.remove(&rnum) {
            Some(open) => self.close_ring(rnum, open, node, start)?,
            None => {
                let order = self.take_pending();
                let span = self.scanner.span_from(start);
                self.open_rings.insert(rnum, OpenRing { node, order, span });
            }
        }
        Ok(())
    }

    /// The marker number: one digit `1`..`9`, or `%` and exactly two digits.
    ///
    /// `%` is restricted to two digits as the reference implementation's own
    /// docstring specifies, in agreement with OpenSMILES, even though that
    /// implementation accepts an arbitrary run (R2.10). Zero is not a marker
    /// number in either spelling — a bare `0` and `%00` name the same absent
    /// marker, so both are refused by the one rule.
    fn read_marker_number(&mut self) -> Result<u16, SmilesError> {
        let start = self.scanner.pos();
        if self.scanner.peek() != Some('%') {
            return match self.scanner.eat_digit() {
                Some(digit) if digit > 0 => Ok(u16::from(digit)),
                _ => Err(self.invalid_marker(start)),
            };
        }
        self.scanner.advance();
        let digits = self.scanner.eat_digits();
        if digits.len() != 2 {
            return Err(self.invalid_marker(start));
        }
        match digits.parse::<u16>() {
            Ok(rnum) if rnum > 0 => Ok(rnum),
            _ => Err(self.invalid_marker(start)),
        }
    }

    /// The malformed-marker error, spanning the marker text read so far.
    fn invalid_marker(&self, start: usize) -> SmilesError {
        let span = self.scanner.span_from(start);
        self.scanner
            .error_at(SmilesErrorKind::CgInvalidRingMarker, span)
    }

    /// Close an open marker, emitting the ring edge at the closing marker.
    ///
    /// The order is the one written before the *opening* marker (R2.5); a
    /// different order written before the closing one is a conflict, not an
    /// override. Whether the closure duplicates an edge the graph already has
    /// is decided by [`CgParser::check_simple_graph`] once the block is read,
    /// not here.
    fn close_ring(
        &mut self,
        rnum: u16,
        open: OpenRing,
        node: usize,
        start: usize,
    ) -> Result<(), SmilesError> {
        let span = self.scanner.span_from(start);
        if let Some((order, at)) = self.pending.take()
            && order != open.order
        {
            let kind = SmilesErrorKind::RingBondConflict { rnum };
            return Err(self.scanner.error_at(kind, Span::new(at, at + 1)));
        }
        self.edges.push(CGEdge {
            i: open.node,
            j: node,
            order: open.order,
            span,
        });
        Ok(())
    }

    // -- the `|n` repeat operator -------------------------------------------

    /// Repeat the preceding unit `n` times in total, chaining the copies.
    ///
    /// The repeat *consumes* the unit: `self.unit` is cleared, so a second `|`
    /// written straight after finds nothing to repeat and is refused as an
    /// unexpected character rather than replaying the same bytes again. The
    /// next node written at branch depth 0 starts a fresh unit.
    fn read_repeat(&mut self) -> Result<(), SmilesError> {
        let at = self.scanner.pos();
        self.scanner.advance(); // '|'
        let count = self.read_repeat_count()?;
        let unit = self.repeatable_unit(at)?;
        let order = self.take_pending();
        let resume = self.scanner.pos();
        let mut previous = unit.anchor;
        for _ in 1..count {
            previous = self.replay(&unit, previous, order, at)?;
        }
        self.scanner.seek(resume);
        self.current = Some(previous);
        self.unit = None;
        Ok(())
    }

    /// The count after `|`: a positive decimal integer, `1` being the
    /// identity (one copy, no chaining edge).
    fn read_repeat_count(&mut self) -> Result<usize, SmilesError> {
        let start = self.scanner.pos();
        let digits = self.scanner.eat_digits();
        if digits.is_empty() {
            let text = self.scanner.peek().map(String::from).unwrap_or_default();
            return Err(self.invalid_count(text, start));
        }
        match digits.parse::<usize>() {
            Ok(count) if count >= 1 => Ok(count),
            _ => Err(self.invalid_count(digits.to_owned(), start)),
        }
    }

    /// The bad-repeat-count error, carrying the text read after `|`.
    fn invalid_count(&self, text: String, start: usize) -> SmilesError {
        let span = self.scanner.span_from(start);
        self.scanner
            .error_at(SmilesErrorKind::CgInvalidRepeatCount(text), span)
    }

    /// The unit `|` may repeat, with the shapes that cannot be replayed
    /// refused.
    ///
    /// A ring marker inside the unit is a one-shot identity and replaying it
    /// would either overwrite the open marker or close it twice; a bonding
    /// descriptor inside the unit, by contrast, is cloned into every copy,
    /// because a descriptor label is a pairing *class* consumed one occurrence
    /// at a time.
    fn repeatable_unit(&self, at: usize) -> Result<RepeatUnit, SmilesError> {
        let span = Span::new(at, at + 1);
        if !self.branches.is_empty() {
            let kind = SmilesErrorKind::CgRepeatOnBranchedNode;
            return Err(self.scanner.error_at(kind, span));
        }
        let Some(unit) = self.unit.as_ref() else {
            return Err(self
                .scanner
                .error_at(SmilesErrorKind::UnexpectedChar('|'), span));
        };
        if unit.ring_touched {
            let kind = SmilesErrorKind::CgRepeatOnRingMarker;
            return Err(self.scanner.error_at(kind, span));
        }
        if unit.branches > 1 {
            let kind = SmilesErrorKind::CgRepeatOnBranchedNode;
            return Err(self.scanner.error_at(kind, span));
        }
        Ok(unit.clone())
    }

    /// Re-read the unit's bytes as one more copy and chain it to `previous`.
    ///
    /// The scanner rewinds over the *full* input, so a node parsed here keeps
    /// the template's span and an error raised here points at the template's
    /// text — the copies share spans, which is why spans in a level are
    /// neither disjoint nor monotonic.
    fn replay(
        &mut self,
        unit: &RepeatUnit,
        previous: usize,
        order: CGBondOrder,
        at: usize,
    ) -> Result<usize, SmilesError> {
        self.scanner.seek(unit.start);
        self.current = None;
        let anchor = self.nodes.len();
        while self.scanner.pos() < unit.end {
            self.step()?;
        }
        self.edges.push(CGEdge {
            i: previous,
            j: anchor,
            order,
            span: Span::new(at, at + 1),
        });
        Ok(anchor)
    }
}

#[cfg(test)]
mod tests {
    use crate::io::smiles::{
        CGBondOrder, CGGraph, DescriptorKind, Notation, SmilesErrorKind, parse_cgsmiles,
    };

    // -- helpers ------------------------------------------------------------

    /// The single resolution level of a `CGsmiles` string that must parse.
    fn level0(text: &str) -> CGGraph {
        let ir =
            parse_cgsmiles(text).unwrap_or_else(|e| panic!("parse_cgsmiles({text:?}) failed: {e}"));
        ir.levels
            .into_iter()
            .next()
            .unwrap_or_else(|| panic!("parse_cgsmiles({text:?}) produced no level"))
    }

    /// `(i, j, order)` of every edge, in parse order.
    fn edges(graph: &CGGraph) -> Vec<(usize, usize, CGBondOrder)> {
        graph.edges.iter().map(|e| (e.i, e.j, e.order)).collect()
    }

    /// `(i, j)` of every edge, in parse order — for fixtures whose expectation
    /// is the connectivity alone.
    fn pairs(graph: &CGGraph) -> Vec<(usize, usize)> {
        graph.edges.iter().map(|e| (e.i, e.j)).collect()
    }

    /// The error kind `text` must be refused with.
    fn kind_of(text: &str) -> SmilesErrorKind {
        parse_cgsmiles(text)
            .err()
            .unwrap_or_else(|| panic!("parse_cgsmiles({text:?}) was accepted"))
            .kind
    }

    // -- F1.1 ---------------------------------------------------------------

    #[test]
    fn test_single_node_block_has_one_named_node_and_no_edge() {
        let graph = level0("{[#PEO]}");
        assert_eq!(graph.nodes.len(), 1);
        assert_eq!(graph.nodes[0].name, "PEO");
        assert!(graph.edges.is_empty(), "edges were {:?}", graph.edges);
    }

    /// One block is one resolution level; 01c lifts this when it appends
    /// resolved levels.
    #[test]
    fn test_one_block_yields_exactly_one_level() {
        let ir = parse_cgsmiles("{[#PEO]}").expect("{[#PEO]} must parse");
        assert_eq!(ir.levels.len(), 1);
    }

    // -- F1.2, F1.3: adjacency and bond symbols -----------------------------

    #[test]
    fn test_adjacent_nodes_are_joined_by_single_edges() {
        let graph = level0("{[#PEO][#PEO][#PEO]}");
        assert_eq!(
            edges(&graph),
            vec![(0, 1, CGBondOrder::Single), (1, 2, CGBondOrder::Single),]
        );
    }

    #[test]
    fn test_bond_symbols_set_the_edge_order() {
        let graph = level0("{[#A]=[#B]#[#C]$[#D]}");
        assert_eq!(
            edges(&graph),
            vec![
                (0, 1, CGBondOrder::Double),
                (1, 2, CGBondOrder::Triple),
                (2, 3, CGBondOrder::Quadruple),
            ]
        );
    }

    // -- F1.5: branches -----------------------------------------------------

    #[test]
    fn test_branch_attaches_to_the_node_written_before_it() {
        let graph = level0("{[#A]([#B])[#C]}");
        assert_eq!(pairs(&graph), vec![(0, 1), (0, 2)]);
    }

    // -- F1.6, F1.14, F1.15: ring markers -----------------------------------

    #[test]
    fn test_ring_closure_edge_is_emitted_last() {
        let graph = level0("{[#A]1[#B][#C]1}");
        assert_eq!(pairs(&graph), vec![(0, 1), (1, 2), (0, 2)]);
    }

    #[test]
    fn test_two_digit_ring_marker_closes_the_ring() {
        let graph = level0("{[#A]%12[#B][#C]%12}");
        assert_eq!(graph.nodes.len(), 3);
        assert_eq!(graph.edges.len(), 3);
    }

    #[test]
    fn test_ring_marker_number_is_reusable_after_it_closes() {
        let graph = level0("{[#A]1[#B][#A]1[#C]}");
        assert_eq!(graph.nodes.len(), 4);
        assert_eq!(graph.edges.len(), 4);
    }

    #[test]
    fn test_ring_bond_order_comes_from_the_opening_marker() {
        let graph = level0("{[#A]=1[#B][#C]1}");
        let ring = graph.edges.last().expect("ring closure edge is missing");
        assert_eq!((ring.i, ring.j, ring.order), (0, 2, CGBondOrder::Double));
    }

    // -- F1.7, F1.11, F1.12: annotations ------------------------------------

    #[test]
    fn test_keyword_charge_is_read_as_a_partial_charge_in_e() {
        let graph = level0("{[#A;q=-0.5;kind=ether]}");
        let q = graph.nodes[0].charge.expect("q annotation was dropped");
        assert!((q - (-0.5)).abs() < 1e-10, "charge was {q}");
    }

    #[test]
    fn test_unknown_annotation_is_retained_verbatim_without_q_or_w() {
        let graph = level0("{[#A;q=-0.5;kind=ether]}");
        assert_eq!(
            graph.nodes[0].annotations,
            vec![("kind".to_owned(), "ether".to_owned())]
        );
    }

    #[test]
    fn test_second_positional_slot_is_the_charge() {
        let graph = level0("{[#A;0.5]}");
        let q = graph.nodes[0].charge.expect("positional q was dropped");
        assert!((q - 0.5).abs() < 1e-10, "charge was {q}");
    }

    #[test]
    fn test_third_positional_slot_is_the_weight_and_a_non_default_one_is_refused() {
        let kind = kind_of("{[#A;0;0.5]}");
        assert!(
            matches!(
                &kind,
                SmilesErrorKind::CgUnsupportedAnnotation { key, value }
                    if key == "w" && value == "0.5"
            ),
            "kind was {kind:?}"
        );
    }

    #[test]
    fn test_default_positional_weight_parses_with_a_zero_charge() {
        let graph = level0("{[#A;0;1]}");
        let q = graph.nodes[0].charge.expect("positional q was dropped");
        assert!(q.abs() < 1e-10, "charge was {q}");
    }

    // -- F1.8: bonding descriptors ------------------------------------------

    #[test]
    fn test_descriptor_binds_to_the_node_written_before_it() {
        let graph = level0("{[#A][$][#B][$]}");
        let kinds: Vec<Vec<DescriptorKind>> = graph
            .nodes
            .iter()
            .map(|n| n.descriptors.iter().map(|d| d.kind).collect())
            .collect();
        assert_eq!(
            kinds,
            vec![
                vec![DescriptorKind::Symmetric],
                vec![DescriptorKind::Symmetric],
            ]
        );
    }

    #[test]
    fn test_descriptor_is_not_a_node_of_the_chain() {
        let graph = level0("{[#A][$][#B][$]}");
        assert_eq!(graph.nodes.len(), 2);
        assert_eq!(graph.edges.len(), 1);
    }

    // -- F1.4, F1.9, F1.13: the `|n` repeat operator ------------------------

    #[test]
    fn test_repeat_chains_one_node_into_a_run() {
        let graph = level0("{[#PEO]|4}");
        assert_eq!(graph.nodes.len(), 4);
        assert_eq!(
            edges(&graph),
            vec![
                (0, 1, CGBondOrder::Single),
                (1, 2, CGBondOrder::Single),
                (2, 3, CGBondOrder::Single),
            ]
        );
    }

    #[test]
    fn test_repeat_count_one_is_the_identity() {
        let graph = level0("{[#A]|1}");
        assert_eq!(graph.nodes.len(), 1);
        assert!(graph.edges.is_empty(), "edges were {:?}", graph.edges);
    }

    /// F1.9 (R2.20): the repeated unit includes its anchor, and successive
    /// anchors are chained — copy 1 is nodes 0,1,2 with edges (0,1) and (1,2),
    /// copy 2 is nodes 3,4,5 with edges (3,4) and (4,5), joined by (0,3). The
    /// expectation is the *parse-order* edge vector, unsorted: the chaining
    /// edge (0,3) is pushed after the copy's own internal edges, so it lands
    /// last rather than second.
    #[test]
    fn test_repeated_branch_unit_is_copied_with_its_anchor() {
        let graph = level0("{[#A]([#B][#C])|2}");
        assert_eq!(graph.nodes.len(), 6);
        assert_eq!(graph.edges.len(), 5);
        assert_eq!(pairs(&graph), vec![(0, 1), (1, 2), (3, 4), (4, 5), (0, 3)]);
    }

    #[test]
    fn test_repeat_of_a_two_node_branch_expands_to_nine_nodes() {
        let graph = level0("{[#A]([#B][#B])|3}");
        assert_eq!(graph.nodes.len(), 9);
        assert_eq!(graph.edges.len(), 8);
    }

    #[test]
    fn test_repeat_with_a_late_anchor_expands_to_seven_nodes() {
        let graph = level0("{[#A][#B]([#C])|3}");
        assert_eq!(graph.nodes.len(), 7);
        assert_eq!(graph.edges.len(), 6);
    }

    #[test]
    fn test_descriptor_inside_a_repeated_unit_is_cloned_into_every_copy() {
        let graph = level0("{[#A][$1]|3}");
        let labels: Vec<Vec<(DescriptorKind, String)>> = graph
            .nodes
            .iter()
            .map(|n| {
                n.descriptors
                    .iter()
                    .map(|d| (d.kind, d.label.clone()))
                    .collect()
            })
            .collect();
        assert_eq!(
            labels,
            vec![
                vec![(DescriptorKind::Symmetric, "1".to_owned())],
                vec![(DescriptorKind::Symmetric, "1".to_owned())],
                vec![(DescriptorKind::Symmetric, "1".to_owned())],
            ]
        );
    }

    /// R2.22: a bond symbol written before `|` is promoted to the order of the
    /// edge that joins each copy to the previous one.
    #[test]
    fn test_bond_symbol_before_the_repeat_sets_the_chaining_order() {
        let graph = level0("{[#A]=|3}");
        assert_eq!(
            edges(&graph),
            vec![(0, 1, CGBondOrder::Double), (1, 2, CGBondOrder::Double),]
        );
    }

    /// Every construct is validated on the first pass over the input, so there
    /// is no replay-only error path: the non-default weight below is refused
    /// while the template is read, before any copy of it exists. The error a
    /// repeated unit produces therefore carries the full input and points
    /// inside the bracket that was actually written.
    #[test]
    fn test_error_inside_a_repeated_unit_carries_the_full_input() {
        let text = "{[#A]([#B;w=2])|2}";
        let err = parse_cgsmiles(text).expect_err("a non-default weight must be refused");
        assert_eq!(err.input, text);
        assert!(
            err.span.start < text.len(),
            "span was {:?} over {} bytes",
            err.span,
            text.len()
        );
        let open = text.find("[#B").expect("fixture writes a [#B bracket");
        let close = open + "[#B;w=2]".len();
        assert!(
            (open..close).contains(&err.span.start),
            "span {:?} is outside the [#B;w=2] bracket at {open}..{close}",
            err.span
        );
    }

    // -- rejections ---------------------------------------------------------

    /// F1.10. 01c deletes this test by name when it accepts further blocks.
    #[test]
    fn second_block_is_trailing_characters() {
        assert!(matches!(
            kind_of("{[#PEO][#PEO]}[#X]"),
            SmilesErrorKind::TrailingCharacters
        ));
    }

    #[test]
    fn test_empty_block_is_refused() {
        assert!(matches!(kind_of("{}"), SmilesErrorKind::CgEmptyBlock));
    }

    #[test]
    fn test_empty_annotation_is_malformed() {
        assert!(matches!(
            kind_of("{[#A;]}"),
            SmilesErrorKind::CgMalformedAnnotation(_)
        ));
    }

    #[test]
    fn test_annotation_without_a_key_is_malformed() {
        assert!(matches!(
            kind_of("{[#A;=1]}"),
            SmilesErrorKind::CgMalformedAnnotation(_)
        ));
    }

    #[test]
    fn test_non_numeric_charge_is_malformed() {
        assert!(matches!(
            kind_of("{[#A;q=x]}"),
            SmilesErrorKind::CgMalformedAnnotation(_)
        ));
    }

    #[test]
    fn test_fourth_positional_slot_is_malformed() {
        assert!(matches!(
            kind_of("{[#A;0;1;x;y]}"),
            SmilesErrorKind::CgMalformedAnnotation(_)
        ));
    }

    #[test]
    fn test_keyword_weight_is_unsupported() {
        let kind = kind_of("{[#A;w=2]}");
        assert!(
            matches!(
                &kind,
                SmilesErrorKind::CgUnsupportedAnnotation { key, value }
                    if key == "w" && value == "2"
            ),
            "kind was {kind:?}"
        );
    }

    #[test]
    fn test_chirality_annotation_is_unsupported() {
        let kind = kind_of("{[#A;x=S]}");
        assert!(
            matches!(
                &kind,
                SmilesErrorKind::CgUnsupportedAnnotation { key, value }
                    if key == "x" && value == "S"
            ),
            "kind was {kind:?}"
        );
    }

    #[test]
    fn test_annotation_on_a_wildcard_node_is_refused() {
        assert!(matches!(
            kind_of("{[#*;q=1]}"),
            SmilesErrorKind::CgAnnotationOnWildcard(_)
        ));
    }

    #[test]
    fn test_zero_bond_order_is_refused() {
        assert!(matches!(
            kind_of("{[#A].[#B]}"),
            SmilesErrorKind::CgInvalidBondOrder
        ));
    }

    #[test]
    fn test_bond_symbol_before_the_block_close_is_dangling() {
        assert!(matches!(
            kind_of("{[#A]=}"),
            SmilesErrorKind::CgDanglingBond
        ));
    }

    #[test]
    fn test_one_digit_percent_marker_is_an_invalid_ring_marker() {
        assert!(matches!(
            kind_of("{[#A]%1}"),
            SmilesErrorKind::CgInvalidRingMarker
        ));
    }

    #[test]
    fn test_ring_closure_duplicating_an_edge_is_refused() {
        assert!(matches!(
            kind_of("{[#A]1[#B]1[#A]}"),
            SmilesErrorKind::CgDuplicateEdge { i: 0, j: 1 }
        ));
    }

    /// A marker opened and closed on one node would bond that node to itself,
    /// which the simple-graph check refuses under the duplicate-edge kind.
    #[test]
    fn test_ring_marker_opened_and_closed_on_one_node_is_a_self_bond() {
        assert!(matches!(
            kind_of("{[#A]11}"),
            SmilesErrorKind::CgDuplicateEdge { i: 0, j: 0 }
        ));
    }

    /// R2.5: the closure takes the order written before the *opening* marker,
    /// so a different order before the closing one is a conflict, not an
    /// override.
    #[test]
    fn test_conflicting_ring_bond_orders_are_refused() {
        assert!(matches!(
            kind_of("{[#A]=1[#B][#C]-1}"),
            SmilesErrorKind::RingBondConflict { .. }
        ));
    }

    #[test]
    fn test_unclosed_ring_marker_is_unmatched() {
        assert!(matches!(
            kind_of("{[#A]1[#B]}"),
            SmilesErrorKind::UnmatchedRingClosure(1)
        ));
    }

    /// Two markers are left open, and which of them is reported is
    /// unspecified — the open markers live in a `HashMap` — so the assertion
    /// is on the kind only, never on the marker number.
    #[test]
    fn test_two_unclosed_ring_markers_report_an_unmatched_closure() {
        assert!(matches!(
            kind_of("{[#A]1[#B]2}"),
            SmilesErrorKind::UnmatchedRingClosure(_)
        ));
    }

    /// A descriptor needs a node to hang on; the block below opens with one.
    #[test]
    fn test_descriptor_without_a_node_to_anchor_it_is_refused() {
        assert!(matches!(
            kind_of("{[$]}"),
            SmilesErrorKind::DanglingDescriptor
        ));
    }

    /// `CGsmiles` writes a descriptor's bond order outside the bracket, so the
    /// `BigSMILES` in-bracket spelling is refused rather than translated.
    #[test]
    fn test_bond_symbol_inside_a_descriptor_bracket_is_refused() {
        assert!(matches!(
            kind_of("{[#A][$=]}"),
            SmilesErrorKind::BondInsideDescriptor
        ));
    }

    /// The annotation run swallows the block's `}` and hits end of input, which
    /// is a missing `]`, not a missing `}`.
    #[test]
    fn test_node_bracket_left_open_after_an_annotation_is_refused() {
        assert!(matches!(
            kind_of("{[#A;k=v}"),
            SmilesErrorKind::UnclosedBracket
        ));
    }

    #[test]
    fn test_zero_repeat_count_is_refused() {
        assert!(matches!(
            kind_of("{[#A]|0}"),
            SmilesErrorKind::CgInvalidRepeatCount(_)
        ));
    }

    #[test]
    fn test_non_numeric_repeat_count_is_refused() {
        assert!(matches!(
            kind_of("{[#A]|x}"),
            SmilesErrorKind::CgInvalidRepeatCount(_)
        ));
    }

    /// A repeat consumes the unit it copies, so a second `|` written straight
    /// after the first has nothing left to repeat and is an unexpected
    /// character — not a silent second replay of the same bytes.
    #[test]
    fn test_stacked_repeat_has_no_unit_left_to_repeat() {
        assert!(matches!(
            kind_of("{[#A]|2|3}"),
            SmilesErrorKind::UnexpectedChar('|')
        ));
    }

    #[test]
    fn test_ring_marker_opened_inside_a_repeated_unit_is_refused() {
        assert!(matches!(
            kind_of("{[#A]1|2[#B]1}"),
            SmilesErrorKind::CgRepeatOnRingMarker
        ));
    }

    #[test]
    fn test_ring_marker_closed_inside_a_repeated_unit_is_refused() {
        assert!(matches!(
            kind_of("{[#A]1[#B]1|2}"),
            SmilesErrorKind::CgRepeatOnRingMarker
        ));
    }

    #[test]
    fn test_repeat_inside_an_open_branch_is_refused() {
        assert!(matches!(
            kind_of("{[#A](|2[#B])}"),
            SmilesErrorKind::CgRepeatOnBranchedNode
        ));
    }

    #[test]
    fn test_repeat_on_a_node_with_two_branches_is_refused() {
        assert!(matches!(
            kind_of("{[#A]([#B])([#C])|3}"),
            SmilesErrorKind::CgRepeatOnBranchedNode
        ));
    }

    #[test]
    fn test_every_cgsmiles_error_is_stamped_with_the_cgsmiles_notation() {
        for text in [
            "{}",
            "{[#A;]}",
            "{[#A].[#B]}",
            "{[#A]|0}",
            "{[#PEO][#PEO]}[#X]",
        ] {
            let err = parse_cgsmiles(text).expect_err("input must be refused");
            assert_eq!(err.notation, Notation::CGsmiles, "input was {text}");
        }
    }
}
