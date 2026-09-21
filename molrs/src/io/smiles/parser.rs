//! Recursive-descent parser for the three line-notation dialects: SMILES,
//! SMARTS, and the SMILES fragment body.
//!
//! The parser directly mirrors the LL(1) grammar:
//!
//! ```text
//! molecule     → component ('.' component)*
//! component    → chain
//! chain        → atom chain_tail*
//! chain_tail   → branch | ring_closure | bonded_atom
//! branch       → '(' bond? chain ')'
//! ring_closure → bond? rnum
//! bonded_atom  → bond? atom
//! atom         → bracket_atom | organic_atom | '*'
//! bracket_atom → '[' isotope? symbol chirality? hcount? charge? class? ']'
//! bond         → '-' | '=' | '#' | '$' | '/' | '\' | ':' | '~' | '@'
//! rnum         → digit | '%' digit digit
//! ```
//!
//! The fragment dialect adds one production and two placements for it,
//! accepted only by [`parse_fragment_smiles`]:
//!
//! ```text
//! descriptor   → '[' ('$' | '<' | '>' | '!') label? ']'
//! label        → alnum*
//! leading_run  → descriptor+ bond?     // before the chain's first atom
//! anchored_run → bond? descriptor+     // anywhere after it
//! ```
//!
//! A descriptor is not a node: it binds to the atom written before it (or, at
//! the start of a chain, to that chain's head), so it is consumed at the
//! `atom` call sites and folded into [`AtomNode::descriptors`] rather than
//! appearing in the tree on its own. The optional `bond` in each run is the
//! order that descriptor annotates, and it takes the only side available: in
//! a leading run there is no atom yet to write it in front of, so it follows
//! the bracket (`[$]=CCC`); everywhere else it precedes it (`CC=[$]`), and a
//! bond written after the bracket is an ordinary bond to the next atom.

use crate::io::smiles::chem::Dialect;
use crate::io::smiles::chem::ast::*;
use crate::io::smiles::chem::scanner::Scanner;
use crate::io::smiles::chem::validation::validate_descriptor;
use crate::io::smiles::error::{SmilesError, SmilesErrorKind};

/// Maximum recursion depth for SMARTS `$(...)` expressions.
const MAX_RECURSION_DEPTH: usize = 16;

/// The set of organic-subset symbols (no brackets required).
///
/// Lowercase entries represent aromatic atoms.
const ORGANIC_SUBSET: &[&str] = &[
    "B", "C", "N", "O", "P", "S", "F", "Cl", "Br", "I", "At", "Ts", "b", "c", "n", "o", "p", "s",
];

/// Parse a plain SMILES string into the shared IR.
///
/// Strict by design: only concrete notation — atoms, bonds, branches and ring
/// closures — is accepted. SMARTS query brackets and the fragment dialect's
/// bonding descriptors are refused rather than reinterpreted. Parsing is
/// syntax only: a bracket atom's element symbol and the pairing of ring
/// digits are checked afterwards, by
/// [`validate_smiles`](crate::io::smiles::validate_smiles).
///
/// # Errors
///
/// Returns a [`SmilesError`] for anything the plain grammar does not accept:
/// [`SmilesErrorKind::EmptyInput`], [`SmilesErrorKind::UnexpectedChar`],
/// [`SmilesErrorKind::UnexpectedEnd`], [`SmilesErrorKind::UnclosedBracket`],
/// [`SmilesErrorKind::UnclosedBranch`],
/// [`SmilesErrorKind::TrailingCharacters`],
/// [`SmilesErrorKind::InvalidElement`] for a letter outside the organic
/// subset, and [`SmilesErrorKind::DescriptorInPlainSmiles`] for a bonding
/// descriptor, whose message points at [`parse_fragment_smiles`].
pub fn parse_smiles(input: &str) -> Result<SmilesIR, SmilesError> {
    Parser::new(input, Dialect::Smiles).parse_molecule()
}

/// Parse a SMARTS pattern into the shared IR.
///
/// This yields SMARTS *syntax* as an IR, for callers that want the pattern as
/// a tree. It is not the frontend of the substructure-matching engine in
/// [`crate::perceive::smarts`], which has a parser of its own and never
/// consumes this one.
///
/// # Errors
///
/// Returns a [`SmilesError`] for anything the SMARTS grammar does not accept:
/// [`SmilesErrorKind::EmptyInput`], [`SmilesErrorKind::UnexpectedChar`],
/// [`SmilesErrorKind::UnexpectedEnd`], [`SmilesErrorKind::UnclosedBracket`],
/// [`SmilesErrorKind::UnclosedBranch`],
/// [`SmilesErrorKind::TrailingCharacters`],
/// [`SmilesErrorKind::InvalidQueryPrimitive`] for an unrecognised query atom,
/// [`SmilesErrorKind::UnclosedRecursive`] for a `$(` with no `)`, and
/// [`SmilesErrorKind::RecursionLimit`] when `$(...)` nests deeper than the
/// parser's limit.
pub fn parse_smarts(input: &str) -> Result<SmilesIR, SmilesError> {
    Parser::new(input, Dialect::Smarts).parse_molecule()
}

/// Parse a SMILES fragment body with `CGsmiles` / `BigSMILES` bonding
/// descriptors into an AST.
///
/// A *fragment body* is one SMILES string standing for a piece of a larger
/// molecule — a monomer, a bead, a building block — with the sites at which it
/// will later be joined to other pieces marked by bonding descriptors. The
/// dialect is therefore plain SMILES plus the descriptor brackets `[$]`,
/// `[<]`, `[>]` and `[!]`, each optionally labelled (`[$a]`) and optionally
/// carrying a bond order written outside the bracket (`CC=[$]`).
///
/// [`parse_smiles`] is the plain sibling of this function and stays strict: it
/// refuses descriptor brackets outright, so no `.smi` line can silently lose
/// one.
///
/// # Where a descriptor lands
///
/// A descriptor is not a node of the chain; it binds to a node and is stored
/// in [`AtomNode::descriptors`](crate::io::smiles::AtomNode). The atom it
/// binds to is the one written immediately before it, or — when it is written
/// before any atom of its chain — that chain's head atom, so `[$]COC[$]`
/// carries a `$` on the first carbon and another on the last. A `(` is not a
/// node either, so a descriptor just inside a branch binds to the atom before
/// the `(`: `N([>])C` puts `>` on the nitrogen. Several descriptors may sit on
/// one atom and are kept in written order: `[>][$1]COC[<]` gives the first
/// carbon the list `[>, $1]`. No valence check is made.
///
/// # Where the bond order is read from
///
/// The order a descriptor annotates is written next to its bracket, not inside
/// it, and which side counts depends on whether any atom of the chain has been
/// parsed yet. Before the first atom there is nothing to the left, so a bond
/// symbol *after* the bracket is the descriptor's order: `[$]=CCC` is a
/// double-bond-annotated `$` on the first carbon over an otherwise single
/// C-C-C chain. Once an atom exists the symbol before the bracket is the
/// order and a symbol after it is an ordinary bond to the next atom, so
/// `CC=[$]` is a double-annotated `$` while `C[$]=CC` is an unannotated `$` on
/// the first carbon plus a double C=C bond. Of a run of brackets only the one
/// the symbol is written against takes the order, so `[>][$1]=CCC` annotates
/// `$1` and `CC=[$][>]` annotates `$`; the rest of the run carries none. The
/// leading form is strictly the less expressive of the two — it cannot spell
/// `C[$]=CC` — which is why
/// [`write_fragment_smiles`](crate::io::smiles::write_fragment_smiles) emits
/// the trailing form only.
///
/// # Errors
///
/// Returns a [`SmilesError`] for any syntax the dialect does not accept,
/// including [`SmilesErrorKind::DanglingDescriptor`] for a descriptor with no
/// atom to bind to, [`SmilesErrorKind::BondInsideDescriptor`] for the
/// `BigSMILES` in-bracket order (`[<=1]`),
/// [`SmilesErrorKind::InvalidDescriptorLabel`],
/// [`SmilesErrorKind::InvalidDescriptorOrder`], and
/// [`SmilesErrorKind::AtomAnnotationUnsupported`] for a `CGsmiles` atom-level
/// annotation (`[C;0.5]`).
pub fn parse_fragment_smiles(input: &str) -> Result<SmilesIR, SmilesError> {
    Parser::new(input, Dialect::FragmentSmiles).parse_molecule()
}

// ---------------------------------------------------------------------------
// Parser internals
// ---------------------------------------------------------------------------

struct Parser<'a> {
    scanner: Scanner<'a>,
    dialect: Dialect,
    depth: usize,
}

/// Which bracket of a descriptor run the out-of-bracket bond symbol touches.
///
/// The order is written outside the brackets, so in a run (`[>][$1]`) only the
/// bracket the symbol is written against takes it: `First` for a symbol before
/// the run (`CC=[$][>]`), `Last` for one after a leading run (`[>][$1]=CCC`).
/// The reader this mirrors processes brackets one at a time, which is the same
/// rule stated position by position.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum AdjacentBracket {
    First,
    Last,
}

impl<'a> Parser<'a> {
    fn new(input: &'a str, dialect: Dialect) -> Self {
        Self {
            // The notation every diagnostic of this parse is labelled with is
            // the dialect's own (`Dialect::notation`), so it is stated once,
            // here, and the parser keeps no second field for it.
            scanner: Scanner::new(input, dialect.notation()),
            dialect,
            depth: 0,
        }
    }

    // -- helpers ------------------------------------------------------------

    fn error(&self, kind: SmilesErrorKind) -> SmilesError {
        self.scanner.error(kind)
    }

    fn error_at(&self, kind: SmilesErrorKind, span: Span) -> SmilesError {
        self.scanner.error_at(kind, span)
    }

    /// True if `ch` can start a bond token.
    fn is_bond_char(ch: char) -> bool {
        matches!(ch, '-' | '=' | '#' | '$' | '/' | '\\' | ':')
    }

    /// True if `ch` can start a bond in SMARTS mode (includes `~` and `@`).
    fn is_bond_char_smarts(ch: char) -> bool {
        Self::is_bond_char(ch) || matches!(ch, '~' | '@' | '!')
    }

    /// True if `ch` can start an atom.
    fn is_atom_start(ch: char) -> bool {
        ch == '[' || ch == '*' || ch.is_ascii_alphabetic()
    }

    /// The plain bond kind a parsed bond slot holds, if it holds one.
    ///
    /// A SMARTS bond query (`!-`, `-,=`) has no single kind; it also never
    /// annotates a descriptor, since the SMARTS dialect has no descriptors.
    fn bond_query_kind(bond: Option<&BondQuery>) -> Option<BondKind> {
        match bond {
            Some(BondQuery::Kind(kind)) => Some(*kind),
            _ => None,
        }
    }

    // -- bonding descriptors (fragment dialect) ------------------------------

    /// True when the cursor sits on a bonding-descriptor bracket: `[`
    /// followed by one of `$ < > !`.
    ///
    /// The lookahead itself runs in every dialect — what to do with it is
    /// [`Parser::take_descriptor_run`]'s decision. Running it in
    /// `Dialect::Smiles` too is what lets plain SMILES name the fragment entry
    /// point, instead of dying on an unexpected `$` inside the bracket parser.
    fn at_descriptor(&self) -> bool {
        self.scanner.peek() == Some('[')
            && matches!(self.scanner.peek_next(), Some('$' | '<' | '>' | '!'))
    }

    /// Consume the run of descriptor brackets at the cursor, if one starts
    /// there.
    ///
    /// Each descriptor comes back with the span of its bracket and with no
    /// order yet: the order is written *outside* the bracket, and whether the
    /// symbol next to the run is one depends on the call site, so
    /// [`Parser::finish_descriptor_run`] applies and validates it.
    ///
    /// `Ok(None)` means there is nothing here to consume — either the cursor
    /// is not on a descriptor bracket, or the dialect is SMARTS, where
    /// `[$(...)]` is recursive SMARTS and `[!C]` is negation and both belong
    /// to the bracket-query parser.
    ///
    /// # Errors
    ///
    /// Returns [`SmilesErrorKind::DescriptorInPlainSmiles`] in
    /// `Dialect::Smiles`, which has no descriptor notation.
    fn take_descriptor_run(
        &mut self,
    ) -> Result<Option<Vec<(BondingDescriptor, Span)>>, SmilesError> {
        if !self.at_descriptor() {
            return Ok(None);
        }
        match self.dialect {
            Dialect::Smarts => Ok(None),
            Dialect::Smiles => Err(self.error(SmilesErrorKind::DescriptorInPlainSmiles)),
            Dialect::FragmentSmiles => {
                let mut run = vec![self.parse_descriptor()?];
                while self.at_descriptor() {
                    run.push(self.parse_descriptor()?);
                }
                Ok(Some(run))
            }
        }
    }

    /// Anchor `order` on the descriptor `adjacent` names and validate the run.
    ///
    /// The order belongs to the one bracket the bond symbol is written against,
    /// and which end of the run that is depends on the side the symbol sits on:
    /// a symbol *before* the run annotates its first bracket (`CC=[$][>]`
    /// carries the double on `$`), one *after* a leading run annotates its last
    /// (`[>][$1]=CCC` carries it on `$1`). Every other descriptor of the run
    /// keeps `order: None`. This is the single construction site of every
    /// [`BondingDescriptor`] the parser emits, so `validate_descriptor` runs
    /// here and nowhere else.
    ///
    /// # Errors
    ///
    /// Returns whatever `validate_descriptor` rejects: an invalid label or an
    /// order no created bond can take.
    fn finish_descriptor_run(
        &self,
        run: Vec<(BondingDescriptor, Span)>,
        order: Option<BondKind>,
        adjacent: AdjacentBracket,
    ) -> Result<Vec<BondingDescriptor>, SmilesError> {
        let ordered = match adjacent {
            AdjacentBracket::First => 0,
            AdjacentBracket::Last => run.len().saturating_sub(1),
        };
        let mut descriptors = Vec::with_capacity(run.len());
        for (index, (mut desc, span)) in run.into_iter().enumerate() {
            if index == ordered {
                desc.order = order;
            }
            validate_descriptor(&desc, span, self.scanner.input(), self.dialect.notation())?;
            descriptors.push(desc);
        }
        Ok(descriptors)
    }

    /// Parse one descriptor bracket: `[` glyph label? `]`.
    fn parse_descriptor(&mut self) -> Result<(BondingDescriptor, Span), SmilesError> {
        let start = self.scanner.pos();
        self.scanner.expect('[')?;

        let kind = match self.scanner.advance() {
            Some('$') => DescriptorKind::Symmetric,
            Some('<') => DescriptorKind::Left,
            Some('>') => DescriptorKind::Right,
            Some('!') => DescriptorKind::Shared,
            Some(c) => return Err(self.error(SmilesErrorKind::UnexpectedChar(c))),
            None => return Err(self.error(SmilesErrorKind::UnexpectedEnd)),
        };

        let label_start = self.scanner.pos();
        loop {
            match self.scanner.peek() {
                Some(']') => break,
                // The order is written outside the bracket (`CC=[$]`);
                // accepting the `BigSMILES` in-bracket spelling `[<=1]` would
                // make two notations mean one thing.
                Some(c) if Self::is_bond_char(c) => {
                    return Err(self.error(SmilesErrorKind::BondInsideDescriptor));
                }
                Some(_) => {
                    self.scanner.advance();
                }
                None => {
                    return Err(self.error_at(
                        SmilesErrorKind::UnclosedBracket,
                        self.scanner.span_from(start),
                    ));
                }
            }
        }
        // The whole run up to `]` is the label; `validate_descriptor` decides
        // whether it is one, so `[$a+]` names the bad label rather than the
        // character.
        let label = self.scanner.input()[label_start..self.scanner.pos()].to_owned();
        self.scanner.advance(); // consume ']'

        Ok((
            BondingDescriptor {
                kind,
                label,
                order: None,
            },
            self.scanner.span_from(start),
        ))
    }

    /// The atom a mid-chain descriptor binds to: the most recent bonded atom,
    /// else the chain head.
    ///
    /// `Branch` and `RingClosure` do not advance it — in `CC(=O)[<]` the `<`
    /// belongs to the carbonyl carbon, not to the branch oxygen — mirroring
    /// how the IR → graph walker tracks its current atom.
    fn anchor<'c>(head: &'c mut AtomNode, tail: &'c mut [ChainElement]) -> &'c mut AtomNode {
        for elem in tail.iter_mut().rev() {
            if let ChainElement::BondedAtom { atom, .. } = elem {
                return atom;
            }
        }
        head
    }

    // -- molecule -----------------------------------------------------------

    fn parse_molecule(&mut self) -> Result<SmilesIR, SmilesError> {
        let start = self.scanner.pos();

        if self.scanner.is_done() {
            return Err(self.error(SmilesErrorKind::EmptyInput));
        }

        let mut components = vec![self.parse_chain()?];

        while self.scanner.peek() == Some('.') {
            self.scanner.advance(); // consume '.'
            components.push(self.parse_chain()?);
        }

        if !self.scanner.is_done() && self.depth == 0 {
            return Err(self.error(SmilesErrorKind::TrailingCharacters));
        }

        Ok(SmilesIR {
            components,
            span: self.scanner.span_from(start),
        })
    }

    // -- chain --------------------------------------------------------------

    /// Parse the descriptor run written before any atom of a chain (R4.2).
    ///
    /// Returns the descriptors that bind to the chain's head atom, empty when
    /// the chain does not open with a run. No atom of the chain exists yet, so
    /// a bond symbol *after* the run is an order (`[$]=CCC` is a double-bonded
    /// `$` on the first carbon over a single C-C chain), not a bond to the
    /// head — the mirror image of the mid-chain rule. It annotates the bracket
    /// it is written against, which here is the run's **last**: `[>][$1]=CCC`
    /// puts the double on `$1` and leaves `>` unannotated.
    ///
    /// # Errors
    ///
    /// Returns [`SmilesErrorKind::DanglingDescriptor`] when no atom follows the
    /// run, plus whatever [`Parser::finish_descriptor_run`] rejects.
    fn parse_leading_descriptors(&mut self) -> Result<Vec<BondingDescriptor>, SmilesError> {
        let Some(run) = self.take_descriptor_run()? else {
            return Ok(Vec::new());
        };
        let order = if self.scanner.peek().is_some_and(Self::is_bond_char) {
            self.parse_bond_kind()?
        } else {
            None
        };
        let descriptors = self.finish_descriptor_run(run, order, AdjacentBracket::Last)?;
        if !self.scanner.peek().is_some_and(Self::is_atom_start) {
            return Err(self.error(SmilesErrorKind::DanglingDescriptor));
        }
        Ok(descriptors)
    }

    /// Consume a mid-chain descriptor run, if one starts at the cursor, and
    /// fold it onto the atom it binds to.
    ///
    /// `order` is the bond kind written *before* the run (`CC=[$]`, R4.4), or
    /// `None` where no bond symbol preceded it. Written before, it annotates
    /// the bracket it touches, which is the run's **first**: `CC=[$][>]` puts
    /// the double on `$` and leaves `>` unannotated. The returned flag says
    /// whether a run was consumed, which is what tells the chain loop to go
    /// round again instead of reading the cursor as a bond, a branch or an
    /// atom.
    ///
    /// # Errors
    ///
    /// Returns whatever [`Parser::take_descriptor_run`] and
    /// [`Parser::finish_descriptor_run`] reject.
    fn take_anchored_descriptors(
        &mut self,
        head: &mut AtomNode,
        tail: &mut [ChainElement],
        order: Option<BondKind>,
    ) -> Result<bool, SmilesError> {
        let Some(run) = self.take_descriptor_run()? else {
            return Ok(false);
        };
        let descriptors = self.finish_descriptor_run(run, order, AdjacentBracket::First)?;
        Self::anchor(head, tail).descriptors.extend(descriptors);
        Ok(true)
    }

    fn parse_chain(&mut self) -> Result<Chain, SmilesError> {
        // R4.2: descriptors written before any atom of this chain bind to its
        // head atom.
        let leading = self.parse_leading_descriptors()?;

        let mut head = self.parse_atom()?;
        head.descriptors = leading;
        let mut tail = Vec::new();

        loop {
            // Mid-chain descriptor: any order was written before the bracket
            // and is therefore consumed by the bond arm below, not here.
            if self.take_anchored_descriptors(&mut head, &mut tail, None)? {
                continue;
            }
            match self.scanner.peek() {
                Some('(') => {
                    let (parent, element) = self.parse_branch()?;
                    if let Some(element) = element {
                        tail.push(element);
                    }
                    // `(` is not a node, so the branch's leading descriptors
                    // belong to this chain's anchor — which a branch never
                    // advances, so pushing first changes nothing.
                    Self::anchor(&mut head, &mut tail)
                        .descriptors
                        .extend(parent);
                }
                Some(c) if c.is_ascii_digit() || c == '%' => {
                    tail.push(self.parse_ring_closure(None)?);
                }
                Some(c)
                    if Self::is_bond_char(c)
                        || (self.dialect == Dialect::Smarts && Self::is_bond_char_smarts(c)) =>
                {
                    let bond = self.parse_bond()?;
                    // A bond symbol immediately before a descriptor bracket is
                    // that descriptor's order (`CC=[$]`, R4.4).
                    if self.take_anchored_descriptors(
                        &mut head,
                        &mut tail,
                        Self::bond_query_kind(bond.as_ref()),
                    )? {
                        continue;
                    }
                    // After a bond: expect atom, ring closure, or (rare) another bond
                    match self.scanner.peek() {
                        Some(c) if c.is_ascii_digit() || c == '%' => {
                            tail.push(self.parse_ring_closure(bond)?);
                        }
                        Some(c) if Self::is_atom_start(c) => {
                            let atom = self.parse_atom()?;
                            tail.push(ChainElement::BondedAtom { bond, atom });
                        }
                        _ => {
                            return Err(self.error(SmilesErrorKind::UnexpectedEnd));
                        }
                    }
                }
                Some(c) if Self::is_atom_start(c) => {
                    let atom = self.parse_atom()?;
                    tail.push(ChainElement::BondedAtom { bond: None, atom });
                }
                _ => break,
            }
        }

        Ok(Chain { head, tail })
    }

    // -- branch -------------------------------------------------------------

    /// Parse a parenthesised branch.
    ///
    /// Returns the descriptors that belong to the **parent** chain's anchor,
    /// and the branch element itself when the branch held anything besides
    /// them. `(` is not a node, so a descriptor written straight after it
    /// binds to the atom before the `(` (R4.2): `N([>])C` puts `>` on the
    /// nitrogen and leaves no branch element at all, while `C([>]N)C` puts it
    /// on the first carbon and still branches to N. An empty branch `C()` is
    /// the error it has always been.
    ///
    /// The order of such a run is written *before* it — `C(=[>])C` is a
    /// double-bonded `>` on the first carbon. A bond symbol written *after* a
    /// branch-leading run is **not** accepted (`C([$]=N)C` is an error), a
    /// deliberate narrowing relative to the `CGsmiles` reference, which reads
    /// that `=` as the run's order. Both readings are spellable here anyway:
    /// `C(=[$]N)C` annotates the descriptor, `C(=N)[$]` bonds the branch.
    fn parse_branch(
        &mut self,
    ) -> Result<(Vec<BondingDescriptor>, Option<ChainElement>), SmilesError> {
        let start = self.scanner.pos();
        self.scanner.expect('(')?;

        // Optional bond at the start of a branch.
        let mut bond = if self.scanner.peek().is_some_and(|c| {
            Self::is_bond_char(c)
                || (self.dialect == Dialect::Smarts && Self::is_bond_char_smarts(c))
        }) {
            self.parse_bond()?
        } else {
            None
        };

        let mut parent = Vec::new();
        if let Some(run) = self.take_descriptor_run()? {
            // That opening bond annotates the descriptor, not the branch:
            // `C(=[>])C` is a double-bonded `>` on the first carbon.
            parent = self.finish_descriptor_run(
                run,
                Self::bond_query_kind(bond.as_ref()),
                AdjacentBracket::First,
            )?;
            bond = None;
            if self.scanner.peek() == Some(')') {
                self.scanner.advance(); // consume ')'
                return Ok((parent, None));
            }
        }

        let chain = self.parse_chain()?;

        if self.scanner.peek() != Some(')') {
            return Err(self.error_at(
                SmilesErrorKind::UnclosedBranch,
                self.scanner.span_from(start),
            ));
        }
        self.scanner.advance(); // consume ')'

        Ok((
            parent,
            Some(ChainElement::Branch {
                bond,
                chain,
                span: self.scanner.span_from(start),
            }),
        ))
    }

    // -- ring closure -------------------------------------------------------

    fn parse_ring_closure(&mut self, bond: Option<BondQuery>) -> Result<ChainElement, SmilesError> {
        let start = self.scanner.pos();
        let rnum = self.parse_rnum()?;
        Ok(ChainElement::RingClosure {
            bond,
            rnum,
            span: self.scanner.span_from(start),
        })
    }

    fn parse_rnum(&mut self) -> Result<u16, SmilesError> {
        match self.scanner.peek() {
            Some('%') => {
                self.scanner.advance(); // consume '%'
                let d1 = self
                    .scanner
                    .eat_digit()
                    .ok_or_else(|| self.error(SmilesErrorKind::UnexpectedEnd))?;
                let d2 = self
                    .scanner
                    .eat_digit()
                    .ok_or_else(|| self.error(SmilesErrorKind::UnexpectedEnd))?;
                Ok(u16::from(d1) * 10 + u16::from(d2))
            }
            Some(c) if c.is_ascii_digit() => {
                let d = self.scanner.eat_digit().unwrap();
                Ok(u16::from(d))
            }
            _ => Err(self.error(SmilesErrorKind::UnexpectedEnd)),
        }
    }

    // -- atom ---------------------------------------------------------------

    fn parse_atom(&mut self) -> Result<AtomNode, SmilesError> {
        let start = self.scanner.pos();
        match self.scanner.peek() {
            Some('[') => self.parse_bracket_atom(start),
            Some('*') => {
                self.scanner.advance();
                Ok(AtomNode {
                    spec: AtomSpec::Wildcard,
                    span: self.scanner.span_from(start),
                    descriptors: Vec::new(),
                })
            }
            Some(c) if c.is_ascii_alphabetic() => self.parse_organic_atom(start),
            Some(c) => Err(self.error(SmilesErrorKind::UnexpectedChar(c))),
            None => Err(self.error(SmilesErrorKind::UnexpectedEnd)),
        }
    }

    fn parse_organic_atom(&mut self, start: usize) -> Result<AtomNode, SmilesError> {
        // Try two-character symbols first (Cl, Br, At, Ts), then one-character.
        let first = self.scanner.advance().unwrap();

        // Check two-character organic symbols.
        if let Some(&second) = self.scanner.peek_byte() {
            let two = format!("{first}{}", second as char);
            if ORGANIC_SUBSET.contains(&two.as_str()) {
                self.scanner.advance();
                let aromatic = first.is_ascii_lowercase();
                return Ok(AtomNode {
                    spec: AtomSpec::Organic {
                        symbol: two,
                        aromatic,
                    },
                    span: self.scanner.span_from(start),
                    descriptors: Vec::new(),
                });
            }
        }

        // One-character organic symbol.
        let one = first.to_string();
        if ORGANIC_SUBSET.contains(&one.as_str()) {
            let aromatic = first.is_ascii_lowercase();
            return Ok(AtomNode {
                spec: AtomSpec::Organic {
                    symbol: one,
                    aromatic,
                },
                span: self.scanner.span_from(start),
                descriptors: Vec::new(),
            });
        }

        Err(self.error_at(
            SmilesErrorKind::InvalidElement(one),
            Span::new(start, self.scanner.pos()),
        ))
    }

    fn parse_bracket_atom(&mut self, start: usize) -> Result<AtomNode, SmilesError> {
        self.scanner.expect('[')?;

        if self.dialect == Dialect::Smarts {
            return self.parse_bracket_atom_smarts(start);
        }

        // --- SMILES bracket atom ---
        let isotope = self.parse_isotope();
        let symbol = self.parse_bracket_symbol()?;
        let chirality = self.parse_chirality();
        let hcount = self.parse_hcount();
        let charge = self.parse_charge()?;
        let atom_class = self.parse_atom_class()?;

        if self.scanner.peek() != Some(']') {
            // `CGsmiles` hangs atom-level annotations off a `;` here — weights
            // `[C;0.5]`, chirality `[C;1;S]`, wildcard overloading
            // `[*;s=C,0]`. The fragment dialect does not support them, and
            // this parser is the one place that has already lexed the bracket,
            // so it names the feature rather than claiming a missing `]`.
            if self.dialect == Dialect::FragmentSmiles && self.scanner.peek() == Some(';') {
                self.scanner.advance(); // consume ';'
                let text_start = self.scanner.pos();
                while self.scanner.peek().is_some_and(|c| c != ']') {
                    self.scanner.advance();
                }
                let text = self.scanner.input()[text_start..self.scanner.pos()].to_owned();
                return Err(self.error_at(
                    SmilesErrorKind::AtomAnnotationUnsupported(text),
                    self.scanner.span_from(start),
                ));
            }
            return Err(self.error_at(
                SmilesErrorKind::UnclosedBracket,
                self.scanner.span_from(start),
            ));
        }
        self.scanner.advance(); // consume ']'

        Ok(AtomNode {
            spec: AtomSpec::Bracket {
                isotope,
                symbol,
                chirality,
                hcount,
                charge,
                atom_class,
            },
            span: self.scanner.span_from(start),
            descriptors: Vec::new(),
        })
    }

    // -- bracket sub-parts --------------------------------------------------

    fn parse_isotope(&mut self) -> Option<u16> {
        if self.scanner.peek()?.is_ascii_digit() {
            let digits = self.scanner.eat_digits();
            digits.parse::<u16>().ok()
        } else {
            None
        }
    }

    fn parse_bracket_symbol(&mut self) -> Result<BracketSymbol, SmilesError> {
        match self.scanner.peek() {
            Some('*') => {
                self.scanner.advance();
                Ok(BracketSymbol::Any)
            }
            Some(c) if c.is_ascii_alphabetic() => {
                let aromatic = c.is_ascii_lowercase();
                let symbol = self.consume_element_symbol();
                Ok(BracketSymbol::Element { symbol, aromatic })
            }
            Some(c) => Err(self.error(SmilesErrorKind::UnexpectedChar(c))),
            None => Err(self.error(SmilesErrorKind::UnclosedBracket)),
        }
    }

    /// Consume an element symbol: one uppercase letter optionally followed by
    /// one lowercase letter.
    fn consume_element_symbol(&mut self) -> String {
        let mut sym = String::new();
        if let Some(c) = self.scanner.advance() {
            sym.push(c);
            // Second letter: must be lowercase to be part of the symbol.
            if let Some(c2) = self.scanner.peek()
                && c2.is_ascii_lowercase()
            {
                sym.push(c2);
                self.scanner.advance();
            }
        }
        sym
    }

    fn parse_chirality(&mut self) -> Option<Chirality> {
        if self.scanner.peek() == Some('@') {
            self.scanner.advance();
            if self.scanner.peek() == Some('@') {
                self.scanner.advance();
                Some(Chirality::Clockwise)
            } else {
                Some(Chirality::CounterClockwise)
            }
        } else {
            None
        }
    }

    fn parse_hcount(&mut self) -> Option<u8> {
        if self.scanner.peek() == Some('H') {
            self.scanner.advance();
            // Optional digit; no digit means H1.
            let n = self.scanner.eat_digit().unwrap_or(1);
            Some(n)
        } else {
            None
        }
    }

    fn parse_charge(&mut self) -> Result<Option<i8>, SmilesError> {
        match self.scanner.peek() {
            Some('+') => {
                self.scanner.advance();
                // ++ means +2
                if self.scanner.peek() == Some('+') {
                    self.scanner.advance();
                    Ok(Some(2))
                } else if let Some(d) = self.scanner.eat_digit() {
                    Ok(Some(d as i8))
                } else {
                    Ok(Some(1))
                }
            }
            Some('-') => {
                self.scanner.advance();
                // -- means -2
                if self.scanner.peek() == Some('-') {
                    self.scanner.advance();
                    Ok(Some(-2))
                } else if let Some(d) = self.scanner.eat_digit() {
                    Ok(Some(-(d as i8)))
                } else {
                    Ok(Some(-1))
                }
            }
            _ => Ok(None),
        }
    }

    fn parse_atom_class(&mut self) -> Result<Option<u16>, SmilesError> {
        if self.scanner.peek() == Some(':') {
            self.scanner.advance();
            let digits = self.scanner.eat_digits();
            if digits.is_empty() {
                return Err(self.error(SmilesErrorKind::UnexpectedEnd));
            }
            Ok(Some(
                digits
                    .parse::<u16>()
                    .map_err(|_| self.error(SmilesErrorKind::InvalidCharge))?,
            ))
        } else {
            Ok(None)
        }
    }

    // -- bond ---------------------------------------------------------------

    /// Parse a single bond kind (no logical operators). Returns `None` if
    /// the next character is not a bond character. Used both directly in
    /// SMILES mode and as the leaf inside SMARTS bond-query parsing.
    fn parse_bond_kind(&mut self) -> Result<Option<BondKind>, SmilesError> {
        match self.scanner.peek() {
            Some('-') => {
                self.scanner.advance();
                Ok(Some(BondKind::Single))
            }
            Some('=') => {
                self.scanner.advance();
                Ok(Some(BondKind::Double))
            }
            Some('#') => {
                self.scanner.advance();
                Ok(Some(BondKind::Triple))
            }
            Some('$') => {
                self.scanner.advance();
                Ok(Some(BondKind::Quadruple))
            }
            Some(':') => {
                self.scanner.advance();
                Ok(Some(BondKind::Aromatic))
            }
            Some('/') => {
                self.scanner.advance();
                Ok(Some(BondKind::Up))
            }
            Some('\\') => {
                self.scanner.advance();
                Ok(Some(BondKind::Down))
            }
            Some('~') if self.dialect == Dialect::Smarts => {
                self.scanner.advance();
                Ok(Some(BondKind::Any))
            }
            Some('@') if self.dialect == Dialect::Smarts => {
                self.scanner.advance();
                Ok(Some(BondKind::Ring))
            }
            _ => Ok(None),
        }
    }

    /// Parse a bond (possibly with SMARTS logical operators). Returns
    /// `Option<BondQuery>` so SMARTS bond operators `!`, `&`, `,` can be
    /// represented faithfully; SMILES inputs always yield
    /// `Some(BondQuery::Kind(_))` or `None`.
    fn parse_bond(&mut self) -> Result<Option<BondQuery>, SmilesError> {
        if self.dialect == Dialect::Smarts {
            self.parse_bond_or()
        } else {
            Ok(self.parse_bond_kind()?.map(BondQuery::Kind))
        }
    }

    /// SMARTS bond `,`-OR: `expr (,expr)*`.
    fn parse_bond_or(&mut self) -> Result<Option<BondQuery>, SmilesError> {
        let Some(head) = self.parse_bond_and()? else {
            return Ok(None);
        };
        let mut parts = vec![head];
        while self.scanner.peek() == Some(',') {
            self.scanner.advance();
            let Some(next) = self.parse_bond_and()? else {
                return Err(self.error(SmilesErrorKind::UnexpectedEnd));
            };
            parts.push(next);
        }
        Ok(Some(if parts.len() == 1 {
            parts.pop().unwrap()
        } else {
            BondQuery::Or(parts)
        }))
    }

    /// SMARTS bond `&`-AND: `expr (&expr)*` (implicit AND via adjacency is
    /// **not** supported on bonds — use `&` explicitly).
    fn parse_bond_and(&mut self) -> Result<Option<BondQuery>, SmilesError> {
        let Some(head) = self.parse_bond_not()? else {
            return Ok(None);
        };
        let mut parts = vec![head];
        while self.scanner.peek() == Some('&') {
            self.scanner.advance();
            let Some(next) = self.parse_bond_not()? else {
                return Err(self.error(SmilesErrorKind::UnexpectedEnd));
            };
            parts.push(next);
        }
        Ok(Some(if parts.len() == 1 {
            parts.pop().unwrap()
        } else {
            BondQuery::And(parts)
        }))
    }

    /// SMARTS bond `!`-NOT (unary). The inner expression is a single bond
    /// kind — nested `!!` is allowed but not `!(...)` groups.
    fn parse_bond_not(&mut self) -> Result<Option<BondQuery>, SmilesError> {
        if self.scanner.peek() == Some('!') {
            self.scanner.advance();
            let Some(inner) = self.parse_bond_not()? else {
                return Err(self.error(SmilesErrorKind::UnexpectedEnd));
            };
            Ok(Some(BondQuery::Not(Box::new(inner))))
        } else {
            Ok(self.parse_bond_kind()?.map(BondQuery::Kind))
        }
    }

    // -----------------------------------------------------------------------
    // SMARTS bracket-atom parsing
    // -----------------------------------------------------------------------

    fn parse_bracket_atom_smarts(&mut self, start: usize) -> Result<AtomNode, SmilesError> {
        let query = self.parse_atom_query_low_and()?;

        if self.scanner.peek() != Some(']') {
            return Err(self.error_at(
                SmilesErrorKind::UnclosedBracket,
                self.scanner.span_from(start),
            ));
        }
        self.scanner.advance(); // consume ']'

        // Optimisation: if the query is a single concrete element with no
        // logical operators, produce a Bracket AtomSpec instead of a Query.
        let spec = self.simplify_query_to_bracket(query);

        Ok(AtomNode {
            spec,
            span: self.scanner.span_from(start),
            descriptors: Vec::new(),
        })
    }

    /// Try to collapse a trivial SMARTS query into a plain `AtomSpec::Bracket`.
    fn simplify_query_to_bracket(&self, query: AtomQuery) -> AtomSpec {
        // Only simplify single-primitive queries without logical ops.
        if let AtomQuery::Primitive(ref prim) = query {
            match prim {
                AtomPrimitive::Element { symbol, aromatic } => {
                    return AtomSpec::Bracket {
                        isotope: None,
                        symbol: BracketSymbol::Element {
                            symbol: symbol.clone(),
                            aromatic: *aromatic,
                        },
                        chirality: None,
                        hcount: None,
                        charge: None,
                        atom_class: None,
                    };
                }
                AtomPrimitive::Wildcard => {
                    return AtomSpec::Bracket {
                        isotope: None,
                        symbol: BracketSymbol::Any,
                        chirality: None,
                        hcount: None,
                        charge: None,
                        atom_class: None,
                    };
                }
                _ => {}
            }
        }
        AtomSpec::Query(query)
    }

    // -- SMARTS query expression parsing ------------------------------------
    //
    // Precedence (lowest to highest):
    //   ;  — low AND
    //   ,  — OR
    //   &  — high AND (also implicit between adjacent primitives)
    //   !  — NOT (unary prefix)

    fn parse_atom_query_low_and(&mut self) -> Result<AtomQuery, SmilesError> {
        let mut parts = vec![self.parse_atom_query_or()?];
        while self.scanner.peek() == Some(';') {
            self.scanner.advance();
            parts.push(self.parse_atom_query_or()?);
        }
        if parts.len() == 1 {
            Ok(parts.pop().unwrap())
        } else {
            Ok(AtomQuery::LowAnd(parts))
        }
    }

    fn parse_atom_query_or(&mut self) -> Result<AtomQuery, SmilesError> {
        let mut parts = vec![self.parse_atom_query_and()?];
        while self.scanner.peek() == Some(',') {
            self.scanner.advance();
            parts.push(self.parse_atom_query_and()?);
        }
        if parts.len() == 1 {
            Ok(parts.pop().unwrap())
        } else {
            Ok(AtomQuery::Or(parts))
        }
    }

    fn parse_atom_query_and(&mut self) -> Result<AtomQuery, SmilesError> {
        let mut parts = vec![self.parse_atom_query_not()?];
        loop {
            // Explicit '&' or implicit adjacency (next char starts a primitive).
            if self.scanner.peek() == Some('&') {
                self.scanner.advance();
                parts.push(self.parse_atom_query_not()?);
            } else if self
                .scanner
                .peek()
                .is_some_and(|c| self.is_smarts_primitive_start(c))
            {
                parts.push(self.parse_atom_query_not()?);
            } else {
                break;
            }
        }
        if parts.len() == 1 {
            Ok(parts.pop().unwrap())
        } else {
            Ok(AtomQuery::And(parts))
        }
    }

    fn parse_atom_query_not(&mut self) -> Result<AtomQuery, SmilesError> {
        if self.scanner.peek() == Some('!') {
            self.scanner.advance();
            let inner = self.parse_atom_query_not()?;
            Ok(AtomQuery::Not(Box::new(inner)))
        } else {
            let prim = self.parse_atom_primitive()?;
            Ok(AtomQuery::Primitive(prim))
        }
    }

    fn is_smarts_primitive_start(&self, ch: char) -> bool {
        // Characters that can start a SMARTS atom primitive inside brackets.
        // `:` introduces an atom-class primitive (`:<n>`, Daylight §3.1).
        ch.is_ascii_alphabetic()
            || ch == '*'
            || ch == '#'
            || ch == '+'
            || ch == '-'
            || ch.is_ascii_digit()
            || ch == '$'
            || ch == '@'
            || ch == '!'
            || ch == ':'
    }

    fn parse_atom_primitive(&mut self) -> Result<AtomPrimitive, SmilesError> {
        match self.scanner.peek() {
            Some(':') => {
                // Atom class / map number — Daylight `[atom:<n>]`.
                self.scanner.advance();
                let digits = self.scanner.eat_digits();
                let n: u16 = digits.parse().map_err(|_| {
                    self.error(SmilesErrorKind::InvalidQueryPrimitive(format!(":{digits}")))
                })?;
                Ok(AtomPrimitive::AtomClass(n))
            }
            Some('*') => {
                self.scanner.advance();
                Ok(AtomPrimitive::Wildcard)
            }
            Some('#') => {
                // Atomic number: #6 means carbon
                self.scanner.advance();
                let digits = self.scanner.eat_digits();
                let _num: u8 = digits.parse().map_err(|_| {
                    self.error(SmilesErrorKind::InvalidQueryPrimitive(format!("#{digits}")))
                })?;
                // Resolve to element symbol (would need Element::by_number).
                // For now, store as element with the number.
                Ok(AtomPrimitive::Element {
                    symbol: format!("#{digits}"),
                    aromatic: false,
                })
            }
            Some('$') => {
                // Recursive SMARTS: $(...)
                self.scanner.advance();
                if self.scanner.peek() != Some('(') {
                    return Err(self.error(SmilesErrorKind::UnexpectedChar(
                        self.scanner.peek().unwrap_or('\0'),
                    )));
                }
                self.scanner.advance(); // consume '('
                self.depth += 1;
                if self.depth > MAX_RECURSION_DEPTH {
                    return Err(self.error(SmilesErrorKind::RecursionLimit));
                }
                let mol = self.parse_molecule()?;
                self.depth -= 1;
                if self.scanner.peek() != Some(')') {
                    return Err(self.error(SmilesErrorKind::UnclosedRecursive));
                }
                self.scanner.advance(); // consume ')'
                Ok(AtomPrimitive::Recursive(Box::new(mol)))
            }
            Some('@') => {
                self.scanner.advance();
                if self.scanner.peek() == Some('@') {
                    self.scanner.advance();
                    Ok(AtomPrimitive::Chirality(Chirality::Clockwise))
                } else {
                    Ok(AtomPrimitive::Chirality(Chirality::CounterClockwise))
                }
            }
            Some('+') => {
                self.scanner.advance();
                if self.scanner.peek() == Some('+') {
                    self.scanner.advance();
                    Ok(AtomPrimitive::Charge(2))
                } else if let Some(d) = self.scanner.eat_digit() {
                    Ok(AtomPrimitive::Charge(d as i8))
                } else {
                    Ok(AtomPrimitive::Charge(1))
                }
            }
            Some('-') => {
                self.scanner.advance();
                if self.scanner.peek() == Some('-') {
                    self.scanner.advance();
                    Ok(AtomPrimitive::Charge(-2))
                } else if let Some(d) = self.scanner.eat_digit() {
                    Ok(AtomPrimitive::Charge(-(d as i8)))
                } else {
                    Ok(AtomPrimitive::Charge(-1))
                }
            }
            Some(c) if c.is_ascii_digit() => {
                // Isotope: bare number
                let digits = self.scanner.eat_digits();
                let iso: u16 = digits.parse().map_err(|_| {
                    self.error(SmilesErrorKind::InvalidQueryPrimitive(digits.to_owned()))
                })?;
                Ok(AtomPrimitive::Isotope(iso))
            }
            Some(c) if c.is_ascii_uppercase() => {
                // Uppercase: element symbol or SMARTS primitive letter
                match c {
                    'A' => {
                        // Could be 'A' (aliphatic wildcard) or element like 'Al', 'Ag', etc.
                        self.scanner.advance();
                        if let Some(c2) = self.scanner.peek() {
                            if c2.is_ascii_lowercase()
                                && c2 != 'l'
                                && c2 != 'g'
                                && c2 != 'r'
                                && c2 != 's'
                                && c2 != 'u'
                                && c2 != 'c'
                                && c2 != 't'
                                && c2 != 'm'
                            {
                                // Not a known two-letter element starting with A
                                return Ok(AtomPrimitive::Aliphatic);
                            }
                            if c2.is_ascii_lowercase() {
                                // Two-letter element: Al, Ag, Ar, As, Au, Ac, At, Am
                                let mut sym = String::from('A');
                                sym.push(c2);
                                self.scanner.advance();
                                return Ok(AtomPrimitive::Element {
                                    symbol: sym,
                                    aromatic: false,
                                });
                            }
                        }
                        Ok(AtomPrimitive::Aliphatic)
                    }
                    'D' => {
                        self.scanner.advance();
                        if let Some(d) = self.scanner.eat_digit() {
                            Ok(AtomPrimitive::Degree(d))
                        } else if self.scanner.peek().is_some_and(|c| c.is_ascii_lowercase()) {
                            // Dy, Db, Ds — two-letter elements
                            let c2 = self.scanner.advance().unwrap();
                            Ok(AtomPrimitive::Element {
                                symbol: format!("D{c2}"),
                                aromatic: false,
                            })
                        } else {
                            Ok(AtomPrimitive::Degree(1))
                        }
                    }
                    'H' => {
                        self.scanner.advance();
                        if let Some(d) = self.scanner.eat_digit() {
                            Ok(AtomPrimitive::HCount(d))
                        } else if self.scanner.peek().is_some_and(|c| {
                            c == 'e' || c == 'f' || c == 'g' || c == 's' || c == 'o'
                        }) {
                            let c2 = self.scanner.advance().unwrap();
                            Ok(AtomPrimitive::Element {
                                symbol: format!("H{c2}"),
                                aromatic: false,
                            })
                        } else {
                            Ok(AtomPrimitive::HCount(1))
                        }
                    }
                    'R' => {
                        self.scanner.advance();
                        if let Some(d) = self.scanner.eat_digit() {
                            Ok(AtomPrimitive::RingMembership(Some(d)))
                        } else if self.scanner.peek().is_some_and(|c| c.is_ascii_lowercase()) {
                            let c2 = self.scanner.advance().unwrap();
                            Ok(AtomPrimitive::Element {
                                symbol: format!("R{c2}"),
                                aromatic: false,
                            })
                        } else {
                            Ok(AtomPrimitive::RingMembership(None))
                        }
                    }
                    'X' => {
                        self.scanner.advance();
                        if let Some(d) = self.scanner.eat_digit() {
                            Ok(AtomPrimitive::TotalConnections(d))
                        } else if self.scanner.peek().is_some_and(|c| c.is_ascii_lowercase()) {
                            let c2 = self.scanner.advance().unwrap();
                            Ok(AtomPrimitive::Element {
                                symbol: format!("X{c2}"),
                                aromatic: false,
                            })
                        } else {
                            Ok(AtomPrimitive::TotalConnections(1))
                        }
                    }
                    _ => {
                        // Generic uppercase: element symbol
                        let sym = self.consume_element_symbol();
                        Ok(AtomPrimitive::Element {
                            symbol: sym,
                            aromatic: false,
                        })
                    }
                }
            }
            Some(c) if c.is_ascii_lowercase() => {
                match c {
                    'h' => {
                        self.scanner.advance();
                        if let Some(d) = self.scanner.eat_digit() {
                            Ok(AtomPrimitive::ImplicitH(d))
                        } else {
                            Ok(AtomPrimitive::ImplicitH(1))
                        }
                    }
                    'r' => {
                        self.scanner.advance();
                        if let Some(d) = self.scanner.eat_digit() {
                            Ok(AtomPrimitive::RingSize(d))
                        } else {
                            // Bare 'r' means "in a ring" — same as R but lowercase
                            Ok(AtomPrimitive::RingMembership(None))
                        }
                    }
                    'v' => {
                        self.scanner.advance();
                        if let Some(d) = self.scanner.eat_digit() {
                            Ok(AtomPrimitive::Valence(d))
                        } else {
                            Ok(AtomPrimitive::Valence(1))
                        }
                    }
                    // Aromatic element symbols: c, n, o, s, p
                    'c' | 'n' | 'o' | 's' | 'p' => {
                        self.scanner.advance();
                        Ok(AtomPrimitive::Element {
                            symbol: c.to_string(),
                            aromatic: true,
                        })
                    }
                    _ => {
                        let sym = c.to_string();
                        self.scanner.advance();
                        Err(self.error(SmilesErrorKind::InvalidQueryPrimitive(sym)))
                    }
                }
            }
            Some(c) => Err(self.error(SmilesErrorKind::UnexpectedChar(c))),
            None => Err(self.error(SmilesErrorKind::UnclosedBracket)),
        }
    }
}

// ==========================================================================
// Tests
// ==========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::smiles::chem::test_support::atom_nodes;
    use crate::io::smiles::error::Notation;

    // -- helpers ------------------------------------------------------------

    fn smiles(input: &str) -> SmilesIR {
        parse_smiles(input).unwrap_or_else(|e| panic!("parse_smiles({input:?}) failed: {e}"))
    }

    fn smarts(input: &str) -> SmilesIR {
        parse_smarts(input).unwrap_or_else(|e| panic!("parse_smarts({input:?}) failed: {e}"))
    }

    fn atom_count(mol: &SmilesIR) -> usize {
        mol.components.iter().map(chain_atom_count).sum()
    }

    fn chain_atom_count(chain: &Chain) -> usize {
        1 + chain
            .tail
            .iter()
            .map(|elem| match elem {
                ChainElement::BondedAtom { .. } => 1,
                ChainElement::Branch { chain, .. } => chain_atom_count(chain),
                ChainElement::RingClosure { .. } => 0,
            })
            .sum::<usize>()
    }

    // -- simple atoms -------------------------------------------------------

    #[test]
    fn test_single_atom() {
        let mol = smiles("C");
        assert_eq!(mol.components.len(), 1);
        assert_eq!(atom_count(&mol), 1);
        assert!(matches!(
            &mol.components[0].head.spec,
            AtomSpec::Organic { symbol, aromatic: false } if symbol == "C"
        ));
    }

    #[test]
    fn test_two_atoms() {
        let mol = smiles("CO");
        assert_eq!(atom_count(&mol), 2);
    }

    #[test]
    fn test_aromatic() {
        let mol = smiles("c");
        assert!(matches!(
            &mol.components[0].head.spec,
            AtomSpec::Organic { aromatic: true, .. }
        ));
    }

    #[test]
    fn test_chlorine() {
        let mol = smiles("Cl");
        assert_eq!(atom_count(&mol), 1);
        assert!(matches!(
            &mol.components[0].head.spec,
            AtomSpec::Organic { symbol, .. } if symbol == "Cl"
        ));
    }

    #[test]
    fn test_bromine() {
        let mol = smiles("Br");
        assert_eq!(atom_count(&mol), 1);
    }

    // -- bonds --------------------------------------------------------------

    #[test]
    fn test_double_bond() {
        let mol = smiles("C=O");
        assert_eq!(atom_count(&mol), 2);
        match &mol.components[0].tail[0] {
            ChainElement::BondedAtom { bond, .. } => {
                assert_eq!(*bond, Some(BondQuery::Kind(BondKind::Double)));
            }
            _ => panic!("expected BondedAtom"),
        }
    }

    #[test]
    fn test_triple_bond() {
        let mol = smiles("C#N");
        match &mol.components[0].tail[0] {
            ChainElement::BondedAtom { bond, .. } => {
                assert_eq!(*bond, Some(BondQuery::Kind(BondKind::Triple)));
            }
            _ => panic!("expected BondedAtom"),
        }
    }

    // -- branches -----------------------------------------------------------

    #[test]
    fn test_branch() {
        let mol = smiles("CC(C)C");
        assert_eq!(atom_count(&mol), 4);
    }

    #[test]
    fn test_branch_with_bond() {
        // CC(=O)O: head=C, tail[0]=BondedAtom(C), tail[1]=Branch(=O), tail[2]=BondedAtom(O)
        let mol = smiles("CC(=O)O");
        assert_eq!(atom_count(&mol), 4);
        match &mol.components[0].tail[1] {
            ChainElement::Branch { bond, .. } => {
                assert_eq!(*bond, Some(BondQuery::Kind(BondKind::Double)));
            }
            _ => panic!("expected Branch at tail[1]"),
        }
    }

    #[test]
    fn test_nested_branch() {
        let mol = smiles("CC(C(C)C)C");
        assert_eq!(atom_count(&mol), 6);
    }

    // -- ring closures ------------------------------------------------------

    #[test]
    fn test_cyclohexane() {
        let mol = smiles("C1CCCCC1");
        assert_eq!(atom_count(&mol), 6);
        // Should have ring closures at positions 0 and 5
        let tail = &mol.components[0].tail;
        assert!(
            tail.iter()
                .any(|e| matches!(e, ChainElement::RingClosure { rnum: 1, .. }))
        );
    }

    #[test]
    fn test_benzene_aromatic() {
        let mol = smiles("c1ccccc1");
        assert_eq!(atom_count(&mol), 6);
    }

    #[test]
    fn test_two_digit_ring() {
        let mol = smiles("C%12CCCCC%12");
        assert_eq!(atom_count(&mol), 6);
        let tail = &mol.components[0].tail;
        assert!(
            tail.iter()
                .any(|e| matches!(e, ChainElement::RingClosure { rnum: 12, .. }))
        );
    }

    // -- bracket atoms ------------------------------------------------------

    #[test]
    fn test_bracket_isotope() {
        let mol = smiles("[13CH4]");
        assert_eq!(atom_count(&mol), 1);
        match &mol.components[0].head.spec {
            AtomSpec::Bracket {
                isotope,
                symbol,
                hcount,
                ..
            } => {
                assert_eq!(*isotope, Some(13));
                assert!(matches!(symbol, BracketSymbol::Element { symbol, .. } if symbol == "C"));
                assert_eq!(*hcount, Some(4));
            }
            _ => panic!("expected Bracket"),
        }
    }

    #[test]
    fn test_bracket_charge_positive() {
        let mol = smiles("[Fe+2]");
        match &mol.components[0].head.spec {
            AtomSpec::Bracket { charge, symbol, .. } => {
                assert_eq!(*charge, Some(2));
                assert!(matches!(symbol, BracketSymbol::Element { symbol, .. } if symbol == "Fe"));
            }
            _ => panic!("expected Bracket"),
        }
    }

    #[test]
    fn test_bracket_charge_negative() {
        let mol = smiles("[O-]");
        match &mol.components[0].head.spec {
            AtomSpec::Bracket { charge, .. } => assert_eq!(*charge, Some(-1)),
            _ => panic!("expected Bracket"),
        }
    }

    #[test]
    fn test_bracket_charge_double_minus() {
        let mol = smiles("[O--]");
        match &mol.components[0].head.spec {
            AtomSpec::Bracket { charge, .. } => assert_eq!(*charge, Some(-2)),
            _ => panic!("expected Bracket"),
        }
    }

    #[test]
    fn test_atom_class() {
        let mol = smiles("[CH3:1]");
        match &mol.components[0].head.spec {
            AtomSpec::Bracket {
                atom_class, hcount, ..
            } => {
                assert_eq!(*atom_class, Some(1));
                assert_eq!(*hcount, Some(3));
            }
            _ => panic!("expected Bracket"),
        }
    }

    // -- stereochemistry ----------------------------------------------------

    #[test]
    fn test_tetrahedral_ccw() {
        let mol = smiles("[C@H](F)(Cl)Br");
        match &mol.components[0].head.spec {
            AtomSpec::Bracket { chirality, .. } => {
                assert_eq!(*chirality, Some(Chirality::CounterClockwise));
            }
            _ => panic!("expected Bracket"),
        }
    }

    #[test]
    fn test_tetrahedral_cw() {
        let mol = smiles("[C@@H](F)(Cl)Br");
        match &mol.components[0].head.spec {
            AtomSpec::Bracket { chirality, .. } => {
                assert_eq!(*chirality, Some(Chirality::Clockwise));
            }
            _ => panic!("expected Bracket"),
        }
    }

    #[test]
    fn test_cis_trans() {
        let mol = smiles("F/C=C/F");
        assert_eq!(atom_count(&mol), 4);
    }

    // -- disconnected components --------------------------------------------

    #[test]
    fn test_disconnected() {
        let mol = smiles("[Na+].[Cl-]");
        assert_eq!(mol.components.len(), 2);
        assert_eq!(atom_count(&mol), 2);
    }

    // -- wildcard -----------------------------------------------------------

    #[test]
    fn test_wildcard() {
        let mol = smiles("*");
        assert!(matches!(&mol.components[0].head.spec, AtomSpec::Wildcard));
    }

    // -- error cases --------------------------------------------------------

    #[test]
    fn test_empty_input() {
        let err = parse_smiles("").unwrap_err();
        assert!(matches!(err.kind, SmilesErrorKind::EmptyInput));
    }

    #[test]
    fn test_unclosed_bracket() {
        let err = parse_smiles("[CH4").unwrap_err();
        assert!(matches!(err.kind, SmilesErrorKind::UnclosedBracket));
    }

    #[test]
    fn test_unclosed_branch() {
        let err = parse_smiles("CC(O").unwrap_err();
        assert!(matches!(err.kind, SmilesErrorKind::UnclosedBranch));
    }

    #[test]
    fn test_trailing_characters() {
        let err = parse_smiles("CC)").unwrap_err();
        assert!(matches!(err.kind, SmilesErrorKind::TrailingCharacters));
    }

    // -- real molecules -----------------------------------------------------

    #[test]
    fn test_ethanol() {
        let mol = smiles("CCO");
        assert_eq!(atom_count(&mol), 3);
    }

    #[test]
    fn test_acetic_acid() {
        let mol = smiles("CC(=O)O");
        assert_eq!(atom_count(&mol), 4);
    }

    #[test]
    fn test_caffeine() {
        // Caffeine SMILES
        let mol = smiles("Cn1cnc2c1c(=O)n(c(=O)n2C)C");
        assert!(atom_count(&mol) > 10);
    }

    #[test]
    fn test_aspirin() {
        let mol = smiles("CC(=O)Oc1ccccc1C(=O)O");
        assert!(atom_count(&mol) > 8);
    }

    // -- SMARTS tests -------------------------------------------------------

    #[test]
    fn test_smarts_wildcard_atom() {
        let mol = smarts("[*]");
        assert_eq!(atom_count(&mol), 1);
    }

    #[test]
    fn test_smarts_not() {
        let mol = smarts("[!C]");
        match &mol.components[0].head.spec {
            AtomSpec::Query(AtomQuery::Not(inner)) => {
                assert!(matches!(
                    inner.as_ref(),
                    AtomQuery::Primitive(AtomPrimitive::Element { symbol, aromatic: false }) if symbol == "C"
                ));
            }
            _ => panic!("expected Query(Not(...))"),
        }
    }

    #[test]
    fn test_smarts_or() {
        let mol = smarts("[C,N]");
        match &mol.components[0].head.spec {
            AtomSpec::Query(AtomQuery::Or(parts)) => {
                assert_eq!(parts.len(), 2);
            }
            _ => panic!("expected Query(Or(...))"),
        }
    }

    #[test]
    fn test_smarts_and_high() {
        let mol = smarts("[C&R]");
        match &mol.components[0].head.spec {
            AtomSpec::Query(AtomQuery::And(parts)) => {
                assert_eq!(parts.len(), 2);
            }
            _ => panic!("expected Query(And(...))"),
        }
    }

    #[test]
    fn test_smarts_low_and() {
        let mol = smarts("[C,N;R]");
        match &mol.components[0].head.spec {
            AtomSpec::Query(AtomQuery::LowAnd(parts)) => {
                assert_eq!(parts.len(), 2);
            }
            _ => panic!("expected Query(LowAnd(...))"),
        }
    }

    #[test]
    fn test_smarts_degree() {
        let mol = smarts("[D3]");
        match &mol.components[0].head.spec {
            AtomSpec::Query(AtomQuery::Primitive(AtomPrimitive::Degree(3))) => {}
            _ => panic!("expected Degree(3)"),
        }
    }

    #[test]
    fn test_smarts_ring_membership() {
        let mol = smarts("[R]");
        match &mol.components[0].head.spec {
            AtomSpec::Query(AtomQuery::Primitive(AtomPrimitive::RingMembership(None))) => {}
            _ => panic!("expected RingMembership(None)"),
        }
    }

    #[test]
    fn test_smarts_any_bond() {
        let mol = smarts("[C]~[N]");
        match &mol.components[0].tail[0] {
            ChainElement::BondedAtom {
                bond: Some(BondQuery::Kind(BondKind::Any)),
                ..
            } => {}
            other => panic!("expected Any bond, got {other:?}"),
        }
    }

    #[test]
    fn test_smarts_recursive() {
        let mol = smarts("[$(CC)]");
        match &mol.components[0].head.spec {
            AtomSpec::Query(AtomQuery::Primitive(AtomPrimitive::Recursive(inner))) => {
                assert_eq!(inner.components.len(), 1);
            }
            _ => panic!("expected Recursive"),
        }
    }

    // -- fragment dialect: helpers ------------------------------------------

    fn fragment(input: &str) -> SmilesIR {
        parse_fragment_smiles(input)
            .unwrap_or_else(|e| panic!("parse_fragment_smiles({input:?}) failed: {e}"))
    }

    fn descriptor(kind: DescriptorKind, label: &str, order: Option<BondKind>) -> BondingDescriptor {
        BondingDescriptor {
            kind,
            label: label.to_owned(),
            order,
        }
    }

    // -- fragment dialect: anchoring (R4.2 / R4.3) --------------------------

    #[test]
    fn test_fragment_symmetric_descriptors_anchor_on_first_and_last_atom() {
        let mol = fragment("[$]COC[$]");
        let nodes = atom_nodes(&mol);
        assert_eq!(nodes.len(), 3);
        let expected = vec![descriptor(DescriptorKind::Symmetric, "", None)];
        assert_eq!(nodes[0].descriptors, expected);
        assert!(nodes[1].descriptors.is_empty());
        assert_eq!(nodes[2].descriptors, expected);
    }

    #[test]
    fn test_fragment_directional_descriptors_anchor_on_chain_atoms() {
        // head N, C, carbonyl C (the anchor the trailing `[<]` binds to), O.
        let mol = fragment("[>]NCC(=O)[<]");
        let nodes = atom_nodes(&mol);
        assert_eq!(nodes.len(), 4);
        assert_eq!(
            nodes[0].descriptors,
            vec![descriptor(DescriptorKind::Right, "", None)]
        );
        assert!(nodes[1].descriptors.is_empty());
        assert_eq!(
            nodes[2].descriptors,
            vec![descriptor(DescriptorKind::Left, "", None)]
        );
        assert!(nodes[3].descriptors.is_empty());
    }

    #[test]
    fn test_fragment_descriptor_labels_are_kept_per_atom() {
        let mol = fragment("Clc[$a]c[$b]");
        let nodes = atom_nodes(&mol);
        assert_eq!(nodes.len(), 3);
        assert_eq!(
            nodes[1].descriptors,
            vec![descriptor(DescriptorKind::Symmetric, "a", None)]
        );
        assert_eq!(
            nodes[2].descriptors,
            vec![descriptor(DescriptorKind::Symmetric, "b", None)]
        );
    }

    #[test]
    fn test_fragment_multiple_descriptors_keep_written_order() {
        let mol = fragment("[>][$1]COC[<]");
        let nodes = atom_nodes(&mol);
        assert_eq!(
            nodes[0].descriptors,
            vec![
                descriptor(DescriptorKind::Right, "", None),
                descriptor(DescriptorKind::Symmetric, "1", None),
            ]
        );
    }

    // -- fragment dialect: out-of-bracket bond order (R4.4) -----------------

    #[test]
    fn test_fragment_bond_before_descriptor_is_the_descriptor_order() {
        let mol = fragment("CC=[$]");
        let nodes = atom_nodes(&mol);
        assert_eq!(nodes.len(), 2);
        assert_eq!(
            nodes[1].descriptors,
            vec![descriptor(
                DescriptorKind::Symmetric,
                "",
                Some(BondKind::Double)
            )]
        );
    }

    #[test]
    fn test_fragment_preceding_run_order_goes_to_the_adjacent_bracket() {
        // A run written *after* the bond symbol: `=` is adjacent to `[$]`, the
        // first bracket of the run, so `$` carries the order and `>` none.
        let mol = fragment("CC=[$][>]");
        let nodes = atom_nodes(&mol);
        assert_eq!(nodes.len(), 2);
        assert_eq!(
            nodes[1].descriptors,
            vec![
                descriptor(DescriptorKind::Symmetric, "", Some(BondKind::Double)),
                descriptor(DescriptorKind::Right, "", None),
            ]
        );
    }

    #[test]
    fn test_fragment_bond_after_leading_descriptor_is_the_descriptor_order() {
        let mol = fragment("[$]=CCC");
        let nodes = atom_nodes(&mol);
        assert_eq!(nodes.len(), 3);
        assert_eq!(
            nodes[0].descriptors,
            vec![descriptor(
                DescriptorKind::Symmetric,
                "",
                Some(BondKind::Double)
            )]
        );
    }

    #[test]
    fn test_fragment_leading_run_order_goes_to_the_adjacent_bracket() {
        // A leading run written *before* the bond symbol: `=` is adjacent to
        // `[$1]`, the last bracket of the run, so `$1` carries the order and
        // `>` none.
        let mol = fragment("[>][$1]=CCC");
        let nodes = atom_nodes(&mol);
        assert_eq!(nodes.len(), 3);
        assert_eq!(
            nodes[0].descriptors,
            vec![
                descriptor(DescriptorKind::Right, "", None),
                descriptor(DescriptorKind::Symmetric, "1", Some(BondKind::Double)),
            ]
        );
    }

    #[test]
    fn test_fragment_leading_descriptor_order_leaves_chain_bonds_single() {
        let mol = fragment("[$]=CCC");
        for elem in &mol.components[0].tail {
            match elem {
                ChainElement::BondedAtom { bond, .. } => assert!(
                    matches!(bond, None | Some(BondQuery::Kind(BondKind::Single))),
                    "expected a single C-C bond, got {bond:?}"
                ),
                other => panic!("expected BondedAtom, got {other:?}"),
            }
        }
    }

    #[test]
    fn test_fragment_midchain_descriptor_takes_no_order_from_following_bond() {
        // Spec-local disambiguation: mid-chain, a bond after the bracket is an
        // ordinary bond to the next atom, not the descriptor's order.
        let mol = fragment("C[$]=CC");
        let nodes = atom_nodes(&mol);
        assert_eq!(
            nodes[0].descriptors,
            vec![descriptor(DescriptorKind::Symmetric, "", None)]
        );
    }

    #[test]
    fn test_fragment_midchain_bond_after_descriptor_bonds_the_next_atom() {
        let mol = fragment("C[$]=CC");
        match &mol.components[0].tail[0] {
            ChainElement::BondedAtom { bond, .. } => {
                assert_eq!(*bond, Some(BondQuery::Kind(BondKind::Double)));
            }
            other => panic!("expected BondedAtom, got {other:?}"),
        }
    }

    // -- fragment dialect: branches -----------------------------------------

    #[test]
    fn test_fragment_branch_descriptor_anchors_on_the_parent_atom() {
        let mol = fragment("N([>])C");
        let nodes = atom_nodes(&mol);
        assert_eq!(nodes.len(), 2);
        assert_eq!(
            nodes[0].descriptors,
            vec![descriptor(DescriptorKind::Right, "", None)]
        );
        assert!(nodes[1].descriptors.is_empty());
    }

    #[test]
    fn test_fragment_descriptor_only_branch_emits_no_chain_element() {
        let mol = fragment("N([>])C");
        let tail = &mol.components[0].tail;
        assert_eq!(tail.len(), 1);
        assert!(matches!(tail[0], ChainElement::BondedAtom { .. }));
    }

    #[test]
    fn test_fragment_symmetric_branch_descriptor_anchors_on_the_parent_atom() {
        let mol = fragment("C([$])O");
        let nodes = atom_nodes(&mol);
        assert_eq!(nodes.len(), 2);
        assert_eq!(
            nodes[0].descriptors,
            vec![descriptor(DescriptorKind::Symmetric, "", None)]
        );
    }

    #[test]
    fn test_fragment_descriptor_in_mixed_branch_anchors_on_the_parent_atom() {
        // `>` lands on the first C, not on the branch atom N.
        let mol = fragment("C([>]N)C");
        let nodes = atom_nodes(&mol);
        assert_eq!(nodes.len(), 3);
        assert_eq!(
            nodes[0].descriptors,
            vec![descriptor(DescriptorKind::Right, "", None)]
        );
        assert!(nodes[1].descriptors.is_empty());
    }

    #[test]
    fn test_fragment_branch_bond_before_descriptor_is_the_descriptor_order() {
        let mol = fragment("C(=[>])C");
        let nodes = atom_nodes(&mol);
        assert_eq!(
            nodes[0].descriptors,
            vec![descriptor(
                DescriptorKind::Right,
                "",
                Some(BondKind::Double)
            )]
        );
    }

    #[test]
    fn test_fragment_empty_branch_is_still_an_error() {
        assert!(parse_fragment_smiles("C()").is_err());
    }

    // -- fragment dialect: the `$` glyph overload ---------------------------

    #[test]
    fn test_fragment_quadruple_bond_is_not_a_descriptor() {
        let mol = fragment("C$C");
        let nodes = atom_nodes(&mol);
        assert_eq!(nodes.len(), 2);
        assert!(nodes.iter().all(|n| n.descriptors.is_empty()));
        match &mol.components[0].tail[0] {
            ChainElement::BondedAtom { bond, .. } => {
                assert_eq!(*bond, Some(BondQuery::Kind(BondKind::Quadruple)));
            }
            other => panic!("expected BondedAtom, got {other:?}"),
        }
    }

    #[test]
    fn test_plain_smiles_quadruple_bond_is_unchanged() {
        let mol = smiles("C$C");
        assert_eq!(atom_count(&mol), 2);
        match &mol.components[0].tail[0] {
            ChainElement::BondedAtom { bond, .. } => {
                assert_eq!(*bond, Some(BondQuery::Kind(BondKind::Quadruple)));
            }
            other => panic!("expected BondedAtom, got {other:?}"),
        }
    }

    // -- dialect isolation ---------------------------------------------------

    #[test]
    fn test_plain_smiles_rejects_descriptors() {
        let err = parse_smiles("[$]COC[$]").unwrap_err();
        assert!(matches!(err.kind, SmilesErrorKind::DescriptorInPlainSmiles));
    }

    #[test]
    fn test_plain_smiles_rejects_recursive_smarts_bracket_as_descriptor() {
        let err = parse_smiles("[$(C)]").unwrap_err();
        assert!(matches!(err.kind, SmilesErrorKind::DescriptorInPlainSmiles));
    }

    #[test]
    fn test_plain_smiles_bracket_atom_unchanged_by_descriptor_lookahead() {
        let mol = smiles("[NH4+]");
        assert_eq!(atom_count(&mol), 1);
        match &mol.components[0].head.spec {
            AtomSpec::Bracket {
                symbol,
                hcount,
                charge,
                ..
            } => {
                assert!(matches!(symbol, BracketSymbol::Element { symbol, .. } if symbol == "N"));
                assert_eq!(*hcount, Some(4));
                assert_eq!(*charge, Some(1));
            }
            other => panic!("expected Bracket, got {other:?}"),
        }
    }

    #[test]
    fn test_smarts_negation_unchanged_by_descriptor_lookahead() {
        let mol = smarts("[!C]");
        assert!(matches!(
            &mol.components[0].head.spec,
            AtomSpec::Query(AtomQuery::Not(_))
        ));
    }

    #[test]
    fn test_smarts_recursive_unchanged_by_descriptor_lookahead() {
        let mol = smarts("[$(C)]");
        assert!(matches!(
            &mol.components[0].head.spec,
            AtomSpec::Query(AtomQuery::Primitive(AtomPrimitive::Recursive(_)))
        ));
    }

    // -- fragment dialect: rejections ---------------------------------------

    #[test]
    fn test_fragment_bigsmiles_in_bracket_order_is_rejected() {
        let err = parse_fragment_smiles("[<=1]").unwrap_err();
        assert!(matches!(err.kind, SmilesErrorKind::BondInsideDescriptor));
    }

    #[test]
    fn test_fragment_in_bracket_single_bond_is_rejected() {
        let err = parse_fragment_smiles("[$-]").unwrap_err();
        assert!(matches!(err.kind, SmilesErrorKind::BondInsideDescriptor));
    }

    #[test]
    fn test_fragment_descriptor_without_any_atom_is_dangling() {
        let err = parse_fragment_smiles("[$]").unwrap_err();
        assert!(matches!(err.kind, SmilesErrorKind::DanglingDescriptor));
    }

    #[test]
    fn test_fragment_descriptor_in_empty_component_is_dangling() {
        let err = parse_fragment_smiles("C.[$]").unwrap_err();
        assert!(matches!(err.kind, SmilesErrorKind::DanglingDescriptor));
    }

    #[test]
    fn test_fragment_non_alphanumeric_label_is_rejected() {
        let err = parse_fragment_smiles("[$a+]").unwrap_err();
        assert!(matches!(
            err.kind,
            SmilesErrorKind::InvalidDescriptorLabel(_)
        ));
    }

    #[test]
    fn test_fragment_aromatic_descriptor_order_is_rejected() {
        let err = parse_fragment_smiles("c:[$]").unwrap_err();
        assert!(matches!(
            err.kind,
            SmilesErrorKind::InvalidDescriptorOrder(BondKind::Aromatic)
        ));
    }

    #[test]
    fn test_fragment_atom_weight_annotation_is_unsupported() {
        let err = parse_fragment_smiles("[C;0.5]").unwrap_err();
        match &err.kind {
            SmilesErrorKind::AtomAnnotationUnsupported(text) => assert_eq!(text, "0.5"),
            other => panic!("expected AtomAnnotationUnsupported, got {other:?}"),
        }
    }

    #[test]
    fn test_fragment_wildcard_overloading_annotation_is_unsupported() {
        let err = parse_fragment_smiles("[*;s=C,0]").unwrap_err();
        assert!(matches!(
            err.kind,
            SmilesErrorKind::AtomAnnotationUnsupported(_)
        ));
    }

    #[test]
    fn test_plain_smiles_annotation_still_reports_unclosed_bracket() {
        let err = parse_smiles("[C;0.5]").unwrap_err();
        assert!(matches!(err.kind, SmilesErrorKind::UnclosedBracket));
    }

    #[test]
    fn test_fragment_shared_descriptor_is_representable() {
        // `[!]` is the shared (kind-agnostic) descriptor: it anchors on its
        // neighbouring atom exactly like `$`/`<`/`>`, with or without a label.
        let mol = fragment("[!]C");
        let nodes = atom_nodes(&mol);
        assert_eq!(nodes.len(), 1);
        assert_eq!(
            nodes[0].descriptors,
            vec![descriptor(DescriptorKind::Shared, "", None)]
        );

        let labelled = fragment("C[!a]");
        let nodes = atom_nodes(&labelled);
        assert_eq!(nodes.len(), 1);
        assert_eq!(
            nodes[0].descriptors,
            vec![descriptor(DescriptorKind::Shared, "a", None)]
        );
    }

    // -- notation stamped at the entry point --------------------------------

    #[test]
    fn test_parse_smiles_error_is_stamped_smiles() {
        let err = parse_smiles("CC(").expect_err("an unclosed branch must be refused");
        assert_eq!(err.notation, Notation::Smiles);
    }

    #[test]
    fn test_parse_fragment_smiles_error_is_stamped_smiles() {
        let err = parse_fragment_smiles("[$]").expect_err("a dangling descriptor must be refused");
        assert_eq!(err.notation, Notation::Smiles);
    }

    #[test]
    fn test_parse_smarts_error_is_stamped_smarts() {
        let err = parse_smarts("[C").expect_err("an unclosed bracket must be refused");
        assert_eq!(err.notation, Notation::Smarts);
    }
}
