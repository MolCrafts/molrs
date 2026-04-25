//! Recursive-descent parser for SMILES and SMARTS notation.
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

use crate::smiles::chem::ast::*;
use crate::smiles::chem::scanner::Scanner;
use crate::smiles::error::{SmilesError, SmilesErrorKind};

/// Controls which grammar extensions are enabled.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ParserMode {
    /// Standard SMILES — only concrete atoms and bonds.
    Smiles,
    /// SMARTS — adds query primitives, logical operators, wildcard/ring bonds.
    Smarts,
}

/// Maximum recursion depth for SMARTS `$(...)` expressions.
const MAX_RECURSION_DEPTH: usize = 16;

/// The set of organic-subset symbols (no brackets required).
///
/// Lowercase entries represent aromatic atoms.
const ORGANIC_SUBSET: &[&str] = &[
    "B", "C", "N", "O", "P", "S", "F", "Cl", "Br", "I", "At", "Ts", "b", "c", "n", "o", "p", "s",
];

/// Parse a SMILES string into an AST.
pub fn parse_smiles(input: &str) -> Result<SmilesIR, SmilesError> {
    Parser::new(input, ParserMode::Smiles).parse_molecule()
}

/// Parse a SMARTS string into an AST.
pub fn parse_smarts(input: &str) -> Result<SmilesIR, SmilesError> {
    Parser::new(input, ParserMode::Smarts).parse_molecule()
}

// ---------------------------------------------------------------------------
// Parser internals
// ---------------------------------------------------------------------------

struct Parser<'a> {
    scanner: Scanner<'a>,
    mode: ParserMode,
    depth: usize,
}

impl<'a> Parser<'a> {
    fn new(input: &'a str, mode: ParserMode) -> Self {
        Self {
            scanner: Scanner::new(input),
            mode,
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

    fn parse_chain(&mut self) -> Result<Chain, SmilesError> {
        let head = self.parse_atom()?;
        let mut tail = Vec::new();

        loop {
            match self.scanner.peek() {
                Some('(') => tail.push(self.parse_branch()?),
                Some(c) if c.is_ascii_digit() || c == '%' => {
                    tail.push(self.parse_ring_closure(None)?);
                }
                Some(c)
                    if Self::is_bond_char(c)
                        || (self.mode == ParserMode::Smarts && Self::is_bond_char_smarts(c)) =>
                {
                    let bond = self.parse_bond()?;
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

    fn parse_branch(&mut self) -> Result<ChainElement, SmilesError> {
        let start = self.scanner.pos();
        self.scanner.expect('(')?;

        // Optional bond at the start of a branch.
        let bond = if self.scanner.peek().is_some_and(|c| {
            Self::is_bond_char(c)
                || (self.mode == ParserMode::Smarts && Self::is_bond_char_smarts(c))
        }) {
            self.parse_bond()?
        } else {
            None
        };

        let chain = self.parse_chain()?;

        if self.scanner.peek() != Some(')') {
            return Err(self.error_at(
                SmilesErrorKind::UnclosedBranch,
                self.scanner.span_from(start),
            ));
        }
        self.scanner.advance(); // consume ')'

        Ok(ChainElement::Branch {
            bond,
            chain,
            span: self.scanner.span_from(start),
        })
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
            });
        }

        Err(self.error_at(
            SmilesErrorKind::InvalidElement(one),
            Span::new(start, self.scanner.pos()),
        ))
    }

    fn parse_bracket_atom(&mut self, start: usize) -> Result<AtomNode, SmilesError> {
        self.scanner.expect('[')?;

        if self.mode == ParserMode::Smarts {
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
            Some('~') if self.mode == ParserMode::Smarts => {
                self.scanner.advance();
                Ok(Some(BondKind::Any))
            }
            Some('@') if self.mode == ParserMode::Smarts => {
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
        if self.mode == ParserMode::Smarts {
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
}
