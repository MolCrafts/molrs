//! SMARTS string → query graph (atoms, bonds, branches, ring closures,
//! recursive `$(...)`).
//!
//! Ported (grammar/semantics only) from RDKit's SMARTS parser under BSD-3:
//! `Code/GraphMol/SmilesParse/SmartsParse.cpp`.
//!
//! # Atom-expression precedence (Daylight)
//!
//! Inside a bracket atom `[...]`, operators bind, weakest first:
//! `;` (low AND) over `,` (OR) over implicit/`&` (high AND) over `!` (NOT).
//! Organic-subset atoms outside brackets (`C`, `c`, `N`, `*`, ...) are a
//! single primitive.

use crate::error::MolRsError;

use super::ast::{AtomPrimitive, AtomQuery, BondPrimitive, BondQuery};

/// A parsed query atom: its query tree + optional atom-map label (`:n`).
#[derive(Debug, Clone)]
pub struct QueryAtom {
    pub query: AtomQuery,
    pub map_label: Option<u32>,
}

/// A parsed query bond between two query-atom indices.
#[derive(Debug, Clone)]
pub struct QueryBond {
    pub a: usize,
    pub b: usize,
    pub query: BondQuery,
}

/// The whole compiled query graph.
#[derive(Debug, Clone, Default)]
pub struct QueryGraph {
    pub atoms: Vec<QueryAtom>,
    pub bonds: Vec<QueryBond>,
    /// Compiled recursive subpatterns, addressed by `AtomQuery::Recursive(i)`.
    pub recursives: Vec<QueryGraph>,
}

impl QueryGraph {
    /// Collect every `%LABEL` context-label appearing anywhere in this query
    /// graph, including inside recursive `$(...)` subpatterns. Order is the
    /// traversal order; duplicates are kept (callers dedup as needed).
    pub fn context_labels(&self) -> Vec<String> {
        let mut out = Vec::new();
        for atom in &self.atoms {
            atom.query.collect_context_labels(&mut out);
        }
        for sub in &self.recursives {
            out.extend(sub.context_labels());
        }
        out
    }
}

/// Parse a SMARTS string into a [`QueryGraph`].
pub fn parse(smarts: &str) -> Result<QueryGraph, MolRsError> {
    let mut p = Parser::new(smarts);
    let g = p.parse_graph()?;
    if !p.at_end() {
        return Err(MolRsError::parse(format!(
            "trailing characters at position {} in SMARTS '{}'",
            p.pos, smarts
        )));
    }
    Ok(g)
}

struct Parser<'s> {
    chars: Vec<char>,
    pos: usize,
    src: &'s str,
    /// Recursive `$(...)` subpatterns accumulated during parsing; moved into
    /// the [`QueryGraph`] when the top-level parse completes.
    recursive_stash: Vec<QueryGraph>,
    /// Whether the next bracket-atom primitive is the *leading* one (no prior
    /// primitive parsed inside the current `[...]`). Disambiguates a leading
    /// `H` (= hydrogen element) from a following `H<n>` (= total-H count), as
    /// in RDKit / molpy: `[H]` is the element, `[CH3]` / `[C;H1]` is a count.
    bracket_first_primitive: bool,
}

/// State threaded through a single connected SMARTS branch tree.
struct GraphState {
    graph: QueryGraph,
    /// open ring closures: digit → (atom index, pending bond query if any)
    ring_bonds: std::collections::HashMap<u32, (usize, Option<BondQuery>)>,
}

impl<'s> Parser<'s> {
    fn new(src: &'s str) -> Self {
        Self {
            chars: src.chars().collect(),
            pos: 0,
            src,
            recursive_stash: Vec::new(),
            bracket_first_primitive: false,
        }
    }

    fn at_end(&self) -> bool {
        self.pos >= self.chars.len()
    }

    fn peek(&self) -> Option<char> {
        self.chars.get(self.pos).copied()
    }

    fn bump(&mut self) -> Option<char> {
        let c = self.peek();
        if c.is_some() {
            self.pos += 1;
        }
        c
    }

    fn err(&self, msg: impl Into<String>) -> MolRsError {
        MolRsError::parse(format!(
            "{} at position {} in SMARTS '{}'",
            msg.into(),
            self.pos,
            self.src
        ))
    }

    // -- top level -----------------------------------------------------------

    fn parse_graph(&mut self) -> Result<QueryGraph, MolRsError> {
        let mut st = GraphState {
            graph: QueryGraph::default(),
            ring_bonds: std::collections::HashMap::new(),
        };
        self.parse_branch(&mut st, None)?;
        if !st.ring_bonds.is_empty() {
            return Err(self.err("unclosed ring bond"));
        }
        if st.graph.atoms.is_empty() {
            return Err(self.err("SMARTS contains no atoms"));
        }
        st.graph.recursives = std::mem::take(&mut self.recursive_stash);
        Ok(st.graph)
    }

    /// Parse a chain/branch. `prev` is the atom index the first atom of this
    /// branch should connect to (None at the very start of a component).
    fn parse_branch(
        &mut self,
        st: &mut GraphState,
        mut prev: Option<usize>,
    ) -> Result<(), MolRsError> {
        // A branch must produce at least one atom unless it's the whole input
        // being empty.
        let mut pending_bond: Option<BondQuery> = None;
        let mut produced = false;

        loop {
            match self.peek() {
                None => break,
                Some('(') => {
                    self.bump();
                    let anchor = prev.ok_or_else(|| self.err("'(' before any atom"))?;
                    self.parse_branch(st, Some(anchor))?;
                    if self.peek() != Some(')') {
                        return Err(self.err("unbalanced '(' — missing ')'"));
                    }
                    self.bump();
                }
                Some(')') => break,
                Some(c) if is_bond_char(c) => {
                    pending_bond = Some(self.parse_bond_expr()?);
                }
                Some(c) if c.is_ascii_digit() || c == '%' => {
                    // Ring closure on the current atom.
                    let anchor =
                        prev.ok_or_else(|| self.err("ring-closure digit before any atom"))?;
                    let digit = self.parse_ring_digit()?;
                    self.handle_ring_closure(st, anchor, digit, pending_bond.take())?;
                }
                Some(_) => {
                    let qatom = self.parse_atom()?;
                    let idx = st.graph.atoms.len();
                    st.graph.atoms.push(qatom);
                    produced = true;
                    if let Some(p) = prev {
                        let bond = pending_bond
                            .take()
                            .unwrap_or(BondQuery::Prim(BondPrimitive::SingleOrAromatic));
                        st.graph.bonds.push(QueryBond {
                            a: p,
                            b: idx,
                            query: bond,
                        });
                    } else if pending_bond.is_some() {
                        return Err(self.err("bond symbol before first atom"));
                    }
                    prev = Some(idx);
                }
            }
        }

        if pending_bond.is_some() {
            return Err(self.err("dangling bond with no following atom"));
        }
        // A leading '(' branch with no atoms is an error; the root may legally
        // be empty only when the whole SMARTS is empty.
        let _ = produced;
        Ok(())
    }

    // -- ring closures -------------------------------------------------------

    fn parse_ring_digit(&mut self) -> Result<u32, MolRsError> {
        match self.peek() {
            Some('%') => {
                self.bump();
                let d1 = self
                    .bump()
                    .filter(|c| c.is_ascii_digit())
                    .ok_or_else(|| self.err("'%' must be followed by two digits"))?;
                let d2 = self
                    .bump()
                    .filter(|c| c.is_ascii_digit())
                    .ok_or_else(|| self.err("'%' must be followed by two digits"))?;
                Ok(format!("{d1}{d2}").parse::<u32>().unwrap())
            }
            Some(c) if c.is_ascii_digit() => {
                self.bump();
                Ok(c.to_digit(10).unwrap())
            }
            _ => Err(self.err("expected ring-closure digit")),
        }
    }

    fn handle_ring_closure(
        &mut self,
        st: &mut GraphState,
        atom: usize,
        digit: u32,
        bond: Option<BondQuery>,
    ) -> Result<(), MolRsError> {
        if let Some((other, open_bond)) = st.ring_bonds.remove(&digit) {
            // Close: pick whichever side specified a bond; default otherwise.
            let q = bond
                .or(open_bond)
                .unwrap_or(BondQuery::Prim(BondPrimitive::SingleOrAromatic));
            if other == atom {
                return Err(self.err("ring bond to self"));
            }
            st.graph.bonds.push(QueryBond {
                a: other,
                b: atom,
                query: q,
            });
        } else {
            st.ring_bonds.insert(digit, (atom, bond));
        }
        Ok(())
    }

    // -- bonds ---------------------------------------------------------------

    fn parse_bond_expr(&mut self) -> Result<BondQuery, MolRsError> {
        // Low-precedence OR over high-precedence AND (';' acts like AND here).
        // SMARTS bond logic supports `,` (or), `;`/`&` (and), `!` (not).
        self.parse_bond_or()
    }

    fn parse_bond_or(&mut self) -> Result<BondQuery, MolRsError> {
        let mut terms = vec![self.parse_bond_and()?];
        while self.peek() == Some(',') {
            self.bump();
            terms.push(self.parse_bond_and()?);
        }
        Ok(if terms.len() == 1 {
            terms.pop().unwrap()
        } else {
            BondQuery::Or(terms)
        })
    }

    fn parse_bond_and(&mut self) -> Result<BondQuery, MolRsError> {
        let mut terms = vec![self.parse_bond_not()?];
        loop {
            match self.peek() {
                Some('&') | Some(';') => {
                    self.bump();
                    terms.push(self.parse_bond_not()?);
                }
                // implicit AND between adjacent bond primitives (e.g. `!@-`)
                Some(c) if is_bond_char(c) && c != ',' => {
                    terms.push(self.parse_bond_not()?);
                }
                _ => break,
            }
        }
        Ok(if terms.len() == 1 {
            terms.pop().unwrap()
        } else {
            BondQuery::And(terms)
        })
    }

    fn parse_bond_not(&mut self) -> Result<BondQuery, MolRsError> {
        if self.peek() == Some('!') {
            self.bump();
            Ok(BondQuery::Not(Box::new(self.parse_bond_not()?)))
        } else {
            self.parse_bond_prim()
        }
    }

    fn parse_bond_prim(&mut self) -> Result<BondQuery, MolRsError> {
        let c = self
            .bump()
            .ok_or_else(|| self.err("expected bond symbol"))?;
        let prim = match c {
            '-' => BondPrimitive::Single,
            '=' => BondPrimitive::Double,
            '#' => BondPrimitive::Triple,
            ':' => BondPrimitive::Aromatic,
            '~' => BondPrimitive::Any,
            '@' => BondPrimitive::InRing,
            other => return Err(self.err(format!("unexpected bond symbol '{other}'"))),
        };
        Ok(BondQuery::Prim(prim))
    }

    // -- atoms ---------------------------------------------------------------

    fn parse_atom(&mut self) -> Result<QueryAtom, MolRsError> {
        match self.peek() {
            Some('[') => self.parse_bracket_atom(),
            Some(_) => self.parse_organic_atom(),
            None => Err(self.err("expected an atom")),
        }
    }

    /// Organic-subset atom outside brackets: element or `*`/`a`/`A`.
    fn parse_organic_atom(&mut self) -> Result<QueryAtom, MolRsError> {
        let query = self.parse_organic_primitive()?;
        Ok(QueryAtom {
            query,
            map_label: None,
        })
    }

    fn parse_organic_primitive(&mut self) -> Result<AtomQuery, MolRsError> {
        if self.peek() == Some('*') {
            self.bump();
            return Ok(AtomQuery::Prim(AtomPrimitive::Any));
        }
        // Two-letter organic-subset element (Cl, Br) then single-letter.
        let (sym, aromatic) = self.read_element_symbol(false)?;
        primitive_for_element(&sym, aromatic)
            .ok_or_else(|| self.err(format!("unknown organic-subset element '{sym}'")))
    }

    fn parse_bracket_atom(&mut self) -> Result<QueryAtom, MolRsError> {
        debug_assert_eq!(self.peek(), Some('['));
        self.bump(); // consume '['
        // The first primitive inside the bracket is in "leading" position, where
        // a bare `H` denotes the hydrogen element rather than an H count.
        self.bracket_first_primitive = true;
        let mut map_label = None;
        let query = self.parse_atom_low(&mut map_label)?;
        if self.peek() != Some(']') {
            return Err(self.err("unbalanced '[' — missing ']'"));
        }
        self.bump(); // consume ']'
        Ok(QueryAtom { query, map_label })
    }

    /// `;`-separated low-precedence AND.
    fn parse_atom_low(&mut self, map: &mut Option<u32>) -> Result<AtomQuery, MolRsError> {
        let mut terms = vec![self.parse_atom_or(map)?];
        while self.peek() == Some(';') {
            self.bump();
            terms.push(self.parse_atom_or(map)?);
        }
        Ok(flatten_and(terms))
    }

    /// `,`-separated OR.
    fn parse_atom_or(&mut self, map: &mut Option<u32>) -> Result<AtomQuery, MolRsError> {
        let mut terms = vec![self.parse_atom_high(map)?];
        while self.peek() == Some(',') {
            self.bump();
            terms.push(self.parse_atom_high(map)?);
        }
        Ok(if terms.len() == 1 {
            terms.pop().unwrap()
        } else {
            AtomQuery::Or(terms)
        })
    }

    /// Implicit / `&` high-precedence AND of NOT-terms.
    fn parse_atom_high(&mut self, map: &mut Option<u32>) -> Result<AtomQuery, MolRsError> {
        let mut terms = vec![self.parse_atom_not(map)?];
        loop {
            match self.peek() {
                Some('&') => {
                    self.bump();
                    terms.push(self.parse_atom_not(map)?);
                }
                // implicit AND: another primitive starts (not a separator/close)
                Some(c) if !matches!(c, ';' | ',' | ']') => {
                    terms.push(self.parse_atom_not(map)?);
                }
                _ => break,
            }
        }
        Ok(flatten_and(terms))
    }

    fn parse_atom_not(&mut self, map: &mut Option<u32>) -> Result<AtomQuery, MolRsError> {
        if self.peek() == Some('!') {
            self.bump();
            Ok(AtomQuery::Not(Box::new(self.parse_atom_not(map)?)))
        } else {
            self.parse_atom_primitive(map)
        }
    }

    /// A single atom primitive inside brackets.
    fn parse_atom_primitive(&mut self, map: &mut Option<u32>) -> Result<AtomQuery, MolRsError> {
        // Consume the "leading primitive" flag: only the first primitive of a
        // bracket atom sees it set. A leading bare `H` is the hydrogen element.
        let leading = self.bracket_first_primitive;
        self.bracket_first_primitive = false;
        let c = self
            .peek()
            .ok_or_else(|| self.err("unexpected end of atom"))?;
        match c {
            '$' => self.parse_recursive(),
            '*' => {
                self.bump();
                Ok(AtomQuery::Prim(AtomPrimitive::Any))
            }
            'a' => {
                self.bump();
                Ok(AtomQuery::Prim(AtomPrimitive::AnyAromatic))
            }
            'A' => {
                self.bump();
                Ok(AtomQuery::Prim(AtomPrimitive::AnyAliphatic))
            }
            '#' => {
                self.bump();
                let n = self
                    .read_u32()
                    .ok_or_else(|| self.err("'#' needs a number"))?;
                Ok(AtomQuery::Prim(AtomPrimitive::AtomicNum(n as u8)))
            }
            'H' => {
                self.bump();
                // RDKit / molpy disambiguation: a *leading* `H` not followed by
                // a digit is the hydrogen element (`[H]`, `[H][C...]`); an `H`
                // after another primitive, or `H<n>`, is the total-H count
                // (`[CH3]`, `[C;H1]`, `[H1]`).
                let next_is_digit = self.peek().is_some_and(|c| c.is_ascii_digit());
                if leading && !next_is_digit {
                    // Hydrogen as an element (atomic number 1).
                    Ok(AtomQuery::Prim(AtomPrimitive::AliphaticElement(1)))
                } else {
                    let n = self.read_u32().unwrap_or(1);
                    Ok(AtomQuery::Prim(AtomPrimitive::TotalH(n)))
                }
            }
            'X' => {
                self.bump();
                let n = self
                    .read_u32()
                    .ok_or_else(|| self.err("'X' needs a number"))?;
                Ok(AtomQuery::Prim(AtomPrimitive::TotalConnections(n)))
            }
            'D' => {
                self.bump();
                let n = self
                    .read_u32()
                    .ok_or_else(|| self.err("'D' needs a number"))?;
                Ok(AtomQuery::Prim(AtomPrimitive::Degree(n)))
            }
            'R' => {
                self.bump();
                let n = self.read_u32();
                Ok(AtomQuery::Prim(AtomPrimitive::RingMembership(n)))
            }
            'r' => {
                self.bump();
                // `r{lo-hi}` / `r{lo-}` / `r{-hi}` ring-size *range* form, vs.
                // the existing `r<n>` (exact smallest-ring) / bare `r` (in any
                // ring). RDKit compares the atom's *smallest* ring size against
                // the range; an acyclic atom has smallest-ring 0, so only the
                // open-low `r{-hi}` form (`0 <= hi`) matches acyclic atoms.
                if self.peek() == Some('{') {
                    let (lo, hi) = self.parse_ring_size_range()?;
                    Ok(AtomQuery::Prim(AtomPrimitive::RingSizeRange { lo, hi }))
                } else {
                    let n = self.read_u32();
                    Ok(AtomQuery::Prim(AtomPrimitive::RingSize(n)))
                }
            }
            'x' => {
                self.bump();
                let n = self
                    .read_u32()
                    .ok_or_else(|| self.err("'x' needs a number"))?;
                Ok(AtomQuery::Prim(AtomPrimitive::RingBondCount(n)))
            }
            '+' | '-' => self.parse_charge(),
            '%' => self.parse_context_label(),
            ':' => {
                // atom-map label
                self.bump();
                let n = self
                    .read_u32()
                    .ok_or_else(|| self.err("':' needs a map number"))?;
                *map = Some(n);
                // A map label carries no match constraint; represent as Any so
                // it composes correctly in an implicit-AND chain.
                Ok(AtomQuery::Prim(AtomPrimitive::Any))
            }
            c if c.is_ascii_alphabetic() => {
                let (sym, aromatic) = self.read_element_symbol(true)?;
                primitive_for_element(&sym, aromatic)
                    .ok_or_else(|| self.err(format!("unknown element '{sym}'")))
            }
            other => Err(self.err(format!("unexpected character '{other}' in atom"))),
        }
    }

    fn parse_recursive(&mut self) -> Result<AtomQuery, MolRsError> {
        debug_assert_eq!(self.peek(), Some('$'));
        self.bump();
        if self.peek() != Some('(') {
            return Err(self.err("'$' must be followed by '('"));
        }
        self.bump();
        // Extract the balanced-paren substring and parse it as a fresh graph.
        let start = self.pos;
        let mut depth = 1usize;
        while let Some(c) = self.peek() {
            match c {
                '(' => depth += 1,
                ')' => {
                    depth -= 1;
                    if depth == 0 {
                        break;
                    }
                }
                _ => {}
            }
            self.bump();
        }
        if depth != 0 {
            return Err(self.err("unbalanced '$(' — missing ')'"));
        }
        let inner: String = self.chars[start..self.pos].iter().collect();
        self.bump(); // consume ')'
        let sub = parse(&inner)?;
        if sub.atoms.is_empty() {
            return Err(self.err("empty recursive SMARTS '$()'"));
        }
        // Stash the sub-graph and refer to it by index; the stash is moved
        // onto the top-level QueryGraph.recursives when parsing finishes.
        Ok(AtomQuery::Recursive(self.stash_recursive(sub)))
    }

    fn stash_recursive(&mut self, sub: QueryGraph) -> usize {
        self.recursive_stash.push(sub);
        self.recursive_stash.len() - 1
    }

    // -- charge --------------------------------------------------------------

    fn parse_charge(&mut self) -> Result<AtomQuery, MolRsError> {
        let sign = self.bump().unwrap();
        let positive = sign == '+';
        // `++`/`--` form, or `+n`/`-n`, or bare `+`/`-`.
        let mut magnitude = 1i32;
        if let Some(n) = self.read_u32() {
            magnitude = n as i32;
        } else {
            // count repeated same-sign chars
            while self.peek() == Some(sign) {
                self.bump();
                magnitude += 1;
            }
        }
        let charge = if positive { magnitude } else { -magnitude };
        Ok(AtomQuery::Prim(AtomPrimitive::Charge(charge)))
    }

    // -- context label -------------------------------------------------------

    /// Parse a `%LABEL` context-label predicate (a molrs extension, only valid
    /// inside a bracket atom). `%` here is **not** a ring closure: standard
    /// SMARTS ring closures (`%nn`) appear at the branch level outside brackets
    /// and require two digits, whereas a context label is `%` followed by a
    /// non-empty identifier (letters / digits / `_`, e.g. `%opls_154`). A bare
    /// `%` or a label beginning with a digit is rejected as malformed.
    fn parse_context_label(&mut self) -> Result<AtomQuery, MolRsError> {
        debug_assert_eq!(self.peek(), Some('%'));
        self.bump(); // consume '%'
        // The first char must be a letter or '_' so a context label never
        // collides with a numeric ring closure (which is parsed at the branch
        // level, never reaching this bracket-atom path).
        match self.peek() {
            Some(c) if c.is_ascii_alphabetic() || c == '_' => {}
            _ => {
                return Err(
                    self.err("'%LABEL' context label needs an identifier (letter or '_' first)")
                );
            }
        }
        let mut name = String::new();
        while let Some(c) = self.peek() {
            if c.is_ascii_alphanumeric() || c == '_' {
                name.push(c);
                self.bump();
            } else {
                break;
            }
        }
        Ok(AtomQuery::Prim(AtomPrimitive::HasContextLabel(name)))
    }

    // -- lexing helpers ------------------------------------------------------

    /// Read an element symbol. When `in_bracket`, multi-letter symbols like
    /// `Cl`, `Br`, `Na` are allowed (capital + lowercase). Outside brackets
    /// only the organic subset two-letter forms (`Cl`, `Br`) are recognized.
    /// Returns `(symbol, aromatic_flag)`.
    fn read_element_symbol(&mut self, in_bracket: bool) -> Result<(String, bool), MolRsError> {
        let first = self.bump().ok_or_else(|| self.err("expected element"))?;
        let aromatic = first.is_ascii_lowercase();
        let mut sym = String::new();
        sym.push(first.to_ascii_uppercase());

        // Two-letter symbol: a following lowercase letter that forms a known
        // element. For aromatic (lowercase-first) atoms, the second letter is
        // not consumed (aromatic symbols here are single-letter: c,n,o,s,p).
        if !aromatic
            && let Some(next) = self.peek()
            && next.is_ascii_lowercase()
        {
            let mut two = sym.clone();
            two.push(next);
            let recognized = crate::system::element::Element::by_symbol(&two).is_some();
            let two_letter_organic = !in_bracket && matches!(two.as_str(), "Cl" | "Br");
            if (in_bracket && recognized) || two_letter_organic {
                self.bump();
                return Ok((two, false));
            }
        }
        Ok((sym, aromatic))
    }

    /// Parse a `{lo-hi}` ring-size range body (cursor is on the opening `{`).
    ///
    /// Accepts `{lo-hi}`, `{lo-}` (open high), and `{-hi}` (open low). Returns
    /// `(lo, hi)` where `lo` defaults to 0 (no lower bound) for the `{-hi}`
    /// form and `hi` is `None` for the `{lo-}` form. Mirrors RDKit's
    /// `RANGE` / `GREATEREQUAL` / `LESSEQUAL` range queries over the smallest
    /// ring size.
    fn parse_ring_size_range(&mut self) -> Result<(u32, Option<u32>), MolRsError> {
        debug_assert_eq!(self.peek(), Some('{'));
        self.bump(); // consume '{'
        let lo = self.read_u32();
        if self.peek() != Some('-') {
            return Err(self.err("ring-size range needs a '-' separator"));
        }
        self.bump(); // consume '-'
        let hi = self.read_u32();
        if self.peek() != Some('}') {
            return Err(self.err("unbalanced ring-size range — missing '}'"));
        }
        self.bump(); // consume '}'
        if lo.is_none() && hi.is_none() {
            return Err(self.err("empty ring-size range 'r{-}'"));
        }
        // Open-low `{-hi}` => lower bound 0; open-high `{lo-}` => hi = None.
        Ok((lo.unwrap_or(0), hi))
    }

    /// Read a (possibly multi-digit) unsigned integer, if present.
    fn read_u32(&mut self) -> Option<u32> {
        let mut s = String::new();
        while let Some(c) = self.peek() {
            if c.is_ascii_digit() {
                s.push(c);
                self.bump();
            } else {
                break;
            }
        }
        if s.is_empty() { None } else { s.parse().ok() }
    }
}

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------

fn flatten_and(mut terms: Vec<AtomQuery>) -> AtomQuery {
    if terms.len() == 1 {
        terms.pop().unwrap()
    } else {
        AtomQuery::And(terms)
    }
}

fn is_bond_char(c: char) -> bool {
    matches!(c, '-' | '=' | '#' | ':' | '~' | '@' | '!' | '/' | '\\')
}

/// Build an atom primitive for an element symbol with a known aromatic flag.
fn primitive_for_element(sym: &str, aromatic: bool) -> Option<AtomQuery> {
    let z = crate::system::element::Element::by_symbol(sym)?.z();
    let prim = if aromatic {
        AtomPrimitive::AromaticElement(z)
    } else {
        AtomPrimitive::AliphaticElement(z)
    };
    Some(AtomQuery::Prim(prim))
}
