//! Intermediate representation (IR) types for SMILES and SMARTS notation.
//!
//! [`SmilesIR`] is a pure syntax tree that captures the notation faithfully
//! without committing to atomistic or coarse-grained semantics.
//! SMARTS is modelled as a superset: [`AtomSpec::Query`] and [`BondQuery`]
//! extend the SMILES-only variants without breaking existing consumers.
//!
//! This module lives under `chem/` because the AST is the shared vocabulary
//! for both the SMILES and SMARTS systems. Language-specific processing
//! (parsing entry points, validation, graph conversion, matching) lives in
//! the `smiles/` and `smarts/` sibling modules.

// ---------------------------------------------------------------------------
// Span
// ---------------------------------------------------------------------------

/// Byte-offset range within the input string.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Span {
    pub start: usize,
    pub end: usize,
}

impl Span {
    /// Create a new span.
    pub fn new(start: usize, end: usize) -> Self {
        Self { start, end }
    }
}

// ---------------------------------------------------------------------------
// Top-level
// ---------------------------------------------------------------------------

/// Intermediate representation produced by the SMILES / SMARTS parser.
///
/// This is a pure syntax tree — it captures the notation faithfully without
/// committing to atomistic or coarse-grained semantics. Convert to
/// [`Atomistic`](crate::atomistic::Atomistic) or
/// [`CoarseGrain`](crate::coarsegrain::CoarseGrain) for domain use.
///
/// Multiple disconnected components are separated by `.` in the input.
#[derive(Debug, Clone, PartialEq)]
pub struct SmilesIR {
    /// Connected components (separated by `.` in the input).
    pub components: Vec<Chain>,
    /// Span covering the entire input.
    pub span: Span,
}

/// A linear chain of atoms with branches and ring closures.
#[derive(Debug, Clone, PartialEq)]
pub struct Chain {
    /// The first atom in the chain.
    pub head: AtomNode,
    /// Subsequent elements: bonded atoms, branches, or ring closures.
    pub tail: Vec<ChainElement>,
}

/// An element following the head atom in a chain.
#[derive(Debug, Clone, PartialEq)]
pub enum ChainElement {
    /// An atom bonded to the previous atom.
    BondedAtom {
        bond: Option<BondKind>,
        atom: AtomNode,
    },
    /// A parenthesised branch: `(` bond? chain `)`.
    Branch {
        bond: Option<BondKind>,
        chain: Chain,
        span: Span,
    },
    /// A ring-closure digit or `%nn`.
    RingClosure {
        bond: Option<BondKind>,
        rnum: u16,
        span: Span,
    },
}

// ---------------------------------------------------------------------------
// Atoms
// ---------------------------------------------------------------------------

/// An atom node carrying its specification and source span.
#[derive(Debug, Clone, PartialEq)]
pub struct AtomNode {
    pub spec: AtomSpec,
    pub span: Span,
}

/// Atom specification — the extensibility point for SMARTS.
#[derive(Debug, Clone, PartialEq)]
pub enum AtomSpec {
    /// Organic-subset shorthand (no brackets): `C`, `N`, `c`, `n`, etc.
    Organic { symbol: String, aromatic: bool },
    /// Bracket atom: `[isotope? symbol chirality? hcount? charge? class?]`.
    Bracket {
        isotope: Option<u16>,
        symbol: BracketSymbol,
        chirality: Option<Chirality>,
        hcount: Option<u8>,
        charge: Option<i8>,
        atom_class: Option<u16>,
    },
    /// Wildcard `*`.
    Wildcard,
    /// SMARTS query expression (logical combination of primitives).
    Query(AtomQuery),
}

/// Symbol inside a bracket atom.
#[derive(Debug, Clone, PartialEq)]
pub enum BracketSymbol {
    /// A concrete element, possibly aromatic.
    Element { symbol: String, aromatic: bool },
    /// `*` inside brackets.
    Any,
    /// `A` — any aliphatic atom (SMARTS).
    Aliphatic,
    /// `a` — any aromatic atom (SMARTS).
    Aromatic,
}

/// Tetrahedral chirality marker.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Chirality {
    /// `@` — counter-clockwise (S).
    CounterClockwise,
    /// `@@` — clockwise (R).
    Clockwise,
}

// ---------------------------------------------------------------------------
// Bonds
// ---------------------------------------------------------------------------

/// Bond kind (covers both SMILES and SMARTS bond types).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BondKind {
    /// `-` explicit single bond.
    Single,
    /// `=` double bond.
    Double,
    /// `#` triple bond.
    Triple,
    /// `$` quadruple bond.
    Quadruple,
    /// `:` aromatic bond.
    Aromatic,
    /// `/` directional up (cis/trans).
    Up,
    /// `\` directional down (cis/trans).
    Down,
    /// `~` any bond (SMARTS wildcard).
    Any,
    /// `@` ring bond (SMARTS).
    Ring,
}

/// SMARTS bond query with logical operators.
#[derive(Debug, Clone, PartialEq)]
pub enum BondQuery {
    Kind(BondKind),
    Not(Box<BondQuery>),
    And(Vec<BondQuery>),
    Or(Vec<BondQuery>),
}

// ---------------------------------------------------------------------------
// SMARTS query algebra
// ---------------------------------------------------------------------------

/// SMARTS atom query primitive.
#[derive(Debug, Clone, PartialEq)]
pub enum AtomPrimitive {
    /// Concrete element (possibly aromatic).
    Element { symbol: String, aromatic: bool },
    /// `*` — any atom.
    Wildcard,
    /// `A` — any aliphatic atom.
    Aliphatic,
    /// `a` — any aromatic atom.
    Aromatic,
    /// `D<n>` — explicit degree (number of explicit bonds).
    Degree(u8),
    /// `X<n>` — total connections (explicit + implicit H).
    TotalConnections(u8),
    /// `H<n>` — total hydrogen count.
    HCount(u8),
    /// `h<n>` — implicit hydrogen count.
    ImplicitH(u8),
    /// `R<n>` — ring membership count (0 = any ring).
    RingMembership(Option<u8>),
    /// `r<n>` — smallest ring size.
    RingSize(u8),
    /// `v<n>` — total valence.
    Valence(u8),
    /// Formal charge.
    Charge(i8),
    /// Isotope (mass number).
    Isotope(u16),
    /// `:<n>` — atom class / map number.
    AtomClass(u16),
    /// Chirality specification.
    Chirality(Chirality),
    /// `$(...)` — recursive SMARTS (environment match).
    Recursive(Box<SmilesIR>),
}

/// SMARTS atom query expression with logical operators.
///
/// Operator precedence (highest to lowest):
/// 1. `!` — NOT (unary)
/// 2. `&` — AND (high precedence, also implicit between adjacent primitives)
/// 3. `,` — OR
/// 4. `;` — AND (low precedence)
#[derive(Debug, Clone, PartialEq)]
pub enum AtomQuery {
    /// A single primitive.
    Primitive(AtomPrimitive),
    /// `!expr` — logical NOT.
    Not(Box<AtomQuery>),
    /// `expr & expr` or implicit adjacency — high-precedence AND.
    And(Vec<AtomQuery>),
    /// `expr , expr` — OR.
    Or(Vec<AtomQuery>),
    /// `expr ; expr` — low-precedence AND.
    LowAnd(Vec<AtomQuery>),
}
