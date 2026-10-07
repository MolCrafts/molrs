//! The line notations' shared grammar: SMILES, SMARTS, and the SMILES
//! fragment body (see [`Dialect`]) — one abstract syntax tree, one byte
//! scanner, one recursive-descent parser, one writer, one error type.
//!
//! Crate-private. The SMILES format's public face is [`crate::io::smiles`]
//! (the IR, its error, the text doors); SMARTS is [`crate::perceive::smarts`]
//! (parse, compile, match, generate); both build on this module, so neither
//! depends on the other.
//!
//! A fourth notation borrows from here without being a `Dialect`: the
//! `CGsmiles` coarse-graph parser ([`crate::io::cgsmiles`]) runs on the same
//! scanner, mints the same `Span`s and reuses the bonding-descriptor check,
//! but it has its own grammar and its own AST, because its token vocabulary
//! collides with SMARTS (`[#NAME]` against the atomic-number primitive `[#6]`,
//! `$` against both a bond order and a descriptor) in ways a shared,
//! mode-switching parser would silently mis-read.

use crate::core::Element;
use error::Notation;

pub(crate) mod ast;
pub(crate) mod error;
#[cfg(test)]
pub(crate) mod fixtures;
pub(crate) mod parser;
pub(crate) mod scanner;
pub(crate) mod validation;
pub(crate) mod writer;

/// The element symbol a SMILES atom symbol denotes.
///
/// SMILES writes aromatic atoms in lowercase (`c`, `n`, `se`); that is
/// notation, not an element symbol. Every consumer that keys off `element` —
/// mass tables, typifiers, force-field parameter lookup — expects the
/// canonical capitalisation, so both validation and graph construction
/// normalise through here.
pub(crate) fn canonical_element_symbol(symbol: &str) -> String {
    let mut chars = symbol.chars();
    match chars.next() {
        None => String::new(),
        Some(first) => first.to_ascii_uppercase().to_string() + chars.as_str(),
    }
}

/// Whether `symbol`, as written in a SMILES atom, names a real element.
///
/// The lookup is on the canonical capitalisation, so the aromatic lowercase
/// `se` is the element `Se` and `[Xx]` is nothing at all. Both stages that
/// decide this question ask here — the parser when it reads a bracket atom,
/// and the SMILES element check when it re-checks an IR it did not build — so
/// the two cannot drift into two answers about the same symbol.
pub(crate) fn is_element_symbol(symbol: &str) -> bool {
    Element::by_symbol(&canonical_element_symbol(symbol)).is_some()
}

/// Which notation the parser or writer speaks.
///
/// The three dialects share one AST, one scanner and one recursive-descent
/// parser; this enum is what that shared machinery branches on, so "which
/// language is this" has a single home rather than one private copy per
/// direction.
///
/// `FragmentSmiles` is a SMILES fragment body extended with `CGsmiles` /
/// `BigSMILES` bonding descriptors ([`BondingDescriptor`](ast::BondingDescriptor)),
/// the bracketed joining-site markers `[$]`, `[<]`, `[>]` and `[!]`. It is
/// selected by [`parser::parse_fragment_smiles`] and
/// [`writer::write_fragment_smiles`] only — the plain-SMILES entry points stay
/// strict, so no `.smi` line can silently carry descriptors.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Dialect {
    /// Standard SMILES — only concrete atoms and bonds.
    Smiles,
    /// SMARTS — adds query primitives, logical operators, wildcard/ring bonds.
    Smarts,
    /// SMILES fragment body with `CGsmiles` / `BigSMILES` bonding descriptors.
    FragmentSmiles,
}

impl Dialect {
    /// The notation a diagnostic about this dialect names.
    ///
    /// Three dialects, two notations: the fragment body is SMILES widened with
    /// bonding descriptors, not a language of its own, so it reports as
    /// SMILES. The mapping is total and lives here alone, so the parser and
    /// the writer cannot drift into two answers — and neither holds a second
    /// notation field beside its `dialect`.
    pub(crate) fn notation(self) -> Notation {
        match self {
            Dialect::Smiles | Dialect::FragmentSmiles => Notation::Smiles,
            Dialect::Smarts => Notation::Smarts,
        }
    }
}
