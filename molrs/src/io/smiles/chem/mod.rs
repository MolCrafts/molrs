//! Shared syntax vocabulary of the three line-notation dialects: SMILES,
//! SMARTS, and the SMILES fragment body (see the crate-internal `Dialect`).
//!
//! The abstract syntax tree (AST), the byte-level scanner, and the validation
//! helpers that are language-agnostic live here. Language-specific parsing,
//! validation and graph conversion live in the sibling `smiles/` module;
//! SMARTS pattern *matching* is an independent engine
//! ([`crate::perceive::smarts`]) that does not use this vocabulary.
//!
//! A fourth notation borrows from here without being a `Dialect`: the
//! `CGsmiles` coarse-graph parser behind
//! [`parse_cgsmiles`](crate::io::smiles::parse_cgsmiles) runs on the same
//! scanner, mints the same `Span`s and reuses the bonding-descriptor check,
//! but it has its own grammar and its own AST, because its token vocabulary
//! collides with SMARTS (`[#NAME]` against the atomic-number primitive `[#6]`,
//! `$` against both a bond order and a descriptor) in ways a shared,
//! mode-switching parser would silently mis-read.
//!
//! Over time this module will grow to host shared element tables, bond-order
//! vocabulary, aromaticity rules, and hybridization rules that the dialects
//! and future consumers (embed torsion library, forcefield typifiers) depend on.

use crate::io::smiles::error::Notation;

pub mod ast;
pub(crate) mod scanner;
#[cfg(test)]
pub(crate) mod test_support;
pub(crate) mod validation;

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
/// selected by [`parse_fragment_smiles`](crate::io::smiles::parse_fragment_smiles)
/// and [`write_fragment_smiles`](crate::io::smiles::write_fragment_smiles)
/// only — the plain-SMILES entry points stay strict, so no `.smi` line can
/// silently carry descriptors.
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
