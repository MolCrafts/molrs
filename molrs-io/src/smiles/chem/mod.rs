//! Shared chemistry presets used by both the SMILES and SMARTS systems.
//!
//! The AST, byte-level scanner, and validation helpers that are language-
//! agnostic live here. Language-specific parsing, validation, graph
//! conversion, and pattern matching live in the sibling `smiles/` and
//! `smarts/` modules.
//!
//! Over time this module will grow to host shared element tables, bond-order
//! vocabulary, aromaticity rules, and hybridization rules that both languages
//! and future consumers (embed torsion library, forcefield typifiers) depend on.

pub mod ast;
pub(crate) mod scanner;
pub(crate) mod validation;
