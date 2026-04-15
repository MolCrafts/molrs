//! The SMARTS system: query parsing, validation, and substructure matching.
//!
//! SMARTS is a *query language* over molecular structures — its AST extends
//! the shared [`chem::ast`](crate::chem::ast) vocabulary with query primitives
//! (`[C;X4]`, `[c;r6]`, `$(…)`, etc.) and logical operators. This module owns
//! everything specific to compiling a SMARTS string into a matchable pattern
//! and running it against a [`MolGraph`](molrs::molgraph) target.
//!
//! The SMILES *serialization format* lives in the [`smiles`](crate::smiles)
//! sibling module. Shared AST vocabulary lives in [`chem`](crate::chem).
//!
//! # Status
//!
//! The parser is functional (see [`parse_smarts`]). The subgraph matcher
//! ([`SmartsPattern::find_all`]) is under active development — the public API
//! is stable but the implementation is in progress. Callers currently receive
//! [`SmartsError::NotYetImplemented`] for matcher operations.

pub mod matcher;
pub mod pattern;
pub mod validate;

pub use crate::parser::parse_smarts;
pub use matcher::{Match, SubstructureMatcher};
pub use pattern::{SmartsError, SmartsPattern};
pub use validate::validate_smarts;
