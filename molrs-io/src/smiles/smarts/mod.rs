//! The SMARTS system: query parsing, validation, and substructure matching.
//!
//! SMARTS is a *query language* over molecular structures — its AST extends
//! the shared [`chem::ast`](crate::smiles::chem::ast) vocabulary with query primitives
//! (`[C;X4]`, `[c;r6]`, `$(…)`, etc.) and logical operators. This module owns
//! everything specific to compiling a SMARTS string into a matchable pattern
//! and running it against a [`MolGraph`](molrs::molgraph) target.
//!
//! The SMILES *serialization format* lives in the [`smiles`](crate::smiles::smiles)
//! sibling module. Shared AST vocabulary lives in [`chem`](crate::smiles::chem).
//!
//! # Status
//!
//! Core matcher is implemented: atom / bond primitives, logical operators
//! (`!`, `&`, `,`, `;`), ring primitives (`R`, `r`), recursive SMARTS
//! (`[$(...)]`).
//!
//! # Warning: silent stereochemistry false negatives
//!
//! Stereo-dependent primitives — atom chirality (`@`, `@@`) and directional
//! bonds (`/`, `\`) — are parsed but **always evaluate to `false`** because
//! the atomistic layer does not yet surface stereo descriptors. A pattern
//! that relies on them will compile cleanly and then fail to match
//! structurally correct targets with no error. See
//! [`predicate`] for the exhaustive list of unsupported primitives, and
//! [`matcher::SubstructureMatcher`] for the public contract. Track the
//! stereo-aware pipeline before querying stereo-sensitive structures.
//!
//! Reference: Daylight SMARTS theory manual — see the "Atomic Primitives"
//! and "Bond Primitives" tables:
//! <https://daylight.com/dayhtml/doc/theory/theory.smarts.html>.

pub mod matcher;
pub mod pattern;
pub(crate) mod predicate;
pub mod validate;

pub use crate::smiles::parser::parse_smarts;
pub use matcher::{Match, SubstructureMatcher};
pub use pattern::{SmartsError, SmartsPattern};
pub use validate::validate_smarts;
