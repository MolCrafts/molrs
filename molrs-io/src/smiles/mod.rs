//! SMILES and SMARTS — sibling chemical-notation systems.
//!
//! This module hosts two closely related but deliberately separate systems:
//!
//! * [`smiles`] — the SMILES serialization format: parse a string into an
//!   intermediate representation, validate it, and convert it into an
//!   atomistic molecular graph.
//! * [`smarts`] — the SMARTS query language: parse a pattern, validate it,
//!   and match it against a target molecule.
//!
//! Both systems share a common chemistry vocabulary (AST types, byte scanner,
//! ring-closure validation) that lives in [`chem`]. Keeping SMILES and SMARTS
//! as siblings rather than treating one as a superset avoids the long-standing
//! class of bugs where concrete-structure semantics silently leak into query
//! evaluation.
//!
//! # Pipeline (SMILES)
//!
//! ```text
//! SMILES string → parse_smiles() → SmilesIR → to_atomistic() → Atomistic
//! ```
//!
//! # Pipeline (SMARTS)
//!
//! ```text
//! SMARTS string → parse_smarts() → SmilesIR → SmartsPattern::compile() → matches()
//! ```
//!
//! # Example
//!
//! ```ignore
//! use molrs_io::smiles::{parse_smiles, to_atomistic};
//!
//! let ir = parse_smiles("CCO").unwrap();
//! let mol = to_atomistic(&ir).unwrap();
//! assert_eq!(mol.n_atoms(), 3);
//! ```

pub mod chem;
pub mod error;
pub mod smarts;
// The serialization-format module retains its `smiles` name to mirror the
// `smarts` sibling. The re-exports below flatten it so callers write
// `molrs_io::smiles::parse_smiles`, not the doubled path.
#[allow(clippy::module_inception)]
pub mod smiles;

// The parser is internally unified: a single `Parser` struct dispatches by
// `ParserMode`. Both sibling modules re-export their respective entry point.
mod parser;

// ---------------------------------------------------------------------------
// Public re-exports (stable surface — downstream callers depend on these).
// ---------------------------------------------------------------------------

pub use chem::ast::{
    AtomNode, AtomPrimitive, AtomQuery, AtomSpec, BondKind, BondQuery, BracketSymbol, Chain,
    ChainElement, Chirality, SmilesIR, Span,
};
pub use error::{SmilesError, SmilesErrorKind};
pub use smarts::{parse_smarts, validate_smarts};
pub use smiles::{parse_smiles, to_atomistic, validate_smiles};
