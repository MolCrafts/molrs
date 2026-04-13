//! SMILES and SMARTS string parsing.
//!
//! This module provides a hand-written recursive-descent parser that converts
//! SMILES and SMARTS notation into an intermediate representation ([`SmilesIR`]).
//! The IR is a pure syntax tree — it does not commit to atomistic or
//! coarse-grained semantics.
//!
//! # Pipeline
//!
//! ```text
//! SMILES string → parse_smiles() → SmilesIR → to_atomistic() → Atomistic
//! ```
//!
//! # Examples
//!
//! ```
//! use molrs_smiles::{parse_smiles, to_atomistic};
//!
//! let ir = parse_smiles("CCO").unwrap();
//! let mol = to_atomistic(&ir).unwrap();
//! assert_eq!(mol.n_atoms(), 3);
//! ```
//!
//! # Phases
//!
//! 1. **Parsing** — `parse_smiles` / `parse_smarts` produce an IR.
//! 2. **Validation** — `validate_smiles` / `validate_smarts` check semantic
//!    correctness (matched ring closures, valid element symbols).
//! 3. **Conversion** — `to_atomistic` converts the IR into an
//!    [`Atomistic`](crate::atomistic::Atomistic) molecular graph.

pub mod ast;
pub mod error;
pub mod to_atomistic;
pub mod validate;

mod parser;
mod scanner;

// Public API re-exports.
pub use ast::{
    AtomNode, AtomPrimitive, AtomQuery, AtomSpec, BondKind, BondQuery, BracketSymbol, Chain,
    ChainElement, Chirality, SmilesIR, Span,
};
pub use error::{SmilesError, SmilesErrorKind};
pub use parser::{parse_smarts, parse_smiles};
pub use to_atomistic::to_atomistic;
pub use validate::{validate_smarts, validate_smiles};
