//! SMILES parsing plus the shared SMARTS parser frontend.
//!
//! This module hosts the SMILES serialization pipeline and exposes
//! [`parse_smarts`] for the canonical SMARTS engine in
//! [`crate::perceive::smarts`]. SMARTS matching lives there; this module owns only
//! syntax parsing shared with SMILES.
//!
//! * [`smiles`] — the SMILES serialization format: parse a string into an
//!   intermediate representation, validate it, and convert it into an
//!   atomistic molecular graph.
//!
//! Both SMILES and SMARTS share a common chemistry vocabulary (AST types, byte
//! scanner, ring-closure validation) that lives in [`chem`].
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
//! SMARTS string → parse_smarts() → SmilesIR → crate::perceive::smarts::SmartsPattern
//! ```
//!
//! # Example
//!
//! ```ignore
//! use molrs::io::smiles::{parse_smiles, to_atomistic};
//!
//! let ir = parse_smiles("CCO").unwrap();
//! let mol = to_atomistic(&ir).unwrap();
//! assert_eq!(mol.n_atoms(), 3);
//! ```

pub mod chem;
pub mod error;
// The serialization-format module retains its `smiles` name internally. The
// re-exports below flatten it so callers write `molrs::io::smiles::parse_smiles`,
// not the doubled path.
pub mod frame_reader;
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
pub use parser::parse_smarts;
pub use smiles::{
    AromaticEmit, HydrogenEmit, LocalSmartsOptions, MultiComponentEmit, NeighborStyle,
    SmilesEmitOptions, from_atomistic, local_smarts_ir, parse_smiles, to_atomistic,
    validate_smiles, write_atomistic_smiles, write_local_smarts, write_smarts, write_smiles,
};
