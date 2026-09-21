//! SMILES serialization plus the SMILES/SMARTS syntax vocabulary they share.
//!
//! SMILES (Simplified Molecular Input Line Entry System) writes a molecule as
//! a single line of text: `CCO` is ethanol, `c1ccccc1` benzene. SMARTS (SMILES
//! Arbitrary Target Specification) is its query language — the same grammar
//! widened with wildcards and logical operators, so that one string describes
//! a *class* of substructures instead of one molecule.
//!
//! This module hosts the SMILES serialization pipeline and, in [`chem`], the
//! syntax vocabulary both line notations share: the abstract-syntax-tree (AST)
//! types, the byte scanner, and grammar validation.
//!
//! The [`smiles`] submodule owns the serialization format itself: parse a
//! string into an intermediate representation (IR), validate it, and convert
//! it into an atomistic molecular graph.
//!
//! [`parse_smarts`] parses SMARTS *syntax* into the shared [`SmilesIR`], for
//! callers that want a SMARTS pattern as an IR. It is **not** the frontend of
//! the matching engine in [`crate::perceive::smarts`]: that engine has its own
//! parser and never consumes this one. Neither module depends on the other.
//!
//! The fragment entry points [`parse_fragment_smiles`],
//! [`fragment_to_atomistic`] and [`write_fragment_smiles`] accept and emit a
//! SMILES **fragment body**: one SMILES string standing for a piece of a
//! larger molecule, whose joining sites are marked by `CGsmiles` /
//! `BigSMILES` **bonding descriptors** — the bracketed operators `[$]`, `[<]`,
//! `[>]` and `[!]`, each naming a place where this fragment may later be
//! bonded to another one (see [`BondingDescriptor`]). The plain entry points
//! refuse a descriptor with an error naming the fragment entry point, so no
//! `.smi` line can silently lose one.
//!
//! # Pipeline (SMILES)
//!
//! ```text
//! SMILES string → parse_smiles() → SmilesIR → to_atomistic() → Atomistic
//! ```
//!
//! # Pipeline (SMARTS syntax)
//!
//! ```text
//! SMARTS string → parse_smarts() → SmilesIR
//! ```
//!
//! # Pipeline (SMILES fragment)
//!
//! ```text
//! fragment body → parse_fragment_smiles() → SmilesIR → fragment_to_atomistic() → (Atomistic, descriptor map)
//! ```
//!
//! # Examples
//!
//! ```
//! use molrs::io::smiles::{parse_smiles, to_atomistic};
//!
//! let ir = parse_smiles("CCO").unwrap();
//! let mol = to_atomistic(&ir).unwrap();
//! assert_eq!(mol.n_atoms(), 3);
//! ```
//!
//! ```
//! use molrs::io::smiles::{parse_fragment_smiles, fragment_to_atomistic, DescriptorKind};
//!
//! let ir = parse_fragment_smiles("[$]COC[$]").unwrap();
//! let (mol, ports) = fragment_to_atomistic(&ir).unwrap();
//! assert_eq!(mol.n_atoms(), 3);
//! assert_eq!(ports.len(), 2);
//! assert!(ports.iter().all(|(_, d)| d.kind == DescriptorKind::Symmetric));
//! ```

pub mod chem;
pub mod error;
pub mod frame_reader;
// The serialization-format module retains its `smiles` name internally. The
// re-exports below flatten it so callers write `molrs::io::smiles::parse_smiles`,
// not the doubled path.
#[allow(clippy::module_inception)]
pub mod smiles;

// The parser is internally unified: one `Parser` struct dispatches on
// `chem::Dialect`, so the three dialects share one grammar implementation.
// `parse_smiles` and `parse_fragment_smiles` reach callers through the
// `smiles` module, which re-exports them; `parse_smarts` has no such module of
// its own (the matching engine in `crate::perceive::smarts` is independent of
// this parser) and is re-exported straight from here.
mod parser;

// ---------------------------------------------------------------------------
// Public re-exports (stable surface — downstream callers depend on these).
//
// Three groups: the AST vocabulary (including `BondingDescriptor` /
// `DescriptorKind`, which a fragment caller reads off the descriptor map), the
// error type and its variants, and the per-stage entry points — one set for
// plain SMILES, one for SMARTS syntax, one for fragment bodies.
// ---------------------------------------------------------------------------

pub use chem::ast::{
    AtomNode, AtomPrimitive, AtomQuery, AtomSpec, BondKind, BondQuery, BondingDescriptor,
    BracketSymbol, Chain, ChainElement, Chirality, DescriptorKind, SmilesIR, Span,
};
pub use error::{SmilesError, SmilesErrorKind};
pub use parser::parse_smarts;
pub use smiles::{
    AromaticEmit, HydrogenEmit, LocalSmartsOptions, MultiComponentEmit, NeighborStyle,
    SmilesEmitOptions, fragment_to_atomistic, from_atomistic, local_smarts_ir,
    parse_fragment_smiles, parse_smiles, to_atomistic, validate_smiles, write_atomistic_smiles,
    write_fragment_smiles, write_local_smarts, write_smarts, write_smiles,
};
