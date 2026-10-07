//! SMILES, the Simplified Molecular Input Line Entry System: a molecule as one
//! line of text (`CCO` is ethanol, `c1ccccc1` benzene).
//!
//! The doors are functions of [`crate::io`]:
//! [`read_smiles_str`](crate::io::read_smiles_str) reads one molecule (connectivity
//! only, a `.`-separated set refused), and
//! [`write_smiles_str`](crate::io::write_smiles_str) writes one, as
//! [`SmilesEmitOptions`] say. This module holds the notation's classes:
//!
//! - [`SmilesIr`], the parsed text — the syntax tree ([`Chain`],
//!   [`AtomNode`], [`AtomSpec`], …): [`SmilesIr::parse`],
//!   [`SmilesIr::from_fragment`] (a fragment body with `CGsmiles` /
//!   `BigSMILES` bonding descriptors `[$]`, `[<]`, `[>]`, `[!]`),
//!   [`SmilesIr::validate`], [`SmilesIr::to_atomistic`],
//!   [`SmilesIr::to_atomistic_with_descriptors`], [`SmilesIr::to_template`]
//!   and [`SmilesIr::from_atomistic`];
//! - [`SmilesError`], every refusal of the SMILES family of notations
//!   (SMILES, SMARTS, CGsmiles: its [`Notation`] says which);
//! - [`SmilesReader`], a `.smi` file read one molecule per line.
//!
//! The grammar is the crate's one line-notation grammar, shared with SMARTS
//! ([`crate::perceive::smarts`]): the syntax tree carries SMARTS query nodes
//! too ([`AtomQuery`], [`BondQuery`]), which a SMILES conversion refuses.
//!
//! # Pipeline
//!
//! ```text
//! SMILES text → SmilesIr::parse() → SmilesIr → SmilesIr::to_atomistic() → Atomistic
//! fragment body → SmilesIr::from_fragment() → SmilesIr
//!     → SmilesIr::to_atomistic_with_descriptors() → (Atomistic, descriptor map)
//!     → SmilesIr::to_template() → Atomistic (ported)
//! ```
//!
//! # Examples
//!
//! ```
//! use molrs::io::smiles::SmilesIr;
//!
//! let ir = SmilesIr::parse("CCO").unwrap();
//! let mol = ir.to_atomistic().unwrap();
//! assert_eq!(mol.n_atoms(), 3);
//! ```
//!
//! ```
//! use molrs::io::smiles::{DescriptorKind, SmilesIr};
//!
//! let ir = SmilesIr::from_fragment("[$]COC[$]").unwrap();
//! let (mol, ports) = ir.to_atomistic_with_descriptors().unwrap();
//! assert_eq!(mol.n_atoms(), 3);
//! assert_eq!(ports.len(), 2);
//! assert!(ports.iter().all(|(_, d)| d.kind == DescriptorKind::Symmetric));
//! ```

pub(crate) mod element_check;
pub(crate) mod emit_options;
pub(crate) mod ir_from_atomistic;
pub(crate) mod ir_to_atomistic;
mod line_reader;

pub use crate::line_notation::ast::{
    AtomNode, AtomPrimitive, AtomQuery, AtomSpec, BondKind, BondQuery, BondingDescriptor,
    BracketSymbol, Chain, ChainElement, Chirality, DescriptorKind, SmilesIr, Span,
};
pub use crate::line_notation::error::{Notation, SmilesError, SmilesErrorKind};
pub use emit_options::{AromaticEmit, HydrogenEmit, MultiComponentEmit, SmilesEmitOptions};
pub use line_reader::SmilesReader;

impl SmilesIr {
    /// Parse SMILES text into its IR.
    ///
    /// # Errors
    ///
    /// A [`SmilesError`] for text that is not SMILES: a syntax error, an
    /// unknown element, a bonding descriptor (a fragment body goes through
    /// [`from_fragment`](Self::from_fragment)).
    pub fn parse(smiles: &str) -> Result<SmilesIr, SmilesError> {
        crate::line_notation::parser::parse_smiles(smiles)
    }

    /// Parse a SMILES fragment body — SMILES plus `CGsmiles` / `BigSMILES`
    /// bonding descriptors (`[<]OCC[>]`, `[$]COC[$]`) — into its IR.
    ///
    /// # Errors
    ///
    /// A [`SmilesError`] for text that is not fragment notation.
    pub fn from_fragment(body: &str) -> Result<SmilesIr, SmilesError> {
        crate::line_notation::parser::parse_fragment_smiles(body)
    }
}
