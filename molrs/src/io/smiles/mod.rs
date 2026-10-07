//! SMILES serialization, the `CGsmiles` coarse-graph notation, and the syntax
//! vocabulary they share with SMARTS.
//!
//! SMILES (Simplified Molecular Input Line Entry System) writes a molecule as
//! a single line of text: `CCO` is ethanol, `c1ccccc1` benzene. SMARTS (SMILES
//! Arbitrary Target Specification) is its query language — the same grammar
//! widened with wildcards and logical operators, so that one string describes
//! a *class* of substructures instead of one molecule.
//!
//! This module hosts the SMILES serialization pipeline and the syntax
//! vocabulary those two atomistic notations share: the abstract-syntax-tree
//! (AST) types ([`SmilesIR`], [`AtomNode`], …), the byte scanner, and grammar
//! validation.
//!
//! The serialization format itself: parse a string into an intermediate
//! representation (IR, [`parse_smiles`]), validate it ([`validate_smiles`]),
//! and convert it into an atomistic molecular graph ([`to_atomistic`]) and
//! back ([`from_atomistic`], [`write_smiles`]). [`read_smiles`] is the
//! one-molecule door: parse, refuse a `.`-separated set, convert.
//!
//! [`parse_smarts`] parses SMARTS *syntax* into the shared [`SmilesIR`]. It is
//! the one SMARTS parser: the matching engine in [`crate::perceive::smarts`]
//! compiles its queries from this IR.
//!
//! [`parse_cgsmiles`] reads the third notation this module hosts. `CGsmiles`
//! writes a molecule at a *coarse-grained* resolution: one node per whole
//! group of atoms — a **bead**, named after the fragment it stands for —
//! instead of one node per atom. It parses such a string — one `{…}` block per
//! resolution, `{[#PEO][#PEO]}` being a two-bead chain and every block after
//! the first a table of the fragment bodies named one level up — into the
//! [`CGSmilesIR`] of the private `cgsmiles` submodule. Expanding that IR into
//! real atoms is a separate step on the value itself,
//! [`CGSmilesIR::to_atomistic`]: it replaces every bead of the lowest
//! resolution with a copy of its fragment body and turns each descriptor pair
//! the reader resolved into one bond. Reading the coarsest resolution alone as
//! a bead graph is another step, [`CGSmilesIR::to_coarsegrain`]: one
//! `CoarseGrain` bead per node of `levels[0]`, one CG bond per edge, and no
//! fragment table needed. What `CGsmiles` shares with the
//! two atomistic notations is the scanner, the [`Span`], the [`SmilesError`]
//! and the [`BondingDescriptor`] vocabulary — not the grammar, not the AST. The
//! token vocabularies overlap adversarially (`[#NAME]` against the SMARTS
//! `[#6]`, `$` against both a bond order and a descriptor), so these are two
//! parsers, not one parser with a mode: a missed check in a mode-switching
//! parser would read one notation as the other instead of failing.
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
//! fragment body → parse_fragment_smiles() → SmilesIR
//!     → fragment_to_atomistic() → (Atomistic, descriptor map)
//! fragment body → parse_fragment_smiles() → SmilesIR
//!     → SmilesIR::to_template() → Atomistic (ported)
//! ```
//!
//! # Pipeline (CGsmiles)
//!
//! ```text
//! CGsmiles string → parse_cgsmiles() → CGSmilesIR
//!     → CGSmilesIR::to_atomistic() → Atomistic
//! CGsmiles string → parse_cgsmiles() → CGSmilesIR
//!     → CGSmilesIR::to_coarsegrain() → CoarseGrain
//! CGsmiles string → parse_cgsmiles() → CGSmilesIR
//!     → CGSmilesIR::templates() → BTreeMap<String, Atomistic> (ported)
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

mod chem;
mod error;
pub mod frame_reader;
// The serialization-format module retains its `smiles` name internally. The
// re-exports below flatten it so callers write `molrs::io::smiles::parse_smiles`,
// not the doubled path.
#[allow(clippy::module_inception)]
mod smiles;

// The parser is internally unified: one `Parser` struct dispatches on
// `chem::Dialect`, so the three dialects share one grammar implementation.
// Its three entry points are re-exported straight from here (the matching
// engine in `crate::perceive::smarts` compiles from `parse_smarts`'s output).
mod parser;

// The `CGsmiles` coarse-graph notation: private like `parser`, reaching
// callers through the re-exports below.
mod cgsmiles;

// ---------------------------------------------------------------------------
// Public re-exports (stable surface — downstream callers depend on these).
//
// Four groups, in the order they appear below: the `CGsmiles` coarse-graph IR
// — the levels (`CGGraph` / `CGNode` / `CGEdge` / `CGBondOrder`, an edge
// carrying `EdgeOrigin` to say whether the notation wrote it or resolution
// derived it) together with the fragment tables that resolve them
// (`CGFragmentDef`, and `FragmentBody` for the two shapes a body may take) and
// the descriptor pairing over them (`ResolvedPair` / `PairEnd`) — and the
// entry point that builds it;
// the AST vocabulary of the two atomistic notations (including
// `BondingDescriptor` / `DescriptorKind`, which a fragment caller reads off
// the descriptor map); the error type, its variants
// and the `Notation` that says which of the three languages raised one; and
// the per-stage entry points — one set for plain SMILES, one for SMARTS
// syntax, one for fragment bodies.
// ---------------------------------------------------------------------------

pub use cgsmiles::{
    CGBondOrder, CGEdge, CGFragmentDef, CGGraph, CGNode, CGSmilesIR, EdgeOrigin, FragmentBody,
    PairEnd, ResolvedPair, parse_cgsmiles,
};
pub use chem::ast::{
    AtomNode, AtomPrimitive, AtomQuery, AtomSpec, BondKind, BondQuery, BondingDescriptor,
    BracketSymbol, Chain, ChainElement, Chirality, DescriptorKind, SmilesIR, Span,
};
pub use error::{Notation, SmilesError, SmilesErrorKind};
pub use parser::{parse_fragment_smiles, parse_smarts, parse_smiles};
pub use smiles::{
    AromaticEmit, HydrogenEmit, LocalSmartsOptions, MultiComponentEmit, NeighborStyle,
    SmilesEmitOptions, fragment_to_atomistic, from_atomistic, local_smarts_ir, read_smiles,
    to_atomistic, validate_smiles, write_atomistic_smiles, write_fragment_smiles,
    write_local_smarts, write_smarts, write_smiles,
};
