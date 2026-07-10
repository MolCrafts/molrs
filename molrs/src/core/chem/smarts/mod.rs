//! SMARTS substructure-matching engine.
//!
//! A parser + backtracking subgraph-isomorphism matcher covering the SMARTS
//! feature subset used by RDKit's ETKDGv3 experimental-torsion preference
//! tables (`torsionPreferences_v2 / _smallrings / _macrocycles`), including
//! recursive SMARTS `[$(...)]`.
//!
//! Match semantics follow RDKit `GetSubstructMatches(uniquify=False)`: every
//! distinct query-atom → mol-atom embedding is reported, ordered by query-atom
//! index. Ported (semantics only) from RDKit under the BSD-3 licence:
//! - `Code/GraphMol/SmilesParse/SmartsParse.cpp` (grammar)
//! - `Code/GraphMol/Substruct/SubstructMatch.cpp` (matching + recursive eval)
//! - `Code/GraphMol/QueryAtom.h` / `QueryBond.h` (query primitives)
//!
//! # Aromaticity convention
//!
//! `Atomistic` carries no aromatic model. Aromatic atoms are perceived as those
//! incident to a bond of order `1.5` (the project convention), unless an
//! explicit `is_aromatic` atom/bond prop is present, which takes precedence.
//! This lets callers transplant a reference perception (e.g. RDKit's) so that
//! aromatic queries (`a`, `c`, `:` bonds) agree exactly.
//!
//! # Supported features
//!
//! - Atom primitives: aliphatic/aromatic elements, `*`, `a`, `A`, `#<n>`,
//!   `H<n>`, `X<n>`, `D<n>`, `R`/`R<n>`, `r<n>`, `+`/`++`/`+<n>`/`-`/`-<n>`,
//!   atom-map `:<n>`.
//! - Atom logic: implicit/`&` high AND, `;` low AND, `,` OR, `!` NOT.
//! - Recursive SMARTS `[$(...)]` (nestable), rooted at the candidate atom.
//! - Bond primitives: `-` `=` `#` `:` `~` `@`, `!`, logical combos
//!   (`!@;-`, `-,:`); default bond = single-or-aromatic.
//! - Branches `( )`, ring closures incl. `%nn`.
//!
//! Out of scope: chirality `@`/`@@`, isotopes, reaction / component SMARTS.
//!
//! # Example
//!
//! ```
//! use molrs::chem::smarts::SmartsPattern;
//! use molrs::{Atom, Atomistic, PropValue};
//!
//! // Acetamide skeleton C-C(=O)-N (no Hs needed for this query).
//! let mut g = Atomistic::new();
//! let c0 = g.add_atom(Atom::xyz("C", 0.0, 0.0, 0.0));
//! let c1 = g.add_atom(Atom::xyz("C", 1.0, 0.0, 0.0));
//! let o = g.add_atom(Atom::xyz("O", 2.0, 0.0, 0.0));
//! let n = g.add_atom(Atom::xyz("N", 1.0, 1.0, 0.0));
//! g.add_bond(c0, c1).unwrap();
//! let bo = g.add_bond(c1, o).unwrap();
//! g.set_bond_prop(bo, "order", PropValue::F64(2.0)).unwrap();
//! g.add_bond(c1, n).unwrap();
//!
//! let pat = SmartsPattern::parse("[$([CX3]=[OX1]):1]~[*:2]").unwrap();
//! assert!(pat.has_match(&g, molrs::MatchOptions::default()));
//! assert_eq!(pat.map_label(0), Some(1));
//! ```

mod ast;
mod matcher;
mod parser;
mod reaction;

use std::collections::HashMap;

use crate::error::MolRsError;
use crate::system::atomistic::{AtomId, Atomistic};

use parser::QueryGraph;

pub use reaction::Reaction;

/// Matching controls for [`SmartsPattern::find`].
#[derive(Debug, Clone, Copy, Default)]
pub struct MatchOptions<'a> {
    /// Optional `%LABEL` context (`atom -> current label`).
    pub labels: Option<&'a HashMap<AtomId, String>>,
    /// Optional root pin for query atom 0.
    pub root: Option<AtomId>,
    /// Optional maximum number of matches to return.
    pub limit: Option<usize>,
}

/// One SMARTS match, indexed by query atom order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SmartsMatch {
    pub atoms: Vec<AtomId>,
}

impl SmartsMatch {
    pub fn atoms(&self) -> &[AtomId] {
        &self.atoms
    }
}

/// A compiled SMARTS query.
#[derive(Debug, Clone)]
pub struct SmartsPattern {
    graph: QueryGraph,
}

impl SmartsPattern {
    /// Parse a SMARTS string. Returns `Err` on any syntax error (never panics).
    pub fn parse(smarts: &str) -> Result<SmartsPattern, MolRsError> {
        let graph = parser::parse(smarts)?;
        Ok(SmartsPattern { graph })
    }

    /// All matches (non-uniquified), controlled by [`MatchOptions`].
    pub fn find(&self, mol: &Atomistic, options: MatchOptions<'_>) -> Vec<SmartsMatch> {
        matcher::find(&self.graph, mol, options)
    }

    /// Whether at least one match exists.
    pub fn has_match(&self, mol: &Atomistic, options: MatchOptions<'_>) -> bool {
        matcher::has_match(&self.graph, mol, options)
    }

    /// Project a match into `{atom_map_label -> molecule atom}`.
    pub fn mapped(&self, m: &SmartsMatch) -> HashMap<u32, AtomId> {
        m.atoms
            .iter()
            .enumerate()
            .filter_map(|(i, &atom)| self.map_label(i).map(|label| (label, atom)))
            .collect()
    }

    /// The SMARTS atom-map label (`:1` etc.) of query atom `query_atom`, or
    /// `None` if unlabelled / out of range.
    pub fn map_label(&self, query_atom: usize) -> Option<u32> {
        self.graph.atoms.get(query_atom).and_then(|a| a.map_label)
    }

    /// Number of query atoms.
    pub fn num_query_atoms(&self) -> usize {
        self.graph.atoms.len()
    }

    /// Every `%LABEL` context-label referenced by this pattern (including those
    /// inside recursive `$(...)` subpatterns), in traversal order with
    /// duplicates kept.
    ///
    /// Iterative typifiers use this to discover a def's dependencies on
    /// previously-assigned labels (e.g. OPLS `%opls_NNN` references).
    pub fn context_labels(&self) -> Vec<String> {
        self.graph.context_labels()
    }
}
