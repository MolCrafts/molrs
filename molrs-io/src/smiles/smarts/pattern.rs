//! Compiled SMARTS pattern — the unit of work handed to the matcher.
//!
//! A [`SmartsPattern`] is produced by compiling a SMARTS string once and then
//! re-used across many target molecules. Compilation parses the SMARTS,
//! validates ring closures, and lowers the IR into a
//! `petgraph::UnGraph<AtomQuery, BondQuery>` that
//! [`subgraph_isomorphisms_iter`] operates on at match time.
//!
//! Multi-component SMARTS (e.g. `C.O`) are supported: each connected
//! component becomes an independent pattern and the matcher requires every
//! component to match.
//!
//! # Example
//!
//! ```
//! use molrs_io::smiles::smarts::{SmartsPattern, SubstructureMatcher};
//! use molrs_io::smiles::{parse_smiles, to_atomistic};
//!
//! let pat = SmartsPattern::compile("[C;X4]").unwrap();
//! let mol = to_atomistic(&parse_smiles("CC").unwrap()).unwrap();
//! // `find_all` takes a `&MolGraph`; `Atomistic` exposes one via `as_molgraph`.
//! let matches = pat.find_all(mol.as_molgraph()).unwrap();
//! assert_eq!(matches.len(), 2); // both sp3 carbons
//! ```

use std::fmt;

use petgraph::graph::UnGraph;

use crate::smiles::chem::ast::{
    AtomNode, AtomPrimitive, AtomQuery, AtomSpec, BondKind, BondQuery, Chain, ChainElement, Span,
};
use crate::smiles::error::{SmilesError, SmilesErrorKind};
use crate::smiles::parser::parse_smarts;

use super::validate::validate_smarts;

/// A compiled SMARTS pattern ready to be matched against target molecules.
///
/// Construct via [`SmartsPattern::compile`]. Compiled patterns are immutable
/// and implement `Send + Sync`, so they can be shared across rayon-parallel
/// pattern-matching passes.
#[derive(Debug, Clone)]
pub struct SmartsPattern {
    /// One connected-component subgraph per `.`-separated component. Every
    /// component must match the target for the overall pattern to match.
    pub(crate) components: Vec<ComponentGraph>,
}

/// A compiled connected component — one `petgraph` query graph plus its
/// declaration-order atom list (so matcher results can be reported in the
/// same order the user typed them).
#[derive(Debug, Clone)]
pub(crate) struct ComponentGraph {
    pub(crate) graph: UnGraph<AtomQuery, BondQuery>,
}

/// Errors produced while compiling or matching a SMARTS pattern.
#[derive(Debug, Clone, PartialEq)]
pub enum SmartsError {
    /// The SMARTS string failed to parse or validate.
    Parse(SmilesError),
    /// A requested feature is not yet implemented in this build.
    ///
    /// Emitted for stereochemistry-dependent queries that have no
    /// representation in the current atomistic layer.
    NotYetImplemented,
}

impl fmt::Display for SmartsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SmartsError::Parse(e) => write!(f, "SMARTS parse error: {e}"),
            SmartsError::NotYetImplemented => {
                write!(f, "SMARTS feature not yet implemented")
            }
        }
    }
}

impl std::error::Error for SmartsError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            SmartsError::Parse(e) => Some(e),
            SmartsError::NotYetImplemented => None,
        }
    }
}

impl From<SmilesError> for SmartsError {
    fn from(e: SmilesError) -> Self {
        SmartsError::Parse(e)
    }
}

impl SmartsPattern {
    /// Construct a pattern from already-compiled components. Used by the
    /// matcher to rebuild recursive-SMARTS inner patterns without going
    /// back through string parsing.
    pub(crate) fn from_components(components: Vec<ComponentGraph>) -> Self {
        Self { components }
    }

    /// Parse and compile a SMARTS pattern string.
    ///
    /// Compilation runs [`parse_smarts`] → [`validate_smarts`] →
    /// per-component lowering into a `petgraph::UnGraph`. The resulting
    /// pattern is thread-safe (`Send + Sync`) and intended to be compiled
    /// once and reused across many targets.
    ///
    /// Reference: Daylight SMARTS theory manual — pattern grammar:
    /// <https://daylight.com/dayhtml/doc/theory/theory.smarts.html>
    ///
    /// # Errors
    ///
    /// Returns [`SmartsError::Parse`] if the input is empty, syntactically
    /// invalid, or fails ring-closure validation.
    ///
    /// # Examples
    ///
    /// ```
    /// use molrs_io::smiles::smarts::SmartsPattern;
    ///
    /// let pat = SmartsPattern::compile("[C;X4]").unwrap();
    /// assert!(SmartsPattern::compile("").is_err());
    /// # let _ = pat;
    /// ```
    pub fn compile(pattern: &str) -> Result<Self, SmartsError> {
        if pattern.is_empty() {
            return Err(SmartsError::Parse(SmilesError::new(
                SmilesErrorKind::EmptyInput,
                Span::new(0, 0),
                pattern,
            )));
        }
        let ir = parse_smarts(pattern)?;
        validate_smarts(&ir, pattern)?;
        let components = ir
            .components
            .iter()
            .map(compile_component)
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self { components })
    }
}

// ---------------------------------------------------------------------------
// Component compiler
// ---------------------------------------------------------------------------

fn compile_component(chain: &Chain) -> Result<ComponentGraph, SmartsError> {
    let mut builder = ComponentBuilder::default();
    builder.build_chain(chain, None)?;
    builder.close_rings()?;
    Ok(ComponentGraph {
        graph: builder.graph,
    })
}

/// Matcher-facing entry point: compile one chain-IR component without
/// re-parsing. Used by the recursive-SMARTS precompiler to materialize
/// inner `$(...)` patterns.
pub(crate) fn compile_component_for_ir(chain: &Chain) -> Result<ComponentGraph, SmartsError> {
    compile_component(chain)
}

#[derive(Default)]
struct ComponentBuilder {
    graph: UnGraph<AtomQuery, BondQuery>,
    /// Maps open ring number → (atom-node, optional bond declared at opener).
    open_rings: std::collections::HashMap<u16, (petgraph::graph::NodeIndex, Option<BondQuery>)>,
}

impl ComponentBuilder {
    fn build_chain(
        &mut self,
        chain: &Chain,
        prev: Option<(petgraph::graph::NodeIndex, Option<BondQuery>)>,
    ) -> Result<petgraph::graph::NodeIndex, SmartsError> {
        let head = self.add_atom_node(&chain.head);

        if let Some((prev_node, bond)) = prev {
            self.graph.add_edge(prev_node, head, bond_or_default(bond));
        }

        let mut current = head;

        for elem in &chain.tail {
            match elem {
                ChainElement::BondedAtom { bond, atom } => {
                    let next = self.add_atom_node(atom);
                    self.graph
                        .add_edge(current, next, bond_or_default(bond.clone()));
                    current = next;
                }
                ChainElement::Branch { bond, chain, .. } => {
                    self.build_chain(chain, Some((current, bond.clone())))?;
                }
                ChainElement::RingClosure { bond, rnum, .. } => {
                    if let Some((opener_node, opener_bond)) = self.open_rings.remove(rnum) {
                        let resolved = opener_bond.or_else(|| bond.clone());
                        self.graph
                            .add_edge(opener_node, current, bond_or_default(resolved));
                    } else {
                        self.open_rings.insert(*rnum, (current, bond.clone()));
                    }
                }
            }
        }

        Ok(head)
    }

    fn add_atom_node(&mut self, node: &AtomNode) -> petgraph::graph::NodeIndex {
        let query = atom_spec_to_query(&node.spec);
        self.graph.add_node(query)
    }

    fn close_rings(&mut self) -> Result<(), SmartsError> {
        if self.open_rings.is_empty() {
            Ok(())
        } else {
            // Ring-closure validation has already run over the IR; any
            // open ring here would indicate an internal bug in the compiler
            // since validate_smarts should have rejected it.
            Err(SmartsError::Parse(SmilesError::new(
                SmilesErrorKind::UnmatchedRingClosure(0),
                Span::new(0, 0),
                "",
            )))
        }
    }
}

/// Lower an [`AtomSpec`] into the matcher's canonical [`AtomQuery`] form.
///
/// Organic-subset atoms, bracket atoms, and explicit query expressions all
/// collapse into `AtomQuery` so the matcher only has to deal with one
/// vocabulary.
fn atom_spec_to_query(spec: &AtomSpec) -> AtomQuery {
    match spec {
        AtomSpec::Organic { symbol, aromatic } => AtomQuery::Primitive(AtomPrimitive::Element {
            symbol: symbol.clone(),
            aromatic: *aromatic,
        }),
        AtomSpec::Wildcard => AtomQuery::Primitive(AtomPrimitive::Wildcard),
        AtomSpec::Bracket {
            isotope,
            symbol,
            chirality,
            hcount,
            charge,
            atom_class,
        } => bracket_to_query(*isotope, symbol, *chirality, *hcount, *charge, *atom_class),
        AtomSpec::Query(q) => q.clone(),
    }
}

fn bracket_to_query(
    isotope: Option<u16>,
    symbol: &crate::smiles::chem::ast::BracketSymbol,
    chirality: Option<crate::smiles::chem::ast::Chirality>,
    hcount: Option<u8>,
    charge: Option<i8>,
    atom_class: Option<u16>,
) -> AtomQuery {
    use crate::smiles::chem::ast::BracketSymbol;

    let mut parts: Vec<AtomQuery> = Vec::new();

    match symbol {
        BracketSymbol::Element { symbol, aromatic } => {
            parts.push(AtomQuery::Primitive(AtomPrimitive::Element {
                symbol: symbol.clone(),
                aromatic: *aromatic,
            }));
        }
        BracketSymbol::Any => parts.push(AtomQuery::Primitive(AtomPrimitive::Wildcard)),
        BracketSymbol::Aliphatic => parts.push(AtomQuery::Primitive(AtomPrimitive::Aliphatic)),
        BracketSymbol::Aromatic => parts.push(AtomQuery::Primitive(AtomPrimitive::Aromatic)),
    }

    if let Some(m) = isotope {
        parts.push(AtomQuery::Primitive(AtomPrimitive::Isotope(m)));
    }
    if let Some(h) = hcount {
        parts.push(AtomQuery::Primitive(AtomPrimitive::HCount(h)));
    }
    if let Some(c) = charge {
        parts.push(AtomQuery::Primitive(AtomPrimitive::Charge(c)));
    }
    if let Some(c) = atom_class {
        parts.push(AtomQuery::Primitive(AtomPrimitive::AtomClass(c)));
    }
    if let Some(ch) = chirality {
        parts.push(AtomQuery::Primitive(AtomPrimitive::Chirality(ch)));
    }

    match parts.len() {
        0 => AtomQuery::Primitive(AtomPrimitive::Wildcard),
        1 => parts.into_iter().next().unwrap(),
        _ => AtomQuery::And(parts),
    }
}

/// Collapse an optional bond-query from the IR into a concrete query. When
/// the input wrote no bond symbol, Daylight treats the gap as
/// "single-or-aromatic".
fn bond_or_default(bond: Option<BondQuery>) -> BondQuery {
    bond.unwrap_or_else(|| {
        BondQuery::Or(vec![
            BondQuery::Kind(BondKind::Single),
            BondQuery::Kind(BondKind::Aromatic),
        ])
    })
}

// Compile-time assertion that SmartsPattern is thread-safe.
const _: fn() = || {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<SmartsPattern>();
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compile_single_carbon_produces_one_node() {
        let pat = SmartsPattern::compile("C").unwrap();
        assert_eq!(pat.components.len(), 1);
        assert_eq!(pat.components[0].graph.node_count(), 1);
        assert_eq!(pat.components[0].graph.edge_count(), 0);
    }

    #[test]
    fn compile_cc_produces_two_nodes_one_edge() {
        let pat = SmartsPattern::compile("CC").unwrap();
        assert_eq!(pat.components[0].graph.node_count(), 2);
        assert_eq!(pat.components[0].graph.edge_count(), 1);
    }

    #[test]
    fn compile_ring_closes_with_extra_edge() {
        let pat = SmartsPattern::compile("C1CCCCC1").unwrap();
        assert_eq!(pat.components[0].graph.node_count(), 6);
        assert_eq!(pat.components[0].graph.edge_count(), 6);
    }

    #[test]
    fn compile_multicomponent_produces_multiple_graphs() {
        let pat = SmartsPattern::compile("C.O").unwrap();
        assert_eq!(pat.components.len(), 2);
    }

    #[test]
    fn empty_string_returns_parse_error() {
        assert!(matches!(
            SmartsPattern::compile(""),
            Err(SmartsError::Parse(_))
        ));
    }
}
