//! Substructure matching: run a compiled [`SmartsPattern`] against a target.
//!
//! The matcher finds every subgraph isomorphism from a compiled pattern onto
//! a target [`MolGraph`]. Each match is the ordered list of target
//! [`AtomId`] indices that correspond to the pattern atoms in declaration
//! order, exposed to the caller as a `Vec<usize>` of target atom indexes.
//!
//! # Algorithm
//!
//! This is a thin wrapper over
//! [`petgraph::algo::subgraph_isomorphisms_iter`], which implements VF2
//! with caller-supplied node and edge matching closures. We build a
//! disposable `UnGraph<AtomId, f64>` for the target on every call so we
//! can hand both graphs to VF2 and then translate node indices back to
//! [`AtomId`] via a side table. Building the target graph is linear in
//! `n_atoms + n_bonds`; if this ever becomes a bottleneck the whole
//! construction can be cached per-target without changing the public API.
//!
//! Recursive SMARTS (`[$(inner)]`) is evaluated out-of-band after VF2
//! reports a candidate mapping: for every pattern atom carrying a recursive
//! primitive, the matcher re-runs the inner pattern with the candidate
//! target atom as the anchor and requires it to match. This keeps the main
//! VF2 loop's predicate closures pure and avoids nested recursion through
//! petgraph.

use molrs::molgraph::{AtomId, MolGraph};
use petgraph::graph::UnGraph;

use crate::smiles::chem::ast::{AtomPrimitive, AtomQuery, BondQuery, SmilesIR};

use super::pattern::{ComponentGraph, SmartsError, SmartsPattern};
use super::predicate::{BondEdge, TargetCtx, eval_atom_query, eval_bond_query};

/// A single substructure match: an ordered list of target-atom indices, one
/// per query atom in the compiled pattern's first component (multi-component
/// patterns return one entry per atom in declaration order, concatenated by
/// component).
///
/// Indices are 0-based positions into the iteration order of
/// [`MolGraph::atoms`](molrs::molgraph::MolGraph::atoms).
///
/// Reference: Daylight SMARTS theory manual — substructure matching:
/// <https://daylight.com/dayhtml/doc/theory/theory.smarts.html>
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Match(
    /// Target-atom indices, in the pattern's declaration order.
    pub Vec<usize>,
);

/// Run substructure matching against a target molecule.
///
/// Kept as a trait (rather than an inherent method on [`SmartsPattern`]) so
/// that specialized matchers can share the same entry point while providing
/// different internal strategies. The `Send + Sync` bound is mandatory —
/// compiled patterns are shared across rayon-parallel passes.
///
/// # Stereochemistry
///
/// Chirality (`@`/`@@`) and directional bonds (`/`, `\`) are **not** yet
/// wired to [`MolGraph`]'s stereo layer; predicates for those primitives
/// return `false`. Patterns that depend on stereochemistry will therefore
/// produce silent false negatives — see [`predicate`](super::predicate)
/// for the exhaustive list of unsupported primitives.
pub trait SubstructureMatcher: Send + Sync {
    /// Return every subgraph isomorphism of `self` onto `target`.
    ///
    /// Multi-component patterns (`C.O`) return the disjoint cartesian
    /// product: one entry per component combination where no target atom
    /// is reused across components.
    ///
    /// # Errors
    ///
    /// Returns [`SmartsError`] only if the pattern could not be evaluated
    /// (e.g. a recursive `$(...)` inner pattern failed to recompile).
    /// Returns an empty vector — not an error — when no matches exist.
    ///
    /// # Examples
    ///
    /// ```
    /// use molrs_io::smiles::smarts::{SmartsPattern, SubstructureMatcher};
    /// use molrs_io::smiles::{parse_smiles, to_atomistic};
    ///
    /// let pat = SmartsPattern::compile("[C;X4]").unwrap();
    /// let mol = to_atomistic(&parse_smiles("CCO").unwrap()).unwrap();
    /// let hits = pat.find_all(mol.as_molgraph()).unwrap();
    /// assert_eq!(hits.len(), 2); // both sp3 carbons
    /// ```
    fn find_all(&self, target: &MolGraph) -> Result<Vec<Match>, SmartsError>;

    /// Return the first match, or `None` if the pattern does not occur.
    ///
    /// # Errors
    ///
    /// Same conditions as [`find_all`](Self::find_all).
    fn find_first(&self, target: &MolGraph) -> Result<Option<Match>, SmartsError> {
        Ok(self.find_all(target)?.into_iter().next())
    }
}

impl SubstructureMatcher for SmartsPattern {
    fn find_all(&self, target: &MolGraph) -> Result<Vec<Match>, SmartsError> {
        if self.components.is_empty() {
            return Ok(Vec::new());
        }

        // Ring detection is shared between `TargetCtx` (atom-level R / r
        // predicates) and `build_target_graph` (bond `@` ring flag).
        // Computing it once and threading it into both saves an O(V+E) pass
        // per `find_all`.
        let ctx = TargetCtx::new(target);
        let target_graph = build_target_graph(target, &ctx.rings);

        if self.components.len() == 1 {
            return match_component(&self.components[0], &target_graph, &ctx);
        }

        // Multi-component SMARTS: compute every component's match set and
        // form the cartesian product, skipping combinations that reuse
        // target atoms across components (Daylight requires disjoint maps).
        let per_component: Vec<Vec<Match>> = self
            .components
            .iter()
            .map(|c| match_component(c, &target_graph, &ctx))
            .collect::<Result<_, _>>()?;

        Ok(cartesian_disjoint_product(&per_component))
    }
}

// ---------------------------------------------------------------------------
// Per-component matching
// ---------------------------------------------------------------------------

fn match_component(
    component: &ComponentGraph,
    target_graph: &TargetGraph,
    ctx: &TargetCtx<'_>,
) -> Result<Vec<Match>, SmartsError> {
    // Precompile any recursive primitives referenced by the component so
    // predicate-time evaluation is allocation-free.
    let recursives = collect_recursive_patterns(&component.graph)?;

    let mut node_match = |q: &AtomQuery, node: &AtomNodeRef| -> bool {
        if !eval_atom_query(q, node.id, ctx) {
            return false;
        }
        // Recursive-SMARTS tail check: for every `$(...)` primitive in the
        // query, the candidate target atom must also satisfy an
        // isomorphism of the inner pattern anchored on itself.
        recursive_roots(q).into_iter().all(|inner_ir| {
            let Some(inner_pat) = recursives
                .iter()
                .find(|(k, _)| std::ptr::eq(*k, inner_ir))
                .map(|(_, v)| v)
            else {
                return false;
            };
            recursive_match_anchored(inner_pat, target_graph, ctx, node.id)
        })
    };

    let mut edge_match = eval_bond_query;

    let pattern_ref = &component.graph;
    let target_ref = &target_graph.graph;
    let iter = petgraph::algo::subgraph_isomorphisms_iter(
        &pattern_ref,
        &target_ref,
        &mut node_match,
        &mut edge_match,
    );

    let Some(iter) = iter else {
        return Ok(Vec::new());
    };

    let matches: Vec<Match> = iter.map(Match).collect();
    Ok(matches)
}

// ---------------------------------------------------------------------------
// Target-graph construction
// ---------------------------------------------------------------------------

struct TargetGraph {
    graph: UnGraph<AtomNodeRef, BondEdge>,
    /// Reverse map of `AtomId → petgraph-node-index`, used to anchor
    /// recursive-SMARTS matches onto the parent candidate atom.
    id_to_index: std::collections::HashMap<AtomId, usize>,
}

/// The node weight stored in the target `UnGraph`. Carrying the `AtomId`
/// on every node lets predicate closures evaluate without side tables.
#[derive(Clone, Copy)]
struct AtomNodeRef {
    id: AtomId,
}

impl TargetGraph {
    fn index_of(&self, id: AtomId) -> usize {
        self.id_to_index.get(&id).copied().unwrap_or(usize::MAX)
    }
}

fn build_target_graph(mol: &MolGraph, rings: &molrs::rings::RingInfo) -> TargetGraph {
    let mut graph = UnGraph::<AtomNodeRef, BondEdge>::with_capacity(mol.n_atoms(), mol.n_bonds());
    let mut id_to_node = std::collections::HashMap::new();
    let mut id_to_index: std::collections::HashMap<AtomId, usize> =
        std::collections::HashMap::new();

    for (idx, (id, _atom)) in mol.atoms().enumerate() {
        let ni = graph.add_node(AtomNodeRef { id });
        id_to_node.insert(id, ni);
        id_to_index.insert(id, idx);
    }

    for (bond_id, bond) in mol.bonds() {
        let [a, b] = bond.atoms;
        let stored_order = bond
            .props
            .get("order")
            .and_then(|v| match v {
                molrs::molgraph::PropValue::F64(f) => Some(*f),
                _ => None,
            })
            .unwrap_or(1.0);
        // The SMILES → Atomistic pipeline does not attach an explicit order
        // to implicit aromatic bonds (e.g. `c1ccccc1`); rescue the aromatic
        // semantics here by upgrading single-order bonds between two aromatic
        // atoms to order 1.5 for matcher purposes.
        let both_aromatic = [a, b].iter().all(|id| {
            mol.get_atom(*id)
                .ok()
                .and_then(|atom| atom.get_f64("aromatic"))
                .map(|v| v == 1.0)
                .unwrap_or(false)
        });
        let order = if both_aromatic && (stored_order - 1.0).abs() < 1e-6 {
            1.5
        } else {
            stored_order
        };
        let in_ring = rings.is_bond_in_ring(bond_id);
        if let (Some(&ai), Some(&bi)) = (id_to_node.get(&a), id_to_node.get(&b)) {
            graph.add_edge(ai, bi, BondEdge { order, in_ring });
        }
    }

    TargetGraph { graph, id_to_index }
}

// ---------------------------------------------------------------------------
// Recursive SMARTS
// ---------------------------------------------------------------------------

/// Compile every distinct inner SMARTS pattern referenced by a component's
/// recursive primitives. Identity is by pointer — each `$(...)` node in the
/// IR is a distinct `SmilesIR` allocation, so pointer equality is both
/// sufficient and cheaper than structural equality.
fn collect_recursive_patterns<'g>(
    component: &'g UnGraph<AtomQuery, BondQuery>,
) -> Result<Vec<(&'g SmilesIR, SmartsPattern)>, SmartsError> {
    let mut out: Vec<(&'g SmilesIR, SmartsPattern)> = Vec::new();
    for node in component.node_weights() {
        for inner in recursive_roots(node).into_iter() {
            if out.iter().any(|(k, _)| std::ptr::eq(*k, inner)) {
                continue;
            }
            let inner_pat = compile_from_ir(inner.clone())?;
            out.push((inner, inner_pat));
        }
    }
    Ok(out)
}

fn compile_from_ir(ir: SmilesIR) -> Result<SmartsPattern, SmartsError> {
    // Round-trip via the public compile() would re-parse a serialized form
    // we don't emit. Instead reuse the component compiler on the IR we
    // already have.
    //
    // A small shim: expose `pattern::compile_component` via a free fn.
    use super::pattern::compile_component_for_ir;

    let components = ir
        .components
        .iter()
        .map(compile_component_for_ir)
        .collect::<Result<Vec<_>, _>>()?;
    Ok(SmartsPattern::from_components(components))
}

/// Walk an `AtomQuery` tree yielding every inner `SmilesIR` that appears in
/// a `Recursive(...)` primitive. This is used to precompile recursives.
fn recursive_roots(query: &AtomQuery) -> Vec<&SmilesIR> {
    let mut out: Vec<&SmilesIR> = Vec::new();
    collect_recursive(query, &mut out);
    out
}

fn collect_recursive<'a>(q: &'a AtomQuery, out: &mut Vec<&'a SmilesIR>) {
    match q {
        AtomQuery::Primitive(AtomPrimitive::Recursive(ir)) => out.push(ir.as_ref()),
        AtomQuery::Primitive(_) => {}
        AtomQuery::Not(inner) => collect_recursive(inner, out),
        AtomQuery::And(parts) | AtomQuery::Or(parts) | AtomQuery::LowAnd(parts) => {
            for p in parts {
                collect_recursive(p, out);
            }
        }
    }
}

fn recursive_match_anchored(
    inner: &SmartsPattern,
    target_graph: &TargetGraph,
    ctx: &TargetCtx<'_>,
    anchor: AtomId,
) -> bool {
    // Run the inner pattern and check whether any match starts at `anchor`.
    // Inner patterns are assumed single-component; multi-component recursion
    // is not part of Daylight's spec.
    let matches = match match_component(&inner.components[0], target_graph, ctx) {
        Ok(m) => m,
        Err(_) => return false,
    };
    let anchor_idx = target_graph.index_of(anchor);
    matches.iter().any(|m| m.0.first() == Some(&anchor_idx))
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn cartesian_disjoint_product(lists: &[Vec<Match>]) -> Vec<Match> {
    if lists.iter().any(|v| v.is_empty()) {
        return Vec::new();
    }
    let mut out: Vec<Match> = vec![Match(Vec::new())];
    for list in lists {
        let mut next = Vec::new();
        for acc in &out {
            for m in list {
                if m.0.iter().any(|i| acc.0.contains(i)) {
                    continue;
                }
                let mut merged = acc.0.clone();
                merged.extend_from_slice(&m.0);
                next.push(Match(merged));
            }
        }
        out = next;
    }
    out
}

// ---------------------------------------------------------------------------
// Debug-only: enforce that SubstructureMatcher is object-safe. A build-time
// error here means `Box<dyn SubstructureMatcher>` lost dyn-compat — see
// matcher.rs module doc for the rationale.
// ---------------------------------------------------------------------------

#[allow(dead_code)]
fn _assert_dyn_matcher(_: &dyn SubstructureMatcher) {}
