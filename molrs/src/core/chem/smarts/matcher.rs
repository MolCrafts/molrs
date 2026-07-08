//! Backtracking subgraph-isomorphism matcher (non-uniquified), with
//! recursive-SMARTS evaluation.
//!
//! Ported (semantics only) from RDKit's substructure matcher under BSD-3:
//! `Code/GraphMol/Substruct/SubstructMatch.cpp`. Match semantics follow
//! `GetSubstructMatches(uniquify=False)`: every distinct query-atom →
//! mol-atom embedding is reported once, ordered by query-atom index, and the
//! enumeration order tracks RDKit's (query atom 0 outer loop, depth-first).
//!
//! The algorithm is a straightforward depth-first backtracking VF2-style
//! match. The query is assumed connected (true for all ETKDG torsion
//! patterns); each query atom after the first connects to an already-placed
//! query atom, so candidates are generated from the neighbourhood of the
//! anchor's image.

use std::collections::HashMap;

use crate::system::atomistic::{AtomId, Atomistic};
use crate::system::molgraph::PropValue;

use super::ast::{BondFacts, MolContext, RecursiveEval};
use super::parser::QueryGraph;
use super::{MatchOptions, SmartsMatch};

/// Resolve bond facts between two molecule atoms, if they are bonded.
fn bond_facts(ctx: &MolContext, a: AtomId, b: AtomId) -> Option<BondFacts> {
    let mol = ctx.mol;
    for (bid, bond) in mol.bonds() {
        let (x, y) = (bond.nodes[0], bond.nodes[1]);
        if (x == a && y == b) || (x == b && y == a) {
            let order = match bond.props.get("order") {
                Some(PropValue::F64(v)) => *v,
                _ => 1.0,
            };
            let aromatic = (order - 1.5).abs() < 1e-6
                || matches!(bond.props.get("is_aromatic"),
                    Some(PropValue::Int(v)) if *v != 0)
                || matches!(bond.props.get("is_aromatic"),
                    Some(PropValue::F64(v)) if *v != 0.0);
            let in_ring = ctx.rings.is_bond_in_ring(bid);
            return Some(BondFacts {
                order,
                aromatic,
                in_ring,
            });
        }
    }
    None
}

/// Adapter giving the AST a way to evaluate recursive `$(...)` subpatterns.
struct RecursiveEvaluator<'g> {
    recursives: &'g [QueryGraph],
}

impl RecursiveEval for RecursiveEvaluator<'_> {
    fn eval_recursive(&self, sub_index: usize, ctx: &MolContext, id: AtomId) -> bool {
        let sub = &self.recursives[sub_index];
        // The recursive subpattern matches iff it has at least one embedding
        // whose first query atom maps to `id` (RDKit roots `$(...)` at the
        // candidate atom).
        let mut found = false;
        enumerate_matches(sub, ctx, Some(id), &mut |_assign| {
            found = true;
            false // stop at the first hit
        });
        found
    }
}

/// Enumerate all embeddings of `query` into the molecule described by `ctx`.
///
/// `root_fix`, when `Some(id)`, constrains query atom 0 to map to `id`
/// (used for recursive SMARTS). `visit` is called for every complete match
/// with the assignment vector (indexed by query-atom); returning `false`
/// stops the enumeration early.
fn enumerate_matches(
    query: &QueryGraph,
    ctx: &MolContext,
    root_fix: Option<AtomId>,
    visit: &mut dyn FnMut(&[AtomId]) -> bool,
) {
    let n = query.atoms.len();
    if n == 0 {
        return;
    }
    let rec = RecursiveEvaluator {
        recursives: &query.recursives,
    };

    // Precompute, for each query atom (>0), the already-earlier query atom it
    // bonds to plus the bond query — the "anchor". The parser always connects
    // a new atom to a prior atom, and ring closures add extra bonds among
    // earlier atoms; we treat the first-seen connection as the anchor and the
    // rest as additional constraints checked at placement time.
    let order: Vec<usize> = (0..n).collect();
    let anchors = build_anchors(query, &order);

    let mol_atoms: Vec<AtomId> = ctx.mol.atoms().map(|(id, _)| id).collect();

    let mut assign: Vec<Option<AtomId>> = vec![None; n];
    let mut used: HashMap<AtomId, bool> = HashMap::new();

    backtrack(
        query,
        ctx,
        &rec,
        &order,
        &anchors,
        &mol_atoms,
        root_fix,
        0,
        &mut assign,
        &mut used,
        visit,
    );
}

/// For each position in `order`, the anchor = the earliest placed query atom
/// it is bonded to (or None for the first atom).
fn build_anchors(query: &QueryGraph, order: &[usize]) -> Vec<Option<usize>> {
    let pos_of: HashMap<usize, usize> = order.iter().enumerate().map(|(p, &q)| (q, p)).collect();
    let mut anchors = vec![None; order.len()];
    for &qa in order {
        let mut best: Option<usize> = None;
        for b in &query.bonds {
            let other = if b.a == qa {
                Some(b.b)
            } else if b.b == qa {
                Some(b.a)
            } else {
                None
            };
            if let Some(o) = other
                && pos_of[&o] < pos_of[&qa]
            {
                best = Some(match best {
                    Some(cur) if pos_of[&cur] <= pos_of[&o] => cur,
                    _ => o,
                });
            }
        }
        anchors[pos_of[&qa]] = best;
    }
    anchors
}

/// All query bonds connecting `qa` to query atoms already placed (position < this).
fn earlier_bonds<'q>(
    query: &'q QueryGraph,
    qa: usize,
    placed: &[bool],
) -> Vec<&'q super::parser::QueryBond> {
    query
        .bonds
        .iter()
        .filter(|b| (b.a == qa && placed[b.b]) || (b.b == qa && placed[b.a]))
        .collect()
}

#[allow(clippy::too_many_arguments)]
fn backtrack(
    query: &QueryGraph,
    ctx: &MolContext,
    rec: &dyn RecursiveEval,
    order: &[usize],
    anchors: &[Option<usize>],
    mol_atoms: &[AtomId],
    root_fix: Option<AtomId>,
    depth: usize,
    assign: &mut Vec<Option<AtomId>>,
    used: &mut HashMap<AtomId, bool>,
    visit: &mut dyn FnMut(&[AtomId]) -> bool,
) -> bool {
    if depth == order.len() {
        let full: Vec<AtomId> = assign.iter().map(|o| o.unwrap()).collect();
        return visit(&full);
    }

    let qa = order[depth];
    let placed: Vec<bool> = assign.iter().map(|o| o.is_some()).collect();

    // Candidate molecule atoms for this query atom.
    let candidates: Vec<AtomId> = if depth == 0 {
        match root_fix {
            Some(id) => vec![id],
            None => mol_atoms.to_vec(),
        }
    } else {
        // Generate from the anchor's image neighbourhood.
        let anchor = anchors[depth].expect("non-root query atom must have an anchor");
        let anchor_img = assign[anchor].expect("anchor must be assigned");
        ctx.mol.neighbors(anchor_img).collect()
    };

    for cand in candidates {
        if *used.get(&cand).unwrap_or(&false) {
            continue;
        }
        // Atom primitive must match.
        if !query.atoms[qa].query.eval(ctx, cand, rec) {
            continue;
        }
        // Every query bond from qa to an already-placed atom must be
        // satisfied by an actual molecule bond matching the bond query.
        let mut bonds_ok = true;
        for qb in earlier_bonds(query, qa, &placed) {
            let other_q = if qb.a == qa { qb.b } else { qb.a };
            let other_img = assign[other_q].unwrap();
            match bond_facts(ctx, cand, other_img) {
                Some(facts) if qb.query.eval(&facts) => {}
                _ => {
                    bonds_ok = false;
                    break;
                }
            }
        }
        if !bonds_ok {
            continue;
        }

        assign[qa] = Some(cand);
        used.insert(cand, true);

        let keep_going = backtrack(
            query,
            ctx,
            rec,
            order,
            anchors,
            mol_atoms,
            root_fix,
            depth + 1,
            assign,
            used,
            visit,
        );

        assign[qa] = None;
        used.insert(cand, false);

        if !keep_going {
            return false;
        }
    }
    true
}

/// Find every non-uniquified embedding of `query` in `mol`.
pub fn find(query: &QueryGraph, mol: &Atomistic, options: MatchOptions<'_>) -> Vec<SmartsMatch> {
    let ctx = match options.labels {
        Some(labels) => MolContext::with_labels(mol, labels),
        None => MolContext::new(mol),
    };
    let Some(limit) = options.limit else {
        let mut out = Vec::new();
        enumerate_matches(query, &ctx, options.root, &mut |assign| {
            out.push(SmartsMatch {
                atoms: assign.to_vec(),
            });
            true
        });
        return out;
    };
    if limit == 0 {
        return Vec::new();
    }
    let mut out = Vec::new();
    enumerate_matches(query, &ctx, options.root, &mut |assign| {
        out.push(SmartsMatch {
            atoms: assign.to_vec(),
        });
        out.len() < limit
    });
    out
}

/// Whether at least one embedding exists.
pub fn has_match(query: &QueryGraph, mol: &Atomistic, mut options: MatchOptions<'_>) -> bool {
    options.limit = Some(1);
    !find(query, mol, options).is_empty()
}
