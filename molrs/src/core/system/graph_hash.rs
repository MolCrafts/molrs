//! Isomorphism-invariant structural graph hash, canonical ordering, and
//! whole-graph isomorphism over a [`MolGraph`].
//!
//! All three primitives are built on the generic [`MolGraph`] adjacency (arity-2
//! relations are the graph edges), so they serve both
//! [`Atomistic`](crate::system::atomistic::Atomistic) and
//! [`CoarseGrain`](crate::system::coarsegrain::CoarseGrain) unchanged — the leaf
//! only supplies its node vocabulary (`element` / `bead_type`).
//!
//! # Algorithm — Weisfeiler–Lehman color refinement
//!
//! - **Initial node color** = a deterministic hash of the node's *label*: element
//!   symbol (atoms) or bead type (CG) + degree (incident edge count) + formal
//!   charge (the `charge` component, `0` when absent) + an aromatic flag (the
//!   `is_aromatic` component when set, else inferred from any incident bond of
//!   order `1.5`, matching the project aromaticity convention).
//! - **Refinement round**: a node's new color = hash of its current color plus the
//!   *sorted multiset* of `(edge label = bond order, neighbor's current color)`.
//!   Rounds repeat until the color partition stops refining (the count of
//!   distinct colors — which is monotone non-decreasing under WL — stabilizes),
//!   bounded by the node count.
//! - [`structural_hash`] = a hash of the *sorted multiset of final colors*, so it
//!   is invariant under any node permutation.
//! - [`canonical_order`] = nodes sorted by `(final color, initial color, stable
//!   handle tiebreak)` — deterministic within a graph, and consistent across two
//!   isomorphic graphs (tied nodes share a color, hence share every label
//!   attribute).
//! - [`is_isomorphic`] = a quick reject on `structural_hash` / edge-count
//!   mismatch, then a WL-color-pruned backtracking bijective match.
//!
//! The hasher is a fixed-constant FNV-1a over little-endian bytes — **not** the
//! std `DefaultHasher` (random-seeded) — so a hash is reproducible across runs
//! and processes and can be trusted as a persistent cache key.

use std::collections::HashMap;

use crate::store::keys;
use crate::system::molgraph::{MolGraph, NodeId, PropValue};

// ---------------------------------------------------------------------------
// Deterministic hashing primitives (fixed-seed FNV-1a, 64-bit)
// ---------------------------------------------------------------------------

const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

/// Fold a raw byte slice into an FNV-1a accumulator.
#[inline]
fn hash_bytes(mut h: u64, bytes: &[u8]) -> u64 {
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(FNV_PRIME);
    }
    h
}

/// Fold one `u64` (as little-endian bytes, so the digest is architecture
/// independent) into an FNV-1a accumulator.
#[inline]
fn hash_u64(h: u64, x: u64) -> u64 {
    hash_bytes(h, &x.to_le_bytes())
}

/// Fold a slice of `u64`s in order into an FNV-1a accumulator.
#[inline]
fn hash_u64_slice(mut h: u64, xs: &[u64]) -> u64 {
    for &x in xs {
        h = hash_u64(h, x);
    }
    h
}

// ---------------------------------------------------------------------------
// Node label helpers
// ---------------------------------------------------------------------------

/// The label string of a node: element symbol for an atom, else bead type for a
/// bead, else the empty string (a bare graph node).
fn node_label_str(g: &MolGraph, id: NodeId) -> String {
    let atom = match g.get_node(id) {
        Ok(a) => a,
        Err(_) => return String::new(),
    };
    if let Some(sym) = atom.get_str(keys::ELEMENT) {
        sym.to_owned()
    } else if let Some(bt) = atom.get_str(keys::BEAD_TYPE) {
        bt.to_owned()
    } else {
        String::new()
    }
}

/// Read a prop as an `f64` flag / number, accepting int or float storage.
#[inline]
fn prop_as_f64(v: Option<&PropValue>) -> Option<f64> {
    v.and_then(PropValue::as_f64)
}

/// The bond order that marks an aromatic bond (project convention). Also used to
/// infer a node's aromatic flag when no explicit `is_aromatic` prop is set.
const AROMATIC_ORDER: f64 = 1.5;

// ---------------------------------------------------------------------------
// GraphView — a dense, precomputed snapshot for the WL kernel
// ---------------------------------------------------------------------------

/// A dense, immutable snapshot of `g`: contiguous node indices, per-node initial
/// WL color, and a labeled adjacency list built once so the refinement loop and
/// the matcher never re-materialize relations.
struct GraphView {
    /// Node handles in dense-index order (row order of the node table).
    nodes: Vec<NodeId>,
    /// Initial WL color per dense node index.
    init_colors: Vec<u64>,
    /// Per node: the `(neighbor dense index, edge-label bits)` of each incident
    /// arity-2 relation.
    adj: Vec<Vec<(usize, u64)>>,
    /// Total number of (undirected) edges.
    n_edges: usize,
}

impl GraphView {
    /// Build the dense snapshot: assign contiguous indices, read edge labels
    /// (bond order bits) from each incident relation, then compute initial
    /// colors from element/bead label + degree + charge + aromatic flag.
    fn build(g: &MolGraph) -> Self {
        let nodes: Vec<NodeId> = g.node_ids().collect();
        let index: HashMap<NodeId, usize> =
            nodes.iter().enumerate().map(|(i, &id)| (id, i)).collect();

        let mut adj: Vec<Vec<(usize, u64)>> = vec![Vec::new(); nodes.len()];
        let mut edge_count = 0usize;
        for (i, &id) in nodes.iter().enumerate() {
            for (kind, rid, other) in g.neighbor_relations(id) {
                let Some(&j) = index.get(&other) else {
                    continue;
                };
                let order = g
                    .get_relation(kind, rid)
                    .ok()
                    .and_then(|r| prop_as_f64(r.props.get(keys::ORDER)))
                    .unwrap_or(1.0);
                adj[i].push((j, order.to_bits()));
            }
            // Each undirected edge is seen once from each endpoint.
            edge_count += adj[i].len();
        }
        let n_edges = edge_count / 2;

        let init_colors = nodes
            .iter()
            .enumerate()
            .map(|(i, &id)| Self::initial_color(g, id, &adj[i]))
            .collect();

        Self {
            nodes,
            init_colors,
            adj,
            n_edges,
        }
    }

    /// Initial color of a node from its label, degree, charge and aromatic flag.
    fn initial_color(g: &MolGraph, id: NodeId, incident: &[(usize, u64)]) -> u64 {
        let atom = g.get_node(id).ok();
        let label = node_label_str(g, id);
        let degree = incident.len() as u64;

        let charge = atom
            .as_ref()
            .and_then(|a| prop_as_f64(a.get(keys::CHARGE)))
            .unwrap_or(0.0);

        // Aromatic: explicit `is_aromatic` prop wins; else infer from any
        // incident bond of order ~= 1.5 (project convention, mirrors `ast.rs`).
        let explicit = match atom.as_ref().and_then(|a| a.get("is_aromatic")) {
            Some(PropValue::Int(v)) => Some(*v != 0),
            Some(PropValue::F64(v)) => Some(*v != 0.0),
            Some(PropValue::Bool(v)) => Some(*v),
            _ => None,
        };
        let aromatic = explicit.unwrap_or_else(|| {
            incident
                .iter()
                .any(|&(_, bits)| (f64::from_bits(bits) - AROMATIC_ORDER).abs() < 1e-6)
        });

        let mut h = FNV_OFFSET;
        h = hash_bytes(h, label.as_bytes());
        h = hash_u64(h, degree);
        h = hash_u64(h, charge.to_bits());
        h = hash_u64(h, aromatic as u64);
        h
    }
}

/// Run WL color refinement to convergence, returning the final color per dense
/// node index.
fn wl_colors(view: &GraphView) -> Vec<u64> {
    let n = view.nodes.len();
    let mut colors = view.init_colors.clone();
    if n == 0 {
        return colors;
    }

    let mut distinct = distinct_count(&colors);
    // WL stabilizes within `n` rounds; break early once the partition (distinct
    // color count, monotone non-decreasing) stops refining.
    for _ in 0..n {
        let mut next = vec![0u64; n];
        for i in 0..n {
            // Sorted multiset of (edge label, neighbor color).
            let mut env: Vec<(u64, u64)> = view.adj[i]
                .iter()
                .map(|&(j, label)| (label, colors[j]))
                .collect();
            env.sort_unstable();

            let mut h = hash_u64(FNV_OFFSET, colors[i]);
            for (label, neigh) in env {
                h = hash_u64(h, label);
                h = hash_u64(h, neigh);
            }
            next[i] = h;
        }
        let next_distinct = distinct_count(&next);
        colors = next;
        if next_distinct == distinct {
            break;
        }
        distinct = next_distinct;
    }
    colors
}

/// Number of distinct values in a color vector.
fn distinct_count(colors: &[u64]) -> usize {
    let mut sorted = colors.to_vec();
    sorted.sort_unstable();
    sorted.dedup();
    sorted.len()
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Isomorphism-invariant 64-bit structural hash of `g`.
///
/// Identical for a node-permuted copy; sensitive to element/bead-type, degree,
/// charge, aromatic flag, bond order, and connectivity.
pub fn structural_hash(g: &MolGraph) -> u64 {
    let view = GraphView::build(g);
    let colors = wl_colors(&view);
    let mut sorted = colors;
    sorted.sort_unstable();

    let mut h = FNV_OFFSET;
    h = hash_u64(h, view.nodes.len() as u64);
    h = hash_u64(h, view.n_edges as u64);
    hash_u64_slice(h, &sorted)
}

/// Deterministic canonical node ordering from the WL refinement.
///
/// Nodes are sorted by `(final color, initial color, stable handle)`. Two
/// isomorphic graphs induce a consistent node bijection by pairing their
/// `canonical_order` lists position-by-position.
pub fn canonical_order(g: &MolGraph) -> Vec<NodeId> {
    let view = GraphView::build(g);
    let colors = wl_colors(&view);

    let mut order: Vec<usize> = (0..view.nodes.len()).collect();
    order.sort_by(|&a, &b| {
        colors[a]
            .cmp(&colors[b])
            .then(view.init_colors[a].cmp(&view.init_colors[b]))
            .then_with(|| node_ffi(view.nodes[a]).cmp(&node_ffi(view.nodes[b])))
    });
    order.into_iter().map(|i| view.nodes[i]).collect()
}

/// Whether `a` and `b` are isomorphic as labeled graphs (element/bead-type +
/// charge + aromatic node labels and bond-order edge labels all preserved).
///
/// Quick-rejects on node-count, edge-count, or `structural_hash` mismatch, then
/// runs a WL-color-pruned backtracking bijective match to resolve the rare hash
/// collision before equality is trusted.
pub fn is_isomorphic(a: &MolGraph, b: &MolGraph) -> bool {
    if a.n_nodes() != b.n_nodes() {
        return false;
    }
    let va = GraphView::build(a);
    let vb = GraphView::build(b);
    if va.n_edges != vb.n_edges {
        return false;
    }

    let ca = wl_colors(&va);
    let cb = wl_colors(&vb);
    {
        let mut sa = ca.clone();
        let mut sb = cb.clone();
        sa.sort_unstable();
        sb.sort_unstable();
        if sa != sb {
            return false;
        }
    }

    let n = va.nodes.len();
    if n == 0 {
        return true;
    }

    // Candidate B nodes by final color.
    let mut by_color: HashMap<u64, Vec<usize>> = HashMap::new();
    for (j, &c) in cb.iter().enumerate() {
        by_color.entry(c).or_default().push(j);
    }
    // Order A nodes rarest-color-first for tighter pruning.
    let mut freq: HashMap<u64, usize> = HashMap::new();
    for &c in &ca {
        *freq.entry(c).or_insert(0) += 1;
    }
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by_key(|&i| (*freq.get(&ca[i]).unwrap_or(&0), i));

    let adj_a = adjacency_map(&va);
    let adj_b = adjacency_map(&vb);

    let mut map_ab = vec![usize::MAX; n];
    let mut map_ba = vec![usize::MAX; n];
    backtrack(
        0,
        &order,
        &ca,
        &cb,
        &by_color,
        &adj_a,
        &adj_b,
        &mut map_ab,
        &mut map_ba,
    )
}

// ---------------------------------------------------------------------------
// Isomorphism backtracking internals
// ---------------------------------------------------------------------------

/// Neighbor → edge-label map per dense node index (O(1) edge lookup).
fn adjacency_map(view: &GraphView) -> Vec<HashMap<usize, u64>> {
    view.adj
        .iter()
        .map(|nbrs| nbrs.iter().map(|&(j, label)| (j, label)).collect())
        .collect()
}

/// Recursive WL-pruned bijective match. `order` drives the A-node visitation
/// order; `map_ab` / `map_ba` are the partial forward / inverse maps.
#[allow(clippy::too_many_arguments)]
fn backtrack(
    depth: usize,
    order: &[usize],
    ca: &[u64],
    cb: &[u64],
    by_color: &HashMap<u64, Vec<usize>>,
    adj_a: &[HashMap<usize, u64>],
    adj_b: &[HashMap<usize, u64>],
    map_ab: &mut [usize],
    map_ba: &mut [usize],
) -> bool {
    if depth == order.len() {
        return true;
    }
    let u = order[depth];
    let Some(candidates) = by_color.get(&ca[u]) else {
        return false;
    };
    for &v in candidates {
        if map_ba[v] != usize::MAX || cb[v] != ca[u] {
            continue;
        }
        if feasible(u, v, adj_a, adj_b, map_ab, map_ba) {
            map_ab[u] = v;
            map_ba[v] = u;
            if backtrack(
                depth + 1,
                order,
                ca,
                cb,
                by_color,
                adj_a,
                adj_b,
                map_ab,
                map_ba,
            ) {
                return true;
            }
            map_ab[u] = usize::MAX;
            map_ba[v] = usize::MAX;
        }
    }
    false
}

/// Whether mapping `u -> v` is consistent with the edges among already-mapped
/// nodes, in both directions (so no A-edge is dropped and no extra B-edge is
/// introduced).
fn feasible(
    u: usize,
    v: usize,
    adj_a: &[HashMap<usize, u64>],
    adj_b: &[HashMap<usize, u64>],
    map_ab: &[usize],
    map_ba: &[usize],
) -> bool {
    // Every already-mapped A-neighbor of u must be a B-neighbor of v, same label.
    for (&w, &label) in &adj_a[u] {
        let mw = map_ab[w];
        if mw != usize::MAX && adj_b[v].get(&mw) != Some(&label) {
            return false;
        }
    }
    // Every already-mapped B-neighbor of v must be an A-neighbor of u, same label.
    for (&x, &label) in &adj_b[v] {
        let mx = map_ba[x];
        if mx != usize::MAX && adj_a[u].get(&mx) != Some(&label) {
            return false;
        }
    }
    true
}

/// Stable per-graph tiebreak key for a node handle.
#[inline]
fn node_ffi(id: NodeId) -> u64 {
    crate::system::molgraph::node_to_u64(id)
}

// =========================================================================
// Tests
// =========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::system::atomistic::{AtomId, Atomistic};
    use crate::system::coarsegrain::CoarseGrain;

    /// Ethanol skeleton C-C-O with explicit H (9 atoms), returned with its
    /// atom handles in build order.
    fn ethanol() -> (Atomistic, Vec<AtomId>) {
        let mut mol = Atomistic::new();
        let ids: Vec<AtomId> = ["C", "C", "O", "H", "H", "H", "H", "H", "H"]
            .iter()
            .map(|e| mol.add_atom_bare(e))
            .collect();
        // C0-C1, C1-O2, O2-H8; H3,H4,H5 on C0; H6,H7 on C1
        for (i, j) in [
            (0, 1),
            (1, 2),
            (2, 8),
            (0, 3),
            (0, 4),
            (0, 5),
            (1, 6),
            (1, 7),
        ] {
            mol.add_bond(ids[i], ids[j]).unwrap();
        }
        (mol, ids)
    }

    /// Rebuild the same molecule with a permuted node-insertion order.
    fn ethanol_permuted() -> Atomistic {
        let mut mol = Atomistic::new();
        // Insert atoms in a shuffled order, remember by original index.
        let perm = [8usize, 2, 1, 0, 7, 6, 5, 4, 3];
        let elems = ["C", "C", "O", "H", "H", "H", "H", "H", "H"];
        let mut by_orig = std::collections::HashMap::new();
        for &orig in &perm {
            let id = mol.add_atom_bare(elems[orig]);
            by_orig.insert(orig, id);
        }
        for (i, j) in [
            (0, 1),
            (1, 2),
            (2, 8),
            (0, 3),
            (0, 4),
            (0, 5),
            (1, 6),
            (1, 7),
        ] {
            mol.add_bond(by_orig[&i], by_orig[&j]).unwrap();
        }
        mol
    }

    #[test]
    fn structural_hash_is_permutation_invariant() {
        let (a, _) = ethanol();
        let b = ethanol_permuted();
        assert_eq!(
            structural_hash(a.as_molgraph()),
            structural_hash(b.as_molgraph()),
            "hash must be invariant under node permutation"
        );
        assert!(is_isomorphic(a.as_molgraph(), b.as_molgraph()));
    }

    #[test]
    fn identical_junctions_hash_equal() {
        let (a, _) = ethanol();
        let (b, _) = ethanol();
        assert_eq!(
            structural_hash(a.as_molgraph()),
            structural_hash(b.as_molgraph())
        );
        assert!(is_isomorphic(a.as_molgraph(), b.as_molgraph()));
    }

    #[test]
    fn element_change_changes_hash() {
        let (a, _) = ethanol();
        // Swap the O for an S — a different local environment.
        let mut mol = Atomistic::new();
        let ids: Vec<AtomId> = ["C", "C", "S", "H", "H", "H", "H", "H", "H"]
            .iter()
            .map(|e| mol.add_atom_bare(e))
            .collect();
        for (i, j) in [
            (0, 1),
            (1, 2),
            (2, 8),
            (0, 3),
            (0, 4),
            (0, 5),
            (1, 6),
            (1, 7),
        ] {
            mol.add_bond(ids[i], ids[j]).unwrap();
        }
        assert_ne!(
            structural_hash(a.as_molgraph()),
            structural_hash(mol.as_molgraph())
        );
        assert!(!is_isomorphic(a.as_molgraph(), mol.as_molgraph()));
    }

    #[test]
    fn charge_change_changes_hash() {
        let (a, ids) = ethanol();
        let mut b = a.clone();
        b.set_atom(ids[2], keys::CHARGE, -1.0).unwrap();
        assert_ne!(
            structural_hash(a.as_molgraph()),
            structural_hash(b.as_molgraph()),
            "charge must affect the hash"
        );
    }

    #[test]
    fn aromatic_flag_changes_hash() {
        let (a, ids) = ethanol();
        let mut b = a.clone();
        b.set_atom(ids[0], "is_aromatic", PropValue::Int(1))
            .unwrap();
        assert_ne!(
            structural_hash(a.as_molgraph()),
            structural_hash(b.as_molgraph()),
            "aromatic flag must affect the hash"
        );
    }

    #[test]
    fn bond_order_change_changes_hash() {
        let (a, _) = ethanol();
        let mut b = a.clone();
        let (bid, _) = b.bonds().next().unwrap();
        b.set_bond_prop(bid, keys::ORDER, 2.0).unwrap();
        assert_ne!(
            structural_hash(a.as_molgraph()),
            structural_hash(b.as_molgraph()),
            "bond order must affect the hash"
        );
    }

    #[test]
    fn non_isomorphic_graphs_rejected() {
        // Ethanol (9 atoms) vs a 9-atom linear chain — same size, different topo.
        let (a, _) = ethanol();
        let mut chain = Atomistic::new();
        let ids: Vec<AtomId> = (0..9).map(|_| chain.add_atom_bare("C")).collect();
        for k in 0..ids.len() - 1 {
            chain.add_bond(ids[k], ids[k + 1]).unwrap();
        }
        assert!(!is_isomorphic(a.as_molgraph(), chain.as_molgraph()));
    }

    #[test]
    fn canonical_order_bijection_is_consistent() {
        let (a, _) = ethanol();
        let b = ethanol_permuted();
        let oa = canonical_order(a.as_molgraph());
        let ob = canonical_order(b.as_molgraph());
        assert_eq!(oa.len(), ob.len());
        // Paired nodes must agree on element + degree.
        for (ia, ib) in oa.iter().zip(ob.iter()) {
            let ea = a.get_atom(*ia).unwrap();
            let eb = b.get_atom(*ib).unwrap();
            assert_eq!(ea.get_str("element"), eb.get_str("element"));
            assert_eq!(a.neighbors(*ia).count(), b.neighbors(*ib).count());
        }
    }

    #[test]
    fn coarsegrain_hashes_by_bead_type() {
        // Two identical 3-bead angular CG molecules hash equal; changing a bead
        // type breaks both the hash and the isomorphism.
        let mut cg1 = CoarseGrain::new();
        let a = cg1.add_bead_bare("W");
        let b = cg1.add_bead_bare("P");
        let c = cg1.add_bead_bare("W");
        cg1.add_bond(a, b).unwrap();
        cg1.add_bond(b, c).unwrap();

        let mut cg2 = CoarseGrain::new();
        let d = cg2.add_bead_bare("W");
        let e = cg2.add_bead_bare("P");
        let f = cg2.add_bead_bare("W");
        cg2.add_bond(e, f).unwrap();
        cg2.add_bond(d, e).unwrap();

        assert_eq!(
            structural_hash(cg1.as_molgraph()),
            structural_hash(cg2.as_molgraph())
        );
        assert!(is_isomorphic(cg1.as_molgraph(), cg2.as_molgraph()));

        let mut cg3 = CoarseGrain::new();
        let g = cg3.add_bead_bare("W");
        let h = cg3.add_bead_bare("Q"); // different centre type
        let i = cg3.add_bead_bare("W");
        cg3.add_bond(g, h).unwrap();
        cg3.add_bond(h, i).unwrap();
        assert_ne!(
            structural_hash(cg1.as_molgraph()),
            structural_hash(cg3.as_molgraph())
        );
        assert!(!is_isomorphic(cg1.as_molgraph(), cg3.as_molgraph()));
    }

    #[test]
    fn empty_graph_is_stable() {
        let a = Atomistic::new();
        let b = Atomistic::new();
        assert_eq!(
            structural_hash(a.as_molgraph()),
            structural_hash(b.as_molgraph())
        );
        assert!(is_isomorphic(a.as_molgraph(), b.as_molgraph()));
        assert!(canonical_order(a.as_molgraph()).is_empty());
    }
}
