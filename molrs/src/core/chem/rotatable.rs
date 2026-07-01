//! Rotatable bond detection and downstream atom BFS.
//!
//! Identifies the rotatable bonds of an [`Atomistic`] and the atoms downstream
//! of a bond (the set rotated during a torsion move). Connectivity, degree, ring
//! membership, and traversal are all delegated to the canonical
//! [`Topology`](crate::system::topology::Topology) snapshot built from the graph
//! bonds — this module no longer re-implements ring perception or BFS, so the
//! result tracks `Topology`'s SSSR (correct for fused/bridged rings) instead of
//! the previous hand-rolled union-find heuristic.

use std::collections::{HashMap, HashSet, VecDeque};

use crate::system::atomistic::{AtomId, Atomistic};
use crate::system::topology::Topology;

/// A rotatable bond between atoms `j` and `k`, with the set of downstream
/// atom indices (0-based positional indices, not `AtomId`s) on the `k`-side.
#[derive(Debug, Clone)]
pub struct RotatableBond {
    /// Axis start atom (positional index).
    pub j: usize,
    /// Axis end atom (positional index).
    pub k: usize,
    /// Atoms on the k-side of the bond (positional indices), rotated during torsion.
    pub downstream: Vec<usize>,
}

/// Build a positional index map from `AtomId` to 0-based index, following the
/// iteration order of [`Atomistic::atoms`].
pub fn atom_id_to_index(graph: &Atomistic) -> HashMap<AtomId, usize> {
    graph
        .atoms()
        .enumerate()
        .map(|(idx, (id, _))| (id, idx))
        .collect()
}

/// Build the [`Topology`] snapshot for `graph`, indexing atoms by the order of
/// [`Atomistic::atoms`] and edges by the order of [`Atomistic::bonds`].
///
/// Because [`Topology::from_edges`] preserves edge insertion order, a
/// [`Topology::bond_ring_mask`] over the returned topology aligns positionally
/// with `graph.bonds()`.
fn build_topology(graph: &Atomistic, id_to_idx: &HashMap<AtomId, usize>) -> Topology {
    let edges: Vec<[usize; 2]> = graph
        .bonds()
        .map(|(_, b)| [id_to_idx[&b.nodes[0]], id_to_idx[&b.nodes[1]]])
        .collect();
    Topology::from_edges(id_to_idx.len(), &edges)
}

/// Shared detection core. Returns the positional `(j, k)` index pair of every
/// rotatable bond together with the `Topology` they were derived from (reused by
/// the caller for downstream BFS) and the reverse `idx -> AtomId` table.
///
/// A bond is rotatable when it is a single bond (order ≈ 1.0, defaulting to 1.0
/// when unset), both endpoints are non-terminal (degree > 1), and it is acyclic
/// (not part of any ring per [`Topology::find_rings`]).
fn scan_rotatable(graph: &Atomistic) -> (Vec<AtomId>, Topology, Vec<(usize, usize)>) {
    let ids: Vec<AtomId> = graph.atoms().map(|(id, _)| id).collect();
    let id_to_idx: HashMap<AtomId, usize> =
        ids.iter().enumerate().map(|(i, &id)| (id, i)).collect();

    // Collect bonds once so the edge index used for `bond_ring_mask` and the
    // index used for per-bond filtering are guaranteed identical.
    let bonds: Vec<_> = graph.bonds().collect();
    let edges: Vec<[usize; 2]> = bonds
        .iter()
        .map(|(_, b)| [id_to_idx[&b.nodes[0]], id_to_idx[&b.nodes[1]]])
        .collect();
    let topo = Topology::from_edges(ids.len(), &edges);
    let ring_mask = topo.find_rings().bond_ring_mask(topo.n_bonds());

    let rotatable = bonds
        .iter()
        .enumerate()
        .filter_map(|(bi, (_, bond))| {
            // Single bond only (order defaults to 1.0 if unset). Accept order
            // stored as either F64 or Int via PropValue::as_f64.
            let order = bond
                .props
                .get("order")
                .and_then(|v| v.as_f64())
                .unwrap_or(1.0);
            if (order - 1.0).abs() > 0.01 {
                return None;
            }

            let j = id_to_idx[&bond.nodes[0]];
            let k = id_to_idx[&bond.nodes[1]];

            // Non-terminal: both endpoints must have degree > 1.
            if topo.degree(j) <= 1 || topo.degree(k) <= 1 {
                return None;
            }

            // Acyclic: bond is not part of any ring.
            if ring_mask[bi] {
                return None;
            }

            Some((j, k))
        })
        .collect();

    (ids, topo, rotatable)
}

/// BFS over `topo` from `k`, never crossing back through `j`. Returns the k-side
/// atom indices (including `k`).
fn downstream_indices(topo: &Topology, j: usize, k: usize) -> Vec<usize> {
    let mut visited = HashSet::new();
    visited.insert(j); // block traversal back through j
    visited.insert(k);

    let mut queue = VecDeque::new();
    queue.push_back(k);

    let mut result = vec![k];

    while let Some(current) = queue.pop_front() {
        for neighbor in topo.neighbors(current) {
            if visited.insert(neighbor) {
                result.push(neighbor);
                queue.push_back(neighbor);
            }
        }
    }

    result
}

/// Detect rotatable bonds in a molecular graph.
///
/// Returns a list of `(AtomId, AtomId)` pairs. See [`scan_rotatable`] for the
/// rotatable-bond criteria.
pub fn detect_rotatable_bonds(graph: &Atomistic) -> Vec<(AtomId, AtomId)> {
    let (ids, _topo, rotatable) = scan_rotatable(graph);
    rotatable
        .into_iter()
        .map(|(j, k)| (ids[j], ids[k]))
        .collect()
}

/// BFS to find all atoms downstream of bond (j, k) on the k-side.
///
/// Starting from `k`, traverses all connected atoms without crossing back
/// through `j`. Returns positional indices (not `AtomId`s).
pub fn downstream_atoms(
    j: AtomId,
    k: AtomId,
    graph: &Atomistic,
    id_to_idx: &HashMap<AtomId, usize>,
) -> Vec<usize> {
    let topo = build_topology(graph, id_to_idx);
    downstream_indices(&topo, id_to_idx[&j], id_to_idx[&k])
}

/// Detect rotatable bonds and return them as [`RotatableBond`] structs with
/// positional indices and downstream atom sets.
pub fn detect_rotatable_bonds_with_downstream(graph: &Atomistic) -> Vec<RotatableBond> {
    let (_ids, topo, rotatable) = scan_rotatable(graph);
    rotatable
        .into_iter()
        .map(|(j, k)| RotatableBond {
            j,
            k,
            downstream: downstream_indices(&topo, j, k),
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::system::molgraph::Atom;

    /// Build a chain graph (topology only, coords irrelevant for detection).
    fn chain(n: usize) -> Atomistic {
        let mut g = Atomistic::new();
        let mut ids = Vec::new();
        for _ in 0..n {
            ids.push(g.add_atom(Atom::new()));
        }
        for i in 0..n - 1 {
            g.add_bond(ids[i], ids[i + 1]).expect("add chain bond");
        }
        g
    }

    #[test]
    fn test_chain_rotatable_bonds() {
        let g = chain(5);
        let bonds = detect_rotatable_bonds(&g);
        assert_eq!(bonds.len(), 2);
    }

    #[test]
    fn test_ring_no_rotatable_bonds() {
        let mut g = Atomistic::new();
        let a = g.add_atom(Atom::new());
        let b = g.add_atom(Atom::new());
        let c = g.add_atom(Atom::new());
        g.add_bond(a, b).expect("add bond");
        g.add_bond(b, c).expect("add bond");
        g.add_bond(c, a).expect("add bond");

        assert_eq!(detect_rotatable_bonds(&g).len(), 0);
    }

    #[test]
    fn test_fused_ring_no_rotatable_bonds() {
        // Naphthalene-like fused bicyclic: two six-membered rings sharing the
        // 0-1 edge. Every bond is a ring bond -> none rotatable. The previous
        // union-find heuristic could mis-mark shared/bridge bonds; SSSR cannot.
        let mut g = Atomistic::new();
        let v: Vec<_> = (0..10).map(|_| g.add_atom(Atom::new())).collect();
        // ring A: 0-1-2-3-4-5-0
        for (a, b) in [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0)] {
            g.add_bond(v[a], v[b]).expect("add bond");
        }
        // ring B: 0-1 shared, plus 1-6-7-8-9-0
        for (a, b) in [(1, 6), (6, 7), (7, 8), (8, 9), (9, 0)] {
            g.add_bond(v[a], v[b]).expect("add bond");
        }
        assert_eq!(detect_rotatable_bonds(&g).len(), 0);
    }

    #[test]
    fn test_downstream_atoms_chain() {
        let g = chain(5);
        let id_to_idx = atom_id_to_index(&g);
        let ids: Vec<AtomId> = g.atoms().map(|(id, _)| id).collect();

        let ds = downstream_atoms(ids[1], ids[2], &g, &id_to_idx);
        assert_eq!(ds.len(), 3);
        assert!(ds.contains(&2));
        assert!(ds.contains(&3));
        assert!(ds.contains(&4));
    }

    #[test]
    fn test_detect_with_downstream() {
        let g = chain(5);
        let bonds = detect_rotatable_bonds_with_downstream(&g);
        assert_eq!(bonds.len(), 2);

        for rb in &bonds {
            assert!(!rb.downstream.is_empty());
            assert!(rb.downstream.contains(&rb.k));
        }
    }

    #[test]
    fn test_two_atoms_no_rotatable() {
        assert_eq!(detect_rotatable_bonds(&chain(2)).len(), 0);
    }

    #[test]
    fn test_branched_molecule() {
        //   B0' - B0 - C - B1 - B1'
        //              |
        //              B2 - B2'
        let mut g = Atomistic::new();
        let center = g.add_atom(Atom::new());
        let b0 = g.add_atom(Atom::new());
        let b0p = g.add_atom(Atom::new());
        let b1 = g.add_atom(Atom::new());
        let b1p = g.add_atom(Atom::new());
        let b2 = g.add_atom(Atom::new());
        let b2p = g.add_atom(Atom::new());

        g.add_bond(center, b0).expect("add bond");
        g.add_bond(b0, b0p).expect("add bond");
        g.add_bond(center, b1).expect("add bond");
        g.add_bond(b1, b1p).expect("add bond");
        g.add_bond(center, b2).expect("add bond");
        g.add_bond(b2, b2p).expect("add bond");

        assert_eq!(detect_rotatable_bonds(&g).len(), 3);
    }
}
