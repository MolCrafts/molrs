//! Rotatable bond detection and downstream atom BFS.
//!
//! Provides utilities to identify rotatable bonds in a [`MolGraph`] and
//! compute the set of atoms downstream of a given bond (for torsion rotation).

use std::collections::{HashSet, VecDeque};

use super::molgraph::{AtomId, MolGraph};

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

/// Detect rotatable bonds in a molecular graph.
///
/// A bond is considered rotatable if:
/// - It is a single bond (order == 1.0)
/// - It is acyclic (removing it disconnects the graph)
/// - Both endpoints have degree > 1 (non-terminal)
///
/// Returns a list of `(AtomId, AtomId)` pairs representing rotatable bonds.
pub fn detect_rotatable_bonds(graph: &MolGraph) -> Vec<(AtomId, AtomId)> {
    let ring_bonds = find_ring_bonds(graph);

    graph
        .bonds()
        .filter_map(|(bid, bond)| {
            let a = bond.atoms[0];
            let b = bond.atoms[1];

            // Single bond only (order defaults to 1.0 if unset)
            let order = bond
                .props
                .get("order")
                .and_then(|v| {
                    if let super::molgraph::PropValue::F64(f) = v {
                        Some(*f)
                    } else {
                        None
                    }
                })
                .unwrap_or(1.0);
            if (order - 1.0).abs() > 0.01 {
                return None;
            }

            // Non-terminal: both endpoints must have degree > 1
            if graph.neighbors(a).count() <= 1 || graph.neighbors(b).count() <= 1 {
                return None;
            }

            // Acyclic: not in a ring
            if ring_bonds.contains(&bid) {
                return None;
            }

            Some((a, b))
        })
        .collect()
}

/// BFS to find all atoms downstream of bond (j, k) on the k-side.
///
/// Starting from `k`, traverses all connected atoms without crossing back
/// through `j`. Returns positional indices (not `AtomId`s).
pub fn downstream_atoms(
    j: AtomId,
    k: AtomId,
    graph: &MolGraph,
    id_to_idx: &std::collections::HashMap<AtomId, usize>,
) -> Vec<usize> {
    let mut visited = HashSet::new();
    visited.insert(j); // block traversal back through j
    visited.insert(k);

    let mut queue = VecDeque::new();
    queue.push_back(k);

    let mut result = vec![id_to_idx[&k]];

    while let Some(current) = queue.pop_front() {
        for neighbor in graph.neighbors(current) {
            if visited.insert(neighbor) {
                result.push(id_to_idx[&neighbor]);
                queue.push_back(neighbor);
            }
        }
    }

    result
}

/// Build a positional index map from AtomId to 0-based index.
pub fn atom_id_to_index(graph: &MolGraph) -> std::collections::HashMap<AtomId, usize> {
    graph
        .atoms()
        .enumerate()
        .map(|(idx, (id, _))| (id, idx))
        .collect()
}

/// Detect rotatable bonds and return them as [`RotatableBond`] structs
/// with positional indices and downstream atom sets.
pub fn detect_rotatable_bonds_with_downstream(graph: &MolGraph) -> Vec<RotatableBond> {
    let id_to_idx = atom_id_to_index(graph);
    let bond_pairs = detect_rotatable_bonds(graph);

    bond_pairs
        .into_iter()
        .map(|(j_id, k_id)| {
            let downstream = downstream_atoms(j_id, k_id, graph, &id_to_idx);
            RotatableBond {
                j: id_to_idx[&j_id],
                k: id_to_idx[&k_id],
                downstream,
            }
        })
        .collect()
}

// --- Ring detection (simple DFS-based) ---

use super::molgraph::BondId;

/// Find all bonds that are part of a ring (cycle) in the graph.
fn find_ring_bonds(graph: &MolGraph) -> HashSet<BondId> {
    let mut ring_bonds = HashSet::new();

    // Union-Find approach: for each bond, if both endpoints are already
    // connected, the bond closes a ring.
    let atom_ids: Vec<AtomId> = graph.atoms().map(|(id, _)| id).collect();
    let mut parent: std::collections::HashMap<AtomId, AtomId> =
        atom_ids.iter().map(|&id| (id, id)).collect();

    fn find(parent: &mut std::collections::HashMap<AtomId, AtomId>, x: AtomId) -> AtomId {
        let p = parent[&x];
        if p == x {
            return x;
        }
        let root = find(parent, p);
        parent.insert(x, root);
        root
    }

    fn union(parent: &mut std::collections::HashMap<AtomId, AtomId>, a: AtomId, b: AtomId) {
        let ra = find(parent, a);
        let rb = find(parent, b);
        if ra != rb {
            parent.insert(ra, rb);
        }
    }

    // First pass: identify ring bonds
    let mut ring_bond_endpoints: Vec<(AtomId, AtomId)> = Vec::new();
    for (bid, bond) in graph.bonds() {
        let a = bond.atoms[0];
        let b = bond.atoms[1];
        let ra = find(&mut parent, a);
        let rb = find(&mut parent, b);
        if ra == rb {
            // This bond closes a cycle
            ring_bonds.insert(bid);
            ring_bond_endpoints.push((a, b));
        } else {
            union(&mut parent, a, b);
        }
    }

    // The union-find only catches one bond per cycle. For complete ring bond
    // detection, we need to mark ALL bonds in every cycle.
    // For each ring-closing bond, BFS to find the shortest path between
    // endpoints (not using the closing bond), then mark all bonds on the path.
    for (a, b) in ring_bond_endpoints {
        mark_ring_path(graph, a, b, &mut ring_bonds);
    }

    ring_bonds
}

/// BFS from `a` to `b` avoiding the direct a-b bond, marking all bonds
/// on the shortest path as ring bonds.
fn mark_ring_path(graph: &MolGraph, a: AtomId, b: AtomId, ring_bonds: &mut HashSet<BondId>) {
    let mut visited = HashSet::new();
    visited.insert(a);
    let mut queue: VecDeque<(AtomId, Vec<AtomId>)> = VecDeque::new();
    queue.push_back((a, vec![a]));

    while let Some((current, path)) = queue.pop_front() {
        for neighbor in graph.neighbors(current) {
            // Skip the direct a-b edge at the start
            if current == a && neighbor == b && path.len() == 1 {
                continue;
            }
            if visited.insert(neighbor) {
                let mut new_path = path.clone();
                new_path.push(neighbor);
                if neighbor == b {
                    // Found path: mark all bonds on this path
                    for window in new_path.windows(2) {
                        let u = window[0];
                        let v = window[1];
                        // Find the bond between u and v
                        for (bid, bond) in graph.bonds() {
                            if (bond.atoms[0] == u && bond.atoms[1] == v)
                                || (bond.atoms[0] == v && bond.atoms[1] == u)
                            {
                                ring_bonds.insert(bid);
                            }
                        }
                    }
                    return;
                }
                queue.push_back((neighbor, new_path));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::molgraph::Atom;

    /// Build a chain graph (topology only, coords irrelevant for detection).
    fn chain(n: usize) -> MolGraph {
        let mut g = MolGraph::new();
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
        let mut g = MolGraph::new();
        let a = g.add_atom(Atom::new());
        let b = g.add_atom(Atom::new());
        let c = g.add_atom(Atom::new());
        g.add_bond(a, b).expect("add bond");
        g.add_bond(b, c).expect("add bond");
        g.add_bond(c, a).expect("add bond");

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
        let mut g = MolGraph::new();
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
