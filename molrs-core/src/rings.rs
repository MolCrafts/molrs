//! Ring detection for molecular graphs.
//!
//! Computes the **Smallest Set of Smallest Rings (SSSR)** — equivalently the
//! minimum cycle basis — using a Horton-style algorithm with petgraph BFS.
//!
//! # Algorithm
//! 1. Build an undirected `petgraph::UnGraph` from the molecular bonds.
//! 2. For each edge `(u, v)`, temporarily remove it and BFS from `u` to `v`.
//!    If a path exists, the path + the removed edge = a candidate cycle.
//! 3. Sort candidates by length (ascending).
//! 4. Greedily select linearly independent cycles via Gaussian elimination
//!    over GF(2) on edge-incidence bit vectors.
//! 5. Result = minimum cycle basis = SSSR.
//!
//! # Complexity
//! O(E × (V + E)) for candidate generation, O(R × E²) for independence check.
//! For typical molecular graphs (small, sparse) this is fast.

use std::collections::{HashMap, HashSet, VecDeque};

use petgraph::graph::{NodeIndex, UnGraph};
use petgraph::visit::EdgeRef;

use super::molgraph::{AtomId, BondId, MolGraph};

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// All ring information for a [`MolGraph`], produced by [`find_rings`].
#[derive(Debug, Clone)]
pub struct RingInfo {
    /// Each ring is an ordered list of `AtomId`s forming a closed path.
    rings: Vec<Vec<AtomId>>,
    /// atom → indices of rings that contain it.
    atom_rings: HashMap<AtomId, Vec<usize>>,
    /// bond → indices of rings that contain it.
    bond_rings: HashMap<BondId, Vec<usize>>,
}

impl RingInfo {
    fn empty() -> Self {
        Self {
            rings: Vec::new(),
            atom_rings: HashMap::new(),
            bond_rings: HashMap::new(),
        }
    }

    /// Whether the atom belongs to any ring.
    pub fn is_atom_in_ring(&self, id: AtomId) -> bool {
        self.atom_rings.get(&id).is_some_and(|v| !v.is_empty())
    }

    /// Whether the bond belongs to any ring.
    pub fn is_bond_in_ring(&self, id: BondId) -> bool {
        self.bond_rings.get(&id).is_some_and(|v| !v.is_empty())
    }

    /// Number of rings containing this atom.
    pub fn num_atom_rings(&self, id: AtomId) -> usize {
        self.atom_rings.get(&id).map_or(0, Vec::len)
    }

    /// Number of rings containing this bond.
    pub fn num_bond_rings(&self, id: BondId) -> usize {
        self.bond_rings.get(&id).map_or(0, Vec::len)
    }

    /// Size (atom count) of every ring, in ascending order.
    pub fn ring_sizes(&self) -> Vec<usize> {
        self.rings.iter().map(Vec::len).collect()
    }

    /// All rings of exactly `n` atoms.
    pub fn rings_of_size(&self, n: usize) -> Vec<&Vec<AtomId>> {
        self.rings.iter().filter(|r| r.len() == n).collect()
    }

    /// Size of the smallest ring containing `id`, if any.
    pub fn smallest_ring_containing_atom(&self, id: AtomId) -> Option<usize> {
        self.atom_rings
            .get(&id)?
            .iter()
            .map(|&ri| self.rings[ri].len())
            .min()
    }

    /// Total number of rings detected.
    pub fn num_rings(&self) -> usize {
        self.rings.len()
    }

    /// All rings as slices of `AtomId`.
    pub fn rings(&self) -> &[Vec<AtomId>] {
        &self.rings
    }
}

// ---------------------------------------------------------------------------
// Public entry point
// ---------------------------------------------------------------------------

/// Compute the ring information (SSSR / minimum cycle basis) for `mol`.
pub fn find_rings(mol: &MolGraph) -> RingInfo {
    if mol.n_atoms() == 0 {
        return RingInfo::empty();
    }

    // ---- 1. Build stable atom/bond orderings --------------------------------
    let atom_vec: Vec<AtomId> = mol.atoms().map(|(id, _)| id).collect();
    let atom_to_idx: HashMap<AtomId, usize> = atom_vec
        .iter()
        .enumerate()
        .map(|(i, &id)| (id, i))
        .collect();

    let bond_vec: Vec<BondId> = mol.bonds().map(|(id, _)| id).collect();

    // ---- 2. Build petgraph::UnGraph -----------------------------------------
    let mut graph = UnGraph::<(), ()>::new_undirected();
    for _ in &atom_vec {
        graph.add_node(());
    }

    // Map from graph edge index (usize) → bond_vec index
    let mut edge_to_bond: Vec<usize> = Vec::with_capacity(bond_vec.len());
    for (bi, &bid) in bond_vec.iter().enumerate() {
        let b = mol.get_bond(bid).expect("bond must exist");
        let u = atom_to_idx[&b.atoms[0]];
        let v = atom_to_idx[&b.atoms[1]];
        graph.add_edge(NodeIndex::new(u), NodeIndex::new(v), ());
        edge_to_bond.push(bi);
    }

    let n_edges = graph.edge_count();

    // ---- 3. Expected cycle basis size = E - V + C ---------------------------
    // (C = number of connected components)
    let n_components = petgraph::algo::connected_components(&graph);
    let cycle_rank = n_edges as isize - graph.node_count() as isize + n_components as isize;
    if cycle_rank <= 0 {
        return RingInfo::empty();
    }
    let cycle_rank = cycle_rank as usize;

    // ---- 4. Generate candidate cycles (Horton-style) ------------------------
    // For each edge, remove it, BFS for shortest path, reconstruct cycle.
    let mut candidates: Vec<Vec<usize>> = Vec::new(); // each is a list of node indices

    for edge_idx in graph.edge_indices() {
        let (u, v) = graph.edge_endpoints(edge_idx).unwrap();

        // BFS from u to v in graph, skipping the direct edge between u and v
        if let Some(path) = bfs_shortest_path(&graph, u, v, edge_idx) {
            // path is [u, ..., v], which forms a cycle with the edge u-v
            candidates.push(path);
        }
    }

    // Sort by cycle length (shortest first)
    candidates.sort_by_key(|c| c.len());

    // ---- 5. Select linearly independent cycles (GF(2) Gaussian elimination) -
    // Represent each cycle as a bit vector over edges.
    // Build a lookup: (min_node, max_node) → edge index for quick edge lookup
    let mut edge_lookup: HashMap<(usize, usize), usize> = HashMap::new();
    for edge_idx in graph.edge_indices() {
        let (a, b) = graph.edge_endpoints(edge_idx).unwrap();
        let key = if a.index() < b.index() {
            (a.index(), b.index())
        } else {
            (b.index(), a.index())
        };
        edge_lookup.insert(key, edge_idx.index());
    }

    let mut basis_vectors: Vec<Vec<u64>> = Vec::new();
    let words = n_edges.div_ceil(64);
    let mut selected_cycles: Vec<Vec<usize>> = Vec::new();

    for cycle in &candidates {
        if selected_cycles.len() >= cycle_rank {
            break;
        }

        // Build edge-incidence bit vector for this cycle
        let mut bitvec = vec![0u64; words];
        let n = cycle.len();
        for i in 0..n {
            let a = cycle[i];
            let b = cycle[(i + 1) % n];
            let key = if a < b { (a, b) } else { (b, a) };
            if let Some(&ei) = edge_lookup.get(&key) {
                bitvec[ei / 64] |= 1u64 << (ei % 64);
            }
        }

        // Check linear independence via Gaussian elimination
        if is_linearly_independent(&mut basis_vectors, bitvec, words) {
            selected_cycles.push(cycle.clone());
        }
    }

    // ---- 6. Convert node-index cycles → AtomId rings ------------------------
    let mut raw_rings: Vec<Vec<AtomId>> = selected_cycles
        .into_iter()
        .map(|cycle| cycle.iter().map(|&ni| atom_vec[ni]).collect())
        .collect();

    // Sort smallest first (SSSR convention).
    raw_rings.sort_by_key(Vec::len);

    // ---- 7. Build reverse-lookup maps ---------------------------------------
    // Fast (AtomId, AtomId) → BondId lookup table.
    let mut bond_map: HashMap<(AtomId, AtomId), BondId> = HashMap::new();
    for &bid in &bond_vec {
        let b = mol.get_bond(bid).expect("bond must exist");
        let [a, bb] = b.atoms;
        bond_map.insert((a, bb), bid);
        bond_map.insert((bb, a), bid);
    }

    let mut atom_rings: HashMap<AtomId, Vec<usize>> = HashMap::new();
    let mut bond_rings: HashMap<BondId, Vec<usize>> = HashMap::new();

    for (ri, ring) in raw_rings.iter().enumerate() {
        let n = ring.len();
        for i in 0..n {
            let a = ring[i];
            let b = ring[(i + 1) % n];
            atom_rings.entry(a).or_default().push(ri);
            if let Some(&bid) = bond_map.get(&(a, b)) {
                bond_rings.entry(bid).or_default().push(ri);
            }
        }
    }

    RingInfo {
        rings: raw_rings,
        atom_rings,
        bond_rings,
    }
}

// ---------------------------------------------------------------------------
// BFS shortest path avoiding a specific edge
// ---------------------------------------------------------------------------

/// BFS from `start` to `goal` in `graph`, but do not traverse edge `skip_edge`.
/// Returns the node-index path `[start, ..., goal]` if reachable.
fn bfs_shortest_path(
    graph: &UnGraph<(), ()>,
    start: NodeIndex,
    goal: NodeIndex,
    skip_edge: petgraph::graph::EdgeIndex,
) -> Option<Vec<usize>> {
    let mut visited = HashSet::new();
    let mut parent: HashMap<NodeIndex, NodeIndex> = HashMap::new();
    let mut queue = VecDeque::new();

    visited.insert(start);
    queue.push_back(start);

    while let Some(current) = queue.pop_front() {
        if current == goal {
            // Reconstruct path
            let mut path = vec![goal.index()];
            let mut node = goal;
            while node != start {
                node = parent[&node];
                path.push(node.index());
            }
            path.reverse();
            return Some(path);
        }

        // Walk edges manually to check edge identity
        let edges = graph.edges(current);
        for edge_ref in edges {
            if edge_ref.id() == skip_edge {
                continue;
            }
            let neighbor = edge_ref.target();
            // In petgraph's undirected graph, edges() can return the source
            // as target, so pick the other endpoint.
            let neighbor = if neighbor == current {
                edge_ref.source()
            } else {
                neighbor
            };
            if visited.insert(neighbor) {
                parent.insert(neighbor, current);
                queue.push_back(neighbor);
            }
        }
    }

    None
}

// ---------------------------------------------------------------------------
// GF(2) linear independence check
// ---------------------------------------------------------------------------

/// Try to add `vec` to `basis`. If `vec` is linearly independent of the
/// existing basis vectors (over GF(2)), add it and return `true`.
/// Otherwise return `false`.
fn is_linearly_independent(basis: &mut Vec<Vec<u64>>, mut vec: Vec<u64>, words: usize) -> bool {
    // Reduce vec against existing basis
    for basis_vec in basis.iter() {
        // Find the leading bit of basis_vec
        let lead = leading_bit(basis_vec, words);
        if let Some(lead) = lead
            && (vec[lead / 64] >> (lead % 64)) & 1 == 1
        {
            // XOR to eliminate
            for w in 0..words {
                vec[w] ^= basis_vec[w];
            }
        }
    }

    // If vec is non-zero, it's linearly independent
    let is_nonzero = vec.iter().any(|&w| w != 0);
    if is_nonzero {
        basis.push(vec);
    }
    is_nonzero
}

/// Find the position of the highest set bit in the bitvector.
fn leading_bit(vec: &[u64], words: usize) -> Option<usize> {
    for w in (0..words).rev() {
        if vec[w] != 0 {
            return Some(w * 64 + (63 - vec[w].leading_zeros() as usize));
        }
    }
    None
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::molgraph::{Atom, MolGraph};

    fn cycle(n: usize) -> MolGraph {
        let mut g = MolGraph::new();
        let ids: Vec<AtomId> = (0..n).map(|_| g.add_atom(Atom::new())).collect();
        for i in 0..n {
            g.add_bond(ids[i], ids[(i + 1) % n])
                .expect("add cycle bond");
        }
        g
    }

    #[test]
    fn test_single_6ring() {
        let g = cycle(6);
        let ri = find_rings(&g);
        assert_eq!(ri.num_rings(), 1);
        assert_eq!(ri.ring_sizes(), vec![6]);
    }

    #[test]
    fn test_linear_no_rings() {
        let mut g = MolGraph::new();
        let ids: Vec<AtomId> = (0..6).map(|_| g.add_atom(Atom::new())).collect();
        for i in 0..5 {
            g.add_bond(ids[i], ids[i + 1]).expect("add chain bond");
        }
        assert_eq!(find_rings(&g).num_rings(), 0);
    }

    #[test]
    fn test_all_atoms_in_6ring() {
        let g = cycle(6);
        let ri = find_rings(&g);
        for (id, _) in g.atoms() {
            assert!(ri.is_atom_in_ring(id));
        }
    }

    #[test]
    fn test_all_bonds_in_6ring() {
        let g = cycle(6);
        let ri = find_rings(&g);
        for (bid, _) in g.bonds() {
            assert!(ri.is_bond_in_ring(bid));
        }
    }

    #[test]
    fn test_empty_mol() {
        assert_eq!(find_rings(&MolGraph::new()).num_rings(), 0);
    }

    #[test]
    fn test_naphthalene() {
        let mut g = MolGraph::new();
        let ids: Vec<AtomId> = (0..10).map(|_| g.add_atom(Atom::new())).collect();
        // Ring A: 0-1-2-3-4-5-0
        for i in 0..5 {
            g.add_bond(ids[i], ids[i + 1]).expect("bond");
        }
        g.add_bond(ids[5], ids[0]).expect("bond");
        // Ring B: 2-3-6-7-8-9-2
        g.add_bond(ids[3], ids[6]).expect("bond");
        g.add_bond(ids[6], ids[7]).expect("bond");
        g.add_bond(ids[7], ids[8]).expect("bond");
        g.add_bond(ids[8], ids[9]).expect("bond");
        g.add_bond(ids[9], ids[2]).expect("bond");

        let ri = find_rings(&g);
        assert_eq!(ri.num_rings(), 2);
        let mut sizes = ri.ring_sizes();
        sizes.sort_unstable();
        assert_eq!(sizes, vec![6, 6]);
    }
}
