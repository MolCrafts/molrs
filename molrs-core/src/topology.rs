//! Graph-based molecular topology using petgraph.
//!
//! Provides graph-based representation of molecular connectivity with
//! automated detection of angles, dihedrals, and impropers via neighbor
//! traversal.

use petgraph::graph::{EdgeIndex, NodeIndex, UnGraph};

/// Graph-based molecular topology.
///
/// Wraps an undirected `petgraph::UnGraph<(), ()>` where vertices are atoms
/// and edges are bonds. Angles, dihedrals, and impropers are detected
/// automatically from bond connectivity using neighbor traversal.
pub struct Topology {
    graph: UnGraph<(), ()>,
}

impl Topology {
    /// Create an empty topology with no atoms or bonds.
    pub fn new() -> Self {
        Self {
            graph: UnGraph::new_undirected(),
        }
    }

    /// Create a topology with `n` atoms and no bonds.
    pub fn with_atoms(n: usize) -> Self {
        let mut graph = UnGraph::new_undirected();
        for _ in 0..n {
            graph.add_node(());
        }
        Self { graph }
    }

    /// Create a topology from edge pairs.
    pub fn from_edges(n_atoms: usize, edges: &[[usize; 2]]) -> Self {
        let mut graph = UnGraph::new_undirected();
        for _ in 0..n_atoms {
            graph.add_node(());
        }
        for e in edges {
            graph.add_edge(NodeIndex::new(e[0]), NodeIndex::new(e[1]), ());
        }
        Self { graph }
    }

    // -----------------------------------------------------------------------
    // Count accessors
    // -----------------------------------------------------------------------

    /// Number of atoms (vertices).
    pub fn n_atoms(&self) -> usize {
        self.graph.node_count()
    }

    /// Number of bonds (edges).
    pub fn n_bonds(&self) -> usize {
        self.graph.edge_count()
    }

    /// Number of unique angles (i-j-k triplets).
    pub fn n_angles(&self) -> usize {
        self.angles().len()
    }

    /// Number of unique proper dihedrals (i-j-k-l quartets).
    pub fn n_dihedrals(&self) -> usize {
        self.dihedrals().len()
    }

    // -----------------------------------------------------------------------
    // List accessors
    // -----------------------------------------------------------------------

    /// All atom indices.
    pub fn atoms(&self) -> Vec<usize> {
        self.graph.node_indices().map(|n| n.index()).collect()
    }

    /// All bond pairs as `[i, j]`.
    pub fn bonds(&self) -> Vec<[usize; 2]> {
        self.graph
            .edge_indices()
            .map(|e| {
                let (a, b) = self.graph.edge_endpoints(e).unwrap();
                [a.index(), b.index()]
            })
            .collect()
    }

    /// All unique angle triplets `[i, j, k]`, deduplicated (i < k).
    pub fn angles(&self) -> Vec<[usize; 3]> {
        let mut result = Vec::new();
        for j in self.graph.node_indices() {
            let neighbors: Vec<NodeIndex> = self.graph.neighbors(j).collect();
            for a in 0..neighbors.len() {
                for b in (a + 1)..neighbors.len() {
                    let i = neighbors[a].index();
                    let k = neighbors[b].index();
                    if i < k {
                        result.push([i, j.index(), k]);
                    } else {
                        result.push([k, j.index(), i]);
                    }
                }
            }
        }
        result
    }

    /// All unique proper dihedral quartets `[i, j, k, l]`, deduplicated (j < k).
    pub fn dihedrals(&self) -> Vec<[usize; 4]> {
        let mut result = Vec::new();
        for edge in self.graph.edge_indices() {
            let (a, b) = self.graph.edge_endpoints(edge).unwrap();
            // Canonical ordering: j < k
            let (j, k) = if a.index() < b.index() {
                (a, b)
            } else {
                (b, a)
            };

            let j_neighbors: Vec<NodeIndex> = self.graph.neighbors(j).filter(|&n| n != k).collect();
            let k_neighbors: Vec<NodeIndex> = self.graph.neighbors(k).filter(|&n| n != j).collect();

            for &i in &j_neighbors {
                for &l in &k_neighbors {
                    if i != l {
                        result.push([i.index(), j.index(), k.index(), l.index()]);
                    }
                }
            }
        }
        result
    }

    /// All unique improper dihedral quartets `[center, i, j, k]`, deduplicated.
    ///
    /// For each atom with degree >= 3, iterate all sorted 3-combinations
    /// of its neighbors.
    pub fn impropers(&self) -> Vec<[usize; 4]> {
        let mut result = Vec::new();
        for center in self.graph.node_indices() {
            let mut neighbors: Vec<usize> =
                self.graph.neighbors(center).map(|n| n.index()).collect();
            if neighbors.len() < 3 {
                continue;
            }
            neighbors.sort_unstable();
            let n = neighbors.len();
            for a in 0..n {
                for b in (a + 1)..n {
                    for c in (b + 1)..n {
                        result.push([center.index(), neighbors[a], neighbors[b], neighbors[c]]);
                    }
                }
            }
        }
        result
    }

    // -----------------------------------------------------------------------
    // Atom operations
    // -----------------------------------------------------------------------

    /// Add a single atom.
    pub fn add_atom(&mut self) {
        self.graph.add_node(());
    }

    /// Add `n` atoms.
    pub fn add_atoms(&mut self, n: usize) {
        for _ in 0..n {
            self.graph.add_node(());
        }
    }

    /// Delete an atom by index.
    ///
    /// Note: petgraph swaps the last node into the removed node's slot,
    /// so indices of other nodes may change.
    pub fn delete_atom(&mut self, idx: usize) {
        self.graph.remove_node(NodeIndex::new(idx));
    }

    // -----------------------------------------------------------------------
    // Bond operations
    // -----------------------------------------------------------------------

    /// Add a bond between atoms `i` and `j` if not already connected.
    pub fn add_bond(&mut self, i: usize, j: usize) {
        let ni = NodeIndex::new(i);
        let nj = NodeIndex::new(j);
        if self.graph.find_edge(ni, nj).is_none() {
            self.graph.add_edge(ni, nj, ());
        }
    }

    /// Add multiple bonds from pairs. Skips duplicates.
    pub fn add_bonds(&mut self, pairs: &[[usize; 2]]) {
        for pair in pairs {
            self.add_bond(pair[0], pair[1]);
        }
    }

    /// Delete a bond by edge index.
    pub fn delete_bond(&mut self, idx: usize) {
        self.graph.remove_edge(EdgeIndex::new(idx));
    }

    // -----------------------------------------------------------------------
    // Angle helpers
    // -----------------------------------------------------------------------

    /// Add an angle by ensuring bonds i-j and j-k exist.
    pub fn add_angle(&mut self, i: usize, j: usize, k: usize) {
        self.add_bond(i, j);
        self.add_bond(j, k);
    }

    /// Add multiple angles from triplets, ensuring all required bonds exist.
    pub fn add_angles(&mut self, triplets: &[[usize; 3]]) {
        for t in triplets {
            self.add_angle(t[0], t[1], t[2]);
        }
    }
}

impl Default for Topology {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_topology() {
        let topo = Topology::new();
        assert_eq!(topo.n_atoms(), 0);
        assert_eq!(topo.n_bonds(), 0);
    }

    #[test]
    fn test_with_atoms() {
        let topo = Topology::with_atoms(5);
        assert_eq!(topo.n_atoms(), 5);
        assert_eq!(topo.n_bonds(), 0);
    }

    #[test]
    fn test_add_atom() {
        let mut topo = Topology::new();
        topo.add_atom();
        assert_eq!(topo.n_atoms(), 1);
        topo.add_atoms(3);
        assert_eq!(topo.n_atoms(), 4);
    }

    #[test]
    fn test_delete_atom() {
        let mut topo = Topology::with_atoms(3);
        topo.delete_atom(1);
        assert_eq!(topo.n_atoms(), 2);
    }

    #[test]
    fn test_add_bond() {
        let mut topo = Topology::with_atoms(3);
        topo.add_bond(0, 1);
        assert_eq!(topo.n_bonds(), 1);

        // Adding same bond again should not create duplicate
        topo.add_bond(0, 1);
        assert_eq!(topo.n_bonds(), 1);
    }

    #[test]
    fn test_add_bonds() {
        let mut topo = Topology::with_atoms(4);
        topo.add_bonds(&[[0, 1], [1, 2], [2, 3]]);
        assert_eq!(topo.n_bonds(), 3);
    }

    #[test]
    fn test_delete_bond() {
        let mut topo = Topology::with_atoms(3);
        topo.add_bond(0, 1);
        topo.add_bond(1, 2);
        assert_eq!(topo.n_bonds(), 2);
        topo.delete_bond(0);
        assert_eq!(topo.n_bonds(), 1);
    }

    #[test]
    fn test_bonds_list() {
        let mut topo = Topology::with_atoms(3);
        topo.add_bond(0, 1);
        topo.add_bond(1, 2);
        let bonds = topo.bonds();
        assert_eq!(bonds.len(), 2);
    }

    #[test]
    fn test_angles_3atom_chain() {
        // 0 - 1 - 2
        let mut topo = Topology::with_atoms(3);
        topo.add_bond(0, 1);
        topo.add_bond(1, 2);
        assert_eq!(topo.n_angles(), 1);
        let angles = topo.angles();
        assert_eq!(angles.len(), 1);
        assert_eq!(angles[0], [0, 1, 2]);
    }

    #[test]
    fn test_dihedrals_4atom_chain() {
        // 0 - 1 - 2 - 3
        let mut topo = Topology::with_atoms(4);
        topo.add_bonds(&[[0, 1], [1, 2], [2, 3]]);
        assert_eq!(topo.n_dihedrals(), 1);
        let dihedrals = topo.dihedrals();
        assert_eq!(dihedrals.len(), 1);
        assert_eq!(dihedrals[0], [0, 1, 2, 3]);
    }

    #[test]
    fn test_impropers_star() {
        // Central atom 0 bonded to 1, 2, 3
        let mut topo = Topology::with_atoms(4);
        topo.add_bond(0, 1);
        topo.add_bond(0, 2);
        topo.add_bond(0, 3);
        let impropers = topo.impropers();
        // One unique improper with center 0
        assert_eq!(impropers.len(), 1);
        assert_eq!(impropers[0][0], 0);
    }

    #[test]
    fn test_add_angle_creates_bonds() {
        let mut topo = Topology::with_atoms(3);
        topo.add_angle(0, 1, 2);
        assert_eq!(topo.n_bonds(), 2);
        assert_eq!(topo.n_angles(), 1);
    }

    #[test]
    fn test_methane_ch4() {
        let mut topo = Topology::with_atoms(5);
        topo.add_bond(0, 1);
        topo.add_bond(0, 2);
        topo.add_bond(0, 3);
        topo.add_bond(0, 4);

        assert_eq!(topo.n_atoms(), 5);
        assert_eq!(topo.n_bonds(), 4);
        assert_eq!(topo.n_angles(), 6);
        assert_eq!(topo.n_dihedrals(), 0);
        assert_eq!(topo.impropers().len(), 4);
    }

    #[test]
    fn test_ethane_c2h6() {
        let mut topo = Topology::with_atoms(8);
        topo.add_bond(0, 1);
        topo.add_bond(0, 2);
        topo.add_bond(0, 3);
        topo.add_bond(0, 4);
        topo.add_bond(1, 5);
        topo.add_bond(1, 6);
        topo.add_bond(1, 7);

        assert_eq!(topo.n_atoms(), 8);
        assert_eq!(topo.n_bonds(), 7);
        assert_eq!(topo.n_angles(), 12);
        assert_eq!(topo.n_dihedrals(), 9);
    }

    #[test]
    fn test_from_edges() {
        let topo = Topology::from_edges(4, &[[0, 1], [1, 2], [2, 3]]);
        assert_eq!(topo.n_atoms(), 4);
        assert_eq!(topo.n_bonds(), 3);
        assert_eq!(topo.n_angles(), 2);
        assert_eq!(topo.n_dihedrals(), 1);
    }
}
