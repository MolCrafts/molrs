//! Hydrogen-bond network topology.
//!
//! Ported from TRAVIS aggregation topology (`src/aggrtopo.cpp` — the connected
//! aggregate / cluster assembly over the per-step bond graph). molrs assembles
//! the undirected graph with the native [`Topology`] connectivity
//! (connected-components BFS) — **no petgraph** (petgraph stays confined to the
//! `smiles` feature). Nodes are caller-defined (typically one per molecule); each
//! H-bond contributes a donor-node ↔ acceptor-node edge.

use crate::core::system::topology::Topology;

/// Connected-component summary of one frame's hydrogen-bond graph.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NetworkResult {
    /// Component sizes (node counts), sorted descending.
    pub component_sizes: Vec<usize>,
    /// Number of connected components (including isolated single nodes).
    pub num_components: usize,
}

/// Assemble the H-bond network over `n_nodes` nodes and the given undirected
/// `edges` (node-index pairs), returning component sizes via [`Topology`].
///
/// Self-loops (`a == b`) and out-of-range endpoints are ignored. Isolated nodes
/// each count as a component of size 1.
pub fn hbond_components(n_nodes: usize, edges: &[(usize, usize)]) -> NetworkResult {
    let mut topo = Topology::with_atoms(n_nodes);
    for &(a, b) in edges {
        if a != b && a < n_nodes && b < n_nodes {
            topo.add_bond(a, b);
        }
    }
    let labels = topo.connected_components();
    // `connected_components` labels every node 0..n with a per-component id
    // (isolated nodes get their own). Histogram the labels into sizes.
    let n_labels = labels
        .iter()
        .copied()
        .max()
        .map(|m| (m + 1) as usize)
        .unwrap_or(0);
    let mut sizes = vec![0usize; n_labels];
    for &l in &labels {
        sizes[l as usize] += 1;
    }
    sizes.sort_unstable_by(|a, b| b.cmp(a));
    NetworkResult {
        num_components: sizes.len(),
        component_sizes: sizes,
    }
}
