//! [`Perceive::linear_paths`] — the ordered node path of every linear
//! component of a graph.
//!
//! A connected component is a *path graph* exactly when no node has
//! degree > 2 and it has no cycle (|E| = |V| − 1). Walking from a node of
//! degree ≤ 1 then lists it in O(|V|). This is a **query**: unlike the `find_*`
//! finders of [`Perceive`] it returns node paths, not an annotated graph.

use std::fmt;

use super::builder::Perceive;
use crate::system::molgraph::{MolGraph, NodeId, node_to_u64};

/// Why [`Perceive::linear_paths`] refuses a graph.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LinearPathError {
    /// `node` has more than two `bonds`, so its component branches.
    Branch {
        /// The first node in row order with degree > 2.
        node: NodeId,
        /// Its number of incident `bonds`.
        degree: usize,
    },
    /// `node` lies on a component with no endpoint: a ring.
    Cycle {
        /// The first node in row order left unvisited by the walks.
        node: NodeId,
    },
}

impl fmt::Display for LinearPathError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Branch { node, degree } => write!(
                f,
                "node {} has {degree} bonds; a linear path allows at most 2",
                node_to_u64(*node)
            ),
            Self::Cycle { node } => write!(
                f,
                "node {} lies on a cycle, not a linear path",
                node_to_u64(*node)
            ),
        }
    }
}

impl std::error::Error for LinearPathError {}

impl Perceive {
    /// The ordered node path of every connected component of `graph`.
    ///
    /// Only the arity-2 relation kind named `bonds` joins nodes; every other
    /// kind (`ports`, angles, …) is ignored. With no `bonds` kind, every node
    /// is its own path. An isolated node is a length-1 path.
    ///
    /// A node's *degree* is the number of `bonds` relations touching it. A
    /// connected component with node set V and bond set E is accepted when it
    /// is a *path graph* — its nodes can be listed in a line in which each
    /// consecutive pair shares exactly one bond and no other bond exists.
    /// Equivalently: no node has degree > 2 and the component has no cycle
    /// (|E| = |V| − 1). Two parallel bonds between the same two
    /// nodes are a cycle.
    ///
    /// This is a query: `graph` is never mutated.
    ///
    /// # Ordering
    ///
    /// Rows are scanned in node row order. Each unvisited node of degree ≤ 1
    /// starts a walk, and the path is emitted from that endpoint. So the paths
    /// come in the row order of their first endpoint, and each path starts at
    /// its lower-row endpoint.
    ///
    /// # Arguments
    ///
    /// * `graph` — any [`MolGraph`] (a `CoarseGrain` or `Atomistic` derefs to
    ///   one).
    ///
    /// # Returns
    ///
    /// One `Vec<NodeId>` per component, ordered as above; `[]` for an empty
    /// graph.
    ///
    /// # Errors
    ///
    /// Checked in this order:
    ///
    /// 1. [`LinearPathError::Branch`] — the first node in row order with
    ///    degree > 2.
    /// 2. [`LinearPathError::Cycle`] — the first node in row order that no
    ///    walk reached (its component has no endpoint).
    ///
    /// # Complexity
    ///
    /// O(N + E) time and memory, with N the node count and E the number of
    /// `bonds` relations of `graph`: one pass builds the adjacency lists, and
    /// the walks visit every node once.
    ///
    /// # Examples
    ///
    /// ```
    /// use molrs::perceive::Perceive;
    /// use molrs::system::coarsegrain::CoarseGrain;
    ///
    /// let mut cg = CoarseGrain::new();
    /// let a = cg.add_bead_bare("A");
    /// let b = cg.add_bead_bare("A");
    /// let c = cg.add_bead_bare("A");
    /// cg.add_bond(c, b).unwrap();
    /// cg.add_bond(b, a).unwrap();
    ///
    /// let paths = Perceive::new().linear_paths(&cg).unwrap();
    /// assert_eq!(paths, vec![vec![a, b, c]]);
    /// ```
    pub fn linear_paths(&self, graph: &MolGraph) -> Result<Vec<Vec<NodeId>>, LinearPathError> {
        let ids: Vec<NodeId> = graph.node_ids().collect();
        let table = graph.node_table();

        let bonds = graph
            .kind_id("bonds")
            .filter(|&kind| graph.arity(kind) == 2);
        let mut adj: Vec<Vec<usize>> = vec![Vec::new(); ids.len()];
        if let Some(kind) = bonds {
            for rid in graph.relation_ids(kind) {
                // A live relation of a registered kind always has endpoints,
                // and every endpoint is a live node with a row.
                let Ok(ends) = graph.relation_nodes(kind, rid) else {
                    continue;
                };
                let (Some(i), Some(j)) = (table.row(ends[0]), table.row(ends[1])) else {
                    continue;
                };
                adj[i].push(j);
                adj[j].push(i);
            }
        }

        if let Some(row) = adj.iter().position(|nbrs| nbrs.len() > 2) {
            return Err(LinearPathError::Branch {
                node: ids[row],
                degree: adj[row].len(),
            });
        }

        let mut visited = vec![false; ids.len()];
        let mut paths = Vec::new();
        for start in 0..ids.len() {
            if visited[start] || adj[start].len() > 1 {
                continue;
            }
            let mut path = vec![ids[start]];
            visited[start] = true;
            let mut cur = start;
            while let Some(&next) = adj[cur].iter().find(|&&n| !visited[n]) {
                visited[next] = true;
                path.push(ids[next]);
                cur = next;
            }
            paths.push(path);
        }

        match visited.iter().position(|&seen| !seen) {
            Some(row) => Err(LinearPathError::Cycle { node: ids[row] }),
            None => Ok(paths),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::system::coarsegrain::{BeadId, CoarseGrain};

    /// A CoarseGrain holding `n` bare beads, in row order.
    fn beads(n: usize) -> (CoarseGrain, Vec<BeadId>) {
        let mut cg = CoarseGrain::new();
        let ids = (0..n).map(|_| cg.add_bead_bare("A")).collect();
        (cg, ids)
    }

    #[test]
    fn empty_graph_has_no_paths() {
        let cg = CoarseGrain::new();
        assert_eq!(Perceive::new().linear_paths(&cg), Ok(vec![]));
    }

    #[test]
    fn one_node_is_a_length_one_path() {
        let (cg, n) = beads(1);
        assert_eq!(Perceive::new().linear_paths(&cg), Ok(vec![vec![n[0]]]));
    }

    #[test]
    fn walk_starts_at_the_first_endpoint_in_row_order() {
        let (mut cg, n) = beads(3);
        let (a, b, c) = (n[0], n[1], n[2]);
        cg.add_bond(c, b).unwrap();
        cg.add_bond(b, a).unwrap();
        assert_eq!(Perceive::new().linear_paths(&cg), Ok(vec![vec![a, b, c]]));
    }

    #[test]
    fn components_come_in_first_endpoint_row_order() {
        // Chain 2-4-0 (endpoints rows 0 and 2), chain 1-5, isolated 3.
        let (mut cg, n) = beads(6);
        cg.add_bond(n[2], n[4]).unwrap();
        cg.add_bond(n[4], n[0]).unwrap();
        cg.add_bond(n[1], n[5]).unwrap();
        assert_eq!(
            Perceive::new().linear_paths(&cg),
            Ok(vec![vec![n[0], n[4], n[2]], vec![n[1], n[5]], vec![n[3]]])
        );
    }

    #[test]
    fn a_three_leaf_star_is_a_branch() {
        let (mut cg, n) = beads(4);
        let centre = n[3];
        for &leaf in &n[..3] {
            cg.add_bond(leaf, centre).unwrap();
        }
        assert_eq!(
            Perceive::new().linear_paths(&cg),
            Err(LinearPathError::Branch {
                node: centre,
                degree: 3
            })
        );
    }

    #[test]
    fn a_triangle_is_a_cycle() {
        let (mut cg, n) = beads(3);
        cg.add_bond(n[0], n[1]).unwrap();
        cg.add_bond(n[1], n[2]).unwrap();
        cg.add_bond(n[2], n[0]).unwrap();
        assert_eq!(
            Perceive::new().linear_paths(&cg),
            Err(LinearPathError::Cycle { node: n[0] })
        );
    }

    #[test]
    fn two_parallel_bonds_are_a_cycle() {
        let (mut cg, n) = beads(3);
        cg.add_bond(n[1], n[2]).unwrap();
        cg.add_bond(n[2], n[1]).unwrap();
        assert_eq!(
            Perceive::new().linear_paths(&cg),
            Err(LinearPathError::Cycle { node: n[1] })
        );
    }

    #[test]
    fn without_a_bonds_kind_every_node_is_its_own_path() {
        let mut graph = MolGraph::new();
        let a = graph.add_node();
        let b = graph.add_node();
        assert_eq!(
            Perceive::new().linear_paths(&graph),
            Ok(vec![vec![a], vec![b]])
        );
    }

    #[test]
    fn a_ports_relation_joins_nothing() {
        let (mut cg, n) = beads(2);
        let ports = cg.register_kind("ports", 2);
        cg.add_relation(ports, &[n[0], n[1]]).unwrap();
        assert_eq!(
            Perceive::new().linear_paths(&cg),
            Ok(vec![vec![n[0]], vec![n[1]]])
        );
    }
}
