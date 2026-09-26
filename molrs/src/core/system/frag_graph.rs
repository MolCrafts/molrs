//! Unit-level assembly topology: [`FragGraph`] nodes name templates and each [`FragEdge`] names the two template port ordinals it joins.
//!
//! A `FragGraph` says **what** to assemble, never how: node `i` names the
//! template of unit `i`, and a [`FragEdge`] joins port ordinal `port_a` of
//! unit `a` to port ordinal `port_b` of unit `b`. An ordinal is an index into
//! the template's [`Fragment::ordered_ports`](super::fragment::Fragment::ordered_ports).
//!
//! A **unit** is one placed copy of a template (an all-atom fragment with
//! numbered attachment points, *ports*); a polymer of 100 monomers is a
//! `FragGraph` of 100 nodes. Construction checks only what the graph itself
//! can see — endpoints in range, no self-edge, no (node, port) on two edges.
//! Whether an ordinal exists in its template is checked by the Assembler
//! (`builder::Assembler`), the first place that holds both the graph and the
//! templates.

use std::collections::HashSet;

use crate::error::MolRsError;

/// One unit-level bond: port ordinal `port_a` of node `a` joins port ordinal
/// `port_b` of node `b`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FragEdge {
    /// First endpoint node index.
    pub a: usize,
    /// Second endpoint node index.
    pub b: usize,
    /// Port ordinal on node `a`.
    pub port_a: usize,
    /// Port ordinal on node `b`.
    pub port_b: usize,
}

/// Unit-level assembly topology: template names on the nodes, port-ordinal
/// pairs on the edges.
///
/// Invariants, established by [`new`](Self::new) and held by every
/// constructor: each edge endpoint is a node index, no edge joins a node to
/// itself, and each (node, port ordinal) appears on at most one edge.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FragGraph {
    nodes: Vec<String>,
    edges: Vec<FragEdge>,
}

impl FragGraph {
    /// Build a graph from template names and edges, kept in the given order.
    ///
    /// # Errors
    ///
    /// Returns [`MolRsError::Validation`] when an edge endpoint is not a node
    /// index, when an edge joins a node to itself, or when one (node, port
    /// ordinal) appears on two edges — on either endpoint side.
    pub fn new(nodes: Vec<String>, edges: Vec<FragEdge>) -> Result<Self, MolRsError> {
        let n = nodes.len();
        let mut used: HashSet<(usize, usize)> = HashSet::with_capacity(2 * edges.len());
        for (k, e) in edges.iter().enumerate() {
            for node in [e.a, e.b] {
                if node >= n {
                    return Err(MolRsError::validation(format!(
                        "frag-graph edge {k} names node {node}, but the graph has {n} nodes"
                    )));
                }
            }
            if e.a == e.b {
                return Err(MolRsError::validation(format!(
                    "frag-graph edge {k} joins node {} to itself",
                    e.a
                )));
            }
            for (node, port) in [(e.a, e.port_a), (e.b, e.port_b)] {
                if !used.insert((node, port)) {
                    return Err(MolRsError::validation(format!(
                        "frag-graph edge {k} reuses port ordinal {port} of node {node}, \
                         already joined by an earlier edge"
                    )));
                }
            }
        }
        Ok(Self { nodes, edges })
    }

    /// A linear chain: edge `(i, i + 1, link.0, link.1)` for each consecutive
    /// pair, so port ordinal `link.0` of each unit joins port ordinal
    /// `link.1` of the next.
    ///
    /// # Errors
    ///
    /// As [`new`](Self::new); with three or more nodes, `link.0 == link.1`
    /// puts one port of each interior unit on two edges and is refused.
    pub fn path(nodes: Vec<String>, link: (usize, usize)) -> Result<Self, MolRsError> {
        let edges = Self::chain_edges(nodes.len(), link);
        Self::new(nodes, edges)
    }

    /// A ring: the [`path`](Self::path) edges plus the closing edge
    /// `(n − 1, 0, link.0, link.1)`.
    ///
    /// # Errors
    ///
    /// Returns [`MolRsError::Validation`] for fewer than three nodes (two
    /// nodes would need two edges between one pair), and otherwise as
    /// [`new`](Self::new).
    pub fn cycle(nodes: Vec<String>, link: (usize, usize)) -> Result<Self, MolRsError> {
        let n = nodes.len();
        if n < 3 {
            return Err(MolRsError::validation(format!(
                "a frag-graph cycle needs at least 3 nodes, got {n}"
            )));
        }
        let mut edges = Self::chain_edges(n, link);
        edges.push(FragEdge {
            a: n - 1,
            b: 0,
            port_a: link.0,
            port_b: link.1,
        });
        Self::new(nodes, edges)
    }

    /// A star: node 0 is `center`, nodes `1..=k` are `k = center_ports.len()`
    /// copies of `arm`, and edge `(0, i + 1, center_ports[i], arm_port)`
    /// joins arm `i`.
    ///
    /// # Errors
    ///
    /// As [`new`](Self::new); a repeated entry in `center_ports` is refused.
    pub fn star(
        center: String,
        center_ports: Vec<usize>,
        arm: String,
        arm_port: usize,
    ) -> Result<Self, MolRsError> {
        let k = center_ports.len();
        let mut nodes = Vec::with_capacity(k + 1);
        nodes.push(center);
        nodes.extend(std::iter::repeat_n(arm, k));
        let edges = center_ports
            .into_iter()
            .enumerate()
            .map(|(i, port_a)| FragEdge {
                a: 0,
                b: i + 1,
                port_a,
                port_b: arm_port,
            })
            .collect();
        Self::new(nodes, edges)
    }

    /// Template name per node; node `i` is unit `i`.
    pub fn nodes(&self) -> &[String] {
        &self.nodes
    }

    /// The edges, in construction order.
    pub fn edges(&self) -> &[FragEdge] {
        &self.edges
    }

    /// Edges `(i, i + 1, link.0, link.1)` for `i` in `0..n − 1`.
    fn chain_edges(n: usize, link: (usize, usize)) -> Vec<FragEdge> {
        (1..n)
            .map(|b| FragEdge {
                a: b - 1,
                b,
                port_a: link.0,
                port_b: link.1,
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::{FragEdge, FragGraph};
    use crate::error::MolRsError;

    fn names(n: usize) -> Vec<String> {
        (0..n).map(|i| format!("T{i}")).collect()
    }

    fn edge(a: usize, b: usize, port_a: usize, port_b: usize) -> FragEdge {
        FragEdge {
            a,
            b,
            port_a,
            port_b,
        }
    }

    #[test]
    fn path_links_consecutive_nodes_with_the_link_ports() {
        let g = FragGraph::path(names(3), (1, 0)).expect("a 3-node path is valid");
        assert_eq!(g.nodes(), names(3).as_slice());
        assert_eq!(g.edges(), &[edge(0, 1, 1, 0), edge(1, 2, 1, 0)]);
    }

    #[test]
    fn cycle_adds_the_closing_edge() {
        let g = FragGraph::cycle(names(3), (1, 0)).expect("a 3-node cycle is valid");
        assert_eq!(g.nodes(), names(3).as_slice());
        assert_eq!(
            g.edges(),
            &[edge(0, 1, 1, 0), edge(1, 2, 1, 0), edge(2, 0, 1, 0)]
        );
    }

    #[test]
    fn cycle_refuses_fewer_than_three_nodes() {
        assert!(FragGraph::cycle(names(2), (1, 0)).is_err());
    }

    #[test]
    fn star_joins_center_ports_to_each_arm() {
        let g = FragGraph::star("C".to_owned(), vec![0, 1, 2], "R".to_owned(), 0)
            .expect("a 3-arm star is valid");
        assert_eq!(
            g.nodes(),
            &[
                "C".to_owned(),
                "R".to_owned(),
                "R".to_owned(),
                "R".to_owned()
            ]
        );
        assert_eq!(
            g.edges(),
            &[edge(0, 1, 0, 0), edge(0, 2, 1, 0), edge(0, 3, 2, 0)]
        );
    }

    #[test]
    fn new_accepts_a_valid_edge_list() {
        let g = FragGraph::new(names(2), vec![edge(0, 1, 1, 0)]).expect("valid");
        assert_eq!(g.nodes(), names(2).as_slice());
        assert_eq!(g.edges(), &[edge(0, 1, 1, 0)]);
    }

    #[test]
    fn new_refuses_an_out_of_range_endpoint() {
        let err = FragGraph::new(names(2), vec![edge(0, 2, 1, 0)])
            .expect_err("node 2 does not exist in a 2-node graph");
        assert!(matches!(err, MolRsError::Validation { .. }), "{err:?}");
    }

    #[test]
    fn new_refuses_a_self_edge() {
        let err = FragGraph::new(names(2), vec![edge(1, 1, 0, 1)])
            .expect_err("an edge from a node to itself is refused");
        assert!(matches!(err, MolRsError::Validation { .. }), "{err:?}");
    }

    #[test]
    fn new_refuses_a_node_port_used_twice() {
        // (0, port 1) appears on both edges.
        let err = FragGraph::new(names(3), vec![edge(0, 1, 1, 0), edge(0, 2, 1, 0)])
            .expect_err("one (node, port) cannot join two edges");
        assert!(matches!(err, MolRsError::Validation { .. }), "{err:?}");
    }

    #[test]
    fn new_refuses_a_node_port_used_twice_across_endpoint_sides() {
        // (1, port 0) is b-side of the first edge and a-side of the second.
        let err = FragGraph::new(names(3), vec![edge(0, 1, 1, 0), edge(1, 2, 0, 0)])
            .expect_err("one (node, port) cannot join two edges");
        assert!(matches!(err, MolRsError::Validation { .. }), "{err:?}");
    }
}
