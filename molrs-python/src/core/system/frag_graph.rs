//! Python binding of the unit-level assembly topology
//! (`molrs::core::system::frag_graph`).

use molrs::system::frag_graph::{FragEdge, FragGraph};
use pyo3::prelude::*;

use crate::helpers::molrs_error_to_pyerr;

/// One edge as it crosses: ``(a, b, port_a, port_b)`` — port ordinal
/// ``port_a`` of node ``a`` joins port ordinal ``port_b`` of node ``b``.
type EdgeTuple = (usize, usize, usize, usize);

/// Unit-level assembly topology: node ``i`` names the template of unit ``i``;
/// each edge ``(a, b, port_a, port_b)`` joins two template port ordinals.
///
/// Construction checks only what the graph itself can see: endpoints in
/// range, no self-edge, no (node, port) on two edges.
///
/// Parameters
/// ----------
/// nodes : list[str]
///     Template name per node.
/// edges : list[tuple[int, int, int, int]]
///     ``(a, b, port_a, port_b)`` per edge, kept in the given order.
///
/// Raises
/// ------
/// ValueError
///     If an edge breaks one of the invariants above.
#[pyclass(module = "molrs", name = "FragGraph", frozen, skip_from_py_object)]
pub struct PyFragGraph {
    pub(crate) inner: FragGraph,
}

impl PyFragGraph {
    pub(crate) fn from_core(inner: FragGraph) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PyFragGraph {
    #[new]
    fn new(nodes: Vec<String>, edges: Vec<EdgeTuple>) -> PyResult<Self> {
        let edges = edges
            .into_iter()
            .map(|(a, b, port_a, port_b)| FragEdge {
                a,
                b,
                port_a,
                port_b,
            })
            .collect();
        FragGraph::new(nodes, edges)
            .map(Self::from_core)
            .map_err(molrs_error_to_pyerr)
    }

    /// A linear chain: edge ``(i, i + 1, link[0], link[1])`` for each
    /// consecutive pair of ``nodes``.
    #[staticmethod]
    fn path(nodes: Vec<String>, link: (usize, usize)) -> PyResult<Self> {
        FragGraph::path(nodes, link)
            .map(Self::from_core)
            .map_err(molrs_error_to_pyerr)
    }

    /// A ring: the :meth:`path` edges plus the closing edge
    /// ``(n - 1, 0, link[0], link[1])``. Needs at least three nodes.
    #[staticmethod]
    fn cycle(nodes: Vec<String>, link: (usize, usize)) -> PyResult<Self> {
        FragGraph::cycle(nodes, link)
            .map(Self::from_core)
            .map_err(molrs_error_to_pyerr)
    }

    /// A star: node 0 is ``center``, nodes ``1..=k`` are copies of ``arm``,
    /// and edge ``(0, i + 1, center_ports[i], arm_port)`` joins arm ``i``.
    #[staticmethod]
    fn star(
        center: String,
        center_ports: Vec<usize>,
        arm: String,
        arm_port: usize,
    ) -> PyResult<Self> {
        FragGraph::star(center, center_ports, arm, arm_port)
            .map(Self::from_core)
            .map_err(molrs_error_to_pyerr)
    }

    /// Template name per node.
    #[getter]
    fn nodes(&self) -> Vec<String> {
        self.inner.nodes().to_vec()
    }

    /// Edges as ``(a, b, port_a, port_b)`` tuples, in construction order.
    #[getter]
    fn edges(&self) -> Vec<EdgeTuple> {
        self.inner
            .edges()
            .iter()
            .map(|e| (e.a, e.b, e.port_a, e.port_b))
            .collect()
    }

    fn __repr__(&self) -> String {
        format!(
            "<FragGraph nodes={} edges={}>",
            self.inner.nodes().len(),
            self.inner.edges().len()
        )
    }
}
