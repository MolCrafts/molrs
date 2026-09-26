//! Python binding of the coarse-to-template unit mapping
//! (`molrs::core::system::mapping`).

use molrs::system::mapping::Mapping;
use molrs::system::molgraph::{node_from_u64, node_to_u64};
use pyo3::exceptions::{PyAttributeError, PyIndexError, PyTypeError};
use pyo3::intern;
use pyo3::prelude::*;

use crate::core::spatial::trace::PyTrace;
use crate::core::system::frag_graph::PyFragGraph;
use crate::core::system::molgraph::PyCoarseGrain;
use crate::helpers::molrs_error_to_pyerr;

/// A :class:`FragGraph` whose every unit is tied to coarse source beads.
///
/// Parameters
/// ----------
/// graph : FragGraph
/// sources : list[list[int | NodeRef]]
///     Per unit, its coarse bead handles (or bead views) in template-bead
///     order.
/// labels : list[list[str]]
///     Per unit, the template bead label of each source bead.
/// rules : list[tuple[str, str]]
///     The ``(coarse type, template label)`` relation that licensed them.
///
/// Raises
/// ------
/// ValueError
///     If the source or label lists do not match the graph's units, a unit
///     has no source, a source appears twice, or a label no rule licenses.
/// TypeError
///     If a rule is not a ``(str, str)`` pair, or a source is neither an
///     ``int`` handle nor a view carrying ``.handle``.
#[pyclass(module = "molrs", name = "Mapping", frozen, skip_from_py_object)]
pub struct PyMapping {
    pub(crate) inner: Mapping,
}

impl PyMapping {
    fn check_unit(&self, unit: usize) -> PyResult<()> {
        if unit < self.inner.n_units() {
            return Ok(());
        }
        Err(PyIndexError::new_err(format!(
            "unit {unit} is out of range for a mapping of {} units",
            self.inner.n_units()
        )))
    }
}

#[pymethods]
impl PyMapping {
    #[new]
    fn new(
        graph: &Bound<'_, PyFragGraph>,
        sources: Vec<Vec<Bound<'_, PyAny>>>,
        labels: Vec<Vec<String>>,
        rules: Vec<(String, String)>,
    ) -> PyResult<Self> {
        let py = graph.py();
        // A bead crosses as its handle, or as a view carrying `.handle`.
        let sources = sources
            .iter()
            .map(|unit| {
                unit.iter()
                    .map(|bead| {
                        let handle = match bead.extract::<u64>() {
                            Ok(handle) => handle,
                            Err(_) => bead
                                .getattr(intern!(py, "handle"))
                                .map_err(|err| {
                                    if err.is_instance_of::<PyAttributeError>(py) {
                                        PyTypeError::new_err(format!(
                                            "a source must be a bead handle (int) or a bead \
                                             view, got {}",
                                            bead.get_type()
                                        ))
                                    } else {
                                        err
                                    }
                                })?
                                .extract::<u64>()?,
                        };
                        Ok(node_from_u64(handle))
                    })
                    .collect::<PyResult<Vec<_>>>()
            })
            .collect::<PyResult<Vec<_>>>()?;
        Mapping::new(graph.get().inner.clone(), sources, labels, rules)
            .map(|inner| Self { inner })
            .map_err(molrs_error_to_pyerr)
    }

    /// The unit-level topology.
    #[getter]
    fn graph(&self) -> PyFragGraph {
        PyFragGraph::from_core(self.inner.graph().clone())
    }

    /// Number of units.
    #[getter]
    fn n_units(&self) -> usize {
        self.inner.n_units()
    }

    /// Unit ``unit``'s coarse bead handles, in template-bead order.
    fn sources(&self, unit: usize) -> PyResult<Vec<u64>> {
        self.check_unit(unit)?;
        Ok(self
            .inner
            .sources(unit)
            .unwrap_or_default()
            .iter()
            .map(|&id| node_to_u64(id))
            .collect())
    }

    /// Unit ``unit``'s template bead labels, one per source bead.
    fn labels(&self, unit: usize) -> PyResult<Vec<String>> {
        self.check_unit(unit)?;
        Ok(self.inner.labels(unit).unwrap_or_default().to_vec())
    }

    /// The ``(coarse type, template label)`` rules.
    #[getter]
    fn rules(&self) -> Vec<(String, String)> {
        self.inner.rules().to_vec()
    }

    /// The trace of this mapping over ``source``: per unit, its source beads'
    /// positions.
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If a source bead is missing from ``source``, lacks coordinates, or
    ///     its current type no rule licenses for its label.
    fn trace(&self, source: PyRef<'_, PyCoarseGrain>) -> PyResult<PyTrace> {
        self.inner
            .trace(source.core())
            .map(|inner| PyTrace { inner })
            .map_err(molrs_error_to_pyerr)
    }

    fn __repr__(&self) -> String {
        format!("<Mapping n_units={}>", self.inner.n_units())
    }
}
