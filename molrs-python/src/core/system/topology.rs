//! Python binding for the bond graph [`Topology`].

use molrs::system::topology::Topology;
use numpy::{IntoPyArray, PyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyType;

use crate::core::store::frame::PyFrame;

/// Bond graph of a frame: atoms are rows ``0..n`` of ``frame["atoms"]``,
/// edges the ``atomi`` / ``atomj`` rows of ``frame["bonds"]``.
#[pyclass(
    module = "molrs",
    name = "Topology",
    frozen,
    skip_from_py_object,
    subclass
)]
pub struct PyTopology {
    inner: Topology,
}

#[pymethods]
impl PyTopology {
    /// Read the bond graph of ``frame``.
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     ``frame`` has no atoms, the bonds block lacks
    ///     ``atomi`` / ``atomj``, or a bond names a row outside the frame.
    #[classmethod]
    fn from_frame(_cls: &Bound<'_, PyType>, frame: &PyFrame) -> PyResult<Self> {
        let inner = frame
            .with_frame(Topology::from_frame)?
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Self { inner })
    }

    /// Number of atoms (graph nodes).
    #[getter]
    fn n_atoms(&self) -> usize {
        self.inner.n_atoms()
    }

    /// Number of bonds (graph edges).
    #[getter]
    fn n_bonds(&self) -> usize {
        self.inner.n_bonds()
    }

    /// Number of connected components; an unbonded atom is its own.
    #[getter]
    fn n_components(&self) -> usize {
        self.inner.n_components()
    }

    /// Per-atom connected-component label, ``0..n_components`` in order of
    /// each component's first atom, as a ``uint32`` array.
    fn connected_components<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<u32>> {
        self.inner
            .connected_components()
            .into_iter()
            .map(|c| u32::try_from(c).expect("a component label is below n_atoms"))
            .collect::<Vec<_>>()
            .into_pyarray(py)
    }
}
