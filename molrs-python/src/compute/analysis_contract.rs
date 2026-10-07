//! The analysis contract (`molrs::compute`): the descriptor-row protocol
//! (`DescriptorRow`) a Python
//! object implements to feed `Pca` / `Kmeans`.

use molrs::op::F;
use numpy::PyReadonlyArray1;
use pyo3::prelude::*;

// ---------------------------------------------------------------------------
// PCA
// ---------------------------------------------------------------------------

/// Row-based descriptor wrapper for PCA/Kmeans input.
///
/// Wrap each row (a 1-D float array) with `DescriptorRow(row)`; then pass a
/// Python list of them to ``Pca.compute`` / ``Kmeans.compute``.
#[pyclass(module = "molrs.compute", name = "DescriptorRow", from_py_object)]
#[derive(Clone)]
pub struct PyDescriptorRow {
    row: Vec<F>,
}

#[pymethods]
impl PyDescriptorRow {
    #[new]
    fn new(values: PyReadonlyArray1<'_, f64>) -> PyResult<Self> {
        let slice = values.as_slice()?;
        Ok(Self {
            row: slice.iter().map(|&v| v as F).collect(),
        })
    }

    fn __len__(&self) -> usize {
        self.row.len()
    }
}

impl molrs::compute::DescriptorRow for PyDescriptorRow {
    fn as_row(&self) -> &[F] {
        &self.row
    }
}

/// Register this domain's classes and functions.
pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyDescriptorRow>()?;
    Ok(())
}
