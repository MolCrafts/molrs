//! Dimensionality reduction and clustering (`molrs::compute::ml`): `Pca2`,
//! `KMeans` and their results.

use super::result::PyDescriptorRow;
use crate::error::py_value_err;
use molrs::compute::{Compute, KMeans, KMeansResult, Pca2, PcaResult};
use molrs::store::Frame as CoreFrame;
use numpy::{IntoPyArray, PyArray1, PyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// Two-component PCA result.
#[pyclass(module = "molrs.compute", name = "PcaResult", from_py_object)]
#[derive(Clone)]
pub struct PyPcaResult {
    pub(crate) inner: PcaResult,
}

#[pymethods]
impl PyPcaResult {
    /// Row-major `(n_rows, 2)` projected coordinates.
    #[getter]
    fn coords<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let n = self.inner.coords.len() / 2;
        let flat: Vec<f64> = self.inner.coords.to_vec();
        Ok(ndarray::Array2::from_shape_vec((n, 2), flat)
            .map_err(|e| PyValueError::new_err(e.to_string()))?
            .into_pyarray(py))
    }

    /// `[var_pc1, var_pc2]` — explained variance per component.
    #[getter]
    fn variance(&self) -> (f64, f64) {
        (self.inner.variance[0], self.inner.variance[1])
    }

    fn __repr__(&self) -> String {
        format!(
            "PcaResult(n_rows={}, variance=[{:.4}, {:.4}])",
            self.inner.coords.len() / 2,
            self.inner.variance[0],
            self.inner.variance[1],
        )
    }
}

/// Two-component PCA calculator.
#[pyclass(module = "molrs.compute", name = "Pca2")]
pub struct PyPca2 {
    inner: Pca2<PyDescriptorRow>,
}

#[pymethods]
impl PyPca2 {
    #[new]
    fn new() -> Self {
        Self {
            inner: Pca2::<PyDescriptorRow>::new(),
        }
    }

    /// Compute PCA over a list of `DescriptorRow` objects.
    fn compute(&self, rows: Vec<PyRef<'_, PyDescriptorRow>>) -> PyResult<PyPcaResult> {
        let owned: Vec<PyDescriptorRow> = rows.iter().map(|r| (*r).clone()).collect();
        // Wrap an empty FrameAccess slice — Pca2 does not touch frames.
        let frames: [&CoreFrame; 0] = [];
        let pca = self.inner.compute(&frames, &owned).map_err(py_value_err)?;
        Ok(PyPcaResult { inner: pca })
    }

    fn __repr__(&self) -> String {
        "Pca2()".to_string()
    }
}

// ---------------------------------------------------------------------------
// KMeans
// ---------------------------------------------------------------------------

/// k-means cluster labels.
#[pyclass(module = "molrs.compute", name = "KMeansResult")]
pub struct PyKMeansResult {
    inner: KMeansResult,
}

#[pymethods]
impl PyKMeansResult {
    #[getter]
    fn labels<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<i32>> {
        ndarray::Array1::from_vec(self.inner.0.clone()).into_pyarray(py)
    }

    fn __len__(&self) -> usize {
        self.inner.0.len()
    }

    fn __repr__(&self) -> String {
        format!("KMeansResult(n={})", self.inner.0.len())
    }
}

/// k-means clustering over a PCA projection (2-D).
#[pyclass(module = "molrs.compute", name = "KMeans")]
pub struct PyKMeans {
    inner: KMeans,
}

#[pymethods]
impl PyKMeans {
    #[new]
    #[pyo3(signature = (k, max_iter = 100, seed = 0))]
    fn new(k: usize, max_iter: usize, seed: u64) -> PyResult<Self> {
        Ok(Self {
            inner: KMeans::new(k, max_iter, seed).map_err(py_value_err)?,
        })
    }

    /// Cluster a `PcaResult`.
    fn compute(&self, pca: &PyPcaResult) -> PyResult<PyKMeansResult> {
        let frames: [&CoreFrame; 0] = [];
        let out = self
            .inner
            .compute(&frames, &pca.inner)
            .map_err(py_value_err)?;
        Ok(PyKMeansResult { inner: out })
    }

    fn __repr__(&self) -> String {
        "KMeans(...)".to_string()
    }
}

/// Register this domain's classes and functions.
pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyPcaResult>()?;
    m.add_class::<PyPca2>()?;
    m.add_class::<PyKMeansResult>()?;
    m.add_class::<PyKMeans>()?;
    Ok(())
}
