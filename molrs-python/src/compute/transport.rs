//! Python wrappers for the ion-transport compute kernels
//! (`Onsager`, pair-survival).  The raw transport Computes
//! (`GreenKuboConductivity`, `EinsteinConductivity`, …) are registered from
//! `fitting.rs` as `molrs.compute.transport.*` classes — do not re-wrap them
//! as free functions or recipe types.

use molrs::compute::Compute;
use molrs::compute::OnsagerCorrelation;
use molrs::compute::{SurvivalMethod, pair_survival_tcf};
use molrs::store::Frame as CoreFrame;
use numpy::{IntoPyArray, PyReadonlyArray2, PyReadonlyArray3};
use pyo3::prelude::*;

use crate::helpers::py_value_err;

/// Empty frame slice for the series-based `OnsagerCorrelation` compute.
const EMPTY_FRAMES: &[&CoreFrame] = &[];

/// Onsager collective mean-displacement cross-correlation.
#[pyclass(module = "molrs.compute.transport", name = "Onsager", frozen)]
pub struct PyOnsager;

#[pymethods]
impl PyOnsager {
    #[staticmethod]
    #[pyo3(signature = (p_i, p_j, dt, max_correlation_time))]
    fn correlation<'py>(
        py: Python<'py>,
        p_i: PyReadonlyArray2<'py, f64>,
        p_j: PyReadonlyArray2<'py, f64>,
        dt: f64,
        max_correlation_time: usize,
    ) -> PyResult<Py<PyAny>> {
        let pi = p_i.as_array().to_owned();
        let pj = p_j.as_array().to_owned();
        let result = OnsagerCorrelation
            .compute(EMPTY_FRAMES, (&pi, &pj, dt, max_correlation_time))
            .map_err(py_value_err)?;
        let dict = pyo3::types::PyDict::new(py);
        dict.set_item("lag_times", result.lag_times.into_pyarray(py))?;
        dict.set_item("correlation", result.correlation.into_pyarray(py))?;
        Ok(dict.into())
    }
}

/// Pair-survival (persistence) time-correlation functions.
#[pyclass(module = "molrs.compute.transport", name = "Persist", frozen)]
pub struct PyPersist;

#[pymethods]
impl PyPersist {
    #[staticmethod]
    #[pyo3(signature = (coords_i, coords_j, box_lengths, r0, r1, method, dt, max_correlation_time, exclude_self=false))]
    #[allow(clippy::too_many_arguments)]
    fn pair_survival_tcf<'py>(
        py: Python<'py>,
        coords_i: PyReadonlyArray3<'py, f64>,
        coords_j: PyReadonlyArray3<'py, f64>,
        box_lengths: PyReadonlyArray2<'py, f64>,
        r0: f64,
        r1: f64,
        method: &str,
        dt: f64,
        max_correlation_time: usize,
        exclude_self: bool,
    ) -> PyResult<Py<PyAny>> {
        let ci = coords_i.as_array().to_owned();
        let cj = coords_j.as_array().to_owned();
        let bl = box_lengths.as_array().to_owned();
        let m = SurvivalMethod::parse(method).map_err(py_value_err)?;
        let result = pair_survival_tcf(
            &ci,
            &cj,
            &bl,
            r0,
            r1,
            m,
            dt,
            max_correlation_time,
            exclude_self,
        )
        .map_err(py_value_err)?;
        let dict = pyo3::types::PyDict::new(py);
        dict.set_item("lag_times", result.lag_times.into_pyarray(py))?;
        dict.set_item("correlation", result.correlation.into_pyarray(py))?;
        Ok(dict.into())
    }
}

pub fn register_transport(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyOnsager>()?;
    m.add_class::<PyPersist>()
}
