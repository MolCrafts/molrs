//! Python wrappers for the ion-transport compute kernels
//! (`Onsager`, pair-survival).  The raw transport Computes
//! (`GreenKuboConductivity`, `EinsteinConductivity`, …) are registered from
//! `fitting.rs` as `molrs.compute.transport.*` classes — do not re-wrap them
//! as free functions or recipe types.

use molrs::compute::OnsagerCorrelation;
use molrs::compute::dynamics::persist;
use molrs::compute::traits::Compute;
use molrs::store::frame::Frame as CoreFrame;
use numpy::{IntoPyArray, PyReadonlyArray2, PyReadonlyArray3};
use pyo3::prelude::*;

use crate::helpers::py_value_err;

/// Empty frame slice for the series-based `OnsagerCorrelation` compute.
const EMPTY_FRAMES: &[&CoreFrame] = &[];

#[pyfunction]
#[pyo3(signature = (p_i, p_j, dt, max_correlation_time))]
pub(crate) fn transport_onsager_correlation<'py>(
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

#[pyfunction]
#[pyo3(signature = (coords_i, coords_j, box_lengths, r0, r1, method, dt, max_correlation_time, exclude_self=false))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn transport_pair_survival_tcf<'py>(
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
    let m = persist::SurvivalMethod::parse(method).map_err(py_value_err)?;
    let result = persist::pair_survival_tcf(
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

pub fn register_transport(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(transport_onsager_correlation, m)?)?;
    m.add_function(wrap_pyfunction!(transport_pair_survival_tcf, m)?)?;
    Ok(())
}
