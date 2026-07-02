//! Python wrappers for the ion-transport compute kernels
//! (`molrs-compute::{onsager, jacf, persist}`).
//!
//! These are the molrs ports of the *tame* trajectory-analysis recipes
//! (onsager coefficients, current-ACF Green–Kubo conductivity, pair-survival
//! persistence). Each returns a plain ``dict`` of NumPy arrays / scalars,
//! matching the dielectric binding style.

use molrs::compute::traits::Compute;
use molrs::compute::{OnsagerCorrelation, persist};
use molrs::store::frame::Frame as CoreFrame;
use numpy::{IntoPyArray, PyReadonlyArray2, PyReadonlyArray3};
use pyo3::prelude::*;

use crate::helpers::py_value_err;

/// Empty frame slice for the series-based `OnsagerCorrelation` compute.
fn no_frames() -> Vec<&'static CoreFrame> {
    Vec::new()
}

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
        .compute(&no_frames(), (&pi, &pj, dt, max_correlation_time))
        .map_err(py_value_err)?;
    let dict = pyo3::types::PyDict::new(py);
    dict.set_item("lag_times", result.lag_times.into_pyarray(py))?;
    dict.set_item("correlation", result.correlation.into_pyarray(py))?;
    Ok(dict.into())
}

// The legacy bundled `transport_green_kubo_conductivity` binding (raw JACF +
// fitted sigma/sigma_running) was removed in compute-fit-03-cleanup. Compose
// `molrs.GreenKuboConductivity` (raw current ACF) with `molrs.RunningIntegral`
// and a caller-applied `1/(3·V·k_B·T)` MD→SI prefactor.

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
