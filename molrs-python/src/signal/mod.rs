//! Python wrappers for the `molrs-signal` crate.
//!
//! Functions are registered directly on the `molrs` module (following the
//! same pattern as `io::read_pdb` etc.) and then re-exported via a Python
//! `molrs/signal.py` facade.

use molrs::signal::{self as sig, WindowType};
use numpy::{IntoPyArray, PyArray1, PyArrayDyn, PyReadonlyArray1, PyReadonlyArrayDyn};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::helpers::py_value_err;

#[pyfunction]
#[pyo3(signature = (data, max_lag))]
pub(crate) fn signal_acf_fft<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<'py, f64>,
    max_lag: usize,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let arr = data.as_array().to_owned();
    let result = sig::acf_fft(&arr, max_lag).map_err(py_value_err)?;
    Ok(result.into_pyarray(py))
}

#[pyfunction]
#[pyo3(signature = (data, window_type, axis = 0))]
pub(crate) fn signal_apply_window<'py>(
    py: Python<'py>,
    data: PyReadonlyArrayDyn<'py, f64>,
    window_type: &str,
    axis: usize,
) -> PyResult<Bound<'py, PyArrayDyn<f64>>> {
    let wt = match window_type {
        "hann" => WindowType::Hann,
        "blackman" => WindowType::Blackman,
        other => {
            return Err(PyValueError::new_err(format!(
                "unknown window type: '{other}', expected 'hann' or 'blackman'"
            )));
        }
    };
    let arr = data.as_array().to_owned();
    let result = sig::apply_window(&arr, wt, axis).map_err(py_value_err)?;
    Ok(result.into_pyarray(py))
}

#[pyfunction]
pub(crate) fn signal_frequency_grid<'py>(
    py: Python<'py>,
    n_fft: usize,
    dt: f64,
) -> Bound<'py, PyArray1<f64>> {
    let result = sig::frequency_grid(n_fft, dt);
    result.into_pyarray(py)
}
