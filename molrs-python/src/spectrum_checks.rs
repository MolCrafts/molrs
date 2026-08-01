//! Thin Python bindings for the dielectric / conductivity spectrum checks.
//! All numerics live in `molrs::compute::spectroscopy` alongside the spectra
//! they judge; this module only marshals numpy arrays in and dicts out.

use numpy::{IntoPyArray, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyAnyMethods, PyDict, PyDictMethods};

use molrs::compute::spectroscopy::{ConductivitySumRule, KramersKronig, RouteAgreement};
use molrs::compute::traits::Check;

use crate::helpers::py_value_err;

#[pyfunction]
pub(crate) fn check_kramers_kronig<'py>(
    py: Python<'py>,
    frequency: PyReadonlyArray1<'py, f64>,
    eps_real: PyReadonlyArray1<'py, f64>,
    eps_imag: PyReadonlyArray1<'py, f64>,
    eps_inf: f64,
) -> PyResult<Py<PyAny>> {
    let out = KramersKronig { eps_inf }
        .check((
            &frequency.as_array().to_owned(),
            &eps_real.as_array().to_owned(),
            &eps_imag.as_array().to_owned(),
        ))
        .map_err(py_value_err)?;

    let dict = PyDict::new(py);
    dict.set_item("passed", out.passed)?;
    dict.set_item("mae", out.mae)?;
    dict.set_item("eps_real_recovered", out.recovered.into_pyarray(py))?;
    Ok(dict.into())
}

#[pyfunction]
pub(crate) fn check_conductivity_sum_rule<'py>(
    py: Python<'py>,
    frequency: PyReadonlyArray1<'py, f64>,
    conductivity: PyReadonlyArray1<'py, f64>,
    current_sq_mean: f64,
    volume: f64,
    temperature: f64,
) -> PyResult<Py<PyAny>> {
    let out = ConductivitySumRule {
        current_sq_mean,
        volume,
        temperature,
    }
    .check((
        &frequency.as_array().to_owned(),
        &conductivity.as_array().to_owned(),
    ))
    .map_err(py_value_err)?;

    let dict = PyDict::new(py);
    dict.set_item("passed", out.passed)?;
    dict.set_item("relative_error", out.relative_error)?;
    dict.set_item("integral", out.integral)?;
    dict.set_item("expected", out.expected)?;
    Ok(dict.into())
}

#[pyfunction]
pub(crate) fn check_route_agreement<'py>(
    py: Python<'py>,
    results: &Bound<'py, PyDict>,
) -> PyResult<Py<PyAny>> {
    let mut entries = Vec::with_capacity(results.len());
    for (key, value) in results.iter() {
        let name: String = key.extract()?;
        let arr: PyReadonlyArray1<f64> = value.extract().map_err(|_| {
            PyValueError::new_err(format!(
                "route_agreement: value for '{name}' must be a 1-D float64 array"
            ))
        })?;
        entries.push((name, arr.as_array().to_owned()));
    }

    let out = RouteAgreement.check(&entries).map_err(py_value_err)?;

    let pairwise = PyDict::new(py);
    for (label, rms) in &out.pairwise {
        pairwise.set_item(label, rms)?;
    }
    let dict = PyDict::new(py);
    dict.set_item("passed", out.passed)?;
    dict.set_item("pairwise_rms", pairwise)?;
    Ok(dict.into())
}

pub fn register_check(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(check_kramers_kronig, m)?)?;
    m.add_function(wrap_pyfunction!(check_conductivity_sum_rule, m)?)?;
    m.add_function(wrap_pyfunction!(check_route_agreement, m)?)?;
    Ok(())
}
