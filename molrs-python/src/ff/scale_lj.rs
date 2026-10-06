//! Python bindings for `molrs::ff::scale_lj` (`molrs.ff.scale_lj`): CL&Pol
//! fragment scaling of Lennard-Jones parameters — the fragment table
//! ([`PyFragmentScaling`], `fragment_scaling_data`), the SAPT pair factor
//! (`compute_k_ij`) and the scaled force field (`scale_lj`).

use std::collections::HashMap;

use pyo3::exceptions::PyKeyError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::error::py_value_err;
use crate::ff::forcefield::PyForceField;

/// CL&Pol fragment scaling data backed by the native force-field layer.
#[pyclass(
    module = "molrs.ff.scale_lj",
    name = "FragmentScaling",
    frozen,
    get_all,
    skip_from_py_object,
    subclass
)]
#[derive(Clone)]
pub struct PyFragmentScaling {
    name: String,
    q: f64,
    mu: f64,
    alpha: f64,
    polarizable: bool,
}

impl From<PyFragmentScaling> for molrs::ff::scale_lj::FragmentScaling {
    fn from(value: PyFragmentScaling) -> Self {
        Self {
            name: value.name,
            q: value.q,
            mu: value.mu,
            alpha: value.alpha,
            polarizable: value.polarizable,
        }
    }
}

impl From<molrs::ff::scale_lj::FragmentScaling> for PyFragmentScaling {
    fn from(value: molrs::ff::scale_lj::FragmentScaling) -> Self {
        Self {
            name: value.name,
            q: value.q,
            mu: value.mu,
            alpha: value.alpha,
            polarizable: value.polarizable,
        }
    }
}

#[pymethods]
impl PyFragmentScaling {
    #[new]
    #[pyo3(signature = (name, q, mu, alpha, polarizable=false))]
    fn new(name: String, q: f64, mu: f64, alpha: f64, polarizable: bool) -> Self {
        Self {
            name,
            q,
            mu,
            alpha,
            polarizable,
        }
    }

    fn __repr__(&self) -> String {
        format!("FragmentScaling(name='{}')", self.name)
    }
}

/// Native SAPT epsilon-scaling factor.
#[pyfunction(name = "compute_k_ij")]
pub fn compute_k_ij_py(
    fr_i: PyRef<'_, PyFragmentScaling>,
    fr_j: PyRef<'_, PyFragmentScaling>,
    r: f64,
) -> PyResult<f64> {
    molrs::ff::scale_lj::compute_k_ij(&fr_i.clone().into(), &fr_j.clone().into(), r)
        .map_err(py_value_err)
}

/// Return the compiled-in CL&Pol fragment table.
#[pyfunction(name = "fragment_scaling_data")]
pub fn fragment_scaling_data_py(py: Python<'_>) -> PyResult<Bound<'_, PyDict>> {
    let result = PyDict::new(py);
    for (name, scaling) in molrs::ff::scale_lj::builtin_fragment_scaling() {
        result.set_item(name, Py::new(py, PyFragmentScaling::from(scaling))?)?;
    }
    Ok(result)
}

/// Clone and scale LJ parameters using native COM and force-field transforms.
#[pyfunction(name = "scale_lj")]
#[pyo3(signature = (ff, fragments, frag_data=None, scale_sigma=false))]
pub fn scale_lj_py(
    py: Python<'_>,
    ff: &Bound<'_, PyForceField>,
    fragments: &Bound<'_, PyDict>,
    frag_data: Option<&Bound<'_, PyDict>>,
    scale_sigma: bool,
) -> PyResult<Py<PyForceField>> {
    let mut native_fragments = Vec::with_capacity(fragments.len());
    for (label, value) in fragments.iter() {
        let name = label.extract::<String>()?;
        let (atom_types, coords, masses) =
            value.extract::<(Vec<String>, Vec<[f64; 3]>, Vec<f64>)>()?;
        native_fragments.push(molrs::ff::scale_lj::FragmentAtoms {
            name,
            atom_types,
            coords,
            masses,
        });
    }

    let mut scaling = HashMap::new();
    if let Some(data) = frag_data {
        for (label, value) in data.iter() {
            let item = value.extract::<PyRef<'_, PyFragmentScaling>>()?;
            scaling.insert(label.extract::<String>()?, item.clone().into());
        }
    } else {
        scaling = molrs::ff::scale_lj::builtin_fragment_scaling();
    }

    let inner =
        molrs::ff::scale_lj::scale_lj(&ff.borrow().inner, &native_fragments, &scaling, scale_sigma)
            .map_err(|error| match error {
                molrs::ff::scale_lj::ScaleLjError::MissingFragment(name) => {
                    PyKeyError::new_err(format!("no scaling data for fragment '{name}'"))
                }
                other => py_value_err(other),
            })?;
    PyForceField::from_core(py, inner)
}

/// Register `molrs.ff.scale_lj`.
pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyFragmentScaling>()?;
    m.add_function(wrap_pyfunction!(compute_k_ij_py, m)?)?;
    m.add_function(wrap_pyfunction!(fragment_scaling_data_py, m)?)?;
    m.add_function(wrap_pyfunction!(scale_lj_py, m)?)?;
    Ok(())
}
