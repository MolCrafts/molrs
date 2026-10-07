//! Python bindings for `molrs::ff::params` (`molrs.ff.params`): the parameter
//! tables molrs ships: the CL&Pol `alpha.ff` polarizability table
//! (`clpol_polarizability`). The AMBER 1-4 divisors are engine constants,
//! `molrs.core.constants.AMBER_SCEE` / `AMBER_SCNB`.

use std::collections::HashMap;
use std::path::PathBuf;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::ff::clpol_scaling::PyFragmentScaling;
use crate::path::path_str;

/// One row's fields under the names CL&Pol (and molpy) use.
fn fields(
    m_d: f64,
    q_d_sign: f64,
    k_d: f64,
    alpha: f64,
    a_thole: f64,
) -> HashMap<&'static str, f64> {
    HashMap::from([
        ("m_D", m_d),
        ("q_D_sign", q_d_sign),
        ("k_D", k_d),
        ("alpha", alpha),
        ("a_thole", a_thole),
    ])
}

/// CL&Pol Drude parameters per atom type, from ``alpha.ff``.
///
/// Without ``path``, the table molrs ships (paduagroup/clandpol ``alpha.ff``
/// 2024/06/05, :data:`molrs::ff::params::CLPOL_POLARIZABILITY`); with it, that
/// file read the same way (``#`` comments, rows ``type m_D q_D k_D alpha
/// a_thole``; a type given twice keeps its last row).
///
/// Returns
/// -------
/// dict[str, dict[str, float]]
///     ``type -> {"m_D", "q_D_sign", "k_D", "alpha", "a_thole"}`` in u, e
///     (sign), kJ/mol/Å² (``k/2 r²`` form), Å³ and dimensionless. A type
///     with ``k_D == 0`` carries no Drude particle.
///
/// Raises
/// ------
/// ValueError
///     The file cannot be read or a row's numbers do not parse.
#[pyfunction]
#[pyo3(signature = (path = None))]
pub fn clpol_polarizability(
    path: Option<PathBuf>,
) -> PyResult<HashMap<String, HashMap<&'static str, f64>>> {
    match path {
        None => Ok(molrs::ff::params::CLPOL_POLARIZABILITY
            .iter()
            .map(|r| {
                (
                    r.type_name.to_owned(),
                    fields(r.m_d, r.q_d_sign, r.k_d, r.alpha, r.a_thole),
                )
            })
            .collect()),
        Some(path) => {
            let rows = molrs::io::forcefield::readers::clpol::read_alpha_ff(path_str(&path)?)
                .map_err(PyValueError::new_err)?;
            Ok(rows
                .into_iter()
                .map(|r| {
                    (
                        r.type_name,
                        fields(r.m_d, r.q_d_sign, r.k_d, r.alpha, r.a_thole),
                    )
                })
                .collect())
        }
    }
}

/// CL&Pol's fragment scaling table (paduagroup/clandpol ``fragment.ff``):
/// each fragment's charge, dipole and polarizability, by fragment name, as
/// :func:`molrs.ff.clpol_scaling.scale_lj` reads it.
#[pyfunction]
fn clpol_fragment_scaling(py: Python<'_>) -> PyResult<Bound<'_, PyDict>> {
    let result = PyDict::new(py);
    for (name, scaling) in molrs::ff::params::clpol_fragment_scaling() {
        result.set_item(name, Py::new(py, PyFragmentScaling::from(scaling))?)?;
    }
    Ok(result)
}

/// Register `molrs.ff.params`.
pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    crate::add_function(
        m,
        "molrs.ff.params",
        wrap_pyfunction!(clpol_polarizability, m)?,
    )?;
    crate::add_function(
        m,
        "molrs.ff.params",
        wrap_pyfunction!(clpol_fragment_scaling, m)?,
    )?;
    Ok(())
}
