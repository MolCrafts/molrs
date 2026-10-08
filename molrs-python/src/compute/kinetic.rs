//! Kinetic observables of one configuration (`molrs::compute`):
//! `kinetic_energy`, `kinetic_temperature`, `center_of_mass_velocity`.

use molrs::compute::{center_of_mass_velocity, kinetic_energy, kinetic_temperature};
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;

use crate::error::py_value_err;

/// Kinetic energy ``0.5 * sum_i m_i |v_i|^2`` of ``(n_atoms, 3)`` velocities.
#[pyfunction(name = "kinetic_energy")]
fn kinetic_energy_py(
    mass: PyReadonlyArray1<'_, f64>,
    vel: PyReadonlyArray2<'_, f64>,
) -> PyResult<f64> {
    kinetic_energy(mass.as_array(), vel.as_array()).map_err(py_value_err)
}

/// Kinetic temperature ``2 K / (n_dof * kb)`` by equipartition; ``kb`` is
/// Boltzmann's constant in the unit of ``kinetic_energy``.
#[pyfunction(name = "kinetic_temperature")]
fn kinetic_temperature_py(kinetic_energy: f64, n_dof: usize, kb: f64) -> PyResult<f64> {
    kinetic_temperature(kinetic_energy, n_dof, kb).map_err(py_value_err)
}

/// Mass-weighted centre-of-mass velocity, shape ``(3,)``.
#[pyfunction(name = "center_of_mass_velocity")]
fn center_of_mass_velocity_py<'py>(
    py: Python<'py>,
    mass: PyReadonlyArray1<'py, f64>,
    vel: PyReadonlyArray2<'py, f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let v = center_of_mass_velocity(mass.as_array(), vel.as_array()).map_err(py_value_err)?;
    Ok(v.into_pyarray(py))
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    for f in [
        wrap_pyfunction!(kinetic_energy_py, m)?,
        wrap_pyfunction!(kinetic_temperature_py, m)?,
        wrap_pyfunction!(center_of_mass_velocity_py, m)?,
    ] {
        crate::add_function(m, "molrs.compute", f)?;
    }
    Ok(())
}
