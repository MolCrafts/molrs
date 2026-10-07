//! Dielectric raw observables (`molrs::compute`): the free functions
//! `dipole_moment`, `current_density`, `static_dielectric_constant` and
//! `decompose_current`, at the same names as in Rust.

use molrs::compute::{
    current_density, decompose_current, dipole_moment, static_dielectric_constant,
};
use numpy::{
    IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArray3,
};
use pyo3::prelude::*;

use crate::error::py_value_err;

/// Total dipole moment `M = Σ q_i r_i` (e·Å) of one configuration: `charges`
/// shape `(n_atoms,)` in e, `positions` shape `(n_atoms, 3)` in Å, unwrapped.
#[pyfunction(name = "dipole_moment")]
fn dipole_moment_py<'py>(
    py: Python<'py>,
    charges: PyReadonlyArray1<'py, f64>,
    positions: PyReadonlyArray2<'py, f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let c = charges.as_array().to_owned();
    let p = positions.as_array().to_owned();
    let result = dipole_moment(&c, &p).map_err(py_value_err)?;
    Ok(result.into_pyarray(py))
}

/// Polarisation current density `(dM/dt) / V`, shape `(n_frames, 3)`, by
/// finite differences of `dipole_moments`; row 0 is NaN.
#[pyfunction(name = "current_density")]
fn current_density_py<'py>(
    py: Python<'py>,
    dipole_moments: PyReadonlyArray2<'py, f64>,
    dt: f64,
    volume: f64,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let dm = dipole_moments.as_array().to_owned();
    let result = current_density(&dm, dt, volume).map_err(py_value_err)?;
    Ok(result.into_pyarray(py))
}

/// Neumann static dielectric constant from dipole fluctuations.
#[pyfunction(name = "static_dielectric_constant")]
fn static_dielectric_constant_py<'py>(
    dipole_moments: PyReadonlyArray2<'py, f64>,
    volume: f64,
    temperature: f64,
    epsilon_inf: f64,
) -> PyResult<f64> {
    let dm = dipole_moments.as_array().to_owned();
    static_dielectric_constant(&dm, volume, temperature, epsilon_inf).map_err(py_value_err)
}

/// Split a `(n_particles, n_frames, 3)` per-particle current into the
/// `water_mask` part and the rest, each `(n_frames, 3)`.
#[allow(clippy::type_complexity)]
#[pyfunction(name = "decompose_current")]
fn decompose_current_py<'py>(
    py: Python<'py>,
    per_particle_current: PyReadonlyArray3<'py, f64>,
    water_mask: PyReadonlyArray1<'py, bool>,
) -> PyResult<(Bound<'py, PyArray2<f64>>, Bound<'py, PyArray2<f64>>)> {
    let current = per_particle_current.as_array().to_owned();
    let mask = water_mask.as_array().to_owned();
    let (j_w, j_i) = decompose_current(&current, &mask).map_err(py_value_err)?;
    Ok((j_w.into_pyarray(py), j_i.into_pyarray(py)))
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    for f in [
        wrap_pyfunction!(dipole_moment_py, m)?,
        wrap_pyfunction!(current_density_py, m)?,
        wrap_pyfunction!(static_dielectric_constant_py, m)?,
        wrap_pyfunction!(decompose_current_py, m)?,
    ] {
        crate::add_function(m, "molrs.compute", f)?;
    }
    Ok(())
}
