//! Python wrappers for `molrs-compute::dielectric`.

use molrs::compute::dielectric as diel;
use numpy::{
    IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArray3,
};
use pyo3::prelude::*;

use crate::helpers::py_value_err;

#[pyfunction]
pub(crate) fn dielectric_compute_dipole_moment<'py>(
    py: Python<'py>,
    charges: PyReadonlyArray1<'py, f64>,
    positions: PyReadonlyArray2<'py, f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let c = charges.as_array().to_owned();
    let p = positions.as_array().to_owned();
    let result = diel::compute_dipole_moment(&c, &p).map_err(py_value_err)?;
    Ok(result.into_pyarray(py))
}

#[pyfunction]
pub(crate) fn dielectric_compute_current_density<'py>(
    py: Python<'py>,
    dipole_moments: PyReadonlyArray2<'py, f64>,
    dt: f64,
    volume: f64,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let dm = dipole_moments.as_array().to_owned();
    let result = diel::compute_current_density(&dm, dt, volume).map_err(py_value_err)?;
    Ok(result.into_pyarray(py))
}

#[pyfunction]
pub(crate) fn dielectric_static_dielectric_constant<'py>(
    dipole_moments: PyReadonlyArray2<'py, f64>,
    volume: f64,
    temperature: f64,
    epsilon_inf: f64,
) -> PyResult<f64> {
    let dm = dipole_moments.as_array().to_owned();
    diel::static_dielectric_constant(&dm, volume, temperature, epsilon_inf).map_err(py_value_err)
}

// The frequency-dependent ε(ω) spectrum free fns (`einstein_helfand_spectrum` /
// `green_kubo_spectrum`) and their deprecated bindings were removed in
// compute-fit-04-dielectric. The ε(ω) transform is now the explicit raw-compute
// + Fit composition: `molrs.DebyeRelaxation` (raw fluctuation dipole ACF) +
// `molrs.EinsteinHelfandSpectrum`, and `molrs.GreenKuboConductivity` (raw
// current ACF) + `molrs.GreenKuboSpectrum` (see `compute_fit.rs`).
//
// The legacy bundled `dielectric_einstein_helfand_conductivity` binding (raw MSD
// + fitted sigma/slope/fit_start/fit_end) was removed in compute-fit-03-cleanup.
// Compose `molrs.EinsteinConductivity` (raw collective-dipole MSD) with
// `molrs.LinearFit` and a caller-applied `slope/(6·V·k_B·T)` MD→SI prefactor.

#[pyfunction]
#[allow(clippy::type_complexity)]
pub(crate) fn dielectric_decompose_current<'py>(
    py: Python<'py>,
    per_particle_current: PyReadonlyArray3<'py, f64>,
    water_mask: PyReadonlyArray1<'py, bool>,
) -> PyResult<(Bound<'py, PyArray2<f64>>, Bound<'py, PyArray2<f64>>)> {
    let current = per_particle_current.as_array().to_owned();
    let mask = water_mask.as_array().to_owned();
    let (j_w, j_i) = diel::decompose_current(&current, &mask).map_err(py_value_err)?;
    Ok((j_w.into_pyarray(py), j_i.into_pyarray(py)))
}

pub fn register_dielectric(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(dielectric_compute_dipole_moment, m)?)?;
    m.add_function(wrap_pyfunction!(dielectric_compute_current_density, m)?)?;
    m.add_function(wrap_pyfunction!(dielectric_static_dielectric_constant, m)?)?;
    m.add_function(wrap_pyfunction!(dielectric_decompose_current, m)?)?;
    Ok(())
}
