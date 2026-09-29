//! Python wrappers for `molrs-compute::dielectric`.

use molrs::compute::dielectric as diel;
use numpy::{
    IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArray3,
};
use pyo3::prelude::*;

use crate::helpers::py_value_err;

/// Raw dielectric kernels: dipole moment, current density, current partition
/// and the Neumann static dielectric constant. The frequency-dependent ε(ω)
/// is the raw-compute + fit composition (`DebyeRelaxation` →
/// `EinsteinHelfandSpectrum`, `GreenKuboConductivity` → `GreenKuboSpectrum`).
#[pyclass(module = "molrs.compute.dielectric", name = "Dielectric", frozen)]
pub struct PyDielectric;

#[pymethods]
impl PyDielectric {
    #[staticmethod]
    fn compute_dipole_moment<'py>(
        py: Python<'py>,
        charges: PyReadonlyArray1<'py, f64>,
        positions: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let c = charges.as_array().to_owned();
        let p = positions.as_array().to_owned();
        let result = diel::compute_dipole_moment(&c, &p).map_err(py_value_err)?;
        Ok(result.into_pyarray(py))
    }

    #[staticmethod]
    fn compute_current_density<'py>(
        py: Python<'py>,
        dipole_moments: PyReadonlyArray2<'py, f64>,
        dt: f64,
        volume: f64,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let dm = dipole_moments.as_array().to_owned();
        let result = diel::compute_current_density(&dm, dt, volume).map_err(py_value_err)?;
        Ok(result.into_pyarray(py))
    }

    #[staticmethod]
    fn static_dielectric_constant<'py>(
        dipole_moments: PyReadonlyArray2<'py, f64>,
        volume: f64,
        temperature: f64,
        epsilon_inf: f64,
    ) -> PyResult<f64> {
        let dm = dipole_moments.as_array().to_owned();
        diel::static_dielectric_constant(&dm, volume, temperature, epsilon_inf)
            .map_err(py_value_err)
    }

    #[allow(clippy::type_complexity)]
    #[staticmethod]
    fn decompose_current<'py>(
        py: Python<'py>,
        per_particle_current: PyReadonlyArray3<'py, f64>,
        water_mask: PyReadonlyArray1<'py, bool>,
    ) -> PyResult<(Bound<'py, PyArray2<f64>>, Bound<'py, PyArray2<f64>>)> {
        let current = per_particle_current.as_array().to_owned();
        let mask = water_mask.as_array().to_owned();
        let (j_w, j_i) = diel::decompose_current(&current, &mask).map_err(py_value_err)?;
        Ok((j_w.into_pyarray(py), j_i.into_pyarray(py)))
    }
}

pub fn register_dielectric(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyDielectric>()
}
