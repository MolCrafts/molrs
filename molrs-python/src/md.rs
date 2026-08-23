//! Python bindings for `molrs::md`.
//!
//! ```text
//! VelocityVerlet(dt, potential=lj, neighbors=nl, mass=mass)
//! LJ.calc_energy / calc_force / eval → (energy, force)
//! ```

use crate::core::spatial::neighborlist::{PyNeighbors, PyVerletSkin};
use crate::helpers::NpF;
use molrs::md::{
    LJ, Langevin, MD_ENERGY, MDState, MaxwellBoltzmann, MdError, Potential, VelocityVerlet,
    energy_to_md, kb_md, preset_energy_to_md,
};
use molrs::types::F;
use ndarray::Array1;
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyAnyMethods;

fn md_err(e: MdError) -> PyErr {
    PyValueError::new_err(e.to_string())
}

fn check_nx3(arr: &PyReadonlyArray2<'_, NpF>, label: &str) -> PyResult<()> {
    if arr.as_array().ncols() != 3 {
        return Err(PyValueError::new_err(format!(
            "{label} must have shape (N, 3)"
        )));
    }
    Ok(())
}

fn extract_state(state: &Bound<'_, PyAny>) -> PyResult<MDState> {
    let pos: PyReadonlyArray2<NpF> = state.getattr("pos")?.extract()?;
    let vel: PyReadonlyArray2<NpF> = state.getattr("vel")?.extract()?;
    let forces: PyReadonlyArray2<NpF> = state.getattr("forces")?.extract()?;
    let energy: F = state.getattr("energy")?.extract()?;
    check_nx3(&pos, "pos")?;
    check_nx3(&vel, "vel")?;
    check_nx3(&forces, "forces")?;
    Ok(MDState {
        pos: pos.as_array().to_owned(),
        vel: vel.as_array().to_owned(),
        forces: forces.as_array().to_owned(),
        energy,
    })
}

fn mass_from(mass: &Bound<'_, PyAny>) -> PyResult<Array1<F>> {
    if let Ok(v) = mass.extract::<F>() {
        if !v.is_finite() || v <= 0.0 {
            return Err(PyValueError::new_err("mass must be strictly positive"));
        }
        return Ok(ndarray::array![v]);
    }
    let arr: PyReadonlyArray1<NpF> = mass.extract().map_err(|_| {
        PyValueError::new_err("mass must be a positive scalar or a 1-D float array")
    })?;
    Ok(arr.as_array().to_owned())
}

#[pyclass(name = "MDState", module = "molrs.md")]
pub struct PyMDState {
    inner: MDState,
}

#[pymethods]
impl PyMDState {
    #[new]
    fn new(
        pos: PyReadonlyArray2<'_, NpF>,
        vel: PyReadonlyArray2<'_, NpF>,
        forces: PyReadonlyArray2<'_, NpF>,
        energy: F,
    ) -> PyResult<Self> {
        check_nx3(&pos, "pos")?;
        check_nx3(&vel, "vel")?;
        check_nx3(&forces, "forces")?;
        Ok(Self {
            inner: MDState {
                pos: pos.as_array().to_owned(),
                vel: vel.as_array().to_owned(),
                forces: forces.as_array().to_owned(),
                energy,
            },
        })
    }

    #[getter]
    fn pos<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<NpF>> {
        self.inner.pos.clone().into_pyarray(py)
    }
    #[getter]
    fn vel<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<NpF>> {
        self.inner.vel.clone().into_pyarray(py)
    }
    #[getter]
    fn forces<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<NpF>> {
        self.inner.forces.clone().into_pyarray(py)
    }
    #[getter]
    fn energy(&self) -> F {
        self.inner.energy
    }
}

#[pyclass(name = "LJ", module = "molrs.md", subclass)]
pub struct PyLJ {
    pub(crate) inner: LJ,
}

#[pymethods]
impl PyLJ {
    #[new]
    #[pyo3(signature = (epsilon, sigma, cutoff, *, n=12, m=6, shifted=true, smeared=false))]
    fn new(
        epsilon: F,
        sigma: F,
        cutoff: F,
        n: i32,
        m: i32,
        shifted: bool,
        smeared: bool,
    ) -> PyResult<Self> {
        Ok(Self {
            inner: LJ::new(epsilon, sigma, cutoff, n, m, shifted, smeared).map_err(md_err)?,
        })
    }

    #[getter]
    fn epsilon(&self) -> F {
        self.inner.epsilon()
    }
    #[getter]
    fn sigma(&self) -> F {
        self.inner.sigma()
    }
    #[getter]
    fn cutoff(&self) -> F {
        self.inner.cutoff()
    }
    #[getter]
    fn n(&self) -> i32 {
        self.inner.n()
    }
    #[getter]
    fn m(&self) -> i32 {
        self.inner.m()
    }
    #[getter]
    fn shifted(&self) -> bool {
        self.inner.shifted()
    }
    #[getter]
    fn smeared(&self) -> bool {
        self.inner.smeared()
    }

    fn pair_energy(&self, r2: F, disp: [F; 3]) -> Option<F> {
        Potential::calc_energy(&self.inner, r2, disp)
    }
    fn pair_force(&self, r2: F, disp: [F; 3]) -> Option<[F; 3]> {
        Potential::calc_force(&self.inner, r2, disp)
    }
    fn pair_eval(&self, r2: F, disp: [F; 3]) -> Option<(F, [F; 3])> {
        Potential::eval(&self.inner, r2, disp)
    }

    fn calc_energy(
        &self,
        neighbors: &mut PyVerletSkin,
        pos: PyReadonlyArray2<'_, NpF>,
    ) -> PyResult<F> {
        check_nx3(&pos, "pos")?;
        let nl = neighbors
            .inner
            .as_mut()
            .ok_or_else(|| PyValueError::new_err("VerletSkin has already been moved"))?;
        self.inner.calc_energy(nl, pos.as_array()).map_err(md_err)
    }

    fn calc_force<'py>(
        &self,
        py: Python<'py>,
        neighbors: &mut PyVerletSkin,
        pos: PyReadonlyArray2<'_, NpF>,
    ) -> PyResult<Bound<'py, PyArray2<NpF>>> {
        check_nx3(&pos, "pos")?;
        let nl = neighbors
            .inner
            .as_mut()
            .ok_or_else(|| PyValueError::new_err("VerletSkin has already been moved"))?;
        let f = self.inner.calc_force(nl, pos.as_array()).map_err(md_err)?;
        Ok(f.into_pyarray(py))
    }

    fn eval<'py>(
        &self,
        py: Python<'py>,
        neighbors: &mut PyVerletSkin,
        pos: PyReadonlyArray2<'_, NpF>,
    ) -> PyResult<(F, Bound<'py, PyArray2<NpF>>)> {
        check_nx3(&pos, "pos")?;
        let nl = neighbors
            .inner
            .as_mut()
            .ok_or_else(|| PyValueError::new_err("VerletSkin has already been moved"))?;
        let (e, f) = self.inner.eval(nl, pos.as_array()).map_err(md_err)?;
        Ok((e, f.into_pyarray(py)))
    }

    fn eval_table<'py>(
        &self,
        py: Python<'py>,
        n_atoms: usize,
        neighbors: &PyNeighbors,
    ) -> PyResult<(F, Bound<'py, PyArray2<NpF>>)> {
        let (e, f) = self
            .inner
            .eval_table(n_atoms, &neighbors.inner)
            .map_err(md_err)?;
        Ok((e, f.into_pyarray(py)))
    }

    #[pyo3(signature = (n_atoms, i, j, disp, dist_sq=None))]
    fn eval_pairs<'py>(
        &self,
        py: Python<'py>,
        n_atoms: usize,
        i: PyReadonlyArray1<'_, u32>,
        j: PyReadonlyArray1<'_, u32>,
        disp: PyReadonlyArray2<'_, NpF>,
        dist_sq: Option<PyReadonlyArray1<'_, NpF>>,
    ) -> PyResult<(F, Bound<'py, PyArray2<NpF>>)> {
        check_nx3(&disp, "disp")?;
        let d2 = match dist_sq.as_ref() {
            Some(a) => Some(a.as_slice()?),
            None => None,
        };
        let (e, f) = self
            .inner
            .eval_pairs(n_atoms, i.as_slice()?, j.as_slice()?, disp.as_array(), d2)
            .map_err(md_err)?;
        Ok((e, f.into_pyarray(py)))
    }
}

#[pyclass(name = "VelocityVerlet", module = "molrs.md", subclass)]
pub struct PyVelocityVerlet {
    inner: VelocityVerlet,
}

#[pymethods]
impl PyVelocityVerlet {
    #[new]
    #[pyo3(signature = (dt, *, potential, neighbors, mass))]
    fn new(
        dt: F,
        potential: &PyLJ,
        neighbors: &mut PyVerletSkin,
        mass: Bound<'_, PyAny>,
    ) -> PyResult<Self> {
        Ok(Self {
            inner: VelocityVerlet::new(
                dt,
                potential.inner,
                neighbors.take()?,
                mass_from(&mass)?.view(),
            )
            .map_err(md_err)?,
        })
    }

    #[getter]
    fn dt(&self) -> F {
        self.inner.dt
    }

    #[getter]
    fn removed_dof(&self) -> usize {
        self.inner.removed_dof()
    }

    fn initial(
        &mut self,
        pos: PyReadonlyArray2<'_, NpF>,
        vel: PyReadonlyArray2<'_, NpF>,
    ) -> PyResult<PyMDState> {
        check_nx3(&pos, "pos")?;
        check_nx3(&vel, "vel")?;
        let state = self
            .inner
            .initial(pos.as_array().to_owned(), vel.as_array().to_owned())
            .map_err(md_err)?;
        Ok(PyMDState { inner: state })
    }

    fn advance(&mut self, state: &Bound<'_, PyAny>) -> PyResult<PyMDState> {
        let next = self.inner.advance(extract_state(state)?).map_err(md_err)?;
        Ok(PyMDState { inner: next })
    }

    fn advance_n(&mut self, state: &Bound<'_, PyAny>, n_steps: usize) -> PyResult<PyMDState> {
        let next = self
            .inner
            .advance_n(extract_state(state)?, n_steps)
            .map_err(md_err)?;
        Ok(PyMDState { inner: next })
    }
}

#[pyclass(name = "Langevin", module = "molrs.md", subclass)]
pub struct PyLangevin {
    inner: Langevin,
}

#[pymethods]
impl PyLangevin {
    #[new]
    #[pyo3(signature = (dt, *, gamma, kbt, potential, neighbors, mass, seed=0))]
    fn new(
        dt: F,
        gamma: F,
        kbt: F,
        potential: &PyLJ,
        neighbors: &mut PyVerletSkin,
        mass: Bound<'_, PyAny>,
        seed: u64,
    ) -> PyResult<Self> {
        Ok(Self {
            inner: Langevin::new(
                dt,
                gamma,
                kbt,
                potential.inner,
                neighbors.take()?,
                mass_from(&mass)?.view(),
                seed,
            )
            .map_err(md_err)?,
        })
    }

    #[getter]
    fn dt(&self) -> F {
        self.inner.dt
    }
    #[getter]
    fn gamma(&self) -> F {
        self.inner.gamma
    }
    #[getter]
    fn c1(&self) -> F {
        self.inner.c1()
    }
    #[getter]
    fn c2(&self) -> F {
        self.inner.c2()
    }
    #[getter]
    fn sigma<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<NpF>> {
        self.inner
            .sigma()
            .view()
            .insert_axis(ndarray::Axis(1))
            .to_owned()
            .into_pyarray(py)
    }
    #[getter]
    fn inv_mass<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<NpF>> {
        self.inner
            .inv_mass()
            .view()
            .insert_axis(ndarray::Axis(1))
            .to_owned()
            .into_pyarray(py)
    }
    #[getter]
    fn removed_dof(&self) -> usize {
        self.inner.removed_dof()
    }

    fn initial(
        &mut self,
        pos: PyReadonlyArray2<'_, NpF>,
        vel: PyReadonlyArray2<'_, NpF>,
    ) -> PyResult<PyMDState> {
        check_nx3(&pos, "pos")?;
        check_nx3(&vel, "vel")?;
        let state = self
            .inner
            .initial(pos.as_array().to_owned(), vel.as_array().to_owned())
            .map_err(md_err)?;
        Ok(PyMDState { inner: state })
    }

    fn step(
        &mut self,
        state: &Bound<'_, PyAny>,
        noise: PyReadonlyArray2<'_, NpF>,
    ) -> PyResult<PyMDState> {
        check_nx3(&noise, "noise")?;
        let next = self
            .inner
            .step(extract_state(state)?, noise.as_array())
            .map_err(md_err)?;
        Ok(PyMDState { inner: next })
    }

    fn advance(&mut self, state: &Bound<'_, PyAny>) -> PyResult<PyMDState> {
        let next = self.inner.advance(extract_state(state)?).map_err(md_err)?;
        Ok(PyMDState { inner: next })
    }

    fn advance_n(&mut self, state: &Bound<'_, PyAny>, n_steps: usize) -> PyResult<PyMDState> {
        let next = self
            .inner
            .advance_n(extract_state(state)?, n_steps)
            .map_err(md_err)?;
        Ok(PyMDState { inner: next })
    }

    fn draw_noise<'py>(&mut self, py: Python<'py>, n_atoms: usize) -> Bound<'py, PyArray2<NpF>> {
        self.inner.draw_noise(n_atoms).into_pyarray(py)
    }
}

#[pyclass(name = "MaxwellBoltzmann", module = "molrs.md", subclass)]
pub struct PyMaxwellBoltzmann {
    inner: MaxwellBoltzmann,
}

#[pymethods]
impl PyMaxwellBoltzmann {
    #[new]
    #[pyo3(signature = (temperature, *, seed=0, remove_com=true))]
    fn new(temperature: F, seed: u64, remove_com: bool) -> PyResult<Self> {
        Ok(Self {
            inner: MaxwellBoltzmann::with_com(temperature, seed, remove_com).map_err(md_err)?,
        })
    }

    #[getter]
    fn temperature(&self) -> F {
        self.inner.temperature
    }
    #[getter]
    fn seed(&self) -> u64 {
        self.inner.seed
    }
    #[getter]
    fn remove_com(&self) -> bool {
        self.inner.remove_com
    }

    fn velocities<'py>(
        &self,
        py: Python<'py>,
        pos: PyReadonlyArray2<'_, NpF>,
        mass: Bound<'_, PyAny>,
    ) -> PyResult<Bound<'py, PyArray2<NpF>>> {
        check_nx3(&pos, "pos")?;
        let vel = self
            .inner
            .velocities(pos.as_array(), mass_from(&mass)?.view())
            .map_err(md_err)?;
        Ok(vel.into_pyarray(py))
    }
}

#[pyfunction(name = "kb_md")]
fn kb_md_py() -> PyResult<F> {
    kb_md().map_err(md_err)
}

#[pyfunction(name = "energy_to_md")]
fn energy_to_md_py(value: F, from_unit: &str) -> PyResult<F> {
    energy_to_md(value, from_unit).map_err(md_err)
}

#[pyfunction(name = "preset_energy_to_md")]
fn preset_energy_to_md_py(style: &str) -> PyResult<F> {
    preset_energy_to_md(style).map_err(md_err)
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("MD_ENERGY", MD_ENERGY)?;
    m.add_class::<PyMDState>()?;
    m.add_class::<PyLJ>()?;
    m.add_class::<PyVelocityVerlet>()?;
    m.add_class::<PyLangevin>()?;
    m.add_class::<PyMaxwellBoltzmann>()?;
    m.add_function(wrap_pyfunction!(kb_md_py, m)?)?;
    m.add_function(wrap_pyfunction!(energy_to_md_py, m)?)?;
    m.add_function(wrap_pyfunction!(preset_energy_to_md_py, m)?)?;
    Ok(())
}
