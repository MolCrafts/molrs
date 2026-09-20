//! Python bindings for `molrs::md`.
//!
//! ```text
//! VelocityVerlet(dt, potential=lj, neighbors=nl, mass=mass)
//! VelocityVerlet(dt, potential=potentials, mass=mass)   # ff Potentials / mix
//! LJCut.pair_energy / pair_force / pair_eval → per-pair
//! LJCut.eval(neighbors, pos) → (energy, forces)
//! class MyPotential(Potential): …  — subclass the abstract base, molrs calls it
//! Potentials  — the collection merging members (molrs.Potentials)
//! ```
//!
//! One `Potential` concept everywhere: `LJCut` (nonbond), the force-field
//! `Potentials` collection, and a duck-typed Python override. MD has no
//! unit knowledge. Integrators own the optional `VerletSkin`.

use std::sync::{Arc, Mutex};

use crate::core::spatial::neighborlist::{PyNeighbors, PyVerletSkin};
use crate::core::spatial::simbox::PyBox;
use crate::ff::PyPotentials;
use crate::helpers::NpF;
use molrs::ff::potential::pair::{LJCut, PairPotential};
use molrs::ff::potential::{Member, Potential};
use molrs::math::Virial;
use molrs::md::{
    Direct, ForceProvider, Langevin, MDState, MaxwellBoltzmann, MdError, MicPairs, VelocityVerlet,
};
use molrs::types::F;
use ndarray::{Array1, Array2};
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::{PyTypeError, PyValueError};
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

/// A Python-side neighbour argument picks the minimum-image provider; the
/// ghost régime is not bound yet.
fn provider(
    members: Members,
    skin: Option<molrs::spatial::neighbors::VerletSkin>,
) -> PyResult<Box<dyn ForceProvider>> {
    // `MicPairs` refuses a kernel whose parameters were resolved against a
    // fixed pair list — it would ignore the neighbour table and answer for the
    // list it was built from, while the driver went on rebuilding and reporting
    // that table. `ForceField.to_potentials` builds exactly such kernels, so
    // this is the path where that mistake is made, and the error says what to
    // build instead.
    Ok(match skin {
        Some(s) => Box::new(MicPairs::from_members(members, s).map_err(md_err)?),
        None => {
            if members.len() != 1 {
                return Err(PyValueError::new_err(
                    "several members need a neighbour table to share; pass neighbors=",
                ));
            }
            let (pot, special) = members.into_iter().next().expect("checked above");
            if !special.is_empty() {
                return Err(PyValueError::new_err(
                    "special-bonds weights apply to a pair table; pass neighbors=",
                ));
            }
            Box::new(Direct::new(pot))
        }
    })
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
        images: numpy::ndarray::Array2::zeros((pos.as_array().nrows(), 3)),
        pos: pos.as_array().to_owned(),
        vel: vel.as_array().to_owned(),
        forces: forces.as_array().to_owned(),
        energy,
        virial: None,
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

/// Dynamical state advanced by the integrators.
///
/// Fields are settable (float64 `(N, 3)` arrays / a float energy) so hooks can
/// replace them wholesale: `state.vel = new_vel`. Getters return copies —
/// in-place slice writes (`state.vel[:] = …`) do NOT write through.
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
                images: numpy::ndarray::Array2::zeros((pos.as_array().nrows(), 3)),
                pos: pos.as_array().to_owned(),
                vel: vel.as_array().to_owned(),
                forces: forces.as_array().to_owned(),
                energy,
                virial: None,
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

    /// Accumulated box crossings ``(N, 3)``, one signed count per lattice
    /// vector.
    ///
    /// A wrapped coordinate on its own has lost the atom's history. `pos +
    /// H·images` is the continuous position, and mean-squared displacement,
    /// diffusion and any other path-dependent quantity read that, not `pos`.
    #[getter]
    fn images<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<i64>> {
        self.inner.images.clone().into_pyarray(py)
    }

    #[setter]
    fn set_images(&mut self, value: PyReadonlyArray2<'_, i64>) -> PyResult<()> {
        let v = value.as_array();
        if v.ncols() != 3 || v.nrows() != self.inner.pos.nrows() {
            return Err(PyValueError::new_err(
                "images must have shape (N, 3) matching pos",
            ));
        }
        self.inner.images = v.to_owned();
        Ok(())
    }

    /// Scalar pressure from the virial, a kinetic energy and a cell volume.
    ///
    /// ``None`` when no virial was tallied — not zero. Units are the caller's,
    /// as everywhere in MD: the result is in whatever `energy / volume` is.
    fn pressure(&self, kinetic: F, volume: F) -> Option<F> {
        self.inner.virial.map(|w| w.pressure(kinetic, volume))
    }

    /// Virial `Σ f ⊗ r` as ``(xx, yy, zz, xy, xz, yz)``, or ``None``.
    ///
    /// ``None`` means the force provider does not tally one — not that it is
    /// zero. A pressure computed from a fabricated zero is wrong and looks
    /// entirely plausible.
    #[getter]
    fn virial(&self) -> Option<[F; 6]> {
        self.inner.virial.map(|w: Virial| w.components)
    }

    #[setter]
    fn set_pos(&mut self, pos: PyReadonlyArray2<'_, NpF>) -> PyResult<()> {
        check_nx3(&pos, "pos")?;
        self.inner.pos = pos.as_array().to_owned();
        Ok(())
    }
    #[setter]
    fn set_vel(&mut self, vel: PyReadonlyArray2<'_, NpF>) -> PyResult<()> {
        check_nx3(&vel, "vel")?;
        self.inner.vel = vel.as_array().to_owned();
        Ok(())
    }
    #[setter]
    fn set_forces(&mut self, forces: PyReadonlyArray2<'_, NpF>) -> PyResult<()> {
        check_nx3(&forces, "forces")?;
        self.inner.forces = forces.as_array().to_owned();
        Ok(())
    }
    #[setter]
    fn set_energy(&mut self, energy: F) {
        self.inner.energy = energy;
    }

    fn __repr__(&self) -> String {
        format!(
            "MDState(n_atoms={}, energy={})",
            self.inner.pos.nrows(),
            self.inner.energy
        )
    }
}

/// LAMMPS ``pair_style lj/cut``: cut Lennard-Jones / Mie pair kernel, and
/// md's nonbond potential (the loop feeds it the current neighbour pairs).
#[pyclass(name = "LJCut", module = "molrs.md", subclass)]
pub struct PyLJCut {
    pub(crate) inner: LJCut,
}

#[pymethods]
impl PyLJCut {
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
            inner: LJCut::new(epsilon, sigma, cutoff, n, m, shifted, smeared)
                .map_err(PyValueError::new_err)?,
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
        self.inner.pair_energy(r2, disp)
    }
    fn pair_force(&self, r2: F, disp: [F; 3]) -> Option<[F; 3]> {
        self.inner.pair_force(r2, disp)
    }
    fn pair_eval(&self, r2: F, disp: [F; 3]) -> Option<(F, [F; 3])> {
        self.inner.pair_eval(r2, disp)
    }

    fn calc_energy_forces<'py>(
        &self,
        py: Python<'py>,
        pos: PyReadonlyArray2<'_, NpF>,
    ) -> PyResult<(F, Bound<'py, PyArray2<NpF>>)> {
        check_nx3(&pos, "pos")?;
        let view = pos.as_array();
        let n = view.nrows();
        // A standard-layout `(N, 3)` array *is* the flat `[x0, y0, z0, …]` a
        // kernel wants; only a strided view is copied.
        let owned: Vec<F>;
        let flat: &[F] = match view.as_slice() {
            Some(slice) => slice,
            None => {
                owned = view.iter().copied().collect();
                &owned
            }
        };
        let (energy, forces) = Potential::calc_energy_forces(&self.inner, flat);
        let arr = Array2::from_shape_vec((n, 3), forces)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok((energy, arr.into_pyarray(py)))
    }

    fn eval<'py>(
        &self,
        py: Python<'py>,
        neighbors: &mut PyVerletSkin,
        pos: PyReadonlyArray2<'_, NpF>,
    ) -> PyResult<(F, Bound<'py, PyArray2<NpF>>)> {
        check_nx3(&pos, "pos")?;
        let nl = neighbors.get_mut()?;
        let (e, f) = self
            .inner
            .eval(nl, pos.as_array())
            .map_err(PyValueError::new_err)?;
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
            .map_err(PyValueError::new_err)?;
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
            .map_err(PyValueError::new_err)?;
        Ok((e, f.into_pyarray(py)))
    }
}

// ---------------------------------------------------------------------------
// The one Potential seam — Python subclasses and post-evaluation error relay.
// ---------------------------------------------------------------------------

/// Shared slot where a Python-subclass potential parks an exception raised
/// mid-evaluation — the `Potential` trait has no error channel, so the
/// evaluation returns NaNs and the Python-facing caller that drove it checks
/// the slot and re-raises the original exception.
pub(crate) type ErrSlot = Arc<Mutex<Option<PyErr>>>;

/// Re-raise the first parked exception, clearing its slot.
pub(crate) fn take_err(slots: &[ErrSlot]) -> PyResult<()> {
    for slot in slots {
        if let Some(err) = slot.lock().expect("error slot poisoned").take() {
            return Err(err);
        }
    }
    Ok(())
}

/// A Python `md.Potential` subclass instance as the one `Potential` concept —
/// the seam for NN / external forces. Holds a reference to the instance and
/// dispatches to its overridden ``calc_energy_forces`` under the GIL.
pub struct SubclassPotential {
    obj: Py<PyAny>,
    error: ErrSlot,
}

impl SubclassPotential {
    fn call(&self, py: Python<'_>, coords: &[F]) -> PyResult<(F, Vec<F>)> {
        let n = coords.len() / 3;
        let pos = Array2::from_shape_vec((n, 3), coords.to_vec())
            .expect("flat coords have 3N elements")
            .into_pyarray(py);
        let result = self
            .obj
            .bind(py)
            .call_method1("calc_energy_forces", (pos,))?;
        let (energy, forces): (F, PyReadonlyArray2<'_, NpF>) = result.extract().map_err(|_| {
            PyValueError::new_err(
                "Potential.calc_energy_forces must return \
                 (energy: float, forces: float64 (N, 3) ndarray)",
            )
        })?;
        let forces = forces.as_array();
        if forces.shape() != [n, 3] {
            return Err(PyValueError::new_err(format!(
                "Potential.calc_energy_forces returned forces shape {:?} for {n} atoms",
                forces.shape()
            )));
        }
        Ok((energy, forces.iter().copied().collect()))
    }
}

impl Potential for SubclassPotential {
    fn calc_energy_forces(&self, coords: &[F]) -> (F, Vec<F>) {
        Python::attach(|py| match self.call(py, coords) {
            Ok(out) => out,
            Err(err) => {
                *self.error.lock().expect("error slot poisoned") = Some(err);
                (F::NAN, vec![F::NAN; coords.len()])
            }
        })
    }
}

/// Move the Rust potential out of any exposed potential class.
///
/// Arm order is a hard invariant: concrete Rust types first, duck-typed
/// fallback last. Putting the fallback first would wrap every `Potentials`
/// as a Python dispatch object.
/// The members a provider will evaluate, with the weights each one takes.
///
/// A [`TypedPotentials`](crate::ff::PyTypedPotentials) already knows both —
/// which kernel is which and how its close neighbours are scaled — because
/// `ForceField::to_typed_potentials` decided it. Anything else is one member
/// that scales nothing.
pub(crate) type Members = Vec<(Member, molrs::md::SpecialWeights)>;

pub(crate) fn take_members(obj: &Bound<'_, PyAny>) -> PyResult<(Members, Vec<ErrSlot>)> {
    if let Ok(typed) = obj.cast::<crate::ff::PyTypedPotentials>() {
        let members = typed.borrow_mut().members.take().ok_or_else(|| {
            PyValueError::new_err(
                "these TypedPotentials were already given to an integrator; \
                 build them again from the force field",
            )
        })?;
        return Ok((members, Vec::new()));
    }
    let (pot, slots) = take_potential(obj)?;
    Ok((vec![(pot, molrs::md::SpecialWeights::default())], slots))
}

pub(crate) fn take_potential(obj: &Bound<'_, PyAny>) -> PyResult<(Member, Vec<ErrSlot>)> {
    // Each arm also settles which part the member plays. A pair kernel and an
    // aggregate of them read a neighbour table; a duck-typed Python object has
    // only `calc_energy_forces`, so it reads coordinates and nothing else —
    // and, being unable to tally a virial over pairs, makes the step's virial
    // `None` rather than a number that moves with the box origin.
    if let Ok(lj) = obj.cast::<PyLJCut>() {
        return Ok((Member::pair(lj.borrow().inner.clone()), Vec::new()));
    }
    if let Ok(pots) = obj.cast::<PyPotentials>() {
        let (inner, slots) = pots.borrow_mut().take_compiled()?;
        return Ok((Member::pair(inner), slots));
    }
    if obj.hasattr("calc_energy_forces")? && obj.getattr("calc_energy_forces")?.is_callable() {
        let error: ErrSlot = Arc::default();
        return Ok((
            Member::plain(SubclassPotential {
                obj: obj.clone().unbind(),
                error: Arc::clone(&error),
            }),
            vec![error],
        ));
    }
    Err(PyTypeError::new_err(
        "expected a potential with callable calc_energy_forces (LJCut, Potentials, or duck-typed)",
    ))
}

// ---------------------------------------------------------------------------
// Integrators
// ---------------------------------------------------------------------------

#[pyclass(name = "VelocityVerlet", module = "molrs.md", subclass)]
pub struct PyVelocityVerlet {
    inner: VelocityVerlet,
    err_slots: Vec<ErrSlot>,
}

#[pymethods]
impl PyVelocityVerlet {
    #[new]
    #[pyo3(signature = (dt, *, potential, neighbors=None, mass, simbox=None))]
    fn new(
        dt: F,
        potential: &Bound<'_, PyAny>,
        neighbors: Option<&Bound<'_, PyVerletSkin>>,
        mass: Bound<'_, PyAny>,
        simbox: Option<PyBox>,
    ) -> PyResult<Self> {
        if potential.cast::<PyLJCut>().is_ok() && neighbors.is_none() {
            return Err(PyValueError::new_err(
                "an LJCut pair kernel needs neighbors= (a VerletSkin)",
            ));
        }
        // Validate mass before moving the potential / neighbour state in.
        let mass = mass_from(&mass)?;
        let (members, err_slots) = take_members(potential)?;
        let skin = match neighbors {
            Some(nl) => Some(nl.borrow_mut().take()?),
            None => None,
        };
        Ok(Self {
            inner: VelocityVerlet::new(
                dt,
                provider(members, skin)?,
                mass.view(),
                simbox.map(|b| b.inner),
            )
            .map_err(md_err)?,
            err_slots,
        })
    }

    #[getter]
    fn dt(&self) -> F {
        self.inner.dt()
    }

    #[getter]
    fn removed_dof(&self) -> usize {
        self.inner.removed_dof()
    }

    /// Number of pair edges in the current list (``None`` without neighbors).
    #[getter]
    fn num_edges(&self) -> Option<usize> {
        self.inner.forces().neighbor_stats().edges
    }

    /// Neighbour-list rebuilds since construction (``None`` without neighbors).
    #[getter]
    fn rebuild_count(&self) -> Option<usize> {
        self.inner.forces().neighbor_stats().rebuilds
    }

    /// Updates since the last rebuild (``None`` without neighbors).
    #[getter]
    fn ago(&self) -> Option<usize> {
        self.inner.forces().neighbor_stats().ago
    }

    fn initial(
        &mut self,
        pos: PyReadonlyArray2<'_, NpF>,
        vel: PyReadonlyArray2<'_, NpF>,
    ) -> PyResult<PyMDState> {
        check_nx3(&pos, "pos")?;
        check_nx3(&vel, "vel")?;
        let result = self
            .inner
            .initial(pos.as_array().to_owned(), vel.as_array().to_owned());
        take_err(&self.err_slots)?;
        Ok(PyMDState {
            inner: result.map_err(md_err)?,
        })
    }

    fn advance(&mut self, state: &Bound<'_, PyAny>) -> PyResult<PyMDState> {
        let result = self.inner.advance(extract_state(state)?);
        take_err(&self.err_slots)?;
        Ok(PyMDState {
            inner: result.map_err(md_err)?,
        })
    }

    fn advance_n(&mut self, state: &Bound<'_, PyAny>, n_steps: usize) -> PyResult<PyMDState> {
        let result = self.inner.advance_n(extract_state(state)?, n_steps);
        take_err(&self.err_slots)?;
        Ok(PyMDState {
            inner: result.map_err(md_err)?,
        })
    }
}

#[pyclass(name = "Langevin", module = "molrs.md", subclass)]
pub struct PyLangevin {
    inner: Langevin,
    err_slots: Vec<ErrSlot>,
}

#[pymethods]
impl PyLangevin {
    #[new]
    #[pyo3(signature = (dt, *, gamma, kbt, potential, neighbors=None, mass, seed=0, simbox=None))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        dt: F,
        gamma: F,
        kbt: F,
        potential: &Bound<'_, PyAny>,
        neighbors: Option<&Bound<'_, PyVerletSkin>>,
        mass: Bound<'_, PyAny>,
        seed: u64,
        simbox: Option<PyBox>,
    ) -> PyResult<Self> {
        if potential.cast::<PyLJCut>().is_ok() && neighbors.is_none() {
            return Err(PyValueError::new_err(
                "an LJCut pair kernel needs neighbors= (a VerletSkin)",
            ));
        }
        // Validate the scheme knobs and mass before moving anything in.
        if gamma <= 0.0 {
            return Err(PyValueError::new_err(
                "Langevin requires gamma > 0; use VelocityVerlet for NVE",
            ));
        }
        if kbt <= 0.0 {
            return Err(PyValueError::new_err("Langevin requires kbt > 0"));
        }
        let mass = mass_from(&mass)?;
        let (members, err_slots) = take_members(potential)?;
        let skin = match neighbors {
            Some(nl) => Some(nl.borrow_mut().take()?),
            None => None,
        };
        Ok(Self {
            inner: Langevin::new(
                dt,
                gamma,
                kbt,
                provider(members, skin)?,
                mass.view(),
                seed,
                simbox.map(|b| b.inner),
            )
            .map_err(md_err)?,
            err_slots,
        })
    }

    #[getter]
    fn dt(&self) -> F {
        self.inner.dt()
    }
    #[getter]
    fn gamma(&self) -> F {
        self.inner.gamma()
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

    /// Number of pair edges in the current list (``None`` without neighbors).
    #[getter]
    fn num_edges(&self) -> Option<usize> {
        self.inner.forces().neighbor_stats().edges
    }

    /// Neighbour-list rebuilds since construction (``None`` without neighbors).
    #[getter]
    fn rebuild_count(&self) -> Option<usize> {
        self.inner.forces().neighbor_stats().rebuilds
    }

    /// Updates since the last rebuild (``None`` without neighbors).
    #[getter]
    fn ago(&self) -> Option<usize> {
        self.inner.forces().neighbor_stats().ago
    }

    fn initial(
        &mut self,
        pos: PyReadonlyArray2<'_, NpF>,
        vel: PyReadonlyArray2<'_, NpF>,
    ) -> PyResult<PyMDState> {
        check_nx3(&pos, "pos")?;
        check_nx3(&vel, "vel")?;
        let result = self
            .inner
            .initial(pos.as_array().to_owned(), vel.as_array().to_owned());
        take_err(&self.err_slots)?;
        Ok(PyMDState {
            inner: result.map_err(md_err)?,
        })
    }

    fn step(
        &mut self,
        state: &Bound<'_, PyAny>,
        noise: PyReadonlyArray2<'_, NpF>,
    ) -> PyResult<PyMDState> {
        check_nx3(&noise, "noise")?;
        let result = self.inner.step(extract_state(state)?, noise.as_array());
        take_err(&self.err_slots)?;
        Ok(PyMDState {
            inner: result.map_err(md_err)?,
        })
    }

    fn advance(&mut self, state: &Bound<'_, PyAny>) -> PyResult<PyMDState> {
        let result = self.inner.advance(extract_state(state)?);
        take_err(&self.err_slots)?;
        Ok(PyMDState {
            inner: result.map_err(md_err)?,
        })
    }

    fn advance_n(&mut self, state: &Bound<'_, PyAny>, n_steps: usize) -> PyResult<PyMDState> {
        let result = self.inner.advance_n(extract_state(state)?, n_steps);
        take_err(&self.err_slots)?;
        Ok(PyMDState {
            inner: result.map_err(md_err)?,
        })
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
    #[pyo3(signature = (kbt, *, seed=0, remove_com=true))]
    fn new(kbt: F, seed: u64, remove_com: bool) -> PyResult<Self> {
        let mut inner = MaxwellBoltzmann::new(kbt, seed).map_err(md_err)?;
        if !remove_com {
            inner = inner.keep_com();
        }
        Ok(Self { inner })
    }

    #[getter]
    fn kbt(&self) -> F {
        self.inner.kbt()
    }
    #[getter]
    fn seed(&self) -> u64 {
        self.inner.seed()
    }
    #[getter]
    fn remove_com(&self) -> bool {
        self.inner.remove_com()
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

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyMDState>()?;
    m.add_class::<PyLJCut>()?;
    m.add_class::<PyVelocityVerlet>()?;
    m.add_class::<PyLangevin>()?;
    m.add_class::<PyMaxwellBoltzmann>()?;
    Ok(())
}
