//! Python bindings for `molrs::optimize` (`molrs.optimize`): the L-BFGS
//! minimizer ([`PyLBFGS`]) and the report it returns ([`PyOptReport`]).
//!
//! An optimizer minimizes a `molrs.ff.potential.Potentials` over a coordinate
//! vector; it neither builds the potentials nor owns a force field.

use pyo3::prelude::*;

use molrs::ff::potential::PotentialCompiler;
use molrs::optimize::{LBFGS, OptReport};

use crate::core::frame::PyFrame;
use crate::ff::ir;
use crate::ff::potential::{PotBacking, PyPotentials, potentials_moved_err};

use ndarray::{Array2, Array3};
use numpy::{PyArray2, PyArray3, PyReadonlyArrayDyn, ToPyArray};

/// Outcome of a geometry optimization, exposed to Python as `molrs.optimize.OptReport`.
#[pyclass(module = "molrs.optimize", name = "OptReport", subclass)]
pub struct PyOptReport {
    inner: OptReport,
}

#[pymethods]
impl PyOptReport {
    /// Whether ``fmax`` convergence was reached within ``max_steps``.
    #[getter]
    fn converged(&self) -> bool {
        self.inner.converged
    }

    /// Number of outer L-BFGS iterations performed.
    #[getter]
    fn n_steps(&self) -> usize {
        self.inner.n_steps
    }

    /// Potential energy at the returned geometry (kcal/mol).
    #[getter]
    fn final_energy(&self) -> f64 {
        self.inner.final_energy
    }

    /// Maximum per-atom force magnitude at the returned geometry
    /// (kcal/mol/angstrom).
    #[getter]
    fn final_fmax(&self) -> f64 {
        self.inner.final_fmax
    }

    fn __repr__(&self) -> String {
        format!(
            "OptReport(converged={}, n_steps={}, final_energy={:.6}, final_fmax={:.6})",
            if self.inner.converged {
                "True"
            } else {
                "False"
            },
            self.inner.n_steps,
            self.inner.final_energy,
            self.inner.final_fmax
        )
    }
}

impl From<OptReport> for PyOptReport {
    fn from(inner: OptReport) -> Self {
        Self { inner }
    }
}

/// L-BFGS geometry optimizer, exposed as `molrs.optimize.LBFGS`.
///
/// Construct with potentials + knobs on ``new``, then ``run`` a :class:`Frame`
/// (primary) or a coordinate array (single / batch by rank).
///
/// Examples
/// --------
/// >>> pots = molrs.ff.potential.PotentialCompiler(molrs.ff.typifier.Mmff94Typifier().forcefield()).compile(frame)
/// >>> opt = molrs.optimize.LBFGS(pots, fmax=0.05, max_steps=500)
/// >>> frame, report = opt.run(frame)
/// >>> coords, report = opt.run(coords)         # (N, 3)
#[pyclass(module = "molrs.optimize", name = "LBFGS", subclass)]
pub struct PyLBFGS {
    potentials: Py<PyPotentials>,
    fmax: f64,
    max_steps: usize,
    max_step: f64,
    memory: usize,
}

#[pymethods]
impl PyLBFGS {
    #[new]
    #[pyo3(signature = (potentials, *, fmax = 0.05, max_steps = 500, max_step = 0.2, memory = 8))]
    fn new(
        potentials: Py<PyPotentials>,
        fmax: f64,
        max_steps: usize,
        max_step: f64,
        memory: usize,
    ) -> Self {
        Self {
            potentials,
            fmax,
            max_steps,
            max_step,
            memory,
        }
    }

    /// Relax a :class:`Frame` or coordinates by L-BFGS.
    ///
    /// * ``Frame`` → ``(Frame, OptReport)`` (frame coordinates updated; a new
    ///   Python frame object is returned with the minimized coords).
    /// * ``(N, 3)`` / ``(3N,)`` → ``((N, 3) array, OptReport)``
    /// * ``(B, N, 3)`` → ``((B, N, 3) array, list[OptReport])``
    fn run<'py>(&self, py: Python<'py>, arg: &Bound<'_, PyAny>) -> PyResult<Bound<'py, PyAny>> {
        // Frame path (primary).
        if let Ok(frame) = arg.extract::<PyRef<'_, PyFrame>>() {
            let mut core = frame.clone_core_frame()?;
            let pots = self.potentials.borrow(py);
            // Compile against this frame if deferred, then minimize with free mask.
            let compiled;
            let pot: &dyn molrs::ff::potential::Potential = match &pots.inner {
                PotBacking::Compiled(p) => p,
                PotBacking::Deferred(ff) => {
                    compiled = PotentialCompiler::new(ff)
                        .compile(&core)
                        .map_err(ir::compile_err)?;
                    &compiled
                }
                PotBacking::Moved => return Err(potentials_moved_err()),
            };
            // Borrowed one-shot on flat coords extracted from frame, then write back.
            let mut xyz = core
                .coords()
                .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
            // `Frame::coords` is a fresh row-major N×3 array: its buffer is the
            // flat `[x0, y0, z0, …]` the minimizer takes.
            let flat = xyz
                .as_slice_mut()
                .expect("Frame::coords returns a standard-layout array");
            let report = LBFGS::minimize(
                pot,
                flat,
                self.fmax,
                self.max_steps,
                self.max_step,
                self.memory,
            )
            .map_err(pyo3::exceptions::PyValueError::new_err)?;
            crate::ff::potential::take_err(&pots.err_slots)?;
            core.set_coords(xyz.view())
                .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
            let out_frame = PyFrame::from_core_frame(core)?;
            return Ok((out_frame, PyOptReport::from(report))
                .into_pyobject(py)?
                .into_any());
        }

        let pots = self.potentials.borrow(py);
        let pot = pots.inner.compiled()?;
        let readonly = arg.extract::<PyReadonlyArrayDyn<'_, f64>>()?;
        let arr = readonly.as_array();
        let shape = arr.shape();
        match shape.len() {
            1 | 2 => {
                let mut flat: Vec<f64> = arr.iter().copied().collect();
                let n_elem = flat.len();
                if !n_elem.is_multiple_of(3) {
                    return Err(pyo3::exceptions::PyValueError::new_err(format!(
                        "coords has {n_elem} elements, not a multiple of 3 (expected (N, 3) or (3N,))"
                    )));
                }
                let report = LBFGS::minimize(
                    pot,
                    &mut flat,
                    self.fmax,
                    self.max_steps,
                    self.max_step,
                    self.memory,
                )
                .map_err(pyo3::exceptions::PyValueError::new_err)?;
                crate::ff::potential::take_err(&pots.err_slots)?;
                let out: Bound<'py, PyArray2<f64>> = Array2::from_shape_vec((n_elem / 3, 3), flat)
                    .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?
                    .to_pyarray(py);
                Ok((out, PyOptReport::from(report))
                    .into_pyobject(py)?
                    .into_any())
            }
            3 => {
                if shape[2] != 3 {
                    return Err(pyo3::exceptions::PyValueError::new_err(format!(
                        "batch coords must be (B, N, 3); trailing axis is {} not 3",
                        shape[2]
                    )));
                }
                let (b, n) = (shape[0], shape[1]);
                let expected = pot.n_atoms();
                if expected != 0 && n != expected {
                    return Err(pyo3::exceptions::PyValueError::new_err(format!(
                        "structure atom count N={n} does not match this Potentials' atom count {expected}"
                    )));
                }
                let mut flat: Vec<f64> = arr.iter().copied().collect();
                let reports = LBFGS::minimize_batch(
                    pot,
                    &mut flat,
                    n,
                    b,
                    self.fmax,
                    self.max_steps,
                    self.max_step,
                    self.memory,
                )
                .map_err(pyo3::exceptions::PyValueError::new_err)?;
                crate::ff::potential::take_err(&pots.err_slots)?;
                let out: Bound<'py, PyArray3<f64>> = Array3::from_shape_vec((b, n, 3), flat)
                    .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?
                    .to_pyarray(py);
                let reports: Vec<PyOptReport> =
                    reports.into_iter().map(PyOptReport::from).collect();
                Ok((out, reports).into_pyobject(py)?.into_any())
            }
            other => Err(pyo3::exceptions::PyValueError::new_err(format!(
                "arg must be Frame, 1-D (3N,), 2-D (N, 3), or 3-D (B, N, 3); got {other}-D array"
            ))),
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "LBFGS(fmax={}, max_steps={}, max_step={}, memory={})",
            self.fmax, self.max_steps, self.max_step, self.memory
        )
    }
}

/// Register `molrs.optimize`.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyOptReport>()?;
    m.add_class::<PyLBFGS>()?;
    Ok(())
}
