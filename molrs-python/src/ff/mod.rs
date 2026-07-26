//! Python wrappers for MMFF force-field typification and compiled potentials.
//!
//! The workflow is:
//!
//! 1. Create a typifier — [`PyMMFF94Typifier`] (MMFF94) or [`PyMMFF94STypifier`]
//!    (MMFF94s, the "static" variant). Both load their embedded parameter set at
//!    construction; the variant is the class, never a flag.
//! 2. Call `typify` to assign atom types + bonded parameters, producing a typed
//!    [`PyAtomistic`] (materialize it with `to_frame()` for a [`PyFrame`]).
//! 3. Build the neighbour list (`molrs.intramolecular_pairs`) and compile with
//!    `typifier.forcefield().to_potentials(frame)` — the same route every other
//!    force field in molrs uses.
//! 4. Use [`PyPotentials::eval`] to evaluate energy and forces on flat
//!    coordinate arrays.
//!
//! There is deliberately **no** one-step `build(mol)` and no free
//! `build_mmff_potentials(mol)`. Both existed, sat adjacent in the same namespace
//! with nothing to tell them apart, and one of them silently omitted the entire
//! electrostatic term (150 kcal/mol on caffeine) because no `ForceField` ever
//! defined `pair/mmff_ele`. A typifier's contract is `typify`; compiling
//! potentials is `ForceField.to_potentials`.
//!
//! The antechamber-derived bindings live in their own modules rather than here:
//! [`atd`] (the ATD atom typifier, one engine over seven `ATOMTYPE_*.DEF` tables)
//! and [`charge`] (the three charge models). This file is already large, and they
//! are self-contained.
//!
//! # References
//!
//! - Halgren, T.A. (1996). J. Comput. Chem. 17, 490-519. (MMFF94 force field)
//! - Halgren, T.A. (1999). J. Comput. Chem. 20, 720-729. (MMFF94s option)

pub mod atd;
pub mod charge;

use std::collections::HashMap;
use std::ffi::CString;
use std::fs;

use pyo3::exceptions::{PyKeyError, PyNotImplementedError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyCapsule, PyDict, PyList, PyTuple};

use molrs::ff::ForceField;
use molrs::ff::potential::{Potentials, extract_coords, write_coords};
use molrs::ff::typifier::mmff::{MMFF94STypifier, MMFF94Typifier};
use molrs::ff::typifier::opls::OPLSAATypifier;
use molrs::optimize::{LBFGS, OptReport};
use molrs_ffi::ForceFieldRef;

use crate::core::store::block::PyBlock;
use crate::core::store::frame::PyFrame;
use crate::core::system::molgraph::PyAtomistic;
use crate::helpers::{NpF, py_value_err};

use ndarray::{Array2, Array3};
use numpy::{PyArray1, PyArray2, PyArray3, PyReadonlyArrayDyn, ToPyArray};

/// Nominal Python base for every graph typifier.
///
/// The algorithm contract already lives in the Rust
/// `molrs::ff::typifier::Typifier` trait. This is its Python nominal
/// counterpart: native typifiers extend it and downstream Python typifiers may
/// subclass it.
#[pyclass(module = "molrs", name = "Typifier", subclass)]
pub struct PyTypifier;

#[pymethods]
impl PyTypifier {
    #[new]
    #[pyo3(signature = (*_args, **_kwargs))]
    fn new(_args: &Bound<'_, PyTuple>, _kwargs: Option<&Bound<'_, PyDict>>) -> Self {
        // Python typifiers inherit this native nominal base and commonly expose
        // their own ``__init__(engine, ...)``.  ``object.__new__`` accepts those
        // subclass constructor arguments; the native base must do the same and
        // leave interpretation to the Python ``__init__``.
        Self
    }

    fn typify(&self, _mol: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        Err(PyNotImplementedError::new_err(
            "Typifier.typify must be implemented by a concrete typifier",
        ))
    }
}

/// Outcome of a geometry optimization, exposed to Python as `molrs.OptReport`.
#[pyclass(module = "molrs", name = "OptReport")]
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

/// Compiled force-field potentials for energy and force evaluation.
///
/// Exposed to Python as `molrs.Potentials`.
///
/// Operates on flat coordinate arrays in the layout
/// ``[x0, y0, z0, x1, y1, z1, ...]`` (length 3N).
///
/// Examples
/// --------
/// >>> typifier = MMFF94Typifier()
/// >>> frame = typifier.typify(mol).to_frame()
/// >>> frame["pairs"] = molrs.intramolecular_pairs(frame)
/// >>> potentials = typifier.forcefield().to_potentials(frame)
/// >>> energy, forces = potentials.eval(coords)
#[pyclass(module = "molrs", name = "Potentials")]
pub struct PyPotentials {
    inner: PotBacking,
}

/// A [`PyPotentials`] is either already compiled against a molecule's topology
/// (the MMFF / pre-bound path) or *deferred*: it holds the force field and binds
/// the topology lazily from the `Frame` passed to ``calc_energy``/``calc_forces``.
/// Deferred is what ``ForceField.to_potentials()`` (no frame) returns, matching
/// the molpy evaluation model where the frame enters at evaluation time.
enum PotBacking {
    Compiled(Potentials),
    Deferred(ForceField),
}

impl PotBacking {
    /// The compiled potentials, or an error if this set is still deferred and
    /// no `Frame` has been supplied to bind its topology.
    fn compiled(&self) -> PyResult<&Potentials> {
        match self {
            PotBacking::Compiled(p) => Ok(p),
            PotBacking::Deferred(_) => Err(PyValueError::new_err(
                "this Potentials is not bound to a molecule; \
                 call calc_energy(frame)/calc_forces(frame) with a Frame, \
                 or build it from a typifier",
            )),
        }
    }
}

/// Force-field definition metadata exposed to Python as `molrs.ForceField`.
#[pyclass(module = "molrs", name = "ForceField", subclass)]
pub struct PyForceField {
    pub(crate) inner: ForceField,
}

/// CL&Pol fragment scaling data backed by the native force-field layer.
#[pyclass(module = "molrs", name = "FragmentScaling", frozen, get_all, skip_from_py_object)]
#[derive(Clone)]
pub struct PyFragmentScaling {
    name: String,
    q: f64,
    mu: f64,
    alpha: f64,
    polarizable: bool,
}

impl From<PyFragmentScaling> for molrs::ff::FragmentScaling {
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

impl From<molrs::ff::FragmentScaling> for PyFragmentScaling {
    fn from(value: molrs::ff::FragmentScaling) -> Self {
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
    molrs::ff::compute_k_ij(&fr_i.clone().into(), &fr_j.clone().into(), r)
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
        native_fragments.push(molrs::ff::FragmentAtoms {
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

    let inner = molrs::ff::scale_lj(
        &ff.borrow().inner,
        &native_fragments,
        &scaling,
        scale_sigma,
    )
    .map_err(|error| match error {
        molrs::ff::ScaleLjError::MissingFragment(name) => {
            PyKeyError::new_err(format!("no scaling data for fragment '{name}'"))
        }
        other => py_value_err(other),
    })?;
    let public = py.import("molrs")?.getattr("ForceField")?;
    let native = py.get_type::<PyForceField>();
    if public.is(&native) {
        return Py::new(py, PyForceField { inner });
    }
    let name = inner.name.clone();
    let object: Py<PyForceField> = public.call1((name,))?.extract()?;
    object.borrow_mut(py).inner = inner;
    Ok(object)
}

/// Convert an optional Python ``dict[str, float]`` of parameters into owned
/// ``(key, value)`` pairs. A missing dict yields no params.
fn params_from_dict(params: Option<&Bound<'_, PyDict>>) -> PyResult<Vec<(String, f64)>> {
    let mut out = Vec::new();
    if let Some(d) = params {
        for (k, v) in d.iter() {
            out.push((k.extract::<String>()?, v.extract::<f64>()?));
        }
    }
    Ok(out)
}

/// Borrow owned param pairs as the `&[(&str, f64)]` the builder API expects.
fn as_pairs(owned: &[(String, f64)]) -> Vec<(&str, f64)> {
    owned.iter().map(|(k, v)| (k.as_str(), *v)).collect()
}

impl PyPotentials {
    /// Evaluate energy + forces against either a [`PyFrame`] (binds topology and
    /// reads coordinates from the frame's ``atoms`` block — the molpy model) or a
    /// flat coordinate array (requires already-compiled potentials).
    fn eval_any(&self, arg: &Bound<'_, PyAny>) -> PyResult<(f64, Vec<NpF>)> {
        if let Ok(frame) = arg.extract::<PyRef<'_, PyFrame>>() {
            let core = frame.clone_core_frame()?;
            let coords = extract_coords(&core).map_err(PyValueError::new_err)?;
            let ef = match &self.inner {
                PotBacking::Compiled(p) => p.calc_energy_forces(&coords),
                PotBacking::Deferred(ff) => ff
                    .to_potentials(&core)
                    .map_err(PyValueError::new_err)?
                    .calc_energy_forces(&coords),
            };
            return Ok(ef);
        }
        let arr = arg.extract::<numpy::PyReadonlyArray1<'_, NpF>>()?;
        let slice = arr.as_slice()?;
        Ok(self.inner.compiled()?.calc_energy_forces(slice))
    }
}

#[pymethods]
impl PyPotentials {
    /// Number of compiled potential kernels, or ``0`` while still deferred
    /// (not yet bound to a molecule).
    fn __len__(&self) -> usize {
        match &self.inner {
            PotBacking::Compiled(p) => p.len(),
            PotBacking::Deferred(_) => 0,
        }
    }

    /// Returns ``(energy, forces)``, forces shape ``(N, 3)``.
    fn calc_energy_forces<'py>(
        &self,
        py: Python<'py>,
        arg: &Bound<'_, PyAny>,
    ) -> PyResult<(f64, Bound<'py, PyArray2<NpF>>)> {
        let (energy, forces) = self.eval_any(arg)?;
        let n = forces.len() / 3;
        Ok((
            energy,
            Array2::from_shape_vec((n, 3), forces)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?
                .to_pyarray(py),
        ))
    }

    /// Evaluate total energy (kcal/mol) against a :class:`Frame` or coordinates.
    fn calc_energy(&self, arg: &Bound<'_, PyAny>) -> PyResult<f64> {
        Ok(self.eval_any(arg)?.0)
    }

    /// Compute forces (= -gradient) in kcal/(mol·Å), shape ``(N, 3)``.
    fn calc_forces<'py>(
        &self,
        py: Python<'py>,
        arg: &Bound<'_, PyAny>,
    ) -> PyResult<Bound<'py, PyArray2<NpF>>> {
        let forces = self.eval_any(arg)?.1;
        let n = forces.len() / 3;
        Ok(Array2::from_shape_vec((n, 3), forces)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?
            .to_pyarray(py))
    }

    fn __repr__(&self) -> String {
        match &self.inner {
            PotBacking::Compiled(p) => format!("Potentials(n_kernels={})", p.len()),
            PotBacking::Deferred(_) => "Potentials(deferred)".to_string(),
        }
    }
}

/// L-BFGS geometry optimizer, exposed as `molrs.LBFGS`.
///
/// Construct with potentials + knobs on ``new``, then ``run`` a :class:`Frame`
/// (primary) or a coordinate array (single / batch by rank).
///
/// Examples
/// --------
/// >>> pots = molrs.MMFF94Typifier().forcefield().to_potentials(frame)
/// >>> opt = molrs.LBFGS(pots, fmax=0.05, max_steps=500)
/// >>> frame, report = opt.run(frame)
/// >>> coords, report = opt.run(coords)         # (N, 3)
#[pyclass(module = "molrs", name = "LBFGS")]
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
    fn run<'py>(
        &self,
        py: Python<'py>,
        arg: &Bound<'_, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        // Frame path (primary).
        if let Ok(frame) = arg.extract::<PyRef<'_, PyFrame>>() {
            let mut core = frame.clone_core_frame()?;
            let pots = self.potentials.borrow(py);
            // Compile against this frame if deferred, then minimize with free mask.
            let compiled;
            let pot: &dyn molrs::ff::potential::Potential = match &pots.inner {
                PotBacking::Compiled(p) => p,
                PotBacking::Deferred(ff) => {
                    compiled = ff
                        .to_potentials(&core)
                        .map_err(pyo3::exceptions::PyValueError::new_err)?;
                    &compiled
                }
            };
            // Borrowed one-shot on flat coords extracted from frame, then write back.
            let mut flat = extract_coords(&core).map_err(pyo3::exceptions::PyValueError::new_err)?;
            let report = LBFGS::minimize(
                pot,
                &mut flat,
                self.fmax,
                self.max_steps,
                self.max_step,
                self.memory,
            )
            .map_err(pyo3::exceptions::PyValueError::new_err)?;
            write_coords(&mut core, &flat).map_err(pyo3::exceptions::PyValueError::new_err)?;
            let out_frame = PyFrame::from_core_frame(core)?;
            return Ok((out_frame, PyOptReport::from(report))
                .into_pyobject(py)?
                .into_any());
        }

        let pots = self.potentials.borrow(py);
        let pot = pots.inner.compiled()?;
        let readonly = arg.extract::<PyReadonlyArrayDyn<'_, NpF>>()?;
        let arr = readonly.as_array();
        let shape = arr.shape();
        match shape.len() {
            1 | 2 => {
                let mut flat: Vec<NpF> = arr.iter().copied().collect();
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
                let out: Bound<'py, PyArray2<NpF>> = Array2::from_shape_vec((n_elem / 3, 3), flat)
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
                let mut flat: Vec<NpF> = arr.iter().copied().collect();
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
                let out: Bound<'py, PyArray3<NpF>> = Array3::from_shape_vec((b, n, 3), flat)
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

/// Bind one MMFF front door to Python.
///
/// MMFF94 and MMFF94s are the same engine over two parameter sets, and molrs
/// exposes them as two **named types** rather than one type with a variant flag —
/// so the binder mirrors that shape exactly: two `#[pyclass]`es, each wrapping its
/// own core typifier, generated from one forwarding body so they cannot drift.
macro_rules! py_mmff_front_door {
    (
        $(#[$doc:meta])*
        $py_ty:ident, $core:ty, $name:literal
    ) => {
        $(#[$doc])*
        #[pyclass(module = "molrs", name = $name, extends = PyTypifier)]
        pub struct $py_ty {
            inner: $core,
        }

        #[pymethods]
        impl $py_ty {
            /// Create the typifier with its embedded parameter tables.
            ///
            /// Never fails: the parameter set is compiled into the extension module.
            #[new]
            fn new() -> (Self, PyTypifier) {
                (
                    Self {
                        inner: <$core>::new(),
                    },
                    PyTypifier,
                )
            }

            /// Assign MMFF atom types (and this variant's bonded parameters) to a
            /// molecular graph.
            ///
            /// Parameters
            /// ----------
            /// mol : Atomistic
            ///     Molecular graph with element symbols and bonds.
            ///
            /// Returns
            /// -------
            /// Atomistic
            ///     Typed molecular graph. Call ``typed.to_frame()`` explicitly when a
            ///     tabular representation is needed. The improper rows carry ``koop``
            ///     (md*A*rad^-2) and the dihedral rows ``v1``/``v2``/``v3``, both
            ///     resolved from *this* class's parameter set.
            ///
            /// Raises
            /// ------
            /// ValueError
            ///     If atom types cannot be determined (e.g. unsupported elements).
            fn typify(&self, py: Python<'_>, mol: &PyAtomistic) -> PyResult<Py<PyAtomistic>> {
                let typed = self
                    .inner
                    .typify(mol.core())
                    .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
                PyAtomistic::from_core(py, typed)
            }

            /// Return the underlying force-field definition.
            ///
            /// This is the seam to the standard compile path — the typifier
            /// labels the graph, the force field compiles it::
            ///
            ///     typed = typifier.typify(mol)
            ///     frame = typed.to_frame()
            ///     frame["pairs"] = molrs.intramolecular_pairs(frame)
            ///     pots  = typifier.forcefield().to_potentials(frame)
            fn forcefield(&self) -> PyForceField {
                PyForceField {
                    inner: self.inner.ff().clone(),
                }
            }

            fn __repr__(&self) -> String {
                format!("{}(forcefield='{}')", $name, self.inner.ff().name)
            }
        }
    };
}

py_mmff_front_door! {
    /// MMFF94 atom-type assigner.
    ///
    /// Exposed to Python as `molrs.MMFF94Typifier`.
    ///
    /// Loads the embedded MMFF94 parameter tables at construction time. Use
    /// :meth:`typify` to label a molecular graph (atom types, partial charges, and
    /// the per-instance force constants the kernels read), then compile it through
    /// :meth:`forcefield` — the standard route, shared with every other force
    /// field in molrs.
    ///
    /// See :class:`MMFF94STypifier` for the "static" variant used in energy
    /// minimization.
    ///
    /// # References
    ///
    /// - Halgren, T.A. (1996). J. Comput. Chem. 17, 490-519.
    ///
    /// Examples
    /// --------
    /// >>> typifier = MMFF94Typifier()
    /// >>> frame = typifier.typify(mol).to_frame()          # labels + charges
    /// >>> frame["pairs"] = molrs.intramolecular_pairs(frame)
    /// >>> pots = typifier.forcefield().to_potentials(frame)
    PyMMFF94Typifier, MMFF94Typifier, "MMFF94Typifier"
}

py_mmff_front_door! {
    /// MMFF94s ("static") atom-type assigner and potential builder.
    ///
    /// Exposed to Python as `molrs.MMFF94STypifier`.
    ///
    /// Identical to :class:`MMFF94Typifier` except on delocalised trivalent
    /// nitrogen (MMFF numeric types 10 ``NC=O`` and 40 ``NC=C``), where MMFF94s
    /// re-parameterises 11 out-of-plane rows and 42 torsion rows so the nitrogen
    /// minimizes to a **planar** geometry — the one seen in crystal structures.
    ///
    /// The mechanism is the out-of-plane force constant ``koop`` (md*A*rad^-2) that
    /// :meth:`typify` bakes onto the improper rows. The kernel evaluates
    /// ``E_oop = 0.5 * 143.9325 * koop * chi**2`` with ``chi`` the Wilson
    /// out-of-plane angle in radians, so ``koop > 0`` makes the planar centre an
    /// energy minimum. MMFF94s sets it to ``+0.015`` (type 10) / ``+0.030``
    /// (type 40); MMFF94's values on those rows run from ``-0.033`` to ``+0.004``.
    ///
    /// All 95 atom types, and every bond / angle / stretch-bend / vdW / charge
    /// parameter, are shared with MMFF94 — so a molecule with no such nitrogen gets
    /// bit-for-bit the same answer from both classes.
    ///
    /// # References
    ///
    /// - Halgren, T.A. (1999). J. Comput. Chem. 20, 720-729. (MMFF94s)
    ///
    /// Examples
    /// --------
    /// >>> typifier = MMFF94STypifier()
    /// >>> typifier.forcefield().name
    /// 'MMFF94s'
    PyMMFF94STypifier, MMFF94STypifier, "MMFF94STypifier"
}

fn oplsaa_source_xml(source: Option<&Bound<'_, PyAny>>) -> PyResult<Option<String>> {
    let Some(source) = source else {
        return Ok(None);
    };
    let raw = match source.extract::<String>() {
        Ok(value) => value,
        Err(_) => source.call_method0("__fspath__")?.extract::<String>()?,
    };
    if raw.trim_start().starts_with('<') {
        return Ok(Some(raw));
    }
    fs::read_to_string(&raw).map(Some).map_err(|err| {
        PyValueError::new_err(format!("failed to read OPLS-AA XML source {raw:?}: {err}"))
    })
}

/// OPLS-AA atom-type assigner and potential builder.
///
/// Exposed to Python as `molrs.OPLSAATypifier`. It loads the embedded canonical
/// OPLS-AA parameter set by default, or reads one XML source at construction.
/// :meth:`typify` returns a typed :class:`Atomistic`; use :meth:`build` for the
/// one-step potential compilation path.
///
/// Parameters
/// ----------
/// source : str or path-like, optional
///     OPLS-AA XML text or a path to an XML file. ``None`` uses the embedded
///     canonical OPLS-AA table.
/// strict : bool, default True
///     When True, a bonded term with no force-field match is an error. When
///     False, such terms are skipped (left unparametrized).
///
/// Examples
/// --------
/// >>> typifier = OPLSAATypifier()
/// >>> typed = typifier.typify(mol)        # typed Atomistic
/// >>> potentials = typifier.build(mol)    # compiled Potentials
#[pyclass(module = "molrs", name = "OPLSAATypifier", extends = PyTypifier)]
pub struct PyOPLSAATypifier {
    inner: OPLSAATypifier,
}

#[pymethods]
impl PyOPLSAATypifier {
    /// Create an OPLS-AA typifier from embedded data, XML text, or an XML path.
    #[new]
    #[pyo3(signature = (source = None, *, strict = true))]
    fn new(
        source: Option<&Bound<'_, PyAny>>,
        strict: bool,
    ) -> PyResult<(Self, PyTypifier)> {
        let typifier = match oplsaa_source_xml(source)? {
            Some(xml) => OPLSAATypifier::from_xml_str(&xml)
                .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?,
            None => OPLSAATypifier::oplsaa().map_err(|e| {
                pyo3::exceptions::PyRuntimeError::new_err(format!(
                    "failed to initialize OPLS-AA: {e}"
                ))
            })?,
        }
        .with_strict(strict);
        Ok((Self { inner: typifier }, PyTypifier))
    }

    /// Assign OPLS-AA atom and bonded-term types to a molecular graph.
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If atom typing fails.
    fn typify(&self, py: Python<'_>, mol: &PyAtomistic) -> PyResult<Py<PyAtomistic>> {
        let labeled = self
            .inner
            .typify(mol.core())
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        PyAtomistic::from_core(py, labeled)
    }

    /// Typify and compile potentials in one step.
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If typification or compilation fails.
    fn build(&self, mol: &PyAtomistic) -> PyResult<PyPotentials> {
        let potentials = self
            .inner
            .build(mol.core())
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        Ok(PyPotentials {
            inner: PotBacking::Compiled(potentials),
        })
    }

    /// Return the underlying force-field definition.
    fn forcefield(&self) -> PyForceField {
        PyForceField {
            inner: self.inner.ff().clone(),
        }
    }

    fn __repr__(&self) -> String {
        format!("OPLSAATypifier(forcefield='{}')", self.inner.ff().name)
    }
}

/// Extract a flat coordinate array from a Frame's ``"atoms"`` block.
///
/// Reads the ``x``, ``y``, ``z`` columns from the ``"atoms"`` block and
/// interleaves them into a flat 1D array: ``[x0, y0, z0, x1, y1, z1, ...]``.
///
/// Parameters
/// ----------
/// frame : Frame
///     Frame with an ``"atoms"`` block containing ``x``, ``y``, ``z``
///     float columns.
///
/// Returns
/// -------
/// numpy.ndarray, shape (3*N,), dtype float
///     Flat coordinate array suitable for :meth:`Potentials.eval`.
///
/// Raises
/// ------
/// ValueError
///     If the ``"atoms"`` block or required columns are missing.
///
/// Examples
/// --------
/// >>> coords = extract_coords(frame)
/// >>> energy, forces = potentials.eval(coords)
#[pyfunction]
#[pyo3(name = "extract_coords")]
pub fn extract_coords_py<'py>(
    py: Python<'py>,
    frame: &PyFrame,
) -> PyResult<Bound<'py, PyArray1<NpF>>> {
    let core_frame = frame.clone_core_frame()?;
    let coords = extract_coords(&core_frame)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    Ok(coords.to_pyarray(py))
}

/// Read a force-field definition from an XML file.
#[pyfunction]
#[pyo3(name = "read_forcefield_xml")]
pub fn read_forcefield_xml_py(path: &str) -> PyResult<PyForceField> {
    let forcefield = molrs::ff::read_forcefield_xml(path)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    Ok(PyForceField { inner: forcefield })
}

/// `Send` wrapper around a `*mut ForceFieldRef` so it can ride inside a
/// `PyCapsule` (whose payload must be `Send`).
///
/// `ForceFieldRef` is `!Send` (it holds an `Rc`) and raw pointers are `!Send`,
/// but the capsule is only ever created, read, and destroyed while the Python
/// GIL is held, so no cross-thread `Rc` access occurs. `#[repr(transparent)]`
/// makes the capsule's `void*` reinterpretable as `*mut *mut ForceFieldRef`,
/// matching the frame convention a consumer resolves (mirrors
/// [`crate::core::store::frame`]'s `FrameRefPtr`).
#[repr(transparent)]
struct ForceFieldRefPtr(*mut ForceFieldRef);

// SAFETY: GIL-guarded, single-threaded use only — see the type-level doc.
unsafe impl Send for ForceFieldRefPtr {}

#[pymethods]
impl PyForceField {
    /// Construct an empty force field. Populate it with the ``def_*style`` /
    /// ``def_*type`` builder methods, or load one with :func:`read_forcefield_xml`.
    #[new]
    #[pyo3(signature = (name = "forcefield", units = "real"))]
    fn new(name: &str, units: &str) -> Self {
        // ``units`` is carried by the Python ergonomic layer (``molrs.ForceField``),
        // accepted here so subclasses can forward their ``(name, units)`` ctor.
        let _ = units;
        Self {
            inner: ForceField::new(name),
        }
    }

    #[getter]
    fn name(&self) -> String {
        self.inner.name.clone()
    }

    fn style_names(&self) -> Vec<String> {
        self.inner
            .styles()
            .iter()
            .map(|style| format!("{}:{}", style.category(), style.name))
            .collect()
    }

    /// Export this force field's FFI handle as a ``PyCapsule``.
    ///
    /// The force-field analogue of :meth:`Frame._ffi_frameref_capsule`. The
    /// capsule wraps a :class:`molrs_ffi.ForceFieldRef` that **shares** this
    /// force field's parameters (one ``Rc`` clone — no deep copy), so a
    /// downstream Rust consumer (e.g. the molpack relaxer) can resolve it and
    /// compile potentials with **no marshalling**. The capsule's ``void*`` is
    /// ``*mut *mut`` :class:`molrs_ffi.ForceFieldRef`, matching the frame
    /// convention; its name is the C string ``"molrs.ForceFieldRef"``. The
    /// capsule's destructor reclaims the boxed handle, dropping its ``Rc``.
    ///
    /// Returns
    /// -------
    /// capsule
    ///     A ``PyCapsule`` named ``"molrs.ForceFieldRef"``.
    fn _ffi_forcefield_capsule<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyCapsule>> {
        // Box a shared handle (Rc clone of this force field) and hand the raw
        // pointer to the capsule. See `ForceFieldRefPtr` for the Send / layout
        // contract.
        let raw = ForceFieldRefPtr(Box::into_raw(Box::new(ForceFieldRef::new(
            self.inner.clone(),
        ))));
        let name = CString::new("molrs.ForceFieldRef").expect("static capsule name");
        PyCapsule::new_with_destructor(py, raw, Some(name), |ptr: ForceFieldRefPtr, _ctx| {
            // SAFETY: `ptr.0` came from `Box::into_raw` above and is reclaimed
            // exactly once when the capsule dies.
            drop(unsafe { Box::from_raw(ptr.0) });
        })
    }

    // -- builder: styles (idempotent find-or-create) -------------------------

    /// Ensure an atom style ``name`` exists.
    fn def_atomstyle(&mut self, name: &str) {
        self.inner.def_atomstyle(name);
    }

    /// Ensure a bond style ``name`` exists.
    fn def_bondstyle(&mut self, name: &str) {
        self.inner.def_bondstyle(name);
    }

    /// Ensure an angle style ``name`` exists.
    fn def_anglestyle(&mut self, name: &str) {
        self.inner.def_anglestyle(name);
    }

    /// Ensure a dihedral style ``name`` exists.
    fn def_dihedralstyle(&mut self, name: &str) {
        self.inner.def_dihedralstyle(name);
    }

    /// Ensure an improper style ``name`` exists.
    fn def_improperstyle(&mut self, name: &str) {
        self.inner.def_improperstyle(name);
    }

    /// Ensure a pair style ``name`` exists, with optional style-level params
    /// (e.g. ``{"cutoff": 10.0}``).
    #[pyo3(signature = (name, params = None))]
    fn def_pairstyle(&mut self, name: &str, params: Option<&Bound<'_, PyDict>>) -> PyResult<()> {
        let owned = params_from_dict(params)?;
        self.inner.def_pairstyle(name, &as_pairs(&owned));
        Ok(())
    }

    /// Ensure a k-space style ``name`` exists, with optional style-level params.
    #[pyo3(signature = (name, params = None))]
    fn def_kspacestyle(&mut self, name: &str, params: Option<&Bound<'_, PyDict>>) -> PyResult<()> {
        let owned = params_from_dict(params)?;
        self.inner.def_kspacestyle(name, &as_pairs(&owned));
        Ok(())
    }

    // -- builder: types ------------------------------------------------------

    /// Define an atom type under atom style ``style``.
    #[pyo3(signature = (style, name, params = None))]
    fn def_atomtype(
        &mut self,
        style: &str,
        name: &str,
        params: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<()> {
        let owned = params_from_dict(params)?;
        self.inner
            .def_atomstyle(style)
            .def_atomtype(name, &as_pairs(&owned));
        Ok(())
    }

    /// Define a bond type ``itom-jtom`` under bond style ``style``.
    #[pyo3(signature = (style, itom, jtom, params = None))]
    fn def_bondtype(
        &mut self,
        style: &str,
        itom: &str,
        jtom: &str,
        params: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<()> {
        let owned = params_from_dict(params)?;
        self.inner
            .def_bondstyle(style)
            .def_bondtype(itom, jtom, &as_pairs(&owned));
        Ok(())
    }

    /// Define an angle type ``itom-jtom-ktom`` under angle style ``style``.
    #[pyo3(signature = (style, itom, jtom, ktom, params = None))]
    fn def_angletype(
        &mut self,
        style: &str,
        itom: &str,
        jtom: &str,
        ktom: &str,
        params: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<()> {
        let owned = params_from_dict(params)?;
        self.inner
            .def_anglestyle(style)
            .def_angletype(itom, jtom, ktom, &as_pairs(&owned));
        Ok(())
    }

    /// Define a dihedral type ``itom-jtom-ktom-ltom`` under dihedral style ``style``.
    #[pyo3(signature = (style, itom, jtom, ktom, ltom, params = None))]
    fn def_dihedraltype(
        &mut self,
        style: &str,
        itom: &str,
        jtom: &str,
        ktom: &str,
        ltom: &str,
        params: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<()> {
        let owned = params_from_dict(params)?;
        self.inner.def_dihedralstyle(style).def_dihedraltype(
            itom,
            jtom,
            ktom,
            ltom,
            &as_pairs(&owned),
        );
        Ok(())
    }

    /// Define an improper type ``itom-jtom-ktom-ltom`` under improper style ``style``.
    #[pyo3(signature = (style, itom, jtom, ktom, ltom, params = None))]
    fn def_impropertype(
        &mut self,
        style: &str,
        itom: &str,
        jtom: &str,
        ktom: &str,
        ltom: &str,
        params: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<()> {
        let owned = params_from_dict(params)?;
        self.inner.def_improperstyle(style).def_impropertype(
            itom,
            jtom,
            ktom,
            ltom,
            &as_pairs(&owned),
        );
        Ok(())
    }

    /// Define a pair type under pair style ``style``. ``jtom`` defaults to a
    /// self-pair (``itom`` against itself).
    #[pyo3(signature = (style, itom, jtom = None, params = None))]
    fn def_pairtype(
        &mut self,
        style: &str,
        itom: &str,
        jtom: Option<&str>,
        params: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<()> {
        let owned = params_from_dict(params)?;
        self.inner
            .def_pairstyle(style, &[])
            .def_pairtype(itom, jtom, &as_pairs(&owned));
        Ok(())
    }

    /// Unified type definition. ``category`` is one of ``atom``/``bond``/
    /// ``angle``/``dihedral``/``improper``/``pair``; ``name`` encodes the atom
    /// types in the dash form for that category (``"A"``, ``"A-B"``,
    /// ``"A-B-C"``, ``"A-B-C-D"``). The name grammar and arity validation live
    /// in ``molrs-ff`` (``ForceField::def_type``); a malformed name raises
    /// ``ValueError`` rather than panicking across the FFI boundary.
    #[pyo3(signature = (category, style, name, params = None))]
    fn def_type(
        &mut self,
        category: &str,
        style: &str,
        name: &str,
        params: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<()> {
        let owned = params_from_dict(params)?;
        let pairs = as_pairs(&owned);
        self.inner
            .def_type(category, style, name, &pairs)
            .map_err(py_value_err)
    }

    // -- read accessors (round-trip + P1-A migration) ------------------------

    /// Style-level params for ``category``/``style`` (e.g. a pair style's cutoff).
    fn style_params<'py>(
        &self,
        py: Python<'py>,
        category: &str,
        style: &str,
    ) -> PyResult<Bound<'py, PyDict>> {
        let s = self
            .inner
            .get_style(category, style)
            .ok_or_else(|| PyValueError::new_err(format!("no {category} style named '{style}'")))?;
        let d = PyDict::new(py);
        for (k, v) in s.params.iter() {
            d.set_item(k, v)?;
        }
        Ok(d)
    }

    /// List ``(type_name, params)`` tuples for ``category``/``style``.
    fn types<'py>(
        &self,
        py: Python<'py>,
        category: &str,
        style: &str,
    ) -> PyResult<Bound<'py, PyList>> {
        let s = self
            .inner
            .get_style(category, style)
            .ok_or_else(|| PyValueError::new_err(format!("no {category} style named '{style}'")))?;
        let out = PyList::empty(py);
        for (name, params) in s.defs.collect_type_params() {
            let d = PyDict::new(py);
            for (k, v) in params.iter() {
                d.set_item(k, v)?;
            }
            for (k, v) in params.iter_strings() {
                d.set_item(k, v)?;
            }
            out.append((name, d))?;
        }
        Ok(out)
    }

    // -- handle-view support (Style/Type live in the Python layer over these) --

    /// Endpoint atom-type names of one type, e.g. ``["CT","CT"]`` for a bond.
    /// ``None`` if no such type; ``[]`` for atom styles.
    fn type_endpoints(
        &self,
        category: &str,
        style: &str,
        name: &str,
    ) -> PyResult<Option<Vec<String>>> {
        let s = self
            .inner
            .get_style(category, style)
            .ok_or_else(|| PyValueError::new_err(format!("no {category} style named '{style}'")))?;
        Ok(s.type_endpoints(name))
    }

    /// Set (or add) a single param on one type. Raises if the type is absent.
    fn set_type_param(
        &mut self,
        category: &str,
        style: &str,
        name: &str,
        key: &str,
        value: f64,
    ) -> PyResult<()> {
        let s = self
            .inner
            .get_style_mut(category, style)
            .ok_or_else(|| PyValueError::new_err(format!("no {category} style named '{style}'")))?;
        if s.set_type_param(name, key, value) {
            Ok(())
        } else {
            Err(PyValueError::new_err(format!(
                "no {category} type named '{name}' in style '{style}'"
            )))
        }
    }

    /// Set (or add) a single **string** param on one type (e.g. ``element``).
    /// Raises if the type is absent.
    fn set_type_str_param(
        &mut self,
        category: &str,
        style: &str,
        name: &str,
        key: &str,
        value: &str,
    ) -> PyResult<()> {
        let s = self
            .inner
            .get_style_mut(category, style)
            .ok_or_else(|| PyValueError::new_err(format!("no {category} style named '{style}'")))?;
        if s.set_type_str_param(name, key, value) {
            Ok(())
        } else {
            Err(PyValueError::new_err(format!(
                "no {category} type named '{name}' in style '{style}'"
            )))
        }
    }

    /// Rename every type ``old`` -> ``new`` in ``(category, style)``; returns count.
    fn rename_type(
        &mut self,
        category: &str,
        style: &str,
        old: &str,
        new: &str,
    ) -> PyResult<usize> {
        let s = self
            .inner
            .get_style_mut(category, style)
            .ok_or_else(|| PyValueError::new_err(format!("no {category} style named '{style}'")))?;
        Ok(s.rename_type(old, new))
    }

    /// Remove every type ``name`` in ``(category, style)``; returns count.
    fn remove_type(&mut self, category: &str, style: &str, name: &str) -> PyResult<usize> {
        let s = self
            .inner
            .get_style_mut(category, style)
            .ok_or_else(|| PyValueError::new_err(format!("no {category} style named '{style}'")))?;
        Ok(s.remove_type(name))
    }

    /// Remove a whole style ``(category, name)``; returns whether one was removed.
    fn remove_style(&mut self, category: &str, name: &str) -> bool {
        self.inner.remove_style(category, name)
    }

    /// Build evaluable :class:`Potentials` from this force field.
    ///
    /// Called with no argument, the result is *deferred*: it captures the force
    /// field and binds a molecule's topology + coordinates later, from the
    /// :class:`Frame` passed to ``calc_energy(frame)`` / ``calc_forces(frame)``
    /// (the molpy evaluation model). Optionally pass a typed ``frame`` here to
    /// bind eagerly.
    ///
    /// The frame (here or at eval) must carry the topology + ``type`` columns
    /// each style resolves (``atoms``/``bonds``/``angles``/``dihedrals``/
    /// ``impropers``/``pairs``), as produced by a typifier or external emitter.
    ///
    /// Parameters
    /// ----------
    /// frame : Frame, optional
    ///     Typed molecular data to bind eagerly. If omitted, binding is deferred
    ///     to evaluation time.
    ///
    /// Returns
    /// -------
    /// Potentials
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If (when binding) a style has no registered kernel, a topology block
    ///     is missing, or a type label is unknown.
    #[pyo3(signature = (frame = None))]
    fn to_potentials(&self, frame: Option<&PyFrame>) -> PyResult<PyPotentials> {
        match frame {
            None => Ok(PyPotentials {
                inner: PotBacking::Deferred(self.inner.clone()),
            }),
            Some(frame) => {
                let core = frame.clone_core_frame()?;
                let potentials = self
                    .inner
                    .to_potentials(&core)
                    .map_err(pyo3::exceptions::PyValueError::new_err)?;
                Ok(PyPotentials {
                    inner: PotBacking::Compiled(potentials),
                })
            }
        }
    }

    /// Project this force field onto the types a typed :class:`Frame` uses.
    ///
    /// Reading a full force field yields every type it defines, but a concrete
    /// typed structure references only a fraction of them. ``subset`` returns a
    /// new, smaller :class:`ForceField` restricted to exactly the types named
    /// in the frame's per-block ``type`` columns
    /// (``atoms``/``bonds``/``angles``/``dihedrals``/``impropers``), leaving the
    /// original force field unchanged. A ``PairType`` is kept iff both of its
    /// endpoint atom types are used; styles left with no types are dropped; type
    /// names are preserved verbatim (no renumbering).
    ///
    /// Parameters
    /// ----------
    /// frame : Frame
    ///     Typed molecular data, as produced by a typifier or an emitter.
    ///
    /// Returns
    /// -------
    /// ForceField
    ///     A new force field containing only the types ``frame`` references.
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If the frame's blocks cannot be read.
    ///
    /// Examples
    /// --------
    /// >>> mini = ff.subset(typed_frame)
    /// >>> len(mini.style_names()) <= len(ff.style_names())
    /// True
    fn subset(&self, frame: &PyFrame) -> PyResult<PyForceField> {
        let core = frame.clone_core_frame()?;
        let pruned = self.inner.subset(&core);
        Ok(PyForceField { inner: pruned })
    }

    fn __repr__(&self) -> String {
        format!(
            "ForceField(name='{}', styles={})",
            self.inner.name,
            self.inner.styles().len()
        )
    }
}

/// Parse a force-field definition from an XML string (same schema as
/// :func:`read_forcefield_xml`).
#[pyfunction]
#[pyo3(name = "read_forcefield_xml_str")]
pub fn read_forcefield_xml_str_py(xml: &str) -> PyResult<PyForceField> {
    let forcefield = molrs::ff::read_forcefield_xml_str(xml)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    Ok(PyForceField { inner: forcefield })
}

/// Read an OPLS-AA / GROMACS force-field XML file into a :class:`ForceField`.
///
/// Parses the OpenMM-style OPLS-AA XML (GROMACS units — nm, kJ/mol,
/// Ryckaert-Bellemans torsions) and normalizes it to molrs units (Å, kcal/mol,
/// radians, e): bond/angle/pair conversions plus the RB → OPLS 4-cosine
/// (``f1..f4``) inversion happen in the reader, so the returned force field is
/// pure molrs units. Distinct from :func:`read_forcefield_xml`, which reads
/// molrs's own native schema.
///
/// Parameters
/// ----------
/// path : str
///     Path to an ``oplsaa.xml`` (OpenMM/GROMACS layout).
///
/// Returns
/// -------
/// ForceField
///
/// Raises
/// ------
/// ValueError
///     On a malformed document, an unknown section, or a missing/non-numeric
///     required attribute (reading is total — never a silent skip).
#[pyfunction]
#[pyo3(name = "read_opls_xml")]
pub fn read_opls_xml_py(path: &str) -> PyResult<PyForceField> {
    use molrs::ff::ForceFieldReader;
    let forcefield = molrs::ff::OplsXmlReader::new()
        .read(path)
        .map_err(pyo3::exceptions::PyValueError::new_err)?;
    Ok(PyForceField { inner: forcefield })
}

/// Parse an OPLS-AA / GROMACS force field from an XML string (same schema and
/// unit normalization as :func:`read_opls_xml`).
#[pyfunction]
#[pyo3(name = "read_opls_xml_str")]
pub fn read_opls_xml_str_py(xml: &str) -> PyResult<PyForceField> {
    use molrs::ff::ForceFieldReader;
    let forcefield = molrs::ff::OplsXmlReader::new()
        .read_str(xml)
        .map_err(pyo3::exceptions::PyValueError::new_err)?;
    Ok(PyForceField { inner: forcefield })
}

/// Read a LAMMPS force-field include (``*.ff``) into a :class:`ForceField`.
///
/// Parses the ``pair_style``/``pair_coeff`` + ``bond_style``/``angle_style``/
/// ``dihedral_style`` (``fourier``) [+ optional ``improper_style``] include that
/// molpy's ``LAMMPSForceFieldWriter`` emits (AMBER/GAFF flavour), normalizing it
/// to molrs units (Å, kcal/mol, radians, e): LAMMPS harmonic ``K`` → molrs
/// ``k = 2K``, angle/phase values stay in degrees (the kernels convert), and the
/// ``fourier`` dihedral maps to the molrs ``periodic`` kernel. AMBER 1-4 scaling
/// (LJ ×0.5, Coulomb ×1/1.2) is recorded on the force field's special bonds.
/// Distinct from :func:`read_forcefield_xml` (molrs's own schema) and
/// :func:`read_opls_xml` (OPLS-AA / GROMACS XML).
///
/// Per-atom charge and mass live in the LAMMPS *data* file, not this include, so
/// they are not read here: Coulomb charges are drawn from the frame at evaluation
/// time and masses are irrelevant to geometry relaxation.
///
/// Parameters
/// ----------
/// path : str
///     Path to a LAMMPS force-field include (``*.ff``).
///
/// Returns
/// -------
/// ForceField
///
/// Raises
/// ------
/// ValueError
///     On an unsupported style, a coefficient before its style declaration, a
///     wrong-arity type label, or a non-numeric parameter (reading is total —
///     never a silent skip).
#[pyfunction]
#[pyo3(name = "read_lammps_forcefield")]
pub fn read_lammps_forcefield_py(path: &str) -> PyResult<PyForceField> {
    use molrs::ff::ForceFieldReader;
    let forcefield = molrs::ff::LammpsFfReader::new()
        .read(path)
        .map_err(pyo3::exceptions::PyValueError::new_err)?;
    Ok(PyForceField { inner: forcefield })
}

/// Parse a LAMMPS force-field include from a string (same format and unit
/// normalization as :func:`read_lammps_forcefield`).
#[pyfunction]
#[pyo3(name = "read_lammps_forcefield_str")]
pub fn read_lammps_forcefield_str_py(text: &str) -> PyResult<PyForceField> {
    use molrs::ff::ForceFieldReader;
    let forcefield = molrs::ff::LammpsFfReader::new()
        .read_str(text)
        .map_err(pyo3::exceptions::PyValueError::new_err)?;
    Ok(PyForceField { inner: forcefield })
}

/// Build the intramolecular non-bonded neighbour list for a typed frame.
///
/// Returns a :class:`Block` with ``atomi`` / ``atomj`` / ``is_14`` columns — the
/// exact list :meth:`ForceField.to_potentials` consumes for the pair (van der
/// Waals + Coulomb) kernels. 1-2 and 1-3 neighbours are excluded (from the
/// frame's ``bonds`` / ``angles`` blocks); 1-4 pairs (from ``dihedrals``) are
/// flagged so the kernels apply the force field's special-bonds scaling.
///
/// Insert the result as the frame's ``"pairs"`` block before
/// :meth:`ForceField.to_potentials` when you need the non-bonded terms — e.g. a
/// single-molecule geometry optimization where intramolecular van der Waals
/// drives chain collapse. (``to_potentials`` silently drops the pair styles when
/// no ``"pairs"`` block is present, so bonded-only optimizations need nothing.)
///
/// Parameters
/// ----------
/// frame : Frame
///     A typed frame with ``atoms`` and the topology blocks
///     (``bonds`` / ``angles`` / ``dihedrals``) used for exclusions.
///
/// Returns
/// -------
/// Block
#[pyfunction]
#[pyo3(name = "intramolecular_pairs")]
pub fn intramolecular_pairs_py(frame: &PyFrame) -> PyResult<PyBlock> {
    let core = frame.clone_core_frame()?;
    PyBlock::from_core_block(molrs::ff::potential::intramolecular_pairs(&core))
}
