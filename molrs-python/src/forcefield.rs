//! Python wrappers for MMFF94 force-field typification and compiled potentials.
//!
//! The workflow is:
//!
//! 1. Create an [`PyMMFFTypifier`] (loads embedded MMFF94 parameters).
//! 2. Call [`PyMMFFTypifier::typify`] to assign atom types, producing a typed
//!    [`PyFrame`].
//! 3. Call [`PyMMFFTypifier::build`] to compile potentials directly from an
//!    [`PyAtomistic`] graph.
//! 4. Use [`PyPotentials::eval`] to evaluate energy and forces on flat
//!    coordinate arrays.
//!
//! # References
//!
//! - Halgren, T.A. (1996). J. Comput. Chem. 17, 490-519. (MMFF94 force field)

use std::ffi::CString;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyCapsule, PyDict, PyList};

use molrs::ff::ForceField;
use molrs::ff::mmff::{MmffForceField, MmffMolProperties, MmffVariant};
use molrs::ff::potential::{Potentials, extract_coords};
use molrs::ff::typifier::Typifier;
use molrs::ff::typifier::mmff::MMFFTypifier;
use molrs::ff::typifier::opls::OplsTypifier;
use molrs::optimize::{LBFGS, LbfgsConfig, OptReport};
use molrs_ffi::ForceFieldRef;

use crate::block::PyBlock;
use crate::frame::PyFrame;
use crate::helpers::{NpF, py_value_err};
use crate::molgraph::PyAtomistic;

use ndarray::{Array2, Array3};
use numpy::{PyArray1, PyArray2, PyArray3, PyReadonlyArrayDyn, ToPyArray};

/// Outcome of a geometry optimization, exposed to Python as `molrs.OptReport`.
#[pyclass(name = "OptReport")]
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
/// >>> typifier = MMFFTypifier()
/// >>> potentials = typifier.build(mol)
/// >>> energy, forces = potentials.eval(coords)
#[pyclass(name = "Potentials")]
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
#[pyclass(name = "ForceField", subclass)]
pub struct PyForceField {
    pub(crate) inner: ForceField,
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
/// Mirrors molpy: construct with the potentials + config, then ``run`` (single
/// or homogeneous batch, dispatched on the input rank).
///
/// Examples
/// --------
/// >>> pots = molrs.build_mmff_potentials(mol)
/// >>> opt = molrs.LBFGS(pots, fmax=0.05)
/// >>> coords, report = opt.run(coords)         # (N, 3)
/// >>> batch, reports = opt.run(batch)          # (B, N, 3)
#[pyclass(name = "LBFGS")]
pub struct PyLBFGS {
    potentials: Py<PyPotentials>,
    cfg: LbfgsConfig,
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
            cfg: LbfgsConfig {
                fmax,
                max_steps,
                max_step,
                memory,
            },
        }
    }

    /// Relax coordinates by L-BFGS. ``(N, 3)`` / ``(3N,)`` -> single structure
    /// returning ``((N, 3) array, OptReport)``; ``(B, N, 3)`` -> homogeneous
    /// batch returning ``((B, N, 3) array, list[OptReport])``. Input is not
    /// mutated.
    fn run<'py>(
        &self,
        py: Python<'py>,
        coords: PyReadonlyArrayDyn<'_, NpF>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let pots = self.potentials.borrow(py);
        let arr = coords.as_array();
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
                let report = LBFGS::new(pots.inner.compiled()?, self.cfg)
                    .run(&mut flat)
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
                let expected = pots.inner.compiled()?.n_atoms();
                if expected != 0 && n != expected {
                    return Err(pyo3::exceptions::PyValueError::new_err(format!(
                        "structure atom count N={n} does not match this Potentials' atom count {expected}"
                    )));
                }
                let mut flat: Vec<NpF> = arr.iter().copied().collect();
                let reports = LBFGS::new(pots.inner.compiled()?, self.cfg)
                    .run_batch(&mut flat, n, b)
                    .map_err(pyo3::exceptions::PyValueError::new_err)?;
                let out: Bound<'py, PyArray3<NpF>> = Array3::from_shape_vec((b, n, 3), flat)
                    .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?
                    .to_pyarray(py);
                let reports: Vec<PyOptReport> =
                    reports.into_iter().map(PyOptReport::from).collect();
                Ok((out, reports).into_pyobject(py)?.into_any())
            }
            other => Err(pyo3::exceptions::PyValueError::new_err(format!(
                "coords must be 1-D (3N,), 2-D (N, 3), or 3-D (B, N, 3); got {other}-D"
            ))),
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "LBFGS(fmax={}, max_steps={}, max_step={}, memory={})",
            self.cfg.fmax, self.cfg.max_steps, self.cfg.max_step, self.cfg.memory
        )
    }
}

/// MMFF94 atom-type assigner and potential builder.
///
/// Exposed to Python as `molrs.MMFFTypifier`.
///
/// Loads embedded MMFF94 parameter tables at construction time. Use
/// :meth:`typify` to assign atom types to a molecular graph, or
/// :meth:`build` as a one-step shortcut that also compiles potentials.
///
/// # References
///
/// - Halgren, T.A. (1996). J. Comput. Chem. 17, 490-519.
///
/// Examples
/// --------
/// >>> typifier = MMFFTypifier()
/// >>> frame = typifier.typify(mol)   # typed Frame
/// >>> potentials = typifier.build(mol)  # compiled Potentials
#[pyclass(name = "MMFFTypifier")]
pub struct PyMMFFTypifier {
    inner: MMFFTypifier,
}

#[pymethods]
impl PyMMFFTypifier {
    /// Create an MMFF94 typifier with embedded parameter tables.
    ///
    /// Returns
    /// -------
    /// MMFFTypifier
    ///
    /// Raises
    /// ------
    /// RuntimeError
    ///     If parameter initialization fails (should not happen with
    ///     embedded data).
    #[new]
    fn new() -> PyResult<Self> {
        let typifier = MMFFTypifier::mmff94().map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("failed to initialize MMFF94: {}", e))
        })?;
        Ok(Self { inner: typifier })
    }

    /// Assign MMFF94 atom types to a molecular graph and return a
    /// ``to_potentials``-ready :class:`Frame`.
    ///
    /// Returns a typed :class:`Frame` with an ``"atoms"`` block containing the
    /// assigned ``type`` (int) column and topology blocks (``"bonds"``,
    /// ``"angles"``, ``"dihedrals"``, ``"impropers"``). The frame is
    /// assembly-complete: the stretch-bend reference lengths (``r0_ij``,
    /// ``r0_kj``, ``theta0``) are merged onto the ``"angles"`` block and the
    /// intramolecular ``"pairs"`` neighbour list is inserted, so the frame can
    /// be passed straight to :meth:`ForceField.to_potentials` (the ``mmff_stbn``
    /// kernel would otherwise error on a bare labeled frame).
    ///
    /// Parameters
    /// ----------
    /// mol : Atomistic
    ///     Molecular graph with element symbols and bonds.
    ///
    /// Returns
    /// -------
    /// Frame
    ///     Typed, ``to_potentials``-ready molecular data.
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If atom types cannot be determined (e.g. unsupported elements).
    fn typify(&self, mol: &PyAtomistic) -> PyResult<PyFrame> {
        // Materialize the labeled graph to a frame that is ready for
        // `ForceField.to_potentials` (stretch-bend merge + pair list applied),
        // matching the assembly `build` performs.
        let frame = self
            .inner
            .typify_frame(mol.core())
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        PyFrame::from_core_frame(frame)
    }

    /// Typify and compile potentials in one step.
    ///
    /// Equivalent to calling :meth:`typify` followed by force-field
    /// compilation, but avoids the intermediate :class:`Frame`.
    ///
    /// Parameters
    /// ----------
    /// mol : Atomistic
    ///     Molecular graph with element symbols and bonds.
    ///
    /// Returns
    /// -------
    /// Potentials
    ///     Compiled energy/force evaluator.
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If typification or compilation fails.
    ///
    /// Examples
    /// --------
    /// >>> potentials = typifier.build(mol)
    /// >>> energy, forces = potentials.eval(coords)
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
        format!("MMFFTypifier(forcefield='{}')", self.inner.ff().name)
    }
}

/// OPLS-AA atom-type assigner and potential builder.
///
/// Exposed to Python as `molrs.OplsTypifier`. Mirrors :class:`MMFFTypifier`:
/// loads the embedded canonical OPLS-AA parameter set at construction, assigns
/// `opls_NNN` atom types by SMARTS matching (:meth:`typify`), assigns bonded
/// parameters (:meth:`typify_full`), and compiles potentials (:meth:`build`).
///
/// Parameters
/// ----------
/// strict : bool, default True
///     When True, a bonded term with no force-field match is an error. When
///     False, the (optional) estimator fills missing parameters.
/// estimator : bool, default False
///     Attach the similarity-based parameter estimator to the no-match seam
///     (only consulted when ``strict=False``).
///
/// Examples
/// --------
/// >>> typifier = OplsTypifier()
/// >>> frame = typifier.typify(mol)        # atom types
/// >>> potentials = typifier.build(mol)    # compiled Potentials
#[pyclass(name = "OplsTypifier")]
pub struct PyOplsTypifier {
    inner: OplsTypifier,
}

#[pymethods]
impl PyOplsTypifier {
    /// Create an OPLS-AA typifier from the embedded canonical parameter set.
    #[new]
    #[pyo3(signature = (strict = true, estimator = false))]
    fn new(strict: bool, estimator: bool) -> PyResult<Self> {
        let mut typifier = OplsTypifier::oplsaa()
            .map_err(|e| {
                pyo3::exceptions::PyRuntimeError::new_err(format!(
                    "failed to initialize OPLS-AA: {e}"
                ))
            })?
            .with_strict(strict);
        if estimator {
            typifier = typifier.with_default_estimator();
        }
        Ok(Self { inner: typifier })
    }

    /// Build a typifier from an OPLS-AA / GROMACS XML string.
    ///
    /// Parameters
    /// ----------
    /// xml : str
    ///     OPLS-AA forcefield XML (OpenMM / GROMACS layout).
    /// strict : bool, default True
    ///     See :class:`OplsTypifier`.
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If the XML cannot be parsed.
    #[staticmethod]
    #[pyo3(signature = (xml, strict = true))]
    fn from_xml_str(xml: &str, strict: bool) -> PyResult<Self> {
        let typifier = OplsTypifier::from_xml_str(xml)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?
            .with_strict(strict);
        Ok(Self { inner: typifier })
    }

    /// Assign OPLS-AA atom types to a molecular graph.
    ///
    /// Returns a typed :class:`Frame` with an ``"atoms"`` block carrying the
    /// assigned ``type`` column. Bonded-parameter assignment is the separate
    /// :meth:`typify_full` step (atoms can type even when not every bonded term
    /// resolves).
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If atom typing fails.
    fn typify(&self, mol: &PyAtomistic) -> PyResult<PyFrame> {
        let labeled = self
            .inner
            .typify(mol.core())
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        PyFrame::from_core_frame(labeled.to_frame())
    }

    /// Assign atom types **and** bonded parameters in one step.
    ///
    /// Returns a typed :class:`Frame` with atom + bonded blocks labeled with the
    /// most specific matching force-field parameters.
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If atom typing or bonded assignment fails (under ``strict``).
    fn typify_full(&self, mol: &PyAtomistic) -> PyResult<PyFrame> {
        let labeled = self
            .inner
            .typify_full(mol.core())
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        PyFrame::from_core_frame(labeled.to_frame())
    }

    /// Typify and compile potentials in one step (``typify_full`` → ``Potentials``).
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
        format!("OplsTypifier(forcefield='{}')", self.inner.ff().name)
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

/// Build ready-to-use MMFF94 potentials for a molecule.
///
/// Uses the assembled :rust:`MmffForceField` energy model (the RDKit-validated
/// MMFF94 path used by conformer generation), wrapped as a :class:`Potentials`
/// so it can be evaluated and minimized directly. This is the recommended way
/// to obtain potentials for :meth:`Potentials.minimize`.
///
/// Parameters
/// ----------
/// mol : Atomistic
///     Molecular graph with element symbols, bonds, and 3D coordinates.
/// variant : str, optional
///     ``"MMFF94"`` (default) or ``"MMFF94s"`` (static variant).
///
/// Returns
/// -------
/// Potentials
///     Compiled energy/force evaluator for ``mol``.
///
/// Raises
/// ------
/// ValueError
///     If ``variant`` is unknown, or MMFF94 typing / assembly fails.
///
/// Examples
/// --------
/// >>> mol = molrs.parse_smiles("CCO").to_atomistic()
/// >>> mol, _ = molrs.Conformer(seed=7).generate(mol)
/// >>> pots = molrs.build_mmff_potentials(mol)
/// >>> coords = molrs.extract_coords(molrs.MMFFTypifier().typify(mol)).reshape(-1, 3)
/// >>> opt, report = pots.minimize(coords, fmax=0.05)
#[pyfunction]
#[pyo3(name = "build_mmff_potentials", signature = (mol, variant = "MMFF94"))]
pub fn build_mmff_potentials_py(mol: &PyAtomistic, variant: &str) -> PyResult<PyPotentials> {
    let var = match variant.to_ascii_uppercase().as_str() {
        "MMFF94" => MmffVariant::Mmff94,
        "MMFF94S" => MmffVariant::Mmff94s,
        other => {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "unknown MMFF variant '{other}' (expected 'MMFF94' or 'MMFF94s')"
            )));
        }
    };
    let core = mol.core();
    let props = MmffMolProperties::compute(core, var)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    let ff = MmffForceField::build(core, &props)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    let mut pots = Potentials::new();
    pots.set_n_atoms(core.n_atoms());
    pots.push(Box::new(ff));
    Ok(PyPotentials {
        inner: PotBacking::Compiled(pots),
    })
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
/// [`crate::frame`]'s `FrameRefPtr`).
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
