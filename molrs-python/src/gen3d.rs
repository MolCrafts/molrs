//! Python wrapper for 3D coordinate generation from molecular graphs.
//!
//! The `generate_3d` pipeline takes an [`PyAtomistic`] molecular graph and
//! produces realistic 3D coordinates through a multi-stage process:
//!
//! 1. **Preprocess** -- perceive rings, assign stereochemistry.
//! 2. **Build initial** -- distance geometry embedding.
//! 3. **Coarse optimize** -- rough energy minimization.
//! 4. **Rotor search** -- systematic torsion scanning.
//! 5. **Final optimize** -- refined minimization.
//! 6. **Stereo check** -- verify stereochemistry is preserved.
//!
//! # References
//!
//! - Riniker, S.; Landrum, G.A. (2015). J. Chem. Inf. Model. 55, 2562-2574.

use pyo3::prelude::*;

use molrs_gen3d::{Gen3DOptions, Gen3DSpeed, StageKind, generate_3d};

use crate::helpers::molrs_error_to_pyerr;
use crate::molgraph::PyAtomistic;

/// Options controlling 3D coordinate generation.
///
/// Exposed to Python as `molrs.Gen3DOptions`.
///
/// Parameters
/// ----------
/// speed : str, optional
///     Quality preset: ``"fast"`` (fewest iterations), ``"medium"`` (default),
///     or ``"better"`` (most thorough search).
/// add_hydrogens : bool, optional
///     If ``True`` (default), add implicit hydrogens before embedding.
/// seed : int | None, optional
///     Random seed for reproducibility. ``None`` uses an arbitrary seed.
///
/// Examples
/// --------
/// >>> opts = Gen3DOptions(speed="fast", add_hydrogens=True, seed=42)
#[pyclass(name = "Gen3DOptions", unsendable, skip_from_py_object)]
#[derive(Clone)]
pub struct PyGen3DOptions {
    pub(crate) inner: Gen3DOptions,
}

#[pymethods]
impl PyGen3DOptions {
    /// Create new generation options.
    ///
    /// Parameters
    /// ----------
    /// speed : str
    ///     ``"fast"``, ``"medium"`` (default), or ``"better"``.
    /// add_hydrogens : bool
    ///     Add implicit hydrogens before embedding. Default ``True``.
    /// seed : int | None
    ///     RNG seed for reproducibility.
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If ``speed`` is not one of the recognized presets.
    #[new]
    #[pyo3(signature = (speed="medium", add_hydrogens=true, seed=None))]
    fn new(speed: &str, add_hydrogens: bool, seed: Option<u64>) -> PyResult<Self> {
        let sp = match speed {
            "fast" => Gen3DSpeed::Fast,
            "medium" => Gen3DSpeed::Medium,
            "better" => Gen3DSpeed::Better,
            other => {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "unknown speed '{}', expected 'fast', 'medium', or 'better'",
                    other
                )));
            }
        };
        Ok(Self {
            inner: Gen3DOptions {
                speed: sp,
                add_hydrogens,
                rng_seed: seed,
                ..Gen3DOptions::default()
            },
        })
    }

    fn __repr__(&self) -> String {
        format!(
            "Gen3DOptions(speed={:?}, add_hydrogens={}, seed={:?})",
            self.inner.speed, self.inner.add_hydrogens, self.inner.rng_seed
        )
    }
}

/// Report for a single stage of the 3D generation pipeline.
///
/// Exposed to Python as `molrs.StageReport`.
///
/// Attributes
/// ----------
/// stage : str
///     Stage name (e.g. ``"build_initial"``, ``"final_optimize"``).
/// energy_before : float | None
///     Energy in kcal/mol before this stage, or ``None`` if not applicable.
/// energy_after : float | None
///     Energy in kcal/mol after this stage, or ``None`` if not applicable.
/// steps : int
///     Number of optimizer steps executed.
/// converged : bool
///     Whether the stage converged.
/// elapsed_ms : int
///     Wall-clock time for this stage in milliseconds.
#[pyclass(name = "StageReport", unsendable)]
pub struct PyStageReport {
    #[pyo3(get)]
    pub stage: String,
    #[pyo3(get)]
    pub energy_before: Option<f64>,
    #[pyo3(get)]
    pub energy_after: Option<f64>,
    #[pyo3(get)]
    pub steps: usize,
    #[pyo3(get)]
    pub converged: bool,
    #[pyo3(get)]
    pub elapsed_ms: u64,
}

#[pymethods]
impl PyStageReport {
    fn __repr__(&self) -> String {
        format!(
            "StageReport(stage='{}', steps={}, converged={})",
            self.stage, self.steps, self.converged
        )
    }
}

/// Aggregate report for the full 3D coordinate generation run.
///
/// Exposed to Python as `molrs.Gen3DReport`.
///
/// Attributes
/// ----------
/// final_energy : float | None
///     Total energy in kcal/mol after all stages, or ``None`` on failure.
/// warnings : list[str]
///     Diagnostic warnings generated during the run.
/// stages : list[StageReport]
///     Per-stage execution reports.
///
/// Examples
/// --------
/// >>> result = generate_3d(mol)
/// >>> report = result.report
/// >>> report.final_energy
/// -12.34
/// >>> for s in report.stages:
/// ...     print(s.stage, s.converged)
#[pyclass(name = "Gen3DReport", unsendable)]
pub struct PyGen3DReport {
    #[pyo3(get)]
    pub final_energy: Option<f64>,
    #[pyo3(get)]
    pub warnings: Vec<String>,
    pub(crate) stages_inner: Vec<PyStageReport>,
}

#[pymethods]
impl PyGen3DReport {
    /// Per-stage execution reports.
    ///
    /// Returns
    /// -------
    /// list[StageReport]
    #[getter]
    fn stages(&self) -> Vec<PyStageReport> {
        self.stages_inner
            .iter()
            .map(|s| PyStageReport {
                stage: s.stage.clone(),
                energy_before: s.energy_before,
                energy_after: s.energy_after,
                steps: s.steps,
                converged: s.converged,
                elapsed_ms: s.elapsed_ms,
            })
            .collect()
    }

    fn __repr__(&self) -> String {
        format!(
            "Gen3DReport(final_energy={:?}, stages={}, warnings={})",
            self.final_energy,
            self.stages_inner.len(),
            self.warnings.len()
        )
    }
}

/// Result container for 3D coordinate generation.
///
/// Exposed to Python as `molrs.Gen3DResult`.
///
/// Both :attr:`mol` and :attr:`report` are move-once accessors: the first
/// read transfers ownership out of this container. Subsequent access raises
/// ``RuntimeError``.
///
/// Attributes
/// ----------
/// mol : Atomistic
///     The molecule with generated 3D coordinates (consumed on first access).
/// report : Gen3DReport
///     Generation diagnostics (consumed on first access).
///
/// Examples
/// --------
/// >>> result = generate_3d(mol)
/// >>> mol_3d = result.mol
/// >>> report = result.report
#[pyclass(name = "Gen3DResult", unsendable)]
pub struct PyGen3DResult {
    pub(crate) mol_inner: Option<PyAtomistic>,
    pub(crate) report_inner: Option<PyGen3DReport>,
}

#[pymethods]
impl PyGen3DResult {
    /// The molecule with 3D coordinates.
    ///
    /// This is a move-once accessor. The first read returns the
    /// :class:`Atomistic`; subsequent reads raise ``RuntimeError``.
    ///
    /// Returns
    /// -------
    /// Atomistic
    ///
    /// Raises
    /// ------
    /// RuntimeError
    ///     If the molecule has already been consumed.
    #[getter]
    fn mol(&mut self) -> PyResult<PyAtomistic> {
        self.mol_inner
            .take()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("mol already consumed"))
    }

    /// Generation report with per-stage metrics.
    ///
    /// This is a move-once accessor. See :attr:`mol` for semantics.
    ///
    /// Returns
    /// -------
    /// Gen3DReport
    ///
    /// Raises
    /// ------
    /// RuntimeError
    ///     If the report has already been consumed.
    #[getter]
    fn report(&mut self) -> PyResult<PyGen3DReport> {
        self.report_inner
            .take()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("report already consumed"))
    }
}

/// Map a `StageKind` enum to a human-readable name.
fn stage_kind_name(kind: StageKind) -> &'static str {
    match kind {
        StageKind::Preprocess => "preprocess",
        StageKind::BuildInitial => "build_initial",
        StageKind::CoarseOptimize => "coarse_optimize",
        StageKind::RotorSearch => "rotor_search",
        StageKind::FinalOptimize => "final_optimize",
        StageKind::StereoCheck => "stereo_check",
    }
}

/// Generate 3D coordinates for a molecular graph.
///
/// Runs the full distance-geometry + optimization pipeline to produce
/// physically reasonable 3D coordinates for the input molecule.
///
/// Parameters
/// ----------
/// mol : Atomistic
///     Input molecular graph (heavy atoms and bonds).
/// options : Gen3DOptions | None, optional
///     Pipeline options. If ``None``, uses ``Gen3DOptions()`` defaults
///     (medium speed, add hydrogens, arbitrary seed).
///
/// Returns
/// -------
/// Gen3DResult
///     Container with ``.mol`` (3D molecule) and ``.report`` (diagnostics).
///
/// Raises
/// ------
/// ValueError
///     If the molecular graph is invalid (e.g. missing element symbols,
///     disconnected fragments without expected bonding).
///
/// Examples
/// --------
/// >>> mol = parse_smiles("CCO").to_atomistic()
/// >>> result = generate_3d(mol, Gen3DOptions(speed="fast", seed=42))
/// >>> mol_3d = result.mol
/// >>> mol_3d.n_atoms   # includes added hydrogens
/// 9
#[pyfunction]
#[pyo3(name = "generate_3d", signature = (mol, options=None))]
pub fn generate_3d_py(
    mol: &PyAtomistic,
    options: Option<&PyGen3DOptions>,
) -> PyResult<PyGen3DResult> {
    let opts = options.map(|o| o.inner.clone()).unwrap_or_default();

    let (result_mol, report) = generate_3d(&mol.inner, &opts).map_err(molrs_error_to_pyerr)?;

    let stages: Vec<PyStageReport> = report
        .stages
        .iter()
        .map(|s| PyStageReport {
            stage: stage_kind_name(s.stage).to_string(),
            energy_before: s.energy_before,
            energy_after: s.energy_after,
            steps: s.steps,
            converged: s.converged,
            elapsed_ms: s.elapsed_ms,
        })
        .collect();

    let py_report = PyGen3DReport {
        final_energy: report.final_energy,
        warnings: report.warnings,
        stages_inner: stages,
    };

    Ok(PyGen3DResult {
        mol_inner: Some(PyAtomistic { inner: result_mol }),
        report_inner: Some(py_report),
    })
}
