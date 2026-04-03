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

use pyo3::prelude::*;

use molrs::forcefield::ForceField;
use molrs::potential::{Potentials, extract_coords};
use molrs::typifier::Typifier;
use molrs::typifier::mmff::MMFFTypifier;

use crate::frame::PyFrame;
use crate::helpers::NpF;
use crate::molgraph::PyAtomistic;

use numpy::{PyArray1, ToPyArray};

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
#[pyclass(name = "Potentials", unsendable)]
pub struct PyPotentials {
    inner: Potentials,
}

/// Force-field definition metadata exposed to Python as `molrs.ForceField`.
#[pyclass(name = "ForceField", unsendable)]
pub struct PyForceField {
    pub(crate) inner: ForceField,
}

#[pymethods]
impl PyPotentials {
    /// Number of individual potential kernels in this compiled set.
    ///
    /// Returns
    /// -------
    /// int
    fn __len__(&self) -> usize {
        self.inner.len()
    }

    /// Evaluate energy and forces for the given coordinates.
    ///
    /// Parameters
    /// ----------
    /// coords : numpy.ndarray, shape (3*N,), dtype float
    ///     Flat coordinate array ``[x0, y0, z0, x1, y1, z1, ...]``.
    ///
    /// Returns
    /// -------
    /// tuple[float, numpy.ndarray]
    ///     ``(energy, forces)`` where ``energy`` is a scalar in kcal/mol and
    ///     ``forces`` is a flat array of the same shape as ``coords`` in
    ///     kcal/(mol*angstrom).
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If ``coords`` is not contiguous in memory.
    ///
    /// Examples
    /// --------
    /// >>> energy, forces = potentials.eval(coords)
    /// >>> energy
    /// -12.34
    fn eval<'py>(
        &self,
        py: Python<'py>,
        coords: numpy::PyReadonlyArray1<'_, NpF>,
    ) -> PyResult<(f64, Bound<'py, PyArray1<NpF>>)> {
        let slice = coords.as_slice()?;
        let (energy, forces) = self.inner.eval(slice);
        let forces_arr = forces.to_pyarray(py);
        Ok((energy as f64, forces_arr))
    }

    /// Evaluate energy only (no force computation).
    ///
    /// Parameters
    /// ----------
    /// coords : numpy.ndarray, shape (3*N,), dtype float
    ///     Flat coordinate array.
    ///
    /// Returns
    /// -------
    /// float
    ///     Energy in kcal/mol.
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If ``coords`` is not contiguous in memory.
    fn energy(&self, coords: numpy::PyReadonlyArray1<'_, NpF>) -> PyResult<f64> {
        let slice = coords.as_slice()?;
        Ok(self.inner.energy(slice) as f64)
    }

    fn __repr__(&self) -> String {
        format!("Potentials(n_kernels={})", self.inner.len())
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
#[pyclass(name = "MMFFTypifier", unsendable)]
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

    /// Assign MMFF94 atom types to a molecular graph.
    ///
    /// Returns a typed :class:`Frame` with an ``"atoms"`` block containing
    /// the assigned ``type`` (int) column and topology blocks
    /// (``"bonds"``, ``"angles"``, ``"dihedrals"``, ``"impropers"``).
    ///
    /// Parameters
    /// ----------
    /// mol : Atomistic
    ///     Molecular graph with element symbols and bonds.
    ///
    /// Returns
    /// -------
    /// Frame
    ///     Typed molecular data.
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If atom types cannot be determined (e.g. unsupported elements).
    fn typify(&self, mol: &PyAtomistic) -> PyResult<PyFrame> {
        let frame = self
            .inner
            .typify(&mol.inner)
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
            .build(&mol.inner)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        Ok(PyPotentials { inner: potentials })
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
    let forcefield = molrs::read_forcefield_xml(path)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    Ok(PyForceField { inner: forcefield })
}

#[pymethods]
impl PyForceField {
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

    fn __repr__(&self) -> String {
        format!(
            "ForceField(name='{}', styles={})",
            self.inner.name,
            self.inner.styles().len()
        )
    }
}
