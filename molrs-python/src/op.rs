//! `molrs.op` — the pure numeric base (`molrs::op`): weighted superposition,
//! centroids and SO(3) sampling over `float64` numpy arrays.
//!
//! Registered as a submodule of `_lib`, like `md`. Points cross as `(k, 3)`
//! arrays and rotations as `(3, 3)` (or `(N, 3, 3)`) row-major matrices. A
//! wrong shape is a `ValueError` naming the argument; a
//! [`SuperposeError`](molrs::op::superpose::SuperposeError) is a `ValueError`
//! carrying the Rust message.

use molrs::op::rigid::Rigid;
use molrs::op::so3;
use molrs::op::superpose::{self, DEFAULT_GAP_TOL, Fit, Freedom};
use molrs::op::types::{Mat3, Vec3};
use ndarray::{Array1, Array2, Array3};
use numpy::{IntoPyArray, PyArray1, PyArray2, PyArray3, PyReadonlyArrayDyn};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::helpers::py_value_err;

// ---------------------------------------------------------------------------
// Array seams shared with the leaf `replicate` and the assembly bindings
// ---------------------------------------------------------------------------

/// Read an `(n, 3)` float64 array as points, refusing any other shape.
pub(crate) fn points_from_array(
    array: &PyReadonlyArrayDyn<'_, f64>,
    name: &str,
) -> PyResult<Vec<Vec3>> {
    let view = array.as_array();
    let shape = view.shape();
    if shape.len() != 2 || shape[1] != 3 {
        return Err(PyValueError::new_err(format!(
            "{name} must have shape (n, 3), got {shape:?}"
        )));
    }
    Ok(view
        .rows()
        .into_iter()
        .map(|row| [row[0], row[1], row[2]])
        .collect())
}

/// Read an `(n, 3, 3)` float64 array as row-major rotation matrices.
pub(crate) fn matrices_from_array(
    array: &PyReadonlyArrayDyn<'_, f64>,
    name: &str,
) -> PyResult<Vec<Mat3>> {
    let view = array.as_array();
    let shape = view.shape();
    if shape.len() != 3 || shape[1] != 3 || shape[2] != 3 {
        return Err(PyValueError::new_err(format!(
            "{name} must have shape (n, 3, 3), got {shape:?}"
        )));
    }
    Ok((0..shape[0])
        .map(|c| std::array::from_fn(|i| std::array::from_fn(|j| view[[c, i, j]])))
        .collect())
}

/// Pair `rotations (N, 3, 3)` with `translations (N, 3)` into rigid motions.
pub(crate) fn rigids_from_arrays(
    rotations: &PyReadonlyArrayDyn<'_, f64>,
    translations: &PyReadonlyArrayDyn<'_, f64>,
) -> PyResult<Vec<Rigid>> {
    let rotations = matrices_from_array(rotations, "rotations")?;
    let translations = points_from_array(translations, "translations")?;
    if rotations.len() != translations.len() {
        return Err(PyValueError::new_err(format!(
            "rotations and translations disagree in count: {} vs {}",
            rotations.len(),
            translations.len()
        )));
    }
    Ok(rotations
        .into_iter()
        .zip(translations)
        .map(|(rotation, translation)| Rigid {
            rotation,
            translation,
        })
        .collect())
}

fn matrix_to_py<'py>(py: Python<'py>, m: &Mat3) -> Bound<'py, PyArray2<f64>> {
    Array2::from_shape_fn((3, 3), |(i, j)| m[i][j]).into_pyarray(py)
}

fn vector_to_py<'py>(py: Python<'py>, v: &Vec3) -> Bound<'py, PyArray1<f64>> {
    Array1::from(v.to_vec()).into_pyarray(py)
}

/// Read `weights`, or `k` ones when `None`.
fn weights_or_uniform(
    weights: Option<PyReadonlyArrayDyn<'_, f64>>,
    k: usize,
) -> PyResult<Vec<f64>> {
    match weights {
        None => Ok(vec![1.0; k]),
        Some(w) => {
            let view = w.as_array();
            if view.ndim() != 1 {
                return Err(PyValueError::new_err(format!(
                    "weights must have shape (k,), got {:?}",
                    view.shape()
                )));
            }
            Ok(view.iter().copied().collect())
        }
    }
}

// ---------------------------------------------------------------------------
// Fit
// ---------------------------------------------------------------------------

/// The best-fit proper rigid motion ``target ≈ rotation @ reference + translation``
/// returned by :func:`molrs.op.superpose`. Frozen.
///
/// Attributes
/// ----------
/// rotation : ndarray, shape (3, 3), float64
/// translation : ndarray, shape (3,), float64
/// rmsd : float
///     Weighted RMSD of the fit.
/// rho : float
///     Scale-free eigen-gap; 0 when ``freedom == "free"``.
/// center : ndarray, shape (3,), float64
///     Weighted target centroid (the point a spin axis passes through).
/// freedom : str
///     ``"unique"``, ``"spin"`` (rotation about ``axis`` is undetermined) or
///     ``"free"`` (no rotation determined; ``rotation`` is the identity).
/// axis : ndarray, shape (3,), float64, or None
///     The unit spin axis; ``None`` unless ``freedom == "spin"``.
#[pyclass(module = "molrs.op", name = "Fit", frozen, skip_from_py_object)]
pub struct PyFit {
    pub(crate) inner: Fit,
}

impl PyFit {
    pub(crate) fn from_core(inner: Fit) -> Self {
        Self { inner }
    }
}

#[pymethods]
impl PyFit {
    #[getter]
    fn rotation<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        matrix_to_py(py, &self.inner.rigid.rotation)
    }

    #[getter]
    fn translation<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        vector_to_py(py, &self.inner.rigid.translation)
    }

    #[getter]
    fn rmsd(&self) -> f64 {
        self.inner.rmsd
    }

    #[getter]
    fn rho(&self) -> f64 {
        self.inner.rho
    }

    #[getter]
    fn center<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        vector_to_py(py, &self.inner.center)
    }

    #[getter]
    fn freedom(&self) -> &'static str {
        match self.inner.freedom {
            Freedom::Unique => "unique",
            Freedom::Spin { .. } => "spin",
            Freedom::Free => "free",
        }
    }

    #[getter]
    fn axis<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArray1<f64>>> {
        match self.inner.freedom {
            Freedom::Spin { axis } => Some(vector_to_py(py, &axis)),
            Freedom::Unique | Freedom::Free => None,
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "Fit(freedom='{}', rmsd={}, rho={})",
            self.freedom(),
            self.inner.rmsd,
            self.inner.rho
        )
    }
}

// ---------------------------------------------------------------------------
// Free operations
// ---------------------------------------------------------------------------

/// Best-fit proper rigid motion mapping ``reference`` onto ``target``.
///
/// Parameters
/// ----------
/// reference, target : ndarray, shape (k, 3), float64
/// weights : ndarray, shape (k,), float64, optional
///     Per-point weights; uniform when omitted. Zero-weight points are dropped.
/// gap_tol : float, keyword-only
///     Eigen-gap below which the rotation is reported ``"spin"``;
///     :data:`DEFAULT_GAP_TOL` by default.
///
/// Returns
/// -------
/// Fit
///
/// Raises
/// ------
/// ValueError
///     On a shape other than ``(k, 3)``, a length mismatch, a negative or
///     non-finite weight, a non-finite coordinate, or no positive weight.
#[pyfunction(name = "superpose")]
#[pyo3(signature = (reference, target, weights=None, *, gap_tol=DEFAULT_GAP_TOL))]
fn py_superpose(
    reference: PyReadonlyArrayDyn<'_, f64>,
    target: PyReadonlyArrayDyn<'_, f64>,
    weights: Option<PyReadonlyArrayDyn<'_, f64>>,
    gap_tol: f64,
) -> PyResult<PyFit> {
    let reference = points_from_array(&reference, "reference")?;
    let target = points_from_array(&target, "target")?;
    let weights = weights_or_uniform(weights, reference.len())?;
    superpose::superpose(&reference, &target, &weights, gap_tol)
        .map(PyFit::from_core)
        .map_err(py_value_err)
}

/// :func:`superpose` of one ``reference`` against each unit of ``targets``.
///
/// Parameters
/// ----------
/// reference : ndarray, shape (k, 3), float64
/// targets : ndarray, shape (N·k, 3), float64
///     The ``N`` target units back to back.
/// weights : ndarray, shape (k,), float64, optional
/// gap_tol : float, keyword-only
///
/// Returns
/// -------
/// list[Fit]
///     One fit per unit, each equal to the single call on that unit.
///
/// Raises
/// ------
/// ValueError
///     If ``targets`` is not a whole number of units, or as :func:`superpose`.
#[pyfunction(name = "superpose_many")]
#[pyo3(signature = (reference, targets, weights=None, *, gap_tol=DEFAULT_GAP_TOL))]
fn py_superpose_many(
    reference: PyReadonlyArrayDyn<'_, f64>,
    targets: PyReadonlyArrayDyn<'_, f64>,
    weights: Option<PyReadonlyArrayDyn<'_, f64>>,
    gap_tol: f64,
) -> PyResult<Vec<PyFit>> {
    let reference = points_from_array(&reference, "reference")?;
    let targets = points_from_array(&targets, "targets")?;
    let weights = weights_or_uniform(weights, reference.len())?;
    superpose::superpose_many(&reference, &targets, &weights, gap_tol)
        .map(|fits| fits.into_iter().map(PyFit::from_core).collect())
        .map_err(py_value_err)
}

/// Weighted centroid ``Σ wᵢ pᵢ / Σ wᵢ``.
///
/// Parameters
/// ----------
/// points : ndarray, shape (k, 3), float64
/// weights : ndarray, shape (k,), float64, optional
///     Uniform when omitted.
///
/// Returns
/// -------
/// ndarray, shape (3,), float64, or None
///     ``None`` when the lengths differ or the total weight is not positive
///     and finite.
#[pyfunction(name = "centroid")]
#[pyo3(signature = (points, weights=None))]
fn py_centroid<'py>(
    py: Python<'py>,
    points: PyReadonlyArrayDyn<'_, f64>,
    weights: Option<PyReadonlyArrayDyn<'_, f64>>,
) -> PyResult<Option<Bound<'py, PyArray1<f64>>>> {
    let points = points_from_array(&points, "points")?;
    let weights = weights_or_uniform(weights, points.len())?;
    Ok(superpose::centroid(&points, &weights).map(|c| vector_to_py(py, &c)))
}

/// Haar-random rotations, one per entry of ``indices``; each depends only on
/// ``(seed, index)``.
///
/// Returns
/// -------
/// ndarray, shape (len(indices), 3, 3), float64
#[pyfunction(name = "random_rotations")]
fn py_random_rotations(py: Python<'_>, seed: u64, indices: Vec<u64>) -> Bound<'_, PyArray3<f64>> {
    let rotations = so3::random_rotations(seed, &indices);
    Array3::from_shape_fn((rotations.len(), 3, 3), |(c, i, j)| rotations[c][i][j]).into_pyarray(py)
}

/// Uniform angles in ``[0, 2π)`` radians, one per entry of ``indices``; each
/// depends only on ``(seed, index)``.
///
/// Returns
/// -------
/// ndarray, shape (len(indices),), float64
#[pyfunction(name = "random_angles")]
fn py_random_angles(py: Python<'_>, seed: u64, indices: Vec<u64>) -> Bound<'_, PyArray1<f64>> {
    so3::random_angles(seed, &indices).into_pyarray(py)
}

/// The rotation of Shoemake's unit quaternion for ``u`` in ``[0, 1]³``;
/// Haar-distributed on SO(3) when ``u`` is uniform.
///
/// Returns
/// -------
/// ndarray, shape (3, 3), float64
///
/// Raises
/// ------
/// ValueError
///     If a component of ``u`` is not finite or lies outside ``[0, 1]``.
#[pyfunction(name = "rotation_from_uniform")]
fn py_rotation_from_uniform(py: Python<'_>, u: [f64; 3]) -> PyResult<Bound<'_, PyArray2<f64>>> {
    // `contains` is false for NaN, and ±inf lies outside [0, 1].
    if !u.iter().all(|c| (0.0..=1.0).contains(c)) {
        return Err(PyValueError::new_err(format!(
            "u must lie in [0, 1]^3 with finite components, got {u:?}"
        )));
    }
    Ok(matrix_to_py(py, &so3::rotation_from_uniform(u)))
}

/// Populate the `molrs.op` submodule.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyFit>()?;
    m.add("DEFAULT_GAP_TOL", DEFAULT_GAP_TOL)?;
    m.add_function(wrap_pyfunction!(py_superpose, m)?)?;
    m.add_function(wrap_pyfunction!(py_superpose_many, m)?)?;
    m.add_function(wrap_pyfunction!(py_centroid, m)?)?;
    m.add_function(wrap_pyfunction!(py_random_rotations, m)?)?;
    m.add_function(wrap_pyfunction!(py_random_angles, m)?)?;
    m.add_function(wrap_pyfunction!(py_rotation_from_uniform, m)?)?;
    Ok(())
}
