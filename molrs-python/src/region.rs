//! Python wrappers for geometric regions.
//!
//! Regions describe spatial domains and support Boolean composition via
//! ``&`` (intersection), ``|`` (union), and ``~`` (complement) operators.
//!
//! | Class           | Description                        |
//! |-----------------|------------------------------------|
//! | `Sphere`        | Solid sphere                       |
//! | `HollowSphere`  | Spherical shell (inner + outer R)  |
//! | `Region`        | Composed region (from ``&``/``|``/``~``) |
//!
//! All lengths are in the same unit as the input coordinates (typically
//! angstroms).

use crate::helpers::NpF;
use molrs::region::region::{AndRegion, HollowSphere, NotRegion, OrRegion, Region, Sphere};
use ndarray::Array1;
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::sync::Arc;

/// Type-erased region handle for dynamic dispatch.
type DynRegion = Arc<dyn Region + Send + Sync>;

/// Test which points are inside a region, returning a boolean mask.
fn contains_impl<'py>(
    region: &dyn Region,
    py: Python<'py>,
    points: PyReadonlyArray2<'_, NpF>,
) -> PyResult<Bound<'py, PyArray1<bool>>> {
    let arr = points.as_array().to_owned();
    if arr.ncols() != 3 {
        return Err(PyValueError::new_err("points must have shape (N, 3)"));
    }
    let mask = region.contains(&arr);
    Ok(mask.into_pyarray(py))
}

/// Return the axis-aligned bounding box of a region as a ``(3, 2)`` array.
fn bounds_impl<'py>(region: &dyn Region, py: Python<'py>) -> Bound<'py, PyArray2<NpF>> {
    region.bounds().into_pyarray(py)
}

/// Solid sphere region.
///
/// Exposed to Python as `molrs.Sphere`.
///
/// Supports Boolean composition: ``&`` (AND), ``|`` (OR), ``~`` (NOT).
///
/// Parameters
/// ----------
/// center : numpy.ndarray, shape (3,), dtype float
///     Center of the sphere.
/// radius : float
///     Radius of the sphere.
///
/// Examples
/// --------
/// >>> s = Sphere(np.array([0, 0, 0]), 5.0)
/// >>> mask = s.contains(points)
/// >>> region = s & ~Sphere(np.array([0, 0, 0]), 2.0)  # shell
#[pyclass(name = "Sphere", from_py_object)]
#[derive(Clone)]
pub struct PySphere {
    inner: Arc<Sphere>,
}

#[pymethods]
impl PySphere {
    /// Create a solid sphere region.
    ///
    /// Parameters
    /// ----------
    /// center : numpy.ndarray, shape (3,), dtype float
    ///     Sphere center.
    /// radius : float
    ///     Sphere radius.
    ///
    /// Returns
    /// -------
    /// Sphere
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If ``center`` does not have length 3.
    #[new]
    fn new(center: PyReadonlyArray1<'_, NpF>, radius: NpF) -> PyResult<Self> {
        let c = center.as_slice()?;
        if c.len() != 3 {
            return Err(PyValueError::new_err("center must have length 3"));
        }
        let center_arr = Array1::from_vec(vec![c[0], c[1], c[2]]);
        Ok(PySphere {
            inner: Arc::new(Sphere::new(center_arr, radius)),
        })
    }

    /// Test which points are inside this sphere.
    ///
    /// Parameters
    /// ----------
    /// points : numpy.ndarray, shape (N, 3), dtype float
    ///     Test points.
    ///
    /// Returns
    /// -------
    /// numpy.ndarray, shape (N,), dtype bool
    ///     ``True`` for points inside the sphere.
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If ``points`` does not have 3 columns.
    fn contains<'py>(
        &self,
        py: Python<'py>,
        points: PyReadonlyArray2<'_, NpF>,
    ) -> PyResult<Bound<'py, PyArray1<bool>>> {
        contains_impl(self.inner.as_ref(), py, points)
    }

    /// Axis-aligned bounding box of this sphere.
    ///
    /// Returns
    /// -------
    /// numpy.ndarray, shape (3, 2), dtype float
    ///     ``[[xmin, xmax], [ymin, ymax], [zmin, zmax]]``.
    fn bounds<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<NpF>> {
        bounds_impl(self.inner.as_ref(), py)
    }

    /// Intersection: ``self & other``.
    fn __and__(&self, other: &Bound<'_, pyo3::types::PyAny>) -> PyResult<PyRegion> {
        let other_region = extract_region(other)?;
        let self_region: DynRegion = self.inner.clone();
        Ok(PyRegion {
            inner: Arc::new(AndRegion::new(self_region, other_region)),
        })
    }

    /// Union: ``self | other``.
    fn __or__(&self, other: &Bound<'_, pyo3::types::PyAny>) -> PyResult<PyRegion> {
        let other_region = extract_region(other)?;
        let self_region: DynRegion = self.inner.clone();
        Ok(PyRegion {
            inner: Arc::new(OrRegion::new(self_region, other_region)),
        })
    }

    /// Complement: ``~self``.
    fn __invert__(&self) -> PyRegion {
        let self_region: DynRegion = self.inner.clone();
        PyRegion {
            inner: Arc::new(NotRegion::new(self_region)),
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "Sphere(center=[{:.2}, {:.2}, {:.2}], radius={:.2})",
            self.inner.center[0], self.inner.center[1], self.inner.center[2], self.inner.radius
        )
    }
}

/// Hollow sphere (spherical shell) region.
///
/// Exposed to Python as `molrs.HollowSphere`.
///
/// A point is inside if its distance from the center is between
/// ``inner_radius`` and ``outer_radius``.
///
/// Parameters
/// ----------
/// center : numpy.ndarray, shape (3,), dtype float
///     Center of the spherical shell.
/// inner_radius : float
///     Inner radius.
/// outer_radius : float
///     Outer radius.
///
/// Examples
/// --------
/// >>> shell = HollowSphere(np.array([0,0,0]), 3.0, 5.0)
/// >>> mask = shell.contains(points)
#[pyclass(name = "HollowSphere", from_py_object)]
#[derive(Clone)]
pub struct PyHollowSphere {
    inner: Arc<HollowSphere>,
}

#[pymethods]
impl PyHollowSphere {
    /// Create a hollow sphere (spherical shell) region.
    ///
    /// Parameters
    /// ----------
    /// center : numpy.ndarray, shape (3,), dtype float
    ///     Shell center.
    /// inner_radius : float
    ///     Inner radius.
    /// outer_radius : float
    ///     Outer radius.
    ///
    /// Returns
    /// -------
    /// HollowSphere
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If ``center`` does not have length 3.
    #[new]
    fn new(
        center: PyReadonlyArray1<'_, NpF>,
        inner_radius: NpF,
        outer_radius: NpF,
    ) -> PyResult<Self> {
        let c = center.as_slice()?;
        if c.len() != 3 {
            return Err(PyValueError::new_err("center must have length 3"));
        }
        let center_arr = Array1::from_vec(vec![c[0], c[1], c[2]]);
        Ok(PyHollowSphere {
            inner: Arc::new(HollowSphere::new(center_arr, inner_radius, outer_radius)),
        })
    }

    /// Test which points are inside this spherical shell.
    ///
    /// Parameters
    /// ----------
    /// points : numpy.ndarray, shape (N, 3), dtype float
    ///     Test points.
    ///
    /// Returns
    /// -------
    /// numpy.ndarray, shape (N,), dtype bool
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If ``points`` does not have 3 columns.
    fn contains<'py>(
        &self,
        py: Python<'py>,
        points: PyReadonlyArray2<'_, NpF>,
    ) -> PyResult<Bound<'py, PyArray1<bool>>> {
        contains_impl(self.inner.as_ref(), py, points)
    }

    /// Axis-aligned bounding box of the outer sphere.
    ///
    /// Returns
    /// -------
    /// numpy.ndarray, shape (3, 2), dtype float
    fn bounds<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<NpF>> {
        bounds_impl(self.inner.as_ref(), py)
    }

    /// Intersection: ``self & other``.
    fn __and__(&self, other: &Bound<'_, pyo3::types::PyAny>) -> PyResult<PyRegion> {
        let other_region = extract_region(other)?;
        let self_region: DynRegion = self.inner.clone();
        Ok(PyRegion {
            inner: Arc::new(AndRegion::new(self_region, other_region)),
        })
    }

    /// Union: ``self | other``.
    fn __or__(&self, other: &Bound<'_, pyo3::types::PyAny>) -> PyResult<PyRegion> {
        let other_region = extract_region(other)?;
        let self_region: DynRegion = self.inner.clone();
        Ok(PyRegion {
            inner: Arc::new(OrRegion::new(self_region, other_region)),
        })
    }

    /// Complement: ``~self``.
    fn __invert__(&self) -> PyRegion {
        let self_region: DynRegion = self.inner.clone();
        PyRegion {
            inner: Arc::new(NotRegion::new(self_region)),
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "HollowSphere(inner_radius={:.2}, outer_radius={:.2})",
            self.inner.inner_radius, self.inner.outer_radius
        )
    }
}

/// Composed region produced by ``&``, ``|``, or ``~`` operators.
///
/// Exposed to Python as `molrs.Region`.
///
/// Cannot be constructed directly. Use Boolean operators on :class:`Sphere`,
/// :class:`HollowSphere`, or other `Region` instances.
///
/// Examples
/// --------
/// >>> shell = Sphere(c, 5.0) & ~Sphere(c, 3.0)
/// >>> shell.contains(points)
#[pyclass(name = "Region", from_py_object)]
#[derive(Clone)]
pub struct PyRegion {
    inner: DynRegion,
}

#[pymethods]
impl PyRegion {
    /// Test which points are inside this composed region.
    ///
    /// Parameters
    /// ----------
    /// points : numpy.ndarray, shape (N, 3), dtype float
    ///     Test points.
    ///
    /// Returns
    /// -------
    /// numpy.ndarray, shape (N,), dtype bool
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If ``points`` does not have 3 columns.
    fn contains<'py>(
        &self,
        py: Python<'py>,
        points: PyReadonlyArray2<'_, NpF>,
    ) -> PyResult<Bound<'py, PyArray1<bool>>> {
        contains_impl(self.inner.as_ref(), py, points)
    }

    /// Axis-aligned bounding box of this region.
    ///
    /// Returns
    /// -------
    /// numpy.ndarray, shape (3, 2), dtype float
    fn bounds<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<NpF>> {
        bounds_impl(self.inner.as_ref(), py)
    }

    /// Intersection: ``self & other``.
    fn __and__(&self, other: &Bound<'_, pyo3::types::PyAny>) -> PyResult<PyRegion> {
        let other_region = extract_region(other)?;
        Ok(PyRegion {
            inner: Arc::new(AndRegion::new(self.inner.clone(), other_region)),
        })
    }

    /// Union: ``self | other``.
    fn __or__(&self, other: &Bound<'_, pyo3::types::PyAny>) -> PyResult<PyRegion> {
        let other_region = extract_region(other)?;
        Ok(PyRegion {
            inner: Arc::new(OrRegion::new(self.inner.clone(), other_region)),
        })
    }

    /// Complement: ``~self``.
    fn __invert__(&self) -> PyRegion {
        PyRegion {
            inner: Arc::new(NotRegion::new(self.inner.clone())),
        }
    }

    fn __repr__(&self) -> String {
        "Region(composed)".to_string()
    }
}

/// Extract a `DynRegion` from any supported Python region object.
fn extract_region(obj: &Bound<'_, pyo3::types::PyAny>) -> PyResult<DynRegion> {
    if let Ok(s) = obj.extract::<PySphere>() {
        return Ok(s.inner.clone() as DynRegion);
    }
    if let Ok(hs) = obj.extract::<PyHollowSphere>() {
        return Ok(hs.inner.clone() as DynRegion);
    }
    if let Ok(r) = obj.extract::<PyRegion>() {
        return Ok(r.inner.clone());
    }
    Err(pyo3::exceptions::PyTypeError::new_err(
        "expected a Sphere, HollowSphere, or Region",
    ))
}
