//! Python wrapper for the simulation box (periodic boundary conditions).
//!
//! [`PyBox`] wraps the Rust [`SimBox`] and exposes construction helpers for
//! cubic, orthorhombic, and fully triclinic cells, plus coordinate
//! transformations (Cartesian <-> fractional), wrapping, and displacement
//! calculations with optional minimum-image convention.
//!
//! All length quantities are in the same units as the stored coordinates
//! (typically angstroms).

use crate::helpers::{NpF, box_error_to_pyerr, parse_origin, parse_pbc};
use molrs::spatial::region::simbox::SimBox;
use molrs::spatial::region::Region;
use molrs::types::F;
use ndarray::array;
use numpy::{IntoPyArray, PyArray1, PyArray2, PyArray3, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// Simulation box with periodic boundary conditions, exposed to Python as
/// `molrs.Box`.
///
/// The box is defined by a 3x3 cell matrix **H** whose columns are the
/// lattice vectors, an origin point, and per-axis PBC flags.
///
/// # Python Examples
///
/// ```python
/// import numpy as np
/// from molrs import Box
///
/// box = Box.cube(10.0)                       # 10 x 10 x 10 cubic
/// box = Box.ortho(np.array([10, 20, 30]))    # orthorhombic
/// print(box.volume())                        # 6000.0
/// ```
#[pyclass(module = "molrs", name = "Box", from_py_object, subclass)]
#[derive(Clone)]
pub struct PyBox {
    pub(crate) inner: SimBox,
}

#[pymethods]
impl PyBox {
    /// Create a fully triclinic simulation box from a cell matrix.
    ///
    /// Parameters
    /// ----------
    /// h : numpy.ndarray, shape (3, 3), dtype float
    ///     Cell matrix with lattice vectors as **columns**.
    /// origin : numpy.ndarray, shape (3,), dtype float, optional
    ///     Origin of the box in Cartesian coordinates. Defaults to
    ///     ``[0, 0, 0]``.
    /// pbc : numpy.ndarray, shape (3,), dtype bool, optional
    ///     Periodic boundary flags for x, y, z. Defaults to
    ///     ``[True, True, True]``.
    ///
    /// Returns
    /// -------
    /// Box
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If ``h`` is not 3x3 or the cell matrix is singular.
    ///
    /// Examples
    /// --------
    /// >>> h = np.eye(3) * 10.0
    /// >>> box = Box(h)
    #[new]
    #[pyo3(signature = (h, origin=None, pbc=None, cell_defined=true))]
    fn new(
        h: PyReadonlyArray2<'_, NpF>,
        origin: Option<PyReadonlyArray1<'_, NpF>>,
        pbc: Option<PyReadonlyArray1<'_, bool>>,
        cell_defined: bool,
    ) -> PyResult<Self> {
        let h_view = h.as_array();
        if h_view.dim() != (3, 3) {
            return Err(PyValueError::new_err("h must be a 3x3 matrix"));
        }
        let h_matrix = h_view.to_owned();
        let origin_vec = parse_origin(origin)?;
        let pbc_array = parse_pbc(pbc)?;

        let inner = SimBox::new_cell(h_matrix, origin_vec, pbc_array, cell_defined)
            .map_err(box_error_to_pyerr)?;
        Ok(PyBox { inner })
    }

    /// Whether the cell is geometrically defined. ``False`` marks a "no-cell"
    /// box (undefined / zero-volume), distinct from ``is_free`` (periodicity).
    #[getter]
    fn cell_defined(&self) -> bool {
        self.inner.is_cell_defined()
    }

    /// Create a cubic simulation box.
    ///
    /// Parameters
    /// ----------
    /// a : float
    ///     Side length of the cube in the same length unit as coordinates.
    /// origin : numpy.ndarray, shape (3,), optional
    ///     Box origin. Defaults to ``[0, 0, 0]``.
    /// pbc : numpy.ndarray, shape (3,), dtype bool, optional
    ///     Periodic boundary flags. Defaults to ``[True, True, True]``.
    ///
    /// Returns
    /// -------
    /// Box
    ///
    /// Examples
    /// --------
    /// >>> box = Box.cube(10.0)
    /// >>> box.volume()
    /// 1000.0
    #[staticmethod]
    #[pyo3(signature = (a, origin=None, pbc=None))]
    fn cube(
        a: NpF,
        origin: Option<PyReadonlyArray1<'_, NpF>>,
        pbc: Option<PyReadonlyArray1<'_, bool>>,
    ) -> PyResult<Self> {
        let origin_vec = parse_origin(origin)?;
        let pbc_array = parse_pbc(pbc)?;
        let inner = SimBox::cube(a, origin_vec, pbc_array).map_err(box_error_to_pyerr)?;
        Ok(PyBox { inner })
    }

    /// Create an orthorhombic (rectangular) simulation box.
    ///
    /// Parameters
    /// ----------
    /// lengths : numpy.ndarray, shape (3,), dtype float
    ///     Side lengths ``[Lx, Ly, Lz]``.
    /// origin : numpy.ndarray, shape (3,), optional
    ///     Box origin. Defaults to ``[0, 0, 0]``.
    /// pbc : numpy.ndarray, shape (3,), dtype bool, optional
    ///     Periodic boundary flags. Defaults to ``[True, True, True]``.
    ///
    /// Returns
    /// -------
    /// Box
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If ``lengths`` does not have exactly 3 elements.
    ///
    /// Examples
    /// --------
    /// >>> box = Box.ortho(np.array([10.0, 20.0, 30.0]))
    #[staticmethod]
    #[pyo3(signature = (lengths, origin=None, pbc=None))]
    fn ortho(
        lengths: PyReadonlyArray1<'_, NpF>,
        origin: Option<PyReadonlyArray1<'_, NpF>>,
        pbc: Option<PyReadonlyArray1<'_, bool>>,
    ) -> PyResult<Self> {
        let lv = lengths.as_slice()?;
        if lv.len() != 3 {
            return Err(PyValueError::new_err("lengths must have length 3"));
        }
        let lengths_arr = array![lv[0], lv[1], lv[2]];
        let origin_vec = parse_origin(origin)?;
        let pbc_array = parse_pbc(pbc)?;
        let inner =
            SimBox::ortho(lengths_arr, origin_vec, pbc_array).map_err(box_error_to_pyerr)?;
        Ok(PyBox { inner })
    }

    /// Create a tight orthorhombic box around a point cloud.
    #[staticmethod]
    #[pyo3(signature = (points, padding, pbc=None))]
    fn from_bounds(
        points: PyReadonlyArray2<'_, NpF>,
        padding: PyReadonlyArray1<'_, NpF>,
        pbc: Option<PyReadonlyArray1<'_, bool>>,
    ) -> PyResult<Self> {
        let points = points.as_array();
        if points.ncols() != 3 {
            return Err(PyValueError::new_err("points must have shape (N,3)"));
        }
        let padding = padding.as_slice()?;
        if padding.len() != 3 {
            return Err(PyValueError::new_err("padding must have length 3"));
        }
        let pbc = parse_pbc(pbc)?;
        let inner = SimBox::from_bounds(
            points,
            [padding[0], padding[1], padding[2]],
            pbc,
        )
        .map_err(box_error_to_pyerr)?;
        Ok(Self { inner })
    }

    /// Volume of the simulation box.
    ///
    /// Returns
    /// -------
    /// float
    ///     Volume in length_unit^3 (e.g. angstrom^3).
    fn volume(&self) -> F {
        self.inner.volume()
    }

    /// ``True`` when the box is free (non-periodic on every axis).
    #[getter]
    fn is_free(&self) -> bool {
        self.inner.is_free()
    }

    /// Geometry style label: ``"free"``, ``"orthogonal"``, or ``"triclinic"``.
    #[getter]
    fn style(&self) -> &'static str {
        self.inner.style()
    }

    /// Return a lattice vector by index.
    ///
    /// Parameters
    /// ----------
    /// index : int
    ///     Lattice vector index: 0, 1, or 2.
    ///
    /// Returns
    /// -------
    /// numpy.ndarray, shape (3,)
    ///     The lattice vector as a Cartesian 3-vector.
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If ``index`` is not 0, 1, or 2.
    fn lattice<'py>(&self, py: Python<'py>, index: usize) -> PyResult<Bound<'py, PyArray1<NpF>>> {
        if index >= 3 {
            return Err(PyValueError::new_err("index must be 0, 1, or 2"));
        }
        let vec = self.inner.lattice(index);
        Ok(vec.into_pyarray(py))
    }

    /// Cell matrix **H** (3x3), lattice vectors as columns.
    ///
    /// Returns
    /// -------
    /// numpy.ndarray, shape (3, 3), dtype float
    #[getter]
    fn h<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<NpF>> {
        self.inner.h_view().to_owned().into_pyarray(py)
    }

    /// Inverse cell matrix.
    #[getter]
    fn inverse<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<NpF>> {
        self.inner.inv_view().to_owned().into_pyarray(py)
    }

    /// Box origin in Cartesian coordinates.
    ///
    /// Returns
    /// -------
    /// numpy.ndarray, shape (3,), dtype float
    #[getter]
    fn origin<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<NpF>> {
        self.inner.origin_view().to_owned().into_pyarray(py)
    }

    /// Periodic boundary condition flags ``[pbc_x, pbc_y, pbc_z]``.
    ///
    /// Returns
    /// -------
    /// numpy.ndarray, shape (3,), dtype bool
    #[getter]
    fn pbc<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<bool>> {
        self.inner.pbc_view().to_owned().into_pyarray(py)
    }

    /// Lengths of the three lattice vectors (property; ``[|a|, |b|, |c|]``).
    #[getter]
    fn lengths<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<NpF>> {
        self.inner.lengths().into_pyarray(py)
    }

    /// Lattice angles ``[alpha, beta, gamma]`` in degrees.
    #[getter]
    fn angles<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<NpF>> {
        self.inner.angles().into_pyarray(py)
    }

    #[staticmethod]
    fn matrix_from_lengths_angles<'py>(
        py: Python<'py>,
        lengths: [NpF; 3],
        angles: [NpF; 3],
    ) -> PyResult<Bound<'py, PyArray2<NpF>>> {
        Ok(SimBox::matrix_from_lengths_angles(lengths, angles)
            .map_err(box_error_to_pyerr)?
            .into_pyarray(py))
    }

    #[staticmethod]
    fn matrix_from_lengths_tilts<'py>(
        py: Python<'py>,
        lengths: [NpF; 3],
        tilts: [NpF; 3],
    ) -> Bound<'py, PyArray2<NpF>> {
        SimBox::matrix_from_lengths_tilts(lengths, tilts).into_pyarray(py)
    }

    #[staticmethod]
    fn restricted_matrix<'py>(
        py: Python<'py>,
        matrix: PyReadonlyArray2<'_, NpF>,
    ) -> PyResult<Bound<'py, PyArray2<NpF>>> {
        Ok(SimBox::restricted_matrix(matrix.as_array())
            .map_err(box_error_to_pyerr)?
            .into_pyarray(py))
    }

    /// Box matrix with lattice vectors as columns, shape ``(3, 3)``.
    /// Alias for ``h`` to mirror the molpy API.
    #[getter]
    fn matrix<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<NpF>> {
        self.h(py)
    }

    /// LAMMPS-convention tilt factors ``(xy, xz, yz)``. Zero on
    /// orthogonal boxes.
    #[getter]
    fn tilts<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<NpF>> {
        let h = self.inner.h_view();
        let arr = ndarray::array![h[(0, 1)], h[(0, 2)], h[(1, 2)]];
        arr.into_pyarray(py)
    }

    /// Perpendicular distances between opposite cell faces.
    #[getter]
    fn nearest_plane_distance<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<NpF>> {
        self.inner.nearest_plane_distance().into_pyarray(py)
    }

    /// Eight Cartesian cell corners.
    fn corners<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<NpF>> {
        self.inner.get_corners().into_pyarray(py)
    }

    /// Per-axis coordinate bounds as ``[[xlo, xhi], [ylo, yhi], [zlo, zhi]]``.
    #[getter]
    fn bounds<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<NpF>> {
        self.inner.bounds().into_pyarray(py)
    }

    /// Minimum-image displacement from ``r1`` to ``r2``.
    fn shortest_vector<'py>(
        &self,
        py: Python<'py>,
        r1: PyReadonlyArray1<'_, NpF>,
        r2: PyReadonlyArray1<'_, NpF>,
    ) -> PyResult<Bound<'py, PyArray1<NpF>>> {
        let r1 = r1.as_array();
        let r2 = r2.as_array();
        if r1.len() != 3 || r2.len() != 3 {
            return Err(PyValueError::new_err("r1 and r2 must have length 3"));
        }
        Ok(self.inner.shortest_vector(r1, r2).into_pyarray(py))
    }

    /// Squared minimum-image distance between two points.
    fn distance_squared(
        &self,
        r1: PyReadonlyArray1<'_, NpF>,
        r2: PyReadonlyArray1<'_, NpF>,
    ) -> PyResult<NpF> {
        let r1 = r1.as_array();
        let r2 = r2.as_array();
        if r1.len() != 3 || r2.len() != 3 {
            return Err(PyValueError::new_err("r1 and r2 must have length 3"));
        }
        Ok(self.inner.calc_distance2(r1, r2))
    }

    /// Minimum-image distance between two points.
    fn distance(
        &self,
        r1: PyReadonlyArray1<'_, NpF>,
        r2: PyReadonlyArray1<'_, NpF>,
    ) -> PyResult<NpF> {
        Ok(self.distance_squared(r1, r2)?.sqrt())
    }

    /// Convert Cartesian coordinates to fractional coordinates.
    ///
    /// Parameters
    /// ----------
    /// xyz : numpy.ndarray, shape (N, 3), dtype float
    ///     Cartesian coordinates.
    ///
    /// Returns
    /// -------
    /// numpy.ndarray, shape (N, 3), dtype float
    ///     Fractional coordinates in the range ``[0, 1)`` for wrapped points.
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If ``xyz`` does not have 3 columns.
    fn to_frac<'py>(
        &self,
        py: Python<'py>,
        xyz: PyReadonlyArray2<'_, NpF>,
    ) -> PyResult<Bound<'py, PyArray2<NpF>>> {
        let view = xyz.as_array();
        if view.ncols() != 3 {
            return Err(PyValueError::new_err("expected shape (N,3)"));
        }
        let frac = self.inner.to_frac(view);
        Ok(frac.into_pyarray(py))
    }

    /// Convert fractional coordinates to Cartesian coordinates.
    ///
    /// Parameters
    /// ----------
    /// xyzs : numpy.ndarray, shape (N, 3), dtype float
    ///     Fractional coordinates.
    ///
    /// Returns
    /// -------
    /// numpy.ndarray, shape (N, 3), dtype float
    ///     Cartesian coordinates.
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If ``xyzs`` does not have 3 columns.
    fn to_cart<'py>(
        &self,
        py: Python<'py>,
        xyzs: PyReadonlyArray2<'_, NpF>,
    ) -> PyResult<Bound<'py, PyArray2<NpF>>> {
        let view = xyzs.as_array();
        if view.ncols() != 3 {
            return Err(PyValueError::new_err("expected shape (N,3)"));
        }
        let cart = self.inner.to_cart(view);
        Ok(cart.into_pyarray(py))
    }

    /// Wrap coordinates into the primary simulation cell.
    ///
    /// Applies periodic wrapping along axes where PBC is enabled.
    ///
    /// Parameters
    /// ----------
    /// xyzu : numpy.ndarray, shape (N, 3), dtype float
    ///     Unwrapped Cartesian coordinates.
    ///
    /// Returns
    /// -------
    /// numpy.ndarray, shape (N, 3), dtype float
    ///     Wrapped Cartesian coordinates.
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If ``xyzu`` does not have 3 columns.
    fn wrap<'py>(
        &self,
        py: Python<'py>,
        xyzu: PyReadonlyArray2<'_, NpF>,
    ) -> PyResult<Bound<'py, PyArray2<NpF>>> {
        let view = xyzu.as_array();
        if view.ncols() != 3 {
            return Err(PyValueError::new_err("expected shape (N,3)"));
        }
        let wrapped = self.inner.wrap(view);
        Ok(wrapped.into_pyarray(py))
    }

    /// Integer periodic image flags for Cartesian coordinates.
    fn images<'py>(
        &self,
        py: Python<'py>,
        xyz: PyReadonlyArray2<'_, NpF>,
    ) -> PyResult<Bound<'py, PyArray2<i64>>> {
        let view = xyz.as_array();
        if view.ncols() != 3 {
            return Err(PyValueError::new_err("expected shape (N,3)"));
        }
        Ok(self.inner.images(view).into_pyarray(py))
    }

    /// Reconstruct unwrapped coordinates from wrapped coordinates and images.
    fn unwrap<'py>(
        &self,
        py: Python<'py>,
        xyz: PyReadonlyArray2<'_, NpF>,
        images: PyReadonlyArray2<'_, i64>,
    ) -> PyResult<Bound<'py, PyArray2<NpF>>> {
        let xyz = xyz.as_array();
        let images = images.as_array();
        if xyz.ncols() != 3 || xyz.raw_dim() != images.raw_dim() {
            return Err(PyValueError::new_err(
                "xyz and images must have identical shape (N,3)",
            ));
        }
        Ok(self.inner.unwrap(xyz, images).into_pyarray(py))
    }

    /// Compute displacement vectors between two point sets.
    ///
    /// Calculates ``xyzu2 - xyzu1`` with optional minimum-image convention
    /// for periodic systems.
    ///
    /// Parameters
    /// ----------
    /// xyzu1 : numpy.ndarray, shape (N, 3), dtype float
    ///     First set of Cartesian coordinates.
    /// xyzu2 : numpy.ndarray, shape (N, 3), dtype float
    ///     Second set of Cartesian coordinates (same shape as ``xyzu1``).
    /// minimum_image : bool, optional
    ///     If ``True``, apply the minimum-image convention to displacements.
    ///     Default is ``False``.
    ///
    /// Returns
    /// -------
    /// numpy.ndarray, shape (N, 3), dtype float
    ///     Displacement vectors ``xyzu2 - xyzu1``.
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If shapes do not match or columns != 3.
    #[pyo3(signature = (xyzu1, xyzu2, minimum_image=false))]
    fn delta<'py>(
        &self,
        py: Python<'py>,
        xyzu1: PyReadonlyArray2<'_, NpF>,
        xyzu2: PyReadonlyArray2<'_, NpF>,
        minimum_image: bool,
    ) -> PyResult<Bound<'py, PyArray2<NpF>>> {
        let v1 = xyzu1.as_array();
        let v2 = xyzu2.as_array();
        if v1.raw_dim() != v2.raw_dim() {
            return Err(PyValueError::new_err(
                "xyzu1 and xyzu2 must have the same shape",
            ));
        }
        if v1.ncols() != 3 {
            return Err(PyValueError::new_err("expected shape (N,3)"));
        }
        let d = self.inner.delta(v1, v2, minimum_image);
        Ok(d.into_pyarray(py))
    }

    /// Row-wise minimum-image distances between equally sized point arrays.
    fn distances<'py>(
        &self,
        py: Python<'py>,
        points1: PyReadonlyArray2<'_, NpF>,
        points2: PyReadonlyArray2<'_, NpF>,
    ) -> PyResult<Bound<'py, PyArray1<NpF>>> {
        let points1 = points1.as_array();
        let points2 = points2.as_array();
        if points1.raw_dim() != points2.raw_dim() || points1.ncols() != 3 {
            return Err(PyValueError::new_err(
                "points1 and points2 must have identical shape (N,3)",
            ));
        }
        Ok(self.inner.distances(points1, points2).into_pyarray(py))
    }

    /// All pairwise minimum-image displacement vectors (`points2 - points1`).
    fn pairwise_delta<'py>(
        &self,
        py: Python<'py>,
        points1: PyReadonlyArray2<'_, NpF>,
        points2: PyReadonlyArray2<'_, NpF>,
    ) -> PyResult<Bound<'py, PyArray3<NpF>>> {
        let points1 = points1.as_array();
        let points2 = points2.as_array();
        if points1.ncols() != 3 || points2.ncols() != 3 {
            return Err(PyValueError::new_err("points must have shape (N,3)"));
        }
        Ok(self.inner.pairwise_delta(points1, points2).into_pyarray(py))
    }

    /// All pairwise minimum-image distances.
    fn pairwise_distances<'py>(
        &self,
        py: Python<'py>,
        points1: PyReadonlyArray2<'_, NpF>,
        points2: PyReadonlyArray2<'_, NpF>,
    ) -> PyResult<Bound<'py, PyArray2<NpF>>> {
        let points1 = points1.as_array();
        let points2 = points2.as_array();
        if points1.ncols() != 3 || points2.ncols() != 3 {
            return Err(PyValueError::new_err("points must have shape (N,3)"));
        }
        Ok(self
            .inner
            .pairwise_distances(points1, points2)
            .into_pyarray(py))
    }

    /// Return a box whose cell matrix is right-multiplied by `transformation`.
    fn transformed(
        &self,
        transformation: PyReadonlyArray2<'_, NpF>,
    ) -> PyResult<Self> {
        let transformation = transformation.as_array();
        if transformation.dim() != (3, 3) {
            return Err(PyValueError::new_err("transformation must have shape (3,3)"));
        }
        let inner = self
            .inner
            .transformed(&transformation.to_owned())
            .map_err(box_error_to_pyerr)?;
        Ok(Self { inner })
    }

    /// Test whether each point lies inside the primary simulation cell.
    ///
    /// Parameters
    /// ----------
    /// xyz : numpy.ndarray, shape (N, 3), dtype float
    ///     Cartesian coordinates.
    ///
    /// Returns
    /// -------
    /// numpy.ndarray, shape (N,), dtype bool
    ///     ``True`` for points inside the cell.
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If ``xyz`` does not have 3 columns.
    fn isin<'py>(
        &self,
        py: Python<'py>,
        xyz: PyReadonlyArray2<'_, NpF>,
    ) -> PyResult<Bound<'py, PyArray1<bool>>> {
        let view = xyz.as_array();
        if view.ncols() != 3 {
            return Err(PyValueError::new_err("expected shape (N,3)"));
        }
        let inside = self.inner.isin(view);
        Ok(inside.into_pyarray(py))
    }

    fn __repr__(&self) -> String {
        format!("Box(volume={:.2})", self.inner.volume())
    }
}
