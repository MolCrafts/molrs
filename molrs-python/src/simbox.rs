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
use molrs::region::simbox::SimBox;
use molrs::types::F;
use ndarray::array;
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
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
#[pyclass(name = "Box", from_py_object)]
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
    #[pyo3(signature = (h, origin=None, pbc=None))]
    fn new(
        h: PyReadonlyArray2<'_, NpF>,
        origin: Option<PyReadonlyArray1<'_, NpF>>,
        pbc: Option<PyReadonlyArray1<'_, bool>>,
    ) -> PyResult<Self> {
        let h_view = h.as_array();
        if h_view.dim() != (3, 3) {
            return Err(PyValueError::new_err("h must be a 3x3 matrix"));
        }
        let h_matrix = h_view.to_owned();
        let origin_vec = parse_origin(origin)?;
        let pbc_array = parse_pbc(pbc)?;

        let inner = SimBox::new(h_matrix, origin_vec, pbc_array).map_err(box_error_to_pyerr)?;
        Ok(PyBox { inner })
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

    /// Volume of the simulation box.
    ///
    /// Returns
    /// -------
    /// float
    ///     Volume in length_unit^3 (e.g. angstrom^3).
    fn volume(&self) -> F {
        self.inner.volume()
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

    /// Lengths of the three lattice vectors.
    ///
    /// Returns
    /// -------
    /// numpy.ndarray, shape (3,), dtype float
    ///     ``[|a|, |b|, |c|]`` in the same length unit as the cell matrix.
    fn lengths<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<NpF>> {
        self.inner.lengths().into_pyarray(py)
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
