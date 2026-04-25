//! Python wrappers for neighbor search: `NeighborQuery`, `LinkedCell`, and
//! `NeighborList`.
//!
//! Two APIs are provided:
//!
//! - **freud-style** (preferred): build a [`PyNeighborQuery`] from reference
//!   points, then call [`PyNeighborQuery::query`] or [`PyNeighborQuery::query_self`]
//!   to obtain a [`PyNeighborList`].
//! - **Legacy**: use [`PyLinkedCell`] directly (backward-compatible wrapper
//!   around `LinkCell`).
//!
//! All distances are in the same length unit as the coordinates (typically
//! angstroms).

use crate::helpers::NpF;
use crate::simbox::PyBox;
use molrs::neighbors::{NeighborList as RsNeighborList, NeighborQuery, QueryMode};
use ndarray::ArrayView1;
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

// ---------------------------------------------------------------------------
// PyNeighborList
// ---------------------------------------------------------------------------

/// Result of a neighbor query, holding all pairs within cutoff.
///
/// Exposed to Python as `molrs.NeighborList`.
///
/// Each pair ``(i, j)`` represents a neighbor relationship where ``i`` is the
/// query-point index and ``j`` is the reference-point index. For self-queries
/// only unique pairs ``i < j`` are stored.
///
/// Attributes
/// ----------
/// query_point_indices : numpy.ndarray[uint32]
///     Query-point index per pair.
/// point_indices : numpy.ndarray[uint32]
///     Reference-point index per pair.
/// distances : numpy.ndarray[float]
///     Euclidean distance per pair.
/// dist_sq : numpy.ndarray[float]
///     Squared distance per pair.
/// n_pairs : int
///     Total number of neighbor pairs.
/// num_points : int
///     Number of reference points.
/// num_query_points : int
///     Number of query points.
/// is_self_query : bool
///     ``True`` if this list was built from a self-query.
///
/// Examples
/// --------
/// >>> nlist = aabb.query_self()
/// >>> nlist.n_pairs
/// 42
/// >>> nlist.distances   # numpy float array
#[pyclass(name = "NeighborList")]
pub struct PyNeighborList {
    pub(crate) inner: RsNeighborList,
}

#[pymethods]
impl PyNeighborList {
    /// Query-point indices, one per pair.
    ///
    /// Returns a zero-copy numpy view into the underlying Rust `Vec<u32>`.
    /// Pinned alive via numpy's `.base` mechanism: the returned array keeps
    /// this `NeighborList` alive for as long as it is referenced from Python.
    ///
    /// Returns
    /// -------
    /// numpy.ndarray, shape (n_pairs,), dtype uint32
    #[getter]
    fn query_point_indices<'py>(slf: Bound<'py, Self>) -> Bound<'py, PyArray1<u32>> {
        let owner = slf.clone().into_any();
        let borrowed = slf.borrow();
        let view = ArrayView1::from(borrowed.inner.query_point_indices());
        unsafe { PyArray1::<u32>::borrow_from_array(&view, owner) }
    }

    /// Reference-point indices, one per pair.
    ///
    /// Zero-copy view; see [`query_point_indices`].
    ///
    /// Returns
    /// -------
    /// numpy.ndarray, shape (n_pairs,), dtype uint32
    #[getter]
    fn point_indices<'py>(slf: Bound<'py, Self>) -> Bound<'py, PyArray1<u32>> {
        let owner = slf.clone().into_any();
        let borrowed = slf.borrow();
        let view = ArrayView1::from(borrowed.inner.point_indices());
        unsafe { PyArray1::<u32>::borrow_from_array(&view, owner) }
    }

    /// Euclidean distances, one per pair.
    ///
    /// This getter *must* allocate because distances are stored as squared
    /// distances internally (`sqrt` applied on access).
    ///
    /// Returns
    /// -------
    /// numpy.ndarray, shape (n_pairs,), dtype float
    #[getter]
    fn distances<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<NpF>> {
        self.inner.distances().into_pyarray(py)
    }

    /// Squared distances, one per pair.
    ///
    /// Zero-copy view into the underlying Rust `Vec<f64>`.
    ///
    /// Returns
    /// -------
    /// numpy.ndarray, shape (n_pairs,), dtype float
    #[getter]
    fn dist_sq<'py>(slf: Bound<'py, Self>) -> Bound<'py, PyArray1<NpF>> {
        let owner = slf.clone().into_any();
        let borrowed = slf.borrow();
        let view = ArrayView1::from(borrowed.inner.dist_sq());
        unsafe { PyArray1::<NpF>::borrow_from_array(&view, owner) }
    }

    /// Number of neighbor pairs.
    ///
    /// Returns
    /// -------
    /// int
    #[getter]
    fn n_pairs(&self) -> usize {
        self.inner.n_pairs()
    }

    /// Number of reference points used to build this list.
    ///
    /// Returns
    /// -------
    /// int
    #[getter]
    fn num_points(&self) -> usize {
        self.inner.num_points()
    }

    /// Number of query points used to build this list.
    ///
    /// Returns
    /// -------
    /// int
    #[getter]
    fn num_query_points(&self) -> usize {
        self.inner.num_query_points()
    }

    /// Whether this list was produced by a self-query (unique pairs only).
    ///
    /// Returns
    /// -------
    /// bool
    #[getter]
    fn is_self_query(&self) -> bool {
        self.inner.mode() == QueryMode::SelfQuery
    }

    /// Return index pairs as an ``(M, 2)`` array.
    ///
    /// Each row is ``[query_point_index, point_index]``. This method exists
    /// for backward compatibility with code that expects an integer pair
    /// array.
    ///
    /// Returns
    /// -------
    /// numpy.ndarray, shape (n_pairs, 2), dtype int64
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     On internal shape error (should not happen).
    fn pairs<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<i64>>> {
        let n = self.inner.n_pairs();
        let qi = self.inner.query_point_indices();
        let pi = self.inner.point_indices();
        let flat: Vec<i64> = (0..n).flat_map(|k| [qi[k] as i64, pi[k] as i64]).collect();
        let array = ndarray::Array2::from_shape_vec((n, 2), flat)
            .map_err(|e| PyValueError::new_err(format!("failed to build output array: {}", e)))?;
        Ok(array.into_pyarray(py))
    }

    fn __repr__(&self) -> String {
        let mode = if self.inner.mode() == QueryMode::SelfQuery {
            "self"
        } else {
            "cross"
        };
        format!(
            "NeighborList(pairs={}, mode={}, num_points={}, num_query_points={})",
            self.inner.n_pairs(),
            mode,
            self.inner.num_points(),
            self.inner.num_query_points(),
        )
    }
}

// ---------------------------------------------------------------------------
// PyNeighborQuery
// ---------------------------------------------------------------------------

/// Spatial neighbor query (freud-style API).
///
/// Exposed to Python as `molrs.NeighborQuery`.
///
/// Build a spatial index from reference points and a simulation box, then
/// query for neighbors within cutoff.
///
/// Parameters
/// ----------
/// box : Box
///     Simulation box describing geometry and periodic boundaries.
/// points : numpy.ndarray, shape (N, 3), dtype float
///     Reference point positions in Cartesian coordinates.
/// cutoff : float
///     Cutoff radius for neighbor search (same unit as coordinates).
///
/// Examples
/// --------
/// >>> nq = NeighborQuery(box, positions, cutoff=3.0)
/// >>> nlist = nq.query(query_positions)   # cross-query
/// >>> nlist = nq.query_self()             # self-query (unique pairs)
#[pyclass(name = "NeighborQuery")]
pub struct PyNeighborQuery {
    inner: NeighborQuery,
}

#[pymethods]
impl PyNeighborQuery {
    /// Build the spatial index.
    ///
    /// Parameters
    /// ----------
    /// box : Box
    ///     Simulation box.
    /// points : numpy.ndarray, shape (N, 3), dtype float
    ///     Reference point positions.
    /// cutoff : float
    ///     Cutoff radius.
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If ``points`` does not have 3 columns.
    #[new]
    #[pyo3(signature = (r#box, points, cutoff))]
    fn new(r#box: &PyBox, points: PyReadonlyArray2<'_, NpF>, cutoff: NpF) -> PyResult<Self> {
        let view = points.as_array();
        if view.ncols() != 3 {
            return Err(PyValueError::new_err("points must have shape (N,3)"));
        }
        let inner = NeighborQuery::new(&r#box.inner, view, cutoff);
        Ok(Self { inner })
    }

    /// Find all neighbor pairs between query points and reference points.
    ///
    /// A pair ``(i, j)`` means query point ``i`` is within cutoff of reference
    /// point ``j``.
    ///
    /// Parameters
    /// ----------
    /// query_points : numpy.ndarray, shape (M, 3), dtype float
    ///     Query point positions.
    ///
    /// Returns
    /// -------
    /// NeighborList
    ///     Neighbor pairs with distances.
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If ``query_points`` does not have 3 columns.
    fn query(&self, query_points: PyReadonlyArray2<'_, NpF>) -> PyResult<PyNeighborList> {
        let view = query_points.as_array();
        if view.ncols() != 3 {
            return Err(PyValueError::new_err("query_points must have shape (M,3)"));
        }
        let nlist = self.inner.query(view);
        Ok(PyNeighborList { inner: nlist })
    }

    /// Find unique neighbor pairs within the reference point set.
    ///
    /// Only pairs ``i < j`` are returned to avoid double-counting.
    ///
    /// Returns
    /// -------
    /// NeighborList
    ///     Unique neighbor pairs with distances.
    fn query_self(&self) -> PyNeighborList {
        let nlist = self.inner.query_self();
        PyNeighborList { inner: nlist }
    }

    fn __repr__(&self) -> String {
        format!(
            "NeighborQuery(num_points={}, cutoff={})",
            self.inner.points().nrows(),
            self.inner.cutoff(),
        )
    }
}

// ---------------------------------------------------------------------------
// PyLinkedCell
// ---------------------------------------------------------------------------

/// Link-cell neighbor list (legacy API), exposed to Python as
/// `molrs.LinkedCell`.
///
/// This is a backward-compatible wrapper around the Rust `LinkCell`.
/// For new code, prefer :class:`NeighborQuery`.
///
/// Parameters
/// ----------
/// points : numpy.ndarray, shape (N, 3), dtype float
///     Particle positions.
/// cutoff : float
///     Cutoff radius for neighbor search.
/// box : Box
///     Simulation box for periodic boundary handling.
///
/// Examples
/// --------
/// >>> lc = LinkedCell(positions, cutoff=3.0, box=simbox)
/// >>> pairs = lc.pairs()   # (M, 2) int64 array
/// >>> lc.update(new_positions, box=simbox)
#[pyclass(name = "LinkedCell")]
pub struct PyLinkedCell {
    pub(crate) inner: molrs::neighbors::LinkCell,
}

#[pymethods]
impl PyLinkedCell {
    /// Build a linked-cell neighbor list.
    ///
    /// Parameters
    /// ----------
    /// points : numpy.ndarray, shape (N, 3), dtype float
    ///     Particle positions.
    /// cutoff : float
    ///     Cutoff radius (same unit as coordinates).
    /// box : Box
    ///     Simulation box for periodic boundaries.
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If ``points`` does not have 3 columns.
    #[new]
    fn new(points: PyReadonlyArray2<'_, NpF>, cutoff: NpF, r#box: &PyBox) -> PyResult<Self> {
        use molrs::neighbors::NbListAlgo;
        let view = points.as_array();
        if view.ncols() != 3 {
            return Err(PyValueError::new_err("points must have shape (N,3)"));
        }
        let mut lc = molrs::neighbors::LinkCell::new().cutoff(cutoff);
        lc.build(view, &r#box.inner);
        Ok(Self { inner: lc })
    }

    /// Compute unique neighbor pairs (``i < j``) within cutoff.
    ///
    /// Returns
    /// -------
    /// numpy.ndarray, shape (M, 2), dtype int64
    ///     Array of index pairs ``[i, j]``.
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     On internal shape error (should not happen).
    fn pairs<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<i64>>> {
        use molrs::neighbors::NbListAlgo;
        let result = self.inner.query();
        let n = result.n_pairs();
        let qi = result.query_point_indices();
        let pi = result.point_indices();
        let flat: Vec<i64> = (0..n).flat_map(|k| [qi[k] as i64, pi[k] as i64]).collect();
        let array = ndarray::Array2::from_shape_vec((n, 2), flat)
            .map_err(|e| PyValueError::new_err(format!("failed to build output array: {}", e)))?;
        Ok(array.into_pyarray(py))
    }

    /// Rebuild the neighbor list with new positions.
    ///
    /// Parameters
    /// ----------
    /// points : numpy.ndarray, shape (N, 3), dtype float
    ///     New particle positions.
    /// box : Box
    ///     Simulation box (may have changed).
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If ``points`` does not have 3 columns.
    fn update(&mut self, points: PyReadonlyArray2<'_, NpF>, r#box: &PyBox) -> PyResult<()> {
        use molrs::neighbors::NbListAlgo;
        let view = points.as_array();
        if view.ncols() != 3 {
            return Err(PyValueError::new_err("points must have shape (N,3)"));
        }
        self.inner.update(view, &r#box.inner);
        Ok(())
    }

    fn __repr__(&self) -> String {
        use molrs::neighbors::NbListAlgo;
        let n = self.inner.query().n_pairs();
        format!("LinkedCell(pairs={})", n)
    }
}
