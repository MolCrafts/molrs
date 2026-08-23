//! Python wrappers for neighbor search: `NeighborList`, `Neighbors`,
//! `NeighborQuery`.
//!
//! The surface mirrors the Rust contract (`molrs::spatial::neighbors`) one to
//! one, because the seam is the place where a renamed concept turns into a
//! silently different answer:
//!
//! - [`PyNeighborList`] (`molrs.NeighborList`) is the **engine**: it owns the
//!   cutoff and the backend that indexes space. `build` / `update` index
//!   coordinates and enumerate nothing; `neighbors(...)` materializes a table.
//! - [`PyNeighbors`] (`molrs.Neighbors`) is the **table**: read-only columns,
//!   one row per pair.
//! - [`PyNeighborQuery`] (`molrs.NeighborQuery`) is the **cross** door: a set of
//!   query points searched against a separate reference set.
//!
//! Two conventions travel with every table and are worth stating here, because
//! both have been a source of silent double counting:
//!
//! - A self search is **half-shell**: each unordered pair appears exactly once,
//!   with `i < j`, and never as `i == j`. A cross-query is directed and has no
//!   such rule.
//! - `dist_sq=True, disp=True` (the binder default, "FULL") names the *columns*
//!   a table keeps. It never means a bidirectional pair list. A column that was
//!   not stored reads back as `None` — never as a fabricated zero, which would
//!   be a physically meaningful value (coincident particles).
//!
//! `disp` is the **unnormalized** minimum-image displacement `r_j - r_i`: its
//! length is the pair distance, not 1. All lengths are in the unit of the input
//! coordinates, which is Å everywhere in molrs, so `dist_sq` is in Å².

use crate::core::spatial::simbox::PyBox;
use crate::helpers::NpF;
use molrs::spatial::neighbors::{
    NeighborList as RsNeighborList, NeighborPolicy, NeighborQuery as RsNeighborQuery,
    Neighbors as RsNeighbors, NeighborsStorage, QueryMode, SkinError, VerletSkin as RsVerletSkin,
};
use ndarray::ArrayView1;
use numpy::{PyArray1, PyArray2, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// Reject a coordinate array that is not `(N, 3)` before it reaches the core,
/// whose own check is an `assert!` — and a panic must not cross this seam.
/// `label` names the offending argument in the error.
fn check_points(points: &PyReadonlyArray2<'_, NpF>, label: &str) -> PyResult<()> {
    if points.as_array().ncols() != 3 {
        return Err(PyValueError::new_err(format!(
            "{label} must have shape (N, 3)"
        )));
    }
    Ok(())
}

/// Reject a non-positive cutoff before the core asserts on it.
fn check_cutoff(cutoff: NpF) -> PyResult<()> {
    if cutoff.is_nan() || cutoff <= 0.0 {
        return Err(PyValueError::new_err(
            "cutoff must be a positive length in Å",
        ));
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// PyNeighbors — the materialized pair table
// ---------------------------------------------------------------------------

/// Materialized neighbor pair table, exposed to Python as `molrs.Neighbors`.
///
/// A column store of every pair within the cutoff: two index columns that are
/// always present, plus whichever physical columns the search was told to keep.
/// Row ``k`` of every column describes the same pair.
///
/// A self search is half-shell — each unordered pair appears once, with
/// ``i < j``. A cross-query (:class:`NeighborQuery`) is directed and has no
/// such rule.
///
/// Optional columns are methods returning ``None`` when the search did not
/// store them; ``None`` is deliberately distinct from a column of zeros, which
/// would mean coincident particles.
///
/// Methods
/// -------
/// query_point_indices() : numpy.ndarray[uint32], shape (n_pairs,)
///     Query-point index ``i`` per pair.
/// point_indices() : numpy.ndarray[uint32], shape (n_pairs,)
///     Reference-point index ``j`` per pair.
/// dist_sq() : numpy.ndarray[float64], shape (n_pairs,), or None
///     Squared minimum-image distance in Å². Take ``np.sqrt`` at the call site
///     for a distance — nothing here hides a per-pair square root.
/// disp() : numpy.ndarray[float64], shape (n_pairs, 3), or None
///     Minimum-image displacement ``r_j - r_i`` in Å, **unnormalized**.
///
/// Attributes
/// ----------
/// n_pairs : int
///     Number of pairs — the row count every column shares.
/// num_points : int
///     Number of reference points.
/// num_query_points : int
///     Number of query points (equal to ``num_points`` for a self search).
/// is_self_query : bool
///     ``True`` when both index columns address the same point set.
///
/// Examples
/// --------
/// >>> neigh = nl.neighbors()
/// >>> i, j = neigh.query_point_indices(), neigh.point_indices()
/// >>> distances = np.sqrt(neigh.dist_sq())
/// >>> directions = neigh.disp() / distances[:, None]
#[pyclass(module = "molrs", name = "Neighbors")]
pub struct PyNeighbors {
    pub(crate) inner: RsNeighbors,
}

#[pymethods]
impl PyNeighbors {
    /// Query-point indices ``i``, one per pair.
    ///
    /// Zero-copy numpy view into the underlying Rust ``Vec<u32>``, pinned alive
    /// through numpy's ``.base`` mechanism: the returned array keeps this table
    /// alive for as long as Python references it.
    ///
    /// Returns
    /// -------
    /// numpy.ndarray, shape (n_pairs,), dtype uint32
    fn query_point_indices<'py>(slf: Bound<'py, Self>) -> Bound<'py, PyArray1<u32>> {
        let owner = slf.clone().into_any();
        let borrowed = slf.borrow();
        let view = ArrayView1::from(borrowed.inner.query_point_indices());
        unsafe { PyArray1::<u32>::borrow_from_array(&view, owner) }
    }

    /// Reference-point indices ``j``, one per pair.
    ///
    /// Zero-copy view; see [`query_point_indices`](Self::query_point_indices).
    ///
    /// Returns
    /// -------
    /// numpy.ndarray, shape (n_pairs,), dtype uint32
    fn point_indices<'py>(slf: Bound<'py, Self>) -> Bound<'py, PyArray1<u32>> {
        let owner = slf.clone().into_any();
        let borrowed = slf.borrow();
        let view = ArrayView1::from(borrowed.inner.point_indices());
        unsafe { PyArray1::<u32>::borrow_from_array(&view, owner) }
    }

    /// Squared minimum-image distances (Å²), one per pair.
    ///
    /// Zero-copy view, or ``None`` when this table never stored the column.
    /// The square root is left to the caller: most consumers compare against a
    /// squared cutoff, and an accessor that hid a per-pair ``sqrt`` would make
    /// that cost invisible.
    ///
    /// Returns
    /// -------
    /// numpy.ndarray, shape (n_pairs,), dtype float64, or None
    fn dist_sq<'py>(slf: Bound<'py, Self>) -> Option<Bound<'py, PyArray1<NpF>>> {
        let owner = slf.clone().into_any();
        let borrowed = slf.borrow();
        let column = borrowed.inner.dist_sq()?;
        let view = ArrayView1::from(column);
        Some(unsafe { PyArray1::<NpF>::borrow_from_array(&view, owner) })
    }

    /// Minimum-image displacements ``r_j - r_i`` (Å) as an ``(n_pairs, 3)``
    /// array.
    ///
    /// Row ``k`` points from ``i`` to ``j`` and is **not** normalized — its
    /// length is the pair distance. Zero-copy view, or ``None`` when this table
    /// never stored the column (which is not the same as a zero vector: that
    /// would mean two coincident particles).
    ///
    /// Returns
    /// -------
    /// numpy.ndarray, shape (n_pairs, 3), dtype float64, or None
    fn disp<'py>(slf: Bound<'py, Self>) -> Option<Bound<'py, PyArray2<NpF>>> {
        let owner = slf.clone().into_any();
        let borrowed = slf.borrow();
        let view = borrowed.inner.disp()?;
        Some(unsafe { PyArray2::<NpF>::borrow_from_array(&view, owner) })
    }

    /// Number of neighbor pairs — the row count every column shares.
    #[getter]
    fn n_pairs(&self) -> usize {
        self.inner.n_pairs()
    }

    /// Number of reference points the search indexed.
    #[getter]
    fn num_points(&self) -> usize {
        self.inner.num_points()
    }

    /// Number of query points; equal to ``num_points`` for a self search.
    #[getter]
    fn num_query_points(&self) -> usize {
        self.inner.num_query_points()
    }

    /// Whether both index columns address the same point set (half-shell).
    #[getter]
    fn is_self_query(&self) -> bool {
        matches!(self.inner.mode(), QueryMode::SelfQuery { .. })
    }

    fn __repr__(&self) -> String {
        let mode = if matches!(self.inner.mode(), QueryMode::SelfQuery { .. }) {
            "self"
        } else {
            "cross"
        };
        let storage = self.inner.storage();
        format!(
            "Neighbors(n_pairs={}, mode={}, num_points={}, num_query_points={}, \
             dist_sq={}, disp={})",
            self.inner.n_pairs(),
            mode,
            self.inner.num_points(),
            self.inner.num_query_points(),
            storage.dist_sq,
            storage.disp,
        )
    }
}

// ---------------------------------------------------------------------------
// PyNeighborList — the engine
// ---------------------------------------------------------------------------

/// Neighbor search over one point set, exposed to Python as
/// `molrs.NeighborList`.
///
/// The engine owns the cutoff, the backend that indexes space, and the box the
/// index was built for, and keeps the two halves of the job apart: ``build`` /
/// ``update`` index coordinates and enumerate nothing, ``neighbors`` turns the
/// index into a :class:`Neighbors` table. The pairs are half-shell — each
/// unordered pair once, with ``i < j``. A directed search against a *second*
/// point set is :class:`NeighborQuery`.
///
/// Parameters
/// ----------
/// cutoff : float
///     Interaction radius in Å. Fixed at construction because it sets the cell
///     width of the index.
///
/// Raises
/// ------
/// ValueError
///     If ``cutoff`` is not positive.
///
/// Examples
/// --------
/// >>> nl = molrs.NeighborList(3.0)          # O(N) cell-list backend
/// >>> nl.build(points, box)                 # index only — no pair table
/// >>> neigh = nl.neighbors()                # both columns (the default)
/// >>> lean = nl.neighbors(disp=False)       # indices + d² only
/// >>> nl.update(moved_points)               # re-index in the same box
#[pyclass(module = "molrs", name = "NeighborList")]
pub struct PyNeighborList {
    pub(crate) inner: RsNeighborList,
}

#[pymethods]
impl PyNeighborList {
    /// Build an engine with the O(N) cell-list backend — the production choice.
    #[new]
    fn new(cutoff: NpF) -> PyResult<Self> {
        check_cutoff(cutoff)?;
        Ok(Self {
            inner: RsNeighborList::new(cutoff),
        })
    }

    /// Build an engine with the O(N²) all-pairs backend.
    ///
    /// Same pairs as the cell list — that is what makes it useful as an oracle
    /// — at a cost that grows with the square of the particle count.
    ///
    /// Returns
    /// -------
    /// NeighborList
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If ``cutoff`` is not positive.
    #[staticmethod]
    fn brute_force(cutoff: NpF) -> PyResult<Self> {
        check_cutoff(cutoff)?;
        Ok(Self {
            inner: RsNeighborList::brute_force(cutoff),
        })
    }

    /// The cutoff distance (Å) fixed at construction.
    #[getter]
    fn cutoff(&self) -> NpF {
        self.inner.cutoff()
    }

    /// Index ``points`` in ``box`` — coordinates **and** box.
    ///
    /// Builds the spatial index and nothing else: no pairs are enumerated and
    /// no table is allocated. ``box`` is retained, so a later :meth:`update`
    /// can re-index new coordinates in the same box; a run whose box changes
    /// every step (a barostat) must call ``build`` again rather than
    /// ``update``.
    ///
    /// Parameters
    /// ----------
    /// points : numpy.ndarray, shape (N, 3), dtype float64
    ///     Cartesian coordinates in Å.
    /// box : Box
    ///     Simulation box supplying the periodicity used for minimum images.
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If ``points`` does not have shape ``(N, 3)``.
    fn build(&mut self, points: PyReadonlyArray2<'_, NpF>, r#box: &PyBox) -> PyResult<()> {
        check_points(&points, "points")?;
        self.inner.build(points.as_array(), &r#box.inner);
        Ok(())
    }

    /// Re-index new coordinates in the box captured by the last :meth:`build`.
    ///
    /// The natural per-step call at fixed volume: the positions move, the box
    /// does not. Index-only, like ``build``. There is no skin — this re-indexes
    /// every time rather than deciding for you that the previous index was
    /// still good enough.
    ///
    /// Parameters
    /// ----------
    /// points : numpy.ndarray, shape (N, 3), dtype float64
    ///     New Cartesian coordinates in Å.
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If ``points`` does not have shape ``(N, 3)``, or if no ``build`` has
    ///     run yet — the box is then unknown, and guessing one would fold
    ///     minimum images against a box the caller never named.
    fn update(&mut self, points: PyReadonlyArray2<'_, NpF>) -> PyResult<()> {
        check_points(&points, "points")?;
        // The core panics on an update before a build (the box is unknown);
        // a panic must not cross this seam, so check the engine's own state
        // and raise instead.
        if !self.inner.is_built() {
            return Err(PyValueError::new_err(
                "NeighborList.update reuses the box of the previous build: \
                 call build(points, box) first",
            ));
        }
        self.inner.update(points.as_array());
        Ok(())
    }

    /// Materialize the pairs into a :class:`Neighbors` table.
    ///
    /// Both physical columns are kept by default, so no downstream analysis is
    /// surprised by a missing one; pass ``False`` for a column this call site
    /// will not read. A dropped column cannot be added afterwards — rerun the
    /// search instead.
    ///
    /// Row order is unspecified: the cell-list backend materializes in
    /// parallel, and the work split decides the order.
    ///
    /// Parameters
    /// ----------
    /// dist_sq : bool, default True
    ///     Keep the squared minimum-image distance column (Å²), 8 B/pair.
    /// disp : bool, default True
    ///     Keep the minimum-image displacement column (Å), 24 B/pair.
    ///
    /// Returns
    /// -------
    /// Neighbors
    #[pyo3(signature = (dist_sq=true, disp=true))]
    fn neighbors(&self, dist_sq: bool, disp: bool) -> PyNeighbors {
        PyNeighbors {
            inner: self.inner.neighbors(NeighborsStorage { dist_sq, disp }),
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "NeighborList(cutoff={}, built={})",
            self.inner.cutoff(),
            self.inner.is_built(),
        )
    }
}

// ---------------------------------------------------------------------------
// PyNeighborQuery — the cross door
// ---------------------------------------------------------------------------

/// Cross-query neighbor search, exposed to Python as `molrs.NeighborQuery`.
///
/// Build a spatial index from reference points once, then answer repeated
/// queries against it. This is the door for the *cross* question — "which
/// reference points lie within the cutoff of **these other** points?" — and is
/// the reason the class exists. A search of one point set against itself
/// belongs to :class:`NeighborList`, which can additionally re-index moved
/// coordinates and choose which columns a table keeps.
///
/// Every table returned here carries both physical columns.
///
/// Parameters
/// ----------
/// box : Box
///     Simulation box describing geometry and periodic boundaries.
/// points : numpy.ndarray, shape (N, 3), dtype float64
///     Reference point positions in Å.
/// cutoff : float
///     Cutoff radius in Å.
///
/// Examples
/// --------
/// >>> nq = molrs.NeighborQuery(box, positions, cutoff=3.0)
/// >>> cross = nq.query(query_positions)   # directed, no ``i < j`` rule
/// >>> half = nq.query_self()              # half-shell over the reference set
#[pyclass(module = "molrs", name = "NeighborQuery")]
pub struct PyNeighborQuery {
    inner: RsNeighborQuery,
}

#[pymethods]
impl PyNeighborQuery {
    /// Build the spatial index.
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If ``points`` does not have shape ``(N, 3)`` or ``cutoff`` is not
    ///     positive.
    #[new]
    #[pyo3(signature = (r#box, points, cutoff))]
    fn new(r#box: &PyBox, points: PyReadonlyArray2<'_, NpF>, cutoff: NpF) -> PyResult<Self> {
        check_points(&points, "points")?;
        check_cutoff(cutoff)?;
        Ok(Self {
            inner: RsNeighborQuery::new(&r#box.inner, points.as_array(), cutoff),
        })
    }

    /// Build a free-boundary spatial index from reference points.
    ///
    /// The non-periodic bounding box is derived from the point cloud, so the
    /// caller does not need to manufacture a simulation box for selections or
    /// other isolated-coordinate queries.
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If ``points`` does not have shape ``(N, 3)`` or ``cutoff`` is not
    ///     positive.
    #[staticmethod]
    #[pyo3(signature = (points, cutoff))]
    fn free(points: PyReadonlyArray2<'_, NpF>, cutoff: NpF) -> PyResult<Self> {
        check_points(&points, "points")?;
        check_cutoff(cutoff)?;
        Ok(Self {
            inner: RsNeighborQuery::free(points.as_array(), cutoff),
        })
    }

    /// Find all pairs where ``i`` indexes ``query_points`` and ``j`` indexes
    /// the reference points.
    ///
    /// Directed: every query point reports all of its reference neighbors, with
    /// no ``i < j`` constraint. Passing the reference coordinates back in means
    /// each point finds itself at distance zero and each pair appears twice —
    /// use :meth:`query_self` when each unordered pair should appear once.
    ///
    /// Returns
    /// -------
    /// Neighbors
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If ``query_points`` does not have shape ``(M, 3)``.
    fn query(&self, query_points: PyReadonlyArray2<'_, NpF>) -> PyResult<PyNeighbors> {
        check_points(&query_points, "query_points")?;
        Ok(PyNeighbors {
            inner: self.inner.query(query_points.as_array()),
        })
    }

    /// Find each unordered pair within the reference point set once (``i < j``).
    ///
    /// Returns
    /// -------
    /// Neighbors
    fn query_self(&self) -> PyNeighbors {
        PyNeighbors {
            inner: self.inner.query_self(),
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "NeighborQuery(num_points={}, cutoff={})",
            self.inner.points().nrows(),
            self.inner.cutoff(),
        )
    }
}

fn skin_err(err: SkinError) -> PyErr {
    PyValueError::new_err(err.to_string())
}

/// A :class:`NeighborList` with Verlet skin — ``VerletSkin(NeighborList, …)``.
///
/// Constructed from a search engine whose cutoff is ``cutoff + skin``. The
/// engine is **moved** into this object. Passing the skin into
/// ``VelocityVerlet(..., neighbors=skin)`` moves it again into the integrator.
#[pyclass(module = "molrs", name = "VerletSkin")]
pub struct PyVerletSkin {
    pub(crate) inner: Option<RsVerletSkin>,
}

impl PyVerletSkin {
    pub(crate) fn take(&mut self) -> PyResult<RsVerletSkin> {
        self.inner
            .take()
            .ok_or_else(|| PyValueError::new_err("VerletSkin has already been moved"))
    }

    fn get(&self) -> PyResult<&RsVerletSkin> {
        self.inner
            .as_ref()
            .ok_or_else(|| PyValueError::new_err("VerletSkin has already been moved"))
    }

    fn get_mut(&mut self) -> PyResult<&mut RsVerletSkin> {
        self.inner
            .as_mut()
            .ok_or_else(|| PyValueError::new_err("VerletSkin has already been moved"))
    }
}

#[pymethods]
impl PyVerletSkin {
    /// Wrap ``neighbors`` (cutoff must equal ``cutoff + skin``) with Verlet policy.
    #[new]
    #[pyo3(signature = (neighbors, cutoff, positions, r#box, *, skin=0.0, every=1, delay=0, check=true))]
    fn new(
        neighbors: &mut PyNeighborList,
        cutoff: NpF,
        positions: PyReadonlyArray2<'_, NpF>,
        r#box: &PyBox,
        skin: NpF,
        every: usize,
        delay: usize,
        check: bool,
    ) -> PyResult<Self> {
        check_points(&positions, "positions")?;
        if cutoff <= 0.0 {
            return Err(PyValueError::new_err("cutoff must be > 0 Å"));
        }
        let search_cutoff = neighbors.inner.cutoff();
        let search = std::mem::replace(&mut neighbors.inner, RsNeighborList::new(search_cutoff));
        let policy = NeighborPolicy {
            skin,
            every,
            delay,
            check,
        };
        let inner = RsVerletSkin::new(
            search,
            cutoff,
            policy,
            positions.as_array(),
            r#box.inner.clone(),
        )
        .map_err(skin_err)?;
        Ok(Self {
            inner: Some(inner),
        })
    }

    #[getter]
    fn cutoff(&self) -> PyResult<NpF> {
        Ok(self.get()?.cutoff)
    }

    #[getter]
    fn skin(&self) -> PyResult<NpF> {
        Ok(self.get()?.skin)
    }

    #[getter]
    fn num_edges(&self) -> PyResult<usize> {
        Ok(self.get()?.num_edges)
    }

    #[getter]
    fn rebuild_count(&self) -> PyResult<usize> {
        Ok(self.get()?.rebuild_count)
    }

    #[getter]
    fn ago(&self) -> PyResult<usize> {
        Ok(self.get()?.ago)
    }

    fn update(&mut self, positions: PyReadonlyArray2<'_, NpF>) -> PyResult<bool> {
        check_points(&positions, "positions")?;
        self.get_mut()?
            .update(positions.as_array())
            .map_err(skin_err)
    }

    fn rebuild(&mut self, positions: PyReadonlyArray2<'_, NpF>) -> PyResult<()> {
        check_points(&positions, "positions")?;
        self.get_mut()?
            .rebuild(positions.as_array())
            .map_err(skin_err)
    }

    fn __repr__(&self) -> String {
        match &self.inner {
            Some(s) => format!(
                "VerletSkin(cutoff={}, skin={}, edges={}, rebuilds={})",
                s.cutoff, s.skin, s.num_edges, s.rebuild_count
            ),
            None => "VerletSkin(<moved>)".into(),
        }
    }
}
