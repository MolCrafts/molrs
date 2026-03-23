//! Neighbor-list algorithms for pairwise distance queries.
//!
//! This module provides a trait-based framework for neighbor search, plus a
//! high-level [`AABBQuery`] wrapper inspired by [freud-analysis](https://freud.readthedocs.io/).
//!
//! ## Algorithms
//!
//! - [`LinkCell`] — O(N) cell-list algorithm, parallelized with rayon when the
//!   `rayon` feature is enabled (default). Recommended for production use.
//! - [`BruteForce`] — O(N²) all-pairs reference implementation, useful for
//!   correctness testing and small systems.
//!
//! Both implement [`NbListAlgo`] and can be used via the lower-level API or
//! wrapped in [`AABBQuery`] for a freud-style workflow.
//!
//! ## High-level API (freud-style)
//!
//! ```ignore
//! let nq = AABBQuery::new(&simbox, points.view(), 3.0);
//! let nlist = nq.query(query_points.view());   // cross-query
//! let nlist = nq.query_self();                  // self-query (i < j)
//!
//! nlist.query_point_indices()   // &[u32]
//! nlist.point_indices()         // &[u32]
//! nlist.distances()             // Vec<F>
//! nlist.vectors()               // FNx3View
//! ```
//!
//! ## Lower-level API
//!
//! ```ignore
//! let mut lc = LinkCell::new().cutoff(3.0);
//! lc.build(points.view(), &simbox);
//! let result = lc.query();
//! ```

use crate::region::simbox::SimBox;
use crate::types::{F, FNx3View};
use ndarray::ArrayView2;

pub mod bruteforce;
mod linkcell;
mod query;

pub use bruteforce::BruteForce;
pub use linkcell::LinkCell;
pub use query::AABBQuery;

// ---------------------------------------------------------------------------
// QueryMode — distinguishes self-query from cross-query
// ---------------------------------------------------------------------------

/// Whether a [`NeighborList`] was produced by a self-query or a cross-query.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QueryMode {
    /// Self-query: pairs within the same point set, with `i < j` (half-shell).
    SelfQuery,
    /// Cross-query: pairs where `i` indexes `query_points` and `j` indexes
    /// `reference_points` (full-shell).
    CrossQuery,
}

// ---------------------------------------------------------------------------
// PairVisitor -- zero-allocation callback for on-demand pair traversal
// ---------------------------------------------------------------------------

/// Callback for on-demand pair traversal (zero-allocation alternative to
/// [`NeighborList`]).
///
/// Implement this trait to process pairs without storing them. Used by
/// [`NbListAlgo::visit_pairs`].
pub trait PairVisitor {
    /// Called for each pair `(i, j)` within the cutoff.
    ///
    /// * `i`, `j` — original particle indices
    /// * `dist_sq` — squared distance between particles
    /// * `diff` — displacement vector `r_j - r_i` (minimum-image)
    fn visit_pair(&mut self, i: u32, j: u32, dist_sq: F, diff: [F; 3]);
}

/// Blanket impl: any `FnMut(u32, u32, F, [F; 3])` is a `PairVisitor`.
impl<T: FnMut(u32, u32, F, [F; 3])> PairVisitor for T {
    #[inline(always)]
    fn visit_pair(&mut self, i: u32, j: u32, dist_sq: F, diff: [F; 3]) {
        self(i, j, dist_sq, diff)
    }
}

// ---------------------------------------------------------------------------
// NbListAlgo trait
// ---------------------------------------------------------------------------

/// Trait implemented by neighbor-list algorithms.
///
/// Each algorithm maintains internal state and caches pair results after
/// [`build`](NbListAlgo::build) or [`update`](NbListAlgo::update), so that
/// [`query`](NbListAlgo::query) returns a cheap reference.
pub trait NbListAlgo {
    /// Build the neighbor list from scratch.
    ///
    /// # Panics
    /// Panics if the cutoff is not set or `points` has wrong shape.
    fn build(&mut self, points: FNx3View<'_>, bx: &SimBox);

    /// Rebuild the neighbor list with new positions.
    ///
    /// Implementations may reuse internal buffers for efficiency.
    fn update(&mut self, points: FNx3View<'_>, bx: &SimBox);

    /// Return a reference to the cached pair results.
    fn query(&self) -> &NeighborList;

    /// Return a reference to the simulation box used in the last build/update.
    fn box_ref(&self) -> &SimBox;

    /// Build the spatial index only — NO pair pre-computation.
    ///
    /// After calling this, [`visit_pairs`](NbListAlgo::visit_pairs) can
    /// traverse pairs on-demand without allocating a [`NeighborList`].
    /// The default falls back to a full [`build`](NbListAlgo::build).
    fn build_index(&mut self, points: FNx3View<'_>, bx: &SimBox) {
        self.build(points, bx);
    }

    /// Rebuild the spatial index only — NO pair pre-computation.
    ///
    /// Equivalent to [`update`](NbListAlgo::update) but skips pair enumeration.
    /// The default falls back to a full [`update`](NbListAlgo::update).
    fn update_index(&mut self, points: FNx3View<'_>, bx: &SimBox) {
        self.update(points, bx);
    }

    /// Traverse pairs on-demand, calling the visitor for each pair within
    /// the cutoff. Zero allocation — no [`NeighborList`] is built.
    ///
    /// The default iterates the pre-stored pairs from [`query`](NbListAlgo::query).
    fn visit_pairs(&self, visitor: &mut dyn PairVisitor) {
        let result = self.query();
        let diff = result.vectors();
        for k in 0..result.n_pairs() {
            visitor.visit_pair(
                result.query_point_indices()[k],
                result.point_indices()[k],
                result.dist_sq()[k],
                [diff[[k, 0]], diff[[k, 1]], diff[[k, 2]]],
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Algorithm-generic wrapper (lower-level API)
// ---------------------------------------------------------------------------

/// Algorithm-generic neighbor list wrapper (lower-level API).
///
/// Thin facade that delegates every call to the underlying [`NbListAlgo`]
/// implementation. Construct via `NbList(algo)`.
#[derive(Debug, Clone)]
pub struct NbList<A: NbListAlgo>(pub A);

impl<A: NbListAlgo> NbList<A> {
    /// Build the neighbor list from scratch. See [`NbListAlgo::build`].
    pub fn build(&mut self, points: FNx3View<'_>, bx: &SimBox) {
        self.0.build(points, bx)
    }

    /// Rebuild with new positions. See [`NbListAlgo::update`].
    pub fn update(&mut self, points: FNx3View<'_>, bx: &SimBox) {
        self.0.update(points, bx)
    }

    /// Return cached pair results. See [`NbListAlgo::query`].
    pub fn query(&self) -> &NeighborList {
        self.0.query()
    }

    /// Return the simulation box. See [`NbListAlgo::box_ref`].
    pub fn box_ref(&self) -> &SimBox {
        self.0.box_ref()
    }

    /// Build spatial index only (no pair pre-computation). See [`NbListAlgo::build_index`].
    pub fn build_index(&mut self, points: FNx3View<'_>, bx: &SimBox) {
        self.0.build_index(points, bx)
    }

    /// Rebuild spatial index only. See [`NbListAlgo::update_index`].
    pub fn update_index(&mut self, points: FNx3View<'_>, bx: &SimBox) {
        self.0.update_index(points, bx)
    }

    /// Traverse pairs on-demand. See [`NbListAlgo::visit_pairs`].
    pub fn visit_pairs(&self, visitor: &mut dyn PairVisitor) {
        self.0.visit_pairs(visitor)
    }
}

// ---------------------------------------------------------------------------
// NeighborList (was NeighborResult)
// ---------------------------------------------------------------------------

/// Cached neighbor query results (freud-style).
///
/// Stores all pairs within the cutoff together with squared distances and
/// displacement vectors. The [`mode`](NeighborList::mode) field records
/// whether this was a self-query or cross-query.
///
/// - **Self-query**: `i < j`, both indices refer to the same point set.
/// - **Cross-query**: `i` indexes `query_points`, `j` indexes `reference_points`.
#[derive(Debug, Clone)]
pub struct NeighborList {
    /// Query point index per pair (i).
    pub(crate) idx_i: Vec<u32>,
    /// Reference point index per pair (j).
    pub(crate) idx_j: Vec<u32>,
    /// Squared distances, one per pair.
    pub(crate) dist_sq: Vec<F>,
    /// Flat displacement vectors `[dx0,dy0,dz0, dx1,dy1,dz1, ...]`, stride-3.
    pub(crate) diff_flat: Vec<F>,
    /// Self-query vs cross-query.
    pub(crate) mode: QueryMode,
    /// Number of reference points (j indices).
    pub(crate) num_points: usize,
    /// Number of query points (i indices).
    pub(crate) num_query_points: usize,
}

impl NeighborList {
    /// Create an empty result with no pairs.
    pub(crate) fn empty() -> Self {
        Self {
            idx_i: Vec::new(),
            idx_j: Vec::new(),
            dist_sq: Vec::new(),
            diff_flat: Vec::new(),
            mode: QueryMode::SelfQuery,
            num_points: 0,
            num_query_points: 0,
        }
    }

    /// Create an empty result tagged with mode and point counts.
    pub(crate) fn with_mode(mode: QueryMode, num_points: usize, num_query_points: usize) -> Self {
        Self {
            idx_i: Vec::new(),
            idx_j: Vec::new(),
            dist_sq: Vec::new(),
            diff_flat: Vec::new(),
            mode,
            num_points,
            num_query_points,
        }
    }

    /// Clear all buffers for reuse.
    pub(crate) fn clear(&mut self) {
        self.idx_i.clear();
        self.idx_j.clear();
        self.dist_sq.clear();
        self.diff_flat.clear();
    }

    /// Push a neighbor pair into the result.
    #[inline(always)]
    pub(crate) fn push(&mut self, i: u32, j: u32, d2: F, dr: [F; 3]) {
        self.idx_i.push(i);
        self.idx_j.push(j);
        self.dist_sq.push(d2);
        self.diff_flat.extend(dr);
    }

    // -----------------------------------------------------------------------
    // Public accessors (freud-style names)
    // -----------------------------------------------------------------------

    /// Number of neighbor pairs.
    #[inline]
    pub fn n_pairs(&self) -> usize {
        self.idx_i.len()
    }

    /// Query mode (self-query or cross-query).
    #[inline]
    pub fn mode(&self) -> QueryMode {
        self.mode
    }

    /// Number of reference points (the point set used to build the spatial index).
    #[inline]
    pub fn num_points(&self) -> usize {
        self.num_points
    }

    /// Number of query points.
    /// - Self-query: same as `num_points`.
    /// - Cross-query: the number of points passed to `query()`.
    #[inline]
    pub fn num_query_points(&self) -> usize {
        self.num_query_points
    }

    /// Query point indices (i), one per pair.
    ///
    /// - Self-query: indices into the single point set.
    /// - Cross-query: indices into `query_points`.
    #[inline]
    pub fn query_point_indices(&self) -> &[u32] {
        &self.idx_i
    }

    /// Reference point indices (j), one per pair.
    ///
    /// - Self-query: indices into the single point set (always > i for same-cell pairs).
    /// - Cross-query: indices into the reference `points`.
    #[inline]
    pub fn point_indices(&self) -> &[u32] {
        &self.idx_j
    }

    /// Actual distances (not squared), one per pair.
    pub fn distances(&self) -> Vec<F> {
        self.dist_sq.iter().map(|&d2| d2.sqrt()).collect()
    }

    /// Squared distances, one per pair.
    #[inline]
    pub fn dist_sq(&self) -> &[F] {
        &self.dist_sq
    }

    /// Displacement vectors as an N x 3 array view.
    #[inline]
    pub fn vectors(&self) -> FNx3View<'_> {
        ArrayView2::from_shape((self.n_pairs(), 3), &self.diff_flat)
            .expect("diff_flat shape mismatch")
    }

    // -----------------------------------------------------------------------
    // Deprecated aliases for backward compatibility
    // -----------------------------------------------------------------------

    /// First particle indices, one per pair.
    #[deprecated(note = "use query_point_indices() instead")]
    #[inline]
    pub fn idx_i(&self) -> &[u32] {
        &self.idx_i
    }

    /// Second particle indices, one per pair.
    #[deprecated(note = "use point_indices() instead")]
    #[inline]
    pub fn idx_j(&self) -> &[u32] {
        &self.idx_j
    }

    /// Displacement vectors as an N x 3 array view.
    #[deprecated(note = "use vectors() instead")]
    #[inline]
    pub fn diff(&self) -> FNx3View<'_> {
        self.vectors()
    }
}

/// Backward-compatible type alias.
#[deprecated(note = "use NeighborList instead")]
pub type NeighborResult = NeighborList;
