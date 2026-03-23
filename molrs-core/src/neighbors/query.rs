//! High-level neighbor query API inspired by freud-analysis.
//!
//! [`AABBQuery`] wraps a [`LinkCell`] spatial index built from reference points
//! and provides two query modes:
//!
//! - [`query`](AABBQuery::query) — cross-query: find all pairs `(i, j)` where
//!   `i` indexes `query_points` and `j` indexes the reference `points`.
//! - [`query_self`](AABBQuery::query_self) — self-query: find unique pairs
//!   `(i, j)` with `i < j` within the same point set.

use crate::neighbors::linkcell::LinkCell;
use crate::neighbors::{NbListAlgo, NeighborList, QueryMode};
use crate::region::simbox::SimBox;
use crate::types::{F, FNx3, FNx3View};

/// Axis-aligned bounding box query — high-level neighbor search.
///
/// Wraps a [`LinkCell`] spatial index built from reference points. Provides
/// both self-query and cross-query methods following the freud-analysis API.
///
/// # Example
///
/// ```ignore
/// let nq = AABBQuery::new(&simbox, points.view(), 3.0);
/// let nlist = nq.query(query_points.view());   // cross-query
/// let nlist = nq.query_self();                  // self-query
/// ```
#[derive(Debug, Clone)]
pub struct AABBQuery {
    /// The underlying cell-list spatial index (built once at construction).
    lc: LinkCell,
    /// Copy of the reference points (owned for self-query).
    points: FNx3,
    /// Copy of the simulation box.
    simbox: SimBox,
    /// Cutoff distance.
    cutoff: F,
}

impl AABBQuery {
    /// Build a spatial index from reference points.
    ///
    /// # Panics
    /// Panics if `cutoff <= 0` or `points` does not have 3 columns.
    pub fn new(simbox: &SimBox, points: FNx3View<'_>, cutoff: F) -> Self {
        assert!(cutoff > 0.0, "cutoff must be positive");
        assert_eq!(points.ncols(), 3, "points must have shape (N, 3)");

        let mut lc = LinkCell::new().cutoff(cutoff);
        lc.build_index(points, simbox);

        Self {
            lc,
            points: points.to_owned(),
            simbox: simbox.clone(),
            cutoff,
        }
    }

    /// Cross-query: find all pairs `(i, j)` where `i` indexes `query_points`
    /// and `j` indexes the reference `points`.
    ///
    /// Returns full-shell results (not half-shell): for each query point, all
    /// reference neighbors are returned, even if `i == j` would duplicate in
    /// the same-point-set case.
    pub fn query(&self, query_points: FNx3View<'_>) -> NeighborList {
        assert_eq!(
            query_points.ncols(),
            3,
            "query_points must have shape (N, 3)"
        );

        let n_query = query_points.nrows();
        let n_ref = self.points.nrows();
        let cutoff_sq = self.cutoff * self.cutoff;

        let mut nlist = NeighborList::with_mode(QueryMode::CrossQuery, n_ref, n_query);

        // For each query point, check all 27 neighboring cells
        for qi in 0..n_query {
            let qp = query_points.row(qi);
            self.lc
                .visit_neighbors_of(qp, &self.simbox, |rj, dist_sq, diff| {
                    if dist_sq <= cutoff_sq {
                        nlist.push(qi as u32, rj, dist_sq, diff);
                    }
                });
        }

        nlist
    }

    /// Self-query: find unique pairs `(i, j)` with `i < j` within the
    /// reference point set.
    ///
    /// Equivalent to building a standard half-shell neighbor list.
    pub fn query_self(&self) -> NeighborList {
        let n = self.points.nrows();

        // Reuse the lower-level LinkCell build which does half-shell iteration
        let mut lc = LinkCell::new().cutoff(self.cutoff);
        lc.build(self.points.view(), &self.simbox);
        let raw = lc.query().clone();

        // Tag with self-query metadata
        NeighborList {
            idx_i: raw.idx_i,
            idx_j: raw.idx_j,
            dist_sq: raw.dist_sq,
            diff_flat: raw.diff_flat,
            mode: QueryMode::SelfQuery,
            num_points: n,
            num_query_points: n,
        }
    }

    /// Reference to the stored simulation box.
    pub fn simbox(&self) -> &SimBox {
        &self.simbox
    }

    /// Reference to the stored reference points.
    pub fn points(&self) -> FNx3View<'_> {
        self.points.view()
    }

    /// The cutoff distance.
    pub fn cutoff(&self) -> F {
        self.cutoff
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::region::simbox::SimBox;
    use ndarray::array;

    #[test]
    fn self_query_matches_linkcell() {
        let bx = SimBox::cube(4.0, array![0.0, 0.0, 0.0], [true, true, true]).expect("invalid box");
        let pts = array![[0.1, 0.2, 0.3], [0.3, 0.2, 0.1], [3.9, 3.8, 3.7]];

        let nq = AABBQuery::new(&bx, pts.view(), 0.5);
        let nlist = nq.query_self();

        assert_eq!(nlist.mode(), QueryMode::SelfQuery);
        assert_eq!(nlist.n_pairs(), 1);
        assert_eq!(nlist.query_point_indices()[0], 0);
        assert_eq!(nlist.point_indices()[0], 1);
        assert_eq!(nlist.num_points(), 3);
        assert_eq!(nlist.num_query_points(), 3);
    }

    #[test]
    fn cross_query_finds_all_neighbors() {
        let bx = SimBox::cube(4.0, array![0.0, 0.0, 0.0], [true, true, true]).expect("invalid box");
        let ref_pts = array![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]];
        let query_pts = array![[0.5, 0.0, 0.0]];

        let nq = AABBQuery::new(&bx, ref_pts.view(), 0.6);
        let nlist = nq.query(query_pts.view());

        assert_eq!(nlist.mode(), QueryMode::CrossQuery);
        assert_eq!(nlist.num_query_points(), 1);
        assert_eq!(nlist.num_points(), 3);
        // query point at 0.5 is within 0.6 of ref points 0 (at 0.0) and 1 (at 1.0)
        assert_eq!(nlist.n_pairs(), 2);
    }

    #[test]
    fn distances_returns_sqrt() {
        let bx = SimBox::cube(4.0, array![0.0, 0.0, 0.0], [true, true, true]).expect("invalid box");
        let pts = array![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]];

        let nq = AABBQuery::new(&bx, pts.view(), 1.5);
        let nlist = nq.query_self();

        assert_eq!(nlist.n_pairs(), 1);
        let dists = nlist.distances();
        assert!((dists[0] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn vectors_alias_works() {
        let bx = SimBox::cube(4.0, array![0.0, 0.0, 0.0], [true, true, true]).expect("invalid box");
        let pts = array![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]];

        let nq = AABBQuery::new(&bx, pts.view(), 1.5);
        let nlist = nq.query_self();

        let vecs = nlist.vectors();
        assert_eq!(vecs.nrows(), 1);
        assert!((vecs[[0, 0]] - 1.0).abs() < 1e-6);
        assert!(vecs[[0, 1]].abs() < 1e-6);
        assert!(vecs[[0, 2]].abs() < 1e-6);
    }

    #[test]
    fn self_query_pbc_boundary() {
        let bx = SimBox::cube(2.0, array![0.0, 0.0, 0.0], [true, true, true]).expect("invalid box");
        let pts = array![[0.1, 0.1, 0.1], [1.9, 1.9, 1.9]];

        let nq = AABBQuery::new(&bx, pts.view(), 0.5);
        let nlist = nq.query_self();

        assert_eq!(nlist.n_pairs(), 1);
    }

    #[test]
    fn cross_query_self_overlap_produces_full_shell() {
        let bx = SimBox::cube(4.0, array![0.0, 0.0, 0.0], [true, true, true]).expect("invalid box");
        let pts = array![[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]];

        // Cross-query with the same point set should produce pairs in both
        // directions (0->1 and 1->0), plus self-pairs (0->0, 1->1), unlike
        // self-query which only produces (0, 1) once.
        let nq = AABBQuery::new(&bx, pts.view(), 0.6);
        let nlist = nq.query(pts.view());

        // Each query point finds both reference points (including itself at dist=0)
        // Total: 2 query x 2 ref = 4 pairs
        assert_eq!(nlist.n_pairs(), 4);

        // Compare with self-query which only produces unique pairs (i<j)
        let self_nlist = nq.query_self();
        assert_eq!(self_nlist.n_pairs(), 1);
    }
}
