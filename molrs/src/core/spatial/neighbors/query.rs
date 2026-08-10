//! High-level neighbor query API inspired by freud-analysis.
//!
//! [`NeighborQuery`] wraps a [`LinkCell`] spatial index built once from a set of
//! *reference points*, and then answers repeated queries against it in two
//! modes:
//!
//! - [`query`](NeighborQuery::query) — cross-query: find all pairs `(i, j)` where
//!   `i` indexes a separate set of *query points* and `j` indexes the reference
//!   points. Directed and full-shell.
//! - [`query_self`](NeighborQuery::query_self) — self-query: find unique pairs
//!   `(i, j)` with `i < j` within the reference point set. Half-shell, so each
//!   unordered pair appears once.
//!
//! Every table returned here carries both physical columns
//! ([`NeighborsStorage::FULL`]): distances in Å² and minimum-image
//! displacements in Å. This wrapper does not expose a column policy — a caller
//! who wants a leaner table drives [`LinkCell::with_storage`] directly, or
//! drops columns afterwards with [`Neighbors::repack`].

use crate::spatial::neighbors::linkcell::LinkCell;
use crate::spatial::neighbors::{NbListAlgo, Neighbors, NeighborsStorage, QueryMode};
use crate::spatial::simbox::SimBox;
use crate::types::{F, FNx3, FNx3View};

/// High-level neighbor search over a fixed set of reference points.
///
/// Wraps a [`LinkCell`] cell-list spatial index built once at construction, so
/// that many queries can be answered against the same reference points without
/// re-indexing. Provides both self-query and cross-query methods following the
/// freud-analysis API. Every returned [`Neighbors`] table uses
/// [`NeighborsStorage::FULL`].
///
/// # Example
///
/// ```ignore
/// let nq = NeighborQuery::new(&simbox, points.view(), 3.0);
/// let nlist = nq.query(query_points.view());   // cross-query
/// let nlist = nq.query_self();                  // self-query
/// ```
#[derive(Debug, Clone)]
pub struct NeighborQuery {
    /// The underlying cell-list spatial index (built once at construction).
    lc: LinkCell,
    /// Copy of the reference points (owned for self-query).
    points: FNx3,
    /// Copy of the simulation box.
    simbox: SimBox,
    /// Cutoff distance.
    cutoff: F,
}

impl NeighborQuery {
    /// Build a spatial index from reference points.
    ///
    /// `points` is an `N × 3` view of Cartesian coordinates (Å), `simbox`
    /// supplies the periodicity used for minimum-image distances, and `cutoff`
    /// (Å) is the interaction radius every later query uses — it is fixed here,
    /// not per query, because it sets the cell size of the index.
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

    /// Build from free-boundary points (no periodic box).
    ///
    /// "Free boundary" means the system is an isolated cluster rather than a
    /// periodically repeated cell, so no pair may wrap around an edge. The
    /// constructor derives a non-periodic bounding box from the point cloud,
    /// padded by `cutoff` (Å) so that every particle sits comfortably inside it
    /// and no minimum-image folding can occur.
    ///
    /// # Panics
    /// Panics if `cutoff <= 0` or `points` does not have 3 columns. An empty
    /// point set is not an error: it yields an empty index whose queries return
    /// no pairs.
    pub fn free(points: FNx3View<'_>, cutoff: F) -> Self {
        let bx =
            SimBox::free(points, cutoff).expect("degenerate point cloud for free-boundary box");
        Self::new(&bx, points, cutoff)
    }

    /// Cross-query: find all pairs `(i, j)` where `i` indexes `query_points`
    /// and `j` indexes the reference `points`.
    ///
    /// Results are full-shell, not half-shell: every query point reports all of
    /// its reference neighbors, with no `i < j` constraint. If `query_points`
    /// holds the same coordinates as the reference set, each point therefore
    /// finds itself at distance zero and each pair appears in both orderings —
    /// use [`query_self`](Self::query_self) when you want each unordered pair
    /// once instead.
    ///
    /// The returned table is tagged [`QueryMode::CrossQuery`] with both point
    /// counts, and carries both physical columns
    /// ([`NeighborsStorage::FULL`]): `dist_sq` in Å², `disp` in Å.
    ///
    /// # Panics
    /// Panics if `query_points` does not have 3 columns.
    pub fn query(&self, query_points: FNx3View<'_>) -> Neighbors {
        assert_eq!(
            query_points.ncols(),
            3,
            "query_points must have shape (N, 3)"
        );

        let n_query = query_points.nrows();
        let n_ref = self.points.nrows();
        let cutoff_sq = self.cutoff * self.cutoff;

        let mut nlist = Neighbors::empty(
            QueryMode::CrossQuery {
                num_query_points: n_query,
                num_points: n_ref,
            },
            NeighborsStorage::FULL,
        );

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
    /// This is the standard half-shell neighbor list: each unordered pair is
    /// reported exactly once and no point pairs with itself. The returned table
    /// is tagged `QueryMode::SelfQuery { num_points }` with the size of the
    /// reference set, and carries both physical columns
    /// ([`NeighborsStorage::FULL`]).
    ///
    /// Note that this rebuilds a [`LinkCell`] over the reference points rather
    /// than reusing the index held for cross-queries, because the two traversals
    /// differ: a self-query walks cell pairs once in a forward direction, while
    /// a cross-query walks the full neighborhood of each query point.
    pub fn query_self(&self) -> Neighbors {
        // Reuse the lower-level LinkCell build, which does half-shell iteration
        // and already tags the result `SelfQuery { num_points }`.
        let mut lc = LinkCell::new().cutoff(self.cutoff);
        lc.build(self.points.view(), &self.simbox);
        lc.query().clone()
    }

    /// Reference to the stored simulation box.
    pub fn simbox(&self) -> &SimBox {
        &self.simbox
    }

    /// Reference to the stored reference points (`N × 3`, Å).
    pub fn points(&self) -> FNx3View<'_> {
        self.points.view()
    }

    /// The cutoff distance (Å) fixed at construction.
    pub fn cutoff(&self) -> F {
        self.cutoff
    }

    /// SoA sibling of [`new`](Self::new): build a spatial index from
    /// column-major `x`/`y`/`z` reference-point slices.
    ///
    /// "SoA" is *structure of arrays*: instead of one `N × 3` array of points,
    /// the coordinates arrive as three length-`N` slices (Å), one per axis.
    ///
    /// Byte-for-byte equivalent to `new` on the same coordinates — the index is
    /// built via [`LinkCell::build_index_soa`] and an owned interleaved copy of
    /// the points is retained for [`query_self`](Self::query_self), exactly as
    /// `new` does. Lets callers holding SoA columns skip the interleave.
    ///
    /// # Panics
    /// Panics if `cutoff <= 0` or the three slices differ in length.
    pub fn from_columns(simbox: &SimBox, xs: &[F], ys: &[F], zs: &[F], cutoff: F) -> Self {
        assert!(cutoff > 0.0, "cutoff must be positive");
        assert!(
            xs.len() == ys.len() && ys.len() == zs.len(),
            "x/y/z slices must have equal length"
        );

        let mut lc = LinkCell::new().cutoff(cutoff);
        lc.build_index_soa(xs, ys, zs, simbox);

        // Owned interleaved copy for self-query, mirroring `new`.
        let n = xs.len();
        let mut points = FNx3::zeros((n, 3));
        for i in 0..n {
            points[[i, 0]] = xs[i];
            points[[i, 1]] = ys[i];
            points[[i, 2]] = zs[i];
        }

        Self {
            lc,
            points,
            simbox: simbox.clone(),
            cutoff,
        }
    }

    /// SoA sibling of [`free`](Self::free): build from free-boundary points
    /// held as column-major `x`/`y`/`z` slices.
    ///
    /// Uses [`SimBox::free_columns`] to derive the same bounding box `free`
    /// would produce from the interleaved points, so the result is
    /// byte-identical.
    pub fn free_columns(xs: &[F], ys: &[F], zs: &[F], cutoff: F) -> Self {
        let bx = SimBox::free_columns(xs, ys, zs, cutoff)
            .expect("degenerate point cloud for free-boundary box");
        Self::from_columns(&bx, xs, ys, zs, cutoff)
    }

    /// SoA sibling of [`query`](Self::query): cross-query reading each query
    /// point from column-major `qx`/`qy`/`qz` slices.
    ///
    /// Same pair order and cutoff test as `query`, so the returned
    /// [`Neighbors`] is byte-identical to `query` on the same coordinates.
    ///
    /// # Panics
    /// Panics if the three query slices differ in length.
    pub fn query_columns(&self, qx: &[F], qy: &[F], qz: &[F]) -> Neighbors {
        assert!(
            qx.len() == qy.len() && qy.len() == qz.len(),
            "query x/y/z slices must have equal length"
        );

        let n_query = qx.len();
        let n_ref = self.points.nrows();
        let cutoff_sq = self.cutoff * self.cutoff;

        let mut nlist = Neighbors::empty(
            QueryMode::CrossQuery {
                num_query_points: n_query,
                num_points: n_ref,
            },
            NeighborsStorage::FULL,
        );

        // For each query point, check all 27 neighboring cells
        for qi in 0..n_query {
            let qp = [qx[qi], qy[qi], qz[qi]];
            self.lc
                .visit_neighbors_of_pt(qp, &self.simbox, |rj, dist_sq, diff| {
                    if dist_sq <= cutoff_sq {
                        nlist.push(qi as u32, rj, dist_sq, diff);
                    }
                });
        }

        nlist
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::spatial::simbox::SimBox;
    use ndarray::array;

    #[test]
    fn self_query_matches_linkcell() {
        let bx = SimBox::cube(4.0, array![0.0, 0.0, 0.0], [true, true, true]).expect("invalid box");
        let pts = array![[0.1, 0.2, 0.3], [0.3, 0.2, 0.1], [3.9, 3.8, 3.7]];

        let nq = NeighborQuery::new(&bx, pts.view(), 0.5);
        let nlist = nq.query_self();

        assert_eq!(nlist.mode(), QueryMode::SelfQuery { num_points: 3 });
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

        let nq = NeighborQuery::new(&bx, ref_pts.view(), 0.6);
        let nlist = nq.query(query_pts.view());

        assert_eq!(
            nlist.mode(),
            QueryMode::CrossQuery {
                num_query_points: 1,
                num_points: 3,
            }
        );
        assert_eq!(nlist.num_query_points(), 1);
        assert_eq!(nlist.num_points(), 3);
        // query point at 0.5 is within 0.6 of ref points 0 (at 0.0) and 1 (at 1.0)
        assert_eq!(nlist.n_pairs(), 2);
    }

    /// The table stores \(d^2\); a caller that wants a distance takes the square
    /// root itself (there is no `distances()` accessor to hide the cost or to
    /// return an empty slice when the column is absent).
    #[test]
    fn dist_sq_square_roots_to_distance() {
        let bx = SimBox::cube(4.0, array![0.0, 0.0, 0.0], [true, true, true]).expect("invalid box");
        let pts = array![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]];

        let nq = NeighborQuery::new(&bx, pts.view(), 1.5);
        let nlist = nq.query_self();

        assert_eq!(nlist.n_pairs(), 1);
        let d2 = nlist.dist_sq().expect("self-query materializes dist_sq");
        assert!((d2[0] - 1.0).abs() < 1e-6);
        assert!((d2[0].sqrt() - 1.0).abs() < 1e-6);
    }

    /// `disp()` hands back the stored MIC displacement \(r_j - r_i\)
    /// (unnormalized), or `None` when the column was never stored.
    #[test]
    fn disp_returns_stored_displacements() {
        let bx = SimBox::cube(4.0, array![0.0, 0.0, 0.0], [true, true, true]).expect("invalid box");
        let pts = array![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]];

        let nq = NeighborQuery::new(&bx, pts.view(), 1.5);
        let nlist = nq.query_self();

        let disp = nlist.disp().expect("self-query materializes disp");
        assert_eq!(disp.nrows(), 1);
        assert_eq!(disp.ncols(), 3);
        assert!((disp[[0, 0]] - 1.0).abs() < 1e-6);
        assert!(disp[[0, 1]].abs() < 1e-6);
        assert!(disp[[0, 2]].abs() < 1e-6);
    }

    #[test]
    fn self_query_pbc_boundary() {
        let bx = SimBox::cube(2.0, array![0.0, 0.0, 0.0], [true, true, true]).expect("invalid box");
        let pts = array![[0.1, 0.1, 0.1], [1.9, 1.9, 1.9]];

        let nq = NeighborQuery::new(&bx, pts.view(), 0.5);
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
        let nq = NeighborQuery::new(&bx, pts.view(), 0.6);
        let nlist = nq.query(pts.view());

        // Each query point finds both reference points (including itself at dist=0)
        // Total: 2 query x 2 ref = 4 pairs
        assert_eq!(nlist.n_pairs(), 4);

        // Compare with self-query which only produces unique pairs (i<j)
        let self_nlist = nq.query_self();
        assert_eq!(self_nlist.n_pairs(), 1);
    }

    #[test]
    fn free_boundary_self_query() {
        let pts = array![[0.0 as F, 0.0, 0.0], [0.5, 0.0, 0.0], [10.0, 10.0, 10.0],];
        let nq = NeighborQuery::free(pts.view(), 1.0);
        let nlist = nq.query_self();

        // Only pts[0] and pts[1] are within cutoff=1.0
        assert_eq!(nlist.n_pairs(), 1);
        let d2 = nlist.dist_sq().expect("self-query materializes dist_sq");
        assert!((d2[0].sqrt() - 0.5).abs() < 1e-5);
    }

    #[test]
    fn free_boundary_cross_query() {
        let ref_pts = array![[0.0 as F, 0.0, 0.0], [1.0, 0.0, 0.0], [5.0, 5.0, 5.0],];
        let query_pts = array![[0.3 as F, 0.0, 0.0]];

        let nq = NeighborQuery::free(ref_pts.view(), 0.5);
        let nlist = nq.query(query_pts.view());

        // query point at 0.3 is within 0.5 of ref[0] (dist=0.3) but not ref[1] (dist=0.7)
        assert_eq!(nlist.n_pairs(), 1);
    }

    #[test]
    fn free_boundary_no_wrap() {
        // Points far apart — should NOT be neighbors (no PBC wrapping)
        let pts = array![[0.0 as F, 0.0, 0.0], [5.0, 5.0, 5.0],];
        let nq = NeighborQuery::free(pts.view(), 1.0);
        let nlist = nq.query_self();
        assert_eq!(nlist.n_pairs(), 0);
    }

    // --- SoA cross-query is bit-identical to the Array2 cross-query ---

    /// Split an `Array2` (N×3) into three column vectors (SoA layout).
    fn columns(pts: &ndarray::Array2<F>) -> (Vec<F>, Vec<F>, Vec<F>) {
        let n = pts.nrows();
        let mut xs = Vec::with_capacity(n);
        let mut ys = Vec::with_capacity(n);
        let mut zs = Vec::with_capacity(n);
        for i in 0..n {
            xs.push(pts[[i, 0]]);
            ys.push(pts[[i, 1]]);
            zs.push(pts[[i, 2]]);
        }
        (xs, ys, zs)
    }

    /// Bitwise (not approximate) equality of two neighbor lists.
    fn assert_bitwise_equal(a: &Neighbors, b: &Neighbors) {
        assert_eq!(a.n_pairs(), b.n_pairs(), "n_pairs differ");
        let d2a = a.dist_sq().expect("query materializes dist_sq");
        let d2b = b.dist_sq().expect("query materializes dist_sq");
        let da = a.disp().expect("query materializes disp");
        let db = b.disp().expect("query materializes disp");
        for k in 0..a.n_pairs() {
            assert_eq!(
                a.query_point_indices()[k],
                b.query_point_indices()[k],
                "idx_i"
            );
            assert_eq!(a.point_indices()[k], b.point_indices()[k], "idx_j");
            assert_eq!(d2a[k], d2b[k], "dist_sq bitwise");
            for d in 0..3 {
                assert_eq!(da[[k, d]], db[[k, d]], "disp[{}] bitwise", d);
            }
        }
    }

    #[test]
    fn columns_query_matches_query_bitwise() {
        // Periodic cube fixture.
        let bx = SimBox::cube(10.0, array![0.0, 0.0, 0.0], [true, true, true]).unwrap();
        let refp = array![
            [1.0, 1.0, 1.0],
            [1.5, 1.0, 1.0],
            [9.5, 1.0, 1.0],
            [5.0, 5.0, 5.0],
            [5.3, 5.0, 5.0],
            [2.2, 8.1, 3.3],
        ];
        let qp = array![[1.2, 1.0, 1.0], [5.1, 5.0, 5.0], [9.9, 1.1, 1.0],];
        let (rx, ry, rz) = columns(&refp);
        let (qx, qy, qz) = columns(&qp);

        let nq_a = NeighborQuery::new(&bx, refp.view(), 2.0);
        let nl_a = nq_a.query(qp.view());

        let nq_s = NeighborQuery::from_columns(&bx, &rx, &ry, &rz, 2.0);
        let nl_s = nq_s.query_columns(&qx, &qy, &qz);

        assert!(nl_a.n_pairs() > 0, "fixture should produce pairs");
        assert_bitwise_equal(&nl_a, &nl_s);

        // Free / non-periodic fixture (query points overlap the reference set).
        let nq_af = NeighborQuery::free(refp.view(), 2.0);
        let nl_af = nq_af.query(qp.view());

        let nq_sf = NeighborQuery::free_columns(&rx, &ry, &rz, 2.0);
        let nl_sf = nq_sf.query_columns(&qx, &qy, &qz);

        assert!(nl_af.n_pairs() > 0, "fixture should produce pairs");
        assert_bitwise_equal(&nl_af, &nl_sf);
    }
}
