//! O(N^2) brute-force neighbor search — reference implementation.
//!
//! Checks every pair `(i, j)` with `i < j` — the half-shell convention, so each
//! unordered pair is examined exactly once. Useful for correctness testing
//! against the cell-list algorithm and for very small systems where the
//! overhead of cell construction is not worthwhile.
//!
//! Because it consults the simulation box directly for every pair and shares no
//! cell-assignment code with [`LinkCell`](crate::spatial::neighbors::LinkCell),
//! a defect in the cell partition cannot cancel against a defect here; that is
//! what makes it usable as an oracle in tests.

use crate::spatial::neighbors::{Backend, Neighbors, NeighborsStorage, PairVisitor, QueryMode};
use crate::spatial::simbox::SimBox;
use crate::types::{F, FNx3, FNx3View};

/// Brute-force O(N^2) neighbor search — the reference
/// [`NeighborList`](crate::spatial::neighbors::NeighborList) backend.
///
/// Iterates over all unique pairs and keeps those within the cutoff. Search
/// from outside this crate goes through
/// [`NeighborList::brute_force`](crate::spatial::neighbors::NeighborList::brute_force),
/// which drives this type.
///
/// The cached table always uses [`NeighborsStorage::FULL`] — both the squared
/// distance (Å²) and the minimum-image displacement (Å) are stored for every
/// pair. Unlike [`LinkCell`](crate::spatial::neighbors::LinkCell) there is no
/// column policy to configure, which is intentional for a reference
/// implementation: an oracle that could omit a column would be less useful for
/// comparison. Drop columns afterwards with
/// [`Neighbors::repack`](crate::spatial::neighbors::Neighbors::repack) if a
/// leaner table is wanted.
#[derive(Debug, Clone)]
pub struct BruteForce {
    /// Interaction cutoff distance (Å). Pairs are kept when their minimum-image
    /// separation is less than or equal to this.
    pub cutoff: F,
    /// Simulation box from the last build/update.
    bx: Option<SimBox>,
    /// Cached pair results.
    result: Neighbors,
    /// Stored positions for visit_pairs (set by update_index).
    stored_pos: FNx3,
}

impl BruteForce {
    /// Create a new `BruteForce` with the given cutoff distance (Å).
    ///
    /// The cutoff is not validated here; a non-positive value is rejected by
    /// the first build, which panics.
    pub fn new(cutoff: F) -> Self {
        Self {
            cutoff,
            bx: None,
            result: Neighbors::empty(
                QueryMode::SelfQuery { num_points: 0 },
                NeighborsStorage::FULL,
            ),
            stored_pos: FNx3::zeros((0, 3)),
        }
    }

    /// Scan all pairs and store those within cutoff.
    ///
    /// `dr` is `MIC(r_j - r_i)` and `d2` its squared length, both from the same
    /// call, so the two stored columns always describe the same periodic image.
    fn compute_pairs(&mut self, points: FNx3View<'_>, bx: &SimBox) {
        let n = points.nrows();
        let cutoff2 = self.cutoff * self.cutoff;

        self.result.clear();

        for i in 0..n {
            let pi = [points[[i, 0]], points[[i, 1]], points[[i, 2]]];
            for j in (i + 1)..n {
                let pj = [points[[j, 0]], points[[j, 1]], points[[j, 2]]];
                let dr = bx.shortest_vector_impl(pi, pj);
                let d2 = dr[0] * dr[0] + dr[1] * dr[1] + dr[2] * dr[2];
                if d2 <= cutoff2 {
                    self.result.push(i as u32, j as u32, d2, dr);
                }
            }
        }

        self.result.mode = QueryMode::SelfQuery { num_points: n };
        self.bx = Some(bx.clone());
    }

    /// Scan every pair and materialize the half-shell table read back by
    /// [`query`](Self::query).
    ///
    /// The index-only path — store the coordinates once, then stream pairs
    /// without a table — is
    /// [`NeighborList`](crate::spatial::neighbors::NeighborList).
    ///
    /// # Panics
    /// Panics if the cutoff is not positive.
    pub fn build(&mut self, points: FNx3View<'_>, bx: &SimBox) {
        assert!(self.cutoff > 0.0, "cutoff must be positive");
        self.compute_pairs(points, bx);
    }

    /// Reference to the pair table materialized by the last
    /// [`build`](Self::build).
    ///
    /// Before the first one this is an empty table tagged
    /// `SelfQuery { num_points: 0 }`, not an error.
    pub fn query(&self) -> &Neighbors {
        &self.result
    }
}

impl Backend for BruteForce {
    /// Store positions and box without computing pairs.
    ///
    /// # Panics
    /// Panics if the cutoff is not positive.
    fn build_index(&mut self, points: FNx3View<'_>, bx: &SimBox) {
        assert!(self.cutoff > 0.0, "cutoff must be positive");
        self.update_index(points, bx);
    }

    /// Store positions and box without computing pairs.
    ///
    /// There is no spatial index to build for an all-pairs search, so this
    /// simply keeps a copy of the coordinates for `visit_pairs` to rescan.
    ///
    /// # Panics
    /// Panics if the cutoff is not positive.
    fn update_index(&mut self, points: FNx3View<'_>, bx: &SimBox) {
        assert!(self.cutoff > 0.0, "cutoff must be positive");
        self.stored_pos = points.to_owned();
        self.bx = Some(bx.clone());
    }

    /// On-demand O(N^2) pair traversal using stored positions.
    ///
    /// The visitor receives the same two quantities the table would store:
    /// `dist_sq` (Å²) and the minimum-image displacement `r_j - r_i` (Å), for
    /// every pair with `i < j` inside the cutoff.
    ///
    /// Three states are possible, and the choice between them is silent:
    ///
    /// - Coordinates were stored by `build_index` or `update_index` — they are
    ///   rescanned here, allocating nothing.
    /// - No coordinates were stored but a cached table exists from a previous
    ///   `build` — its rows are replayed instead.
    /// - Neither is available (nothing has been built at all) — the visitor is
    ///   never called and no error is raised.
    fn visit_pairs(&self, visitor: &mut dyn PairVisitor) {
        let bx = match &self.bx {
            Some(b) => b,
            None => return,
        };
        let n = self.stored_pos.nrows();
        if n == 0 {
            // Fall back to pre-stored result if no stored_pos
            if self.result.n_pairs() > 0 {
                let dist_sq = self
                    .result
                    .dist_sq()
                    .expect("BruteForce materializes dist_sq");
                let disp = self.result.disp().expect("BruteForce materializes disp");
                for k in 0..self.result.n_pairs() {
                    visitor.visit_pair(
                        self.result.query_point_indices()[k],
                        self.result.point_indices()[k],
                        dist_sq[k],
                        [disp[[k, 0]], disp[[k, 1]], disp[[k, 2]]],
                    );
                }
            }
            return;
        }

        let cutoff2 = self.cutoff * self.cutoff;
        for i in 0..n {
            let pi = [
                self.stored_pos[[i, 0]],
                self.stored_pos[[i, 1]],
                self.stored_pos[[i, 2]],
            ];
            for j in (i + 1)..n {
                let pj = [
                    self.stored_pos[[j, 0]],
                    self.stored_pos[[j, 1]],
                    self.stored_pos[[j, 2]],
                ];
                let dr = bx.shortest_vector_impl(pi, pj);
                let d2 = dr[0] * dr[0] + dr[1] * dr[1] + dr[2] * dr[2];
                if d2 <= cutoff2 {
                    visitor.visit_pair(i as u32, j as u32, d2, dr);
                }
            }
        }
    }
}
