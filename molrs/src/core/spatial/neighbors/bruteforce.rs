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

use crate::spatial::neighbors::{Backend, PairVisitor};
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
/// It stores coordinates, not pairs: every traversal rescans them, so there is
/// no cached table to go stale and no column policy to configure. Which columns
/// a materialized table keeps is named by the caller at
/// [`NeighborList::neighbors`](crate::spatial::neighbors::NeighborList::neighbors),
/// exactly as it is for the cell list.
#[derive(Debug, Clone)]
pub struct BruteForce {
    /// Interaction cutoff distance (Å). Pairs are kept when their minimum-image
    /// separation is less than or equal to this.
    pub cutoff: F,
    /// Simulation box from the last index call.
    bx: Option<SimBox>,
    /// Stored positions for visit_pairs (set by update_index).
    stored_pos: FNx3,
}

impl BruteForce {
    /// Create a new `BruteForce` with the given cutoff distance (Å).
    ///
    /// The cutoff is not validated here; a non-positive value is rejected by
    /// the first index call (`build_index` / `update_index`), which panics.
    pub fn new(cutoff: F) -> Self {
        Self {
            cutoff,
            bx: None,
            stored_pos: FNx3::zeros((0, 3)),
        }
    }
}

impl Backend for BruteForce {
    /// Store positions and box without computing pairs.
    ///
    /// # Panics
    /// Panics if the cutoff is not positive.
    fn cutoff(&self) -> F {
        self.cutoff
    }

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

    /// On-demand O(N^2) pair traversal using the stored positions.
    ///
    /// The visitor receives the same two quantities a table would store:
    /// `dist_sq` (Å²) and the minimum-image displacement `r_j - r_i` (Å), for
    /// every pair with `i < j` inside the cutoff.
    ///
    /// There is exactly one source of pairs — a fresh rescan of the
    /// coordinates the last `build_index` / `update_index` stored — so nothing
    /// here can hand back a stale answer. Without such a call there is no box
    /// to fold minimum images against, and the visitor is simply never called,
    /// which is the [`Backend`] contract for an unindexed backend.
    fn visit_pairs(&self, visitor: &mut dyn PairVisitor) {
        let Some(bx) = &self.bx else {
            return;
        };
        let n = self.stored_pos.nrows();
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
