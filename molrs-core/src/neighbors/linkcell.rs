//! Cell-list neighbor search — O(N) with sorted-particle layout and half-shell
//! iteration.
//!
//! Particles are counting-sorted by cell index so that all particles in the
//! same cell occupy a contiguous slice of `sorted_idx` / `sorted_pos`.
//! This gives excellent cache locality during the pair search compared to a
//! linked-list layout.

use crate::neighbors::{NbListAlgo, NeighborList, PairVisitor, QueryMode};
use crate::region::simbox::SimBox;
use crate::types::{F, FNx3, FNx3View};
use ndarray::array;

/// Cell-list neighbor search algorithm.
///
/// Partitions space into a regular grid of cells whose width >= cutoff, then
/// searches only neighboring cells for pair interactions.  Uses half-shell
/// iteration so each pair is found exactly once.
///
/// With the `rayon` feature (default), the pair search is parallelized across
/// cells via `rayon::par_iter`.
///
/// # Usage
///
/// ```ignore
/// let lc = LinkCell::new().cutoff(3.0);
/// ```
#[derive(Debug, Clone)]
pub struct LinkCell {
    /// Interaction cutoff distance.
    pub cutoff: F,
    /// Number of cells along each axis `[nx, ny, nz]`.
    celldim: [u32; 3],
    /// `cell_start[c]` = index into `sorted_idx` where cell `c` begins.
    /// Length = n_cells + 1 (sentinel at end).
    cell_start: Vec<u32>,
    /// Original particle indices, sorted by cell assignment.
    sorted_idx: Vec<u32>,
    /// Particle positions in sorted order (N x 3) for cache-friendly access.
    sorted_pos: FNx3,
    /// CSR-style forward neighbor cell indices (flat).
    fwd_neighbors: Vec<u32>,
    /// CSR-style offsets into `fwd_neighbors`, length = n_cells + 1.
    fwd_neighbor_offsets: Vec<u32>,
    /// CSR-style ALL neighbor cell indices (flat) — for cross-query.
    all_neighbors: Vec<u32>,
    /// CSR-style offsets into `all_neighbors`, length = n_cells + 1.
    all_neighbor_offsets: Vec<u32>,
    /// Simulation box from the last build/update.
    bx: SimBox,
    /// Cached pair results.
    result: NeighborList,
    /// Reusable cursor buffer for counting-sort scatter (avoids allocation).
    cursor: Vec<u32>,
    /// Reusable cell-assignment buffer (avoids cloning sorted_idx).
    cell_of: Vec<u32>,
}

impl Default for LinkCell {
    fn default() -> Self {
        Self::new()
    }
}

impl LinkCell {
    /// Create a new `LinkCell` with zero cutoff (must be set via [`cutoff`](Self::cutoff)).
    pub fn new() -> Self {
        Self {
            cutoff: 0.0,
            celldim: [0; 3],
            cell_start: Vec::new(),
            sorted_idx: Vec::new(),
            sorted_pos: FNx3::zeros((0, 3)),
            fwd_neighbors: Vec::new(),
            fwd_neighbor_offsets: Vec::new(),
            all_neighbors: Vec::new(),
            all_neighbor_offsets: Vec::new(),
            bx: SimBox::cube(1.0, array![0.0 as F, 0.0, 0.0], [false, false, false])
                .expect("dummy box"),
            result: NeighborList::empty(),
            cursor: Vec::new(),
            cell_of: Vec::new(),
        }
    }

    /// Set the cutoff distance (builder pattern).
    pub fn cutoff(mut self, cutoff: F) -> Self {
        self.cutoff = cutoff;
        self
    }

    /// Visit all reference-point neighbors of an arbitrary query point.
    ///
    /// Used by [`AABBQuery::query`](super::AABBQuery::query) for cross-query.
    /// Calls `callback(ref_index, dist_sq, [dx, dy, dz])` for each reference
    /// point within range.
    pub(crate) fn visit_neighbors_of<C>(
        &self,
        query_point: ndarray::ArrayView1<'_, F>,
        bx: &SimBox,
        mut callback: C,
    ) where
        C: FnMut(u32, F, [F; 3]),
    {
        let n_cells = (self.celldim[0] * self.celldim[1] * self.celldim[2]) as usize;
        if n_cells == 0 {
            return;
        }

        // Find the cell containing the query point
        let query_cell = get_cell(bx, query_point, self.celldim);

        // Check the query cell itself
        let start = self.cell_start[query_cell] as usize;
        let end = self.cell_start[query_cell + 1] as usize;
        for si in start..end {
            let oj = self.sorted_idx[si];
            let pj = self.sorted_pos.row(si);
            let dr = bx.shortest_vector_fast(query_point, pj);
            let d2 = dr[0] * dr[0] + dr[1] * dr[1] + dr[2] * dr[2];
            callback(oj, d2, [dr[0], dr[1], dr[2]]);
        }

        // Check all neighbor cells (using the all-neighbors table)
        let nb_start = self.all_neighbor_offsets[query_cell] as usize;
        let nb_end = self.all_neighbor_offsets[query_cell + 1] as usize;
        for &nc in &self.all_neighbors[nb_start..nb_end] {
            let nc_start = self.cell_start[nc as usize] as usize;
            let nc_end = self.cell_start[nc as usize + 1] as usize;
            for sj in nc_start..nc_end {
                let oj = self.sorted_idx[sj];
                let pj = self.sorted_pos.row(sj);
                let dr = bx.shortest_vector_fast(query_point, pj);
                let d2 = dr[0] * dr[0] + dr[1] * dr[1] + dr[2] * dr[2];
                callback(oj, d2, [dr[0], dr[1], dr[2]]);
            }
        }
    }
}

impl NbListAlgo for LinkCell {
    fn build(&mut self, points: FNx3View<'_>, bx: &SimBox) {
        assert!(self.cutoff > 0.0, "cutoff must be positive");
        self.update(points, bx);
    }

    fn update(&mut self, points: FNx3View<'_>, bx: &SimBox) {
        assert!(self.cutoff > 0.0, "cutoff must be positive");
        assert!(points.ncols() == 3, "points must have shape (N, 3)");

        self.counting_sort(points, bx);
        let n_cells = (self.celldim[0] * self.celldim[1] * self.celldim[2]) as usize;
        let n_points = points.nrows();

        // Pair search — parallel by default, serial fallback without rayon.
        #[cfg(feature = "rayon")]
        self.compute_pairs_parallel(n_cells);
        #[cfg(not(feature = "rayon"))]
        self.compute_pairs_serial(n_cells);

        // Tag result as self-query
        self.result.mode = QueryMode::SelfQuery;
        self.result.num_points = n_points;
        self.result.num_query_points = n_points;
    }

    #[inline]
    fn query(&self) -> &NeighborList {
        &self.result
    }

    #[inline]
    fn box_ref(&self) -> &SimBox {
        &self.bx
    }

    fn build_index(&mut self, points: FNx3View<'_>, bx: &SimBox) {
        assert!(self.cutoff > 0.0, "cutoff must be positive");
        self.update_index(points, bx);
    }

    /// Rebuild counting-sort index only — NO pair enumeration.
    ///
    /// After this, [`visit_pairs`] can traverse pairs on-demand.
    fn update_index(&mut self, points: FNx3View<'_>, bx: &SimBox) {
        assert!(self.cutoff > 0.0, "cutoff must be positive");
        assert!(points.ncols() == 3, "points must have shape (N, 3)");
        self.counting_sort(points, bx);
    }

    /// On-demand pair traversal — zero allocation.
    ///
    /// Same half-shell iteration as `compute_pairs_serial` but calls the
    /// visitor instead of building a [`NeighborList`].
    #[allow(clippy::needless_range_loop)]
    fn visit_pairs(&self, visitor: &mut dyn PairVisitor) {
        let n_cells = (self.celldim[0] * self.celldim[1] * self.celldim[2]) as usize;
        if n_cells == 0 {
            return;
        }
        let cutoff2 = self.cutoff * self.cutoff;

        for cell in 0..n_cells {
            let start = self.cell_start[cell] as usize;
            let end = self.cell_start[cell + 1] as usize;

            // Self-cell pairs
            for si in start..end {
                let pi = self.sorted_pos.row(si);
                let oi = self.sorted_idx[si];
                for sj in (si + 1)..end {
                    let pj = self.sorted_pos.row(sj);
                    let dr = self.bx.shortest_vector_fast(pi, pj);
                    let d2 = dr[0] * dr[0] + dr[1] * dr[1] + dr[2] * dr[2];
                    if d2 <= cutoff2 {
                        visitor.visit_pair(oi, self.sorted_idx[sj], d2, [dr[0], dr[1], dr[2]]);
                    }
                }
            }

            // Forward neighbor cells
            let fwd_start = self.fwd_neighbor_offsets[cell] as usize;
            let fwd_end = self.fwd_neighbor_offsets[cell + 1] as usize;

            for si in start..end {
                let pi = self.sorted_pos.row(si);
                let oi = self.sorted_idx[si];

                for &nc in &self.fwd_neighbors[fwd_start..fwd_end] {
                    let nc_start = self.cell_start[nc as usize] as usize;
                    let nc_end = self.cell_start[nc as usize + 1] as usize;

                    for sj in nc_start..nc_end {
                        let oj = self.sorted_idx[sj];
                        let pj = self.sorted_pos.row(sj);
                        let dr = self.bx.shortest_vector_fast(pi, pj);
                        let d2 = dr[0] * dr[0] + dr[1] * dr[1] + dr[2] * dr[2];
                        if d2 <= cutoff2 {
                            if oi < oj {
                                visitor.visit_pair(oi, oj, d2, [dr[0], dr[1], dr[2]]);
                            } else {
                                visitor.visit_pair(oj, oi, d2, [-dr[0], -dr[1], -dr[2]]);
                            }
                        }
                    }
                }
            }
        }
    }
}

impl LinkCell {
    /// Counting sort particles by cell index — shared by `update` and `update_index`.
    ///
    /// Sets up `celldim`, `bx`, `cell_start`, `sorted_idx`, `sorted_pos`,
    /// and the forward-neighbor table. Does NOT compute pairs.
    fn counting_sort(&mut self, points: FNx3View<'_>, bx: &SimBox) {
        let cutoff = self.cutoff;
        let npd = bx.nearest_plane_distance();
        let celldim = [
            ((npd[0] / cutoff).floor() as u32).max(1),
            ((npd[1] / cutoff).floor() as u32).max(1),
            ((npd[2] / cutoff).floor() as u32).max(1),
        ];
        let n_cells = (celldim[0] * celldim[1] * celldim[2]) as usize;
        let n_points = points.nrows();

        // Rebuild neighbor tables only when grid dimensions change.
        if celldim != self.celldim {
            let (fwd_neighbors, fwd_neighbor_offsets) = build_fwd_neighbors(celldim);
            let (all_neighbors, all_neighbor_offsets) = build_all_neighbors(celldim);
            self.fwd_neighbors = fwd_neighbors;
            self.fwd_neighbor_offsets = fwd_neighbor_offsets;
            self.all_neighbors = all_neighbors;
            self.all_neighbor_offsets = all_neighbor_offsets;
        }

        self.celldim = celldim;
        self.bx = bx.clone();

        // --- Counting sort particles by cell index ---

        // 1) Count particles per cell.
        self.cell_start.clear();
        self.cell_start.resize(n_cells + 1, 0);

        // Temporary cell assignment per particle (reuse sorted_idx buffer).
        self.sorted_idx.resize(n_points, 0);
        for i in 0..n_points {
            let cell = get_cell(bx, points.row(i), celldim);
            self.sorted_idx[i] = cell as u32;
            self.cell_start[cell] += 1;
        }

        // 2) Prefix sum -> cell_start[c] = offset where cell c begins.
        let mut acc = 0u32;
        for c in 0..n_cells {
            let count = self.cell_start[c];
            self.cell_start[c] = acc;
            acc += count;
        }
        self.cell_start[n_cells] = acc;
        debug_assert_eq!(acc as usize, n_points);

        // 3) Scatter particles into sorted order.
        // Reuse cursor/cell_of buffers to avoid allocation.
        self.cursor.resize(n_cells, 0);
        self.cursor.copy_from_slice(&self.cell_start[..n_cells]);

        self.cell_of.resize(n_points, 0);
        self.cell_of.copy_from_slice(&self.sorted_idx);

        if self.sorted_pos.nrows() != n_points {
            self.sorted_pos = FNx3::zeros((n_points, 3));
        }
        for i in 0..n_points {
            let cell = self.cell_of[i] as usize;
            let dst = self.cursor[cell] as usize;
            self.cursor[cell] += 1;
            self.sorted_idx[dst] = i as u32;
            self.sorted_pos[[dst, 0]] = points[[i, 0]];
            self.sorted_pos[[dst, 1]] = points[[i, 1]];
            self.sorted_pos[[dst, 2]] = points[[i, 2]];
        }
    }

    /// Serial half-shell pair search over all cells.
    #[cfg(not(feature = "rayon"))]
    fn compute_pairs_serial(&mut self, n_cells: usize) {
        let cutoff2 = self.cutoff * self.cutoff;
        self.result.clear();

        for cell in 0..n_cells {
            let start = self.cell_start[cell] as usize;
            let end = self.cell_start[cell + 1] as usize;

            // Self-cell pairs: iterate over all unique pairs within this cell.
            for si in start..end {
                let pi = self.sorted_pos.row(si);
                let oi = self.sorted_idx[si];
                for sj in (si + 1)..end {
                    let pj = self.sorted_pos.row(sj);
                    let dr = self.bx.shortest_vector_fast(pi, pj);
                    let d2 = dr[0] * dr[0] + dr[1] * dr[1] + dr[2] * dr[2];
                    if d2 <= cutoff2 {
                        self.result
                            .push(oi, self.sorted_idx[sj], d2, [dr[0], dr[1], dr[2]]);
                    }
                }
            }

            // Forward neighbor cells.
            let fwd_start = self.fwd_neighbor_offsets[cell] as usize;
            let fwd_end = self.fwd_neighbor_offsets[cell + 1] as usize;

            for si in start..end {
                let pi = self.sorted_pos.row(si);
                let oi = self.sorted_idx[si];

                for &nc in &self.fwd_neighbors[fwd_start..fwd_end] {
                    let nc_start = self.cell_start[nc as usize] as usize;
                    let nc_end = self.cell_start[nc as usize + 1] as usize;

                    for sj in nc_start..nc_end {
                        let oj = self.sorted_idx[sj];
                        let pj = self.sorted_pos.row(sj);
                        let dr = self.bx.shortest_vector_fast(pi, pj);
                        let d2 = dr[0] * dr[0] + dr[1] * dr[1] + dr[2] * dr[2];
                        if d2 <= cutoff2 {
                            // Canonicalize: smaller original index first.
                            if oi < oj {
                                self.result.push(oi, oj, d2, [dr[0], dr[1], dr[2]]);
                            } else {
                                self.result.push(oj, oi, d2, [-dr[0], -dr[1], -dr[2]]);
                            }
                        }
                    }
                }
            }
        }
    }

    /// Parallel half-shell pair search: each rayon thread processes a chunk of
    /// cells into thread-local buffers, then results are merged.
    #[cfg(feature = "rayon")]
    #[allow(clippy::needless_range_loop)]
    fn compute_pairs_parallel(&mut self, n_cells: usize) {
        use rayon::prelude::*;

        let cutoff2 = self.cutoff * self.cutoff;

        let cell_start = &self.cell_start;
        let sorted_idx = &self.sorted_idx;
        let sorted_pos = &self.sorted_pos;
        let bx = &self.bx;
        let fwd_neighbors = &self.fwd_neighbors;
        let fwd_neighbor_offsets = &self.fwd_neighbor_offsets;

        let merged = (0..n_cells)
            .into_par_iter()
            .fold(NeighborList::empty, |mut acc, cell| {
                let start = cell_start[cell] as usize;
                let end = cell_start[cell + 1] as usize;

                // Self-cell pairs.
                for si in start..end {
                    let pi = sorted_pos.row(si);
                    let oi = sorted_idx[si];
                    for sj in (si + 1)..end {
                        let pj = sorted_pos.row(sj);
                        let dr = bx.shortest_vector_fast(pi, pj);
                        let d2 = dr[0] * dr[0] + dr[1] * dr[1] + dr[2] * dr[2];
                        if d2 <= cutoff2 {
                            acc.push(oi, sorted_idx[sj], d2, [dr[0], dr[1], dr[2]]);
                        }
                    }
                }

                // Forward neighbor cells.
                let fwd_start = fwd_neighbor_offsets[cell] as usize;
                let fwd_end = fwd_neighbor_offsets[cell + 1] as usize;

                for si in start..end {
                    let pi = sorted_pos.row(si);
                    let oi = sorted_idx[si];

                    for &nc in &fwd_neighbors[fwd_start..fwd_end] {
                        let nc_start = cell_start[nc as usize] as usize;
                        let nc_end = cell_start[nc as usize + 1] as usize;

                        for sj in nc_start..nc_end {
                            let oj = sorted_idx[sj];
                            let pj = sorted_pos.row(sj);
                            let dr = bx.shortest_vector_fast(pi, pj);
                            let d2 = dr[0] * dr[0] + dr[1] * dr[1] + dr[2] * dr[2];
                            if d2 <= cutoff2 {
                                if oi < oj {
                                    acc.push(oi, oj, d2, [dr[0], dr[1], dr[2]]);
                                } else {
                                    acc.push(oj, oi, d2, [-dr[0], -dr[1], -dr[2]]);
                                }
                            }
                        }
                    }
                }

                acc
            })
            .reduce(NeighborList::empty, |mut a, b| {
                a.idx_i.extend_from_slice(&b.idx_i);
                a.idx_j.extend_from_slice(&b.idx_j);
                a.dist_sq.extend_from_slice(&b.dist_sq);
                a.diff_flat.extend_from_slice(&b.diff_flat);
                a
            });

        self.result = merged;
    }
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Map a position to its cell index using fractional coordinates.
#[inline(always)]
fn get_cell(bx: &SimBox, r: ndarray::ArrayView1<'_, F>, celldim: [u32; 3]) -> usize {
    let frac = bx.make_fractional_fast(r);
    let cx = (frac[0] * celldim[0] as F).floor() as u32 % celldim[0];
    let cy = (frac[1] * celldim[1] as F).floor() as u32 % celldim[1];
    let cz = (frac[2] * celldim[2] as F).floor() as u32 % celldim[2];
    (cz * celldim[1] * celldim[0] + cy * celldim[0] + cx) as usize
}

// ---------------------------------------------------------------------------
// Forward-neighbor table construction (half-shell, for self-query)
// ---------------------------------------------------------------------------

/// Build the CSR-encoded forward-neighbor table for the given grid dimensions.
fn build_fwd_neighbors(celldim: [u32; 3]) -> (Vec<u32>, Vec<u32>) {
    let n_cells = (celldim[0] * celldim[1] * celldim[2]) as usize;
    let mut offsets = Vec::with_capacity(n_cells + 1);
    let mut neighbors_all = Vec::with_capacity(n_cells * 13);
    offsets.push(0);

    for cell in 0..n_cells {
        let idx = cell as u32;
        let cx = (idx % (celldim[1] * celldim[0])) % celldim[0];
        let cy = (idx % (celldim[1] * celldim[0])) / celldim[0];
        let cz = idx / (celldim[1] * celldim[0]);

        let (si, ei) = stencil_range(cx, celldim[0]);
        let (sj, ej) = stencil_range(cy, celldim[1]);
        let (sk, ek) = stencil_range(cz, celldim[2]);

        let mark = neighbors_all.len();
        for nk in sk..=ek {
            for nj in sj..=ej {
                for ni in si..=ei {
                    let wi = wrap(ni, celldim[0]);
                    let wj = wrap(nj, celldim[1]);
                    let wk = wrap(nk, celldim[2]);
                    let nc = (wk * celldim[1] * celldim[0] + wj * celldim[0] + wi) as usize;
                    if nc > cell {
                        neighbors_all.push(nc as u32);
                    }
                }
            }
        }
        neighbors_all[mark..].sort_unstable();
        neighbors_all.dedup();
        offsets.push(neighbors_all.len() as u32);
    }

    (neighbors_all, offsets)
}

// ---------------------------------------------------------------------------
// All-neighbor table construction (full-shell, for cross-query)
// ---------------------------------------------------------------------------

/// Build the CSR-encoded all-neighbor table (all 26 surrounding cells, excluding self).
fn build_all_neighbors(celldim: [u32; 3]) -> (Vec<u32>, Vec<u32>) {
    let n_cells = (celldim[0] * celldim[1] * celldim[2]) as usize;
    let mut offsets = Vec::with_capacity(n_cells + 1);
    let mut neighbors_all = Vec::with_capacity(n_cells * 26);
    offsets.push(0);

    for cell in 0..n_cells {
        let idx = cell as u32;
        let cx = (idx % (celldim[1] * celldim[0])) % celldim[0];
        let cy = (idx % (celldim[1] * celldim[0])) / celldim[0];
        let cz = idx / (celldim[1] * celldim[0]);

        let (si, ei) = stencil_range(cx, celldim[0]);
        let (sj, ej) = stencil_range(cy, celldim[1]);
        let (sk, ek) = stencil_range(cz, celldim[2]);

        let mark = neighbors_all.len();
        for nk in sk..=ek {
            for nj in sj..=ej {
                for ni in si..=ei {
                    let wi = wrap(ni, celldim[0]);
                    let wj = wrap(nj, celldim[1]);
                    let wk = wrap(nk, celldim[2]);
                    let nc = (wk * celldim[1] * celldim[0] + wj * celldim[0] + wi) as usize;
                    if nc != cell {
                        neighbors_all.push(nc as u32);
                    }
                }
            }
        }
        neighbors_all[mark..].sort_unstable();
        neighbors_all.dedup();
        offsets.push(neighbors_all.len() as u32);
    }

    (neighbors_all, offsets)
}

/// Compute the stencil search range `[start, end]` for one axis.
#[inline(always)]
fn stencil_range(c: u32, dim: u32) -> (i32, i32) {
    if dim < 2 {
        (c as i32, c as i32)
    } else if dim < 3 {
        (c as i32, c as i32 + 1)
    } else {
        (c as i32 - 1, c as i32 + 1)
    }
}

/// Wrap a signed cell index into `[0, dim)` with periodic boundary conditions.
#[inline(always)]
fn wrap(idx: i32, dim: u32) -> u32 {
    let d = dim as i32;
    let mut v = idx % d;
    if v < 0 {
        v += d;
    }
    v as u32
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::neighbors::NbList;
    use crate::region::simbox::SimBox;
    use ndarray::array;

    #[test]
    fn linked_cell_basic_pairs() {
        let bx = SimBox::cube(4.0, array![0.0, 0.0, 0.0], [true, true, true])
            .expect("invalid box length");
        let pts = array![[0.1, 0.2, 0.3], [0.3, 0.2, 0.1], [3.9, 3.8, 3.7]];
        let mut nl = NbList(LinkCell::new().cutoff(0.5));
        nl.build(pts.view(), &bx);
        let res = nl.query();
        assert_eq!(res.n_pairs(), 1);
        assert_eq!(res.query_point_indices()[0], 0);
        assert_eq!(res.point_indices()[0], 1);
    }

    #[test]
    fn linked_cell_pbc_boundary() {
        let bx = SimBox::cube(2.0, array![0.0, 0.0, 0.0], [true, true, true])
            .expect("invalid box length");
        let pts = array![[0.1, 0.1, 0.1], [1.9, 1.9, 1.9]];
        let mut nl = NbList(LinkCell::new().cutoff(0.5));
        nl.build(pts.view(), &bx);
        let res = nl.query();
        assert_eq!(res.n_pairs(), 1);
    }

    #[test]
    fn linked_cell_no_duplicates() {
        let bx = SimBox::cube(3.0, array![0.0, 0.0, 0.0], [true, true, true])
            .expect("invalid box length");
        let pts = array![[0.1, 0.1, 0.1], [0.2, 0.2, 0.2], [0.3, 0.3, 0.3]];
        let mut nl = NbList(LinkCell::new().cutoff(1.0));
        nl.build(pts.view(), &bx);
        let res = nl.query();
        let mut seen = std::collections::HashSet::new();
        for k in 0..res.n_pairs() {
            let i = res.query_point_indices()[k];
            let j = res.point_indices()[k];
            assert!(i < j);
            assert!(seen.insert((i, j)));
        }
    }

    #[test]
    fn linked_cell_cutoff_edge_included() {
        let bx = SimBox::cube(3.0, array![0.0, 0.0, 0.0], [true, true, true])
            .expect("invalid box length");
        let pts = array![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]];
        let mut nl = NbList(LinkCell::new().cutoff(1.0));
        nl.build(pts.view(), &bx);
        let res = nl.query();
        assert_eq!(res.n_pairs(), 1);
    }

    #[test]
    fn linked_cell_deterministic_order() {
        let bx = SimBox::cube(4.0, array![0.0, 0.0, 0.0], [true, true, true])
            .expect("invalid box length");
        let pts = array![[0.1, 0.2, 0.3], [0.4, 0.2, 0.3], [1.1, 1.2, 1.3]];
        let mut nl = NbList(LinkCell::new().cutoff(0.5));
        nl.build(pts.view(), &bx);
        let res1_i = nl.query().query_point_indices().to_vec();
        let res1_j = nl.query().point_indices().to_vec();
        let res2_i = nl.query().query_point_indices().to_vec();
        let res2_j = nl.query().point_indices().to_vec();
        assert_eq!(res1_i, res2_i);
        assert_eq!(res1_j, res2_j);
    }

    /// Verify that `visit_pairs` on a LinkCell built with `update_index`
    /// produces the same pair set as the full `build` + `query` path.
    #[test]
    fn visit_pairs_matches_query() {
        let bx = SimBox::cube(3.0, array![0.0, 0.0, 0.0], [true, true, true])
            .expect("invalid box length");
        let pts = array![
            [0.1, 0.2, 0.3],
            [0.4, 0.2, 0.3],
            [1.1, 1.2, 1.3],
            [2.9, 2.8, 2.7]
        ];

        // Full build (pairs pre-stored)
        let mut lc_full = LinkCell::new().cutoff(0.6);
        lc_full.build(pts.view(), &bx);
        let res = lc_full.query();
        let diff = res.vectors();
        let mut full_pairs: Vec<(u32, u32, F, [F; 3])> = (0..res.n_pairs())
            .map(|k| {
                (
                    res.query_point_indices()[k],
                    res.point_indices()[k],
                    res.dist_sq()[k],
                    [diff[[k, 0]], diff[[k, 1]], diff[[k, 2]]],
                )
            })
            .collect();
        full_pairs.sort_by_key(|(i, j, _, _)| (*i, *j));

        // Index-only build + visit_pairs
        let mut lc_index = LinkCell::new().cutoff(0.6);
        lc_index.build_index(pts.view(), &bx);
        let mut visit_pairs: Vec<(u32, u32, F, [F; 3])> = Vec::new();
        lc_index.visit_pairs(&mut |i, j, d2, diff| {
            visit_pairs.push((i, j, d2, diff));
        });
        visit_pairs.sort_by_key(|(i, j, _, _)| (*i, *j));

        assert_eq!(
            full_pairs.len(),
            visit_pairs.len(),
            "pair count mismatch: full={}, visit={}",
            full_pairs.len(),
            visit_pairs.len()
        );
        for (a, b) in full_pairs.iter().zip(visit_pairs.iter()) {
            assert_eq!(a.0, b.0);
            assert_eq!(a.1, b.1);
            assert!(
                (a.2 - b.2).abs() < 1e-6,
                "d2 mismatch for ({},{}): {} vs {}",
                a.0,
                a.1,
                a.2,
                b.2
            );
            for d in 0..3 {
                assert!(
                    (a.3[d] - b.3[d]).abs() < 1e-6,
                    "diff[{}] mismatch for ({},{})",
                    d,
                    a.0,
                    a.1
                );
            }
        }
    }

    #[test]
    fn neighborlist_algorithm_switch_consistency() {
        let bx = SimBox::cube(3.0, array![0.0, 0.0, 0.0], [true, true, true])
            .expect("invalid box length");
        let pts = array![
            [0.1, 0.2, 0.3],
            [0.4, 0.2, 0.3],
            [1.1, 1.2, 1.3],
            [2.9, 2.8, 2.7]
        ];

        let mut lc = NbList(LinkCell::new().cutoff(0.6));
        lc.build(pts.view(), &bx);
        let res_lc = lc.query();

        let mut bf = NbList(crate::neighbors::bruteforce::BruteForce::new(0.6));
        bf.build(pts.view(), &bx);
        let res_bf = bf.query();

        let diff_lc = res_lc.vectors();
        let diff_bf = res_bf.vectors();

        let mut lc_pairs: Vec<(u32, u32, F, [F; 3])> = (0..res_lc.n_pairs())
            .map(|k| {
                (
                    res_lc.query_point_indices()[k],
                    res_lc.point_indices()[k],
                    res_lc.dist_sq()[k],
                    [diff_lc[[k, 0]], diff_lc[[k, 1]], diff_lc[[k, 2]]],
                )
            })
            .collect();
        lc_pairs.sort_by_key(|(i, j, _, _)| (*i, *j));

        let mut bf_pairs: Vec<(u32, u32, F, [F; 3])> = (0..res_bf.n_pairs())
            .map(|k| {
                (
                    res_bf.query_point_indices()[k],
                    res_bf.point_indices()[k],
                    res_bf.dist_sq()[k],
                    [diff_bf[[k, 0]], diff_bf[[k, 1]], diff_bf[[k, 2]]],
                )
            })
            .collect();
        bf_pairs.sort_by_key(|(i, j, _, _)| (*i, *j));

        assert_eq!(lc_pairs.len(), bf_pairs.len());
        for (a, b) in lc_pairs.iter().zip(bf_pairs.iter()) {
            assert_eq!(a.0, b.0);
            assert_eq!(a.1, b.1);
            assert!((a.2 - b.2).abs() < 1e-6);
            for d in 0..3 {
                assert!((a.3[d] - b.3[d]).abs() < 1e-6);
            }
        }
    }
}
