//! Cell-list neighbor search — O(N) with sorted-particle layout and half-shell
//! iteration.
//!
//! Particles are counting-sorted by cell index so that all particles in the
//! same cell occupy a contiguous slice of `sorted_idx` / `sorted_pos`.
//! This gives excellent cache locality during the pair search compared to a
//! linked-list layout.
//!
//! Only **occupied cells** are visited during pair search, so sparse systems
//! (few particles, many cells) pay O(N), not O(n_cells).

use crate::spatial::neighbors::{CellGrid, NbListAlgo, NeighborList, PairVisitor, QueryMode};
use crate::spatial::region::simbox::SimBox;
use crate::types::{F, FNx3View};
use ndarray::array;

/// Cell-list neighbor search algorithm.
///
/// Partitions space into a regular grid of cells whose width >= cutoff, then
/// searches only neighboring cells for pair interactions.  Uses half-shell
/// iteration so each pair is found exactly once.
///
/// With the `rayon` feature (default), the pair search is parallelized across
/// occupied cells via `rayon::par_iter`.
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
    /// Cell partition of the box: dimensions + per-axis periodicity.
    grid: CellGrid,
    /// `cell_start[c]` = index into `sorted_idx` where cell `c` begins.
    /// Length = n_cells + 1 (sentinel at end).
    cell_start: Vec<u32>,
    /// Original particle indices, sorted by cell assignment.
    sorted_idx: Vec<u32>,
    /// Particle positions in sorted order, flat `[x0,y0,z0, x1,y1,z1, ...]`.
    /// Stored as raw `Vec<F>` (not `Array2`) to eliminate per-row view overhead
    /// in the tight pair loop.
    sorted_pos: Vec<F>,
    /// Indices of non-empty cells — pair search only iterates these.
    occupied_cells: Vec<u32>,
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
            grid: CellGrid::with_dims([1; 3], [false; 3]),
            cell_start: Vec::new(),
            sorted_idx: Vec::new(),
            sorted_pos: Vec::new(),
            occupied_cells: Vec::new(),
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
    /// Used by [`NeighborQuery::query`](super::NeighborQuery::query) for cross-query.
    /// Calls `callback(ref_index, dist_sq, [dx, dy, dz])` for each reference
    /// point within range.
    pub(crate) fn visit_neighbors_of<C>(
        &self,
        query_point: ndarray::ArrayView1<'_, F>,
        bx: &SimBox,
        callback: C,
    ) where
        C: FnMut(u32, F, [F; 3]),
    {
        self.visit_neighbors_of_pt(
            [query_point[0], query_point[1], query_point[2]],
            bx,
            callback,
        );
    }

    /// Zero-view variant of [`visit_neighbors_of`](Self::visit_neighbors_of):
    /// reads the query point as a stack `[F; 3]` instead of an `ArrayView1`.
    ///
    /// Bit-identical to `visit_neighbors_of` for the same coordinate — the cell
    /// assignment routes through
    /// [`make_fractional_fast_arr3`](SimBox::make_fractional_fast_arr3), the
    /// stack mirror of the `ArrayView1` fractional helper, and all downstream
    /// math is unchanged. Used by
    /// [`NeighborQuery::query_columns`](super::NeighborQuery::query_columns).
    pub(crate) fn visit_neighbors_of_pt<C>(&self, query_point: [F; 3], bx: &SimBox, mut callback: C)
    where
        C: FnMut(u32, F, [F; 3]),
    {
        if self.cell_start.is_empty() {
            return;
        }

        let query_cell = self.grid.cell_of(bx, query_point);
        let qp = query_point;

        // Check the query cell itself + all 26 neighbor cells
        let mut buf = [0usize; 27];
        let n_all = self.grid.stencil_all(query_cell, &mut buf);
        let all_cells = &buf[..n_all];
        for nc in std::iter::once(query_cell).chain(all_cells.iter().copied()) {
            let start = self.cell_start[nc] as usize;
            let end = self.cell_start[nc + 1] as usize;
            for si in start..end {
                let oj = self.sorted_idx[si];
                let pj = pos_at(&self.sorted_pos, si);
                let dr = bx.shortest_vector_impl(qp, pj);
                let d2 = dr[0] * dr[0] + dr[1] * dr[1] + dr[2] * dr[2];
                callback(oj, d2, dr);
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
        let n_points = points.nrows();

        #[cfg(feature = "rayon")]
        self.compute_pairs_parallel();
        #[cfg(not(feature = "rayon"))]
        self.compute_pairs_serial();

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

    fn update_index(&mut self, points: FNx3View<'_>, bx: &SimBox) {
        assert!(self.cutoff > 0.0, "cutoff must be positive");
        assert!(points.ncols() == 3, "points must have shape (N, 3)");
        self.counting_sort(points, bx);
    }

    /// On-demand pair traversal — zero allocation.
    ///
    /// Same half-shell iteration as `compute_pairs_serial` but calls the
    /// visitor instead of building a [`NeighborList`].
    fn visit_pairs(&self, visitor: &mut dyn PairVisitor) {
        if self.occupied_cells.is_empty() {
            return;
        }
        let cutoff2 = self.cutoff * self.cutoff;
        let mut fwd_buf = [0usize; 27];

        for &cell in &self.occupied_cells {
            let cell = cell as usize;
            let start = self.cell_start[cell] as usize;
            let end = self.cell_start[cell + 1] as usize;

            // Self-cell pairs
            for si in start..end {
                let pi = pos_at(&self.sorted_pos, si);
                let oi = self.sorted_idx[si];
                for sj in (si + 1)..end {
                    let pj = pos_at(&self.sorted_pos, sj);
                    let dr = self.bx.shortest_vector_impl(pi, pj);
                    let d2 = dr[0] * dr[0] + dr[1] * dr[1] + dr[2] * dr[2];
                    if d2 <= cutoff2 {
                        visitor.visit_pair(oi, self.sorted_idx[sj], d2, dr);
                    }
                }
            }

            // Forward neighbor cells (stack buffer, no alloc)
            let n_fwd = self.grid.stencil_forward(cell, &mut fwd_buf);
            let fwd = &fwd_buf[..n_fwd];
            for si in start..end {
                let pi = pos_at(&self.sorted_pos, si);
                let oi = self.sorted_idx[si];

                for &nc in fwd {
                    let nc_start = self.cell_start[nc] as usize;
                    let nc_end = self.cell_start[nc + 1] as usize;

                    for sj in nc_start..nc_end {
                        let oj = self.sorted_idx[sj];
                        let pj = pos_at(&self.sorted_pos, sj);
                        let dr = self.bx.shortest_vector_impl(pi, pj);
                        let d2 = dr[0] * dr[0] + dr[1] * dr[1] + dr[2] * dr[2];
                        if d2 <= cutoff2 {
                            if oi < oj {
                                visitor.visit_pair(oi, oj, d2, dr);
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
    /// SoA sibling of [`build`](NbListAlgo::build): build the self-query pair
    /// list from column-major `x`/`y`/`z` slices.
    ///
    /// Shares the same counting-sort + pair-search core as `build`, so the
    /// resulting [`NeighborList`] is byte-identical to `build` on the same
    /// coordinates. Lets callers holding SoA columns skip the interleave into
    /// an owned `Array2`.
    ///
    /// # Panics
    /// Panics if the cutoff is not positive or the three slices differ in length.
    pub fn build_soa(&mut self, xs: &[F], ys: &[F], zs: &[F], bx: &SimBox) {
        assert!(self.cutoff > 0.0, "cutoff must be positive");
        assert!(
            xs.len() == ys.len() && ys.len() == zs.len(),
            "x/y/z slices must have equal length"
        );

        self.counting_sort_soa(xs, ys, zs, bx);
        let n_points = xs.len();

        #[cfg(feature = "rayon")]
        self.compute_pairs_parallel();
        #[cfg(not(feature = "rayon"))]
        self.compute_pairs_serial();

        self.result.mode = QueryMode::SelfQuery;
        self.result.num_points = n_points;
        self.result.num_query_points = n_points;
    }

    /// SoA sibling of [`build_index`](NbListAlgo::build_index): build the
    /// spatial index only (no pair pre-computation) from column-major slices.
    ///
    /// # Panics
    /// Panics if the cutoff is not positive or the three slices differ in length.
    pub fn build_index_soa(&mut self, xs: &[F], ys: &[F], zs: &[F], bx: &SimBox) {
        assert!(self.cutoff > 0.0, "cutoff must be positive");
        assert!(
            xs.len() == ys.len() && ys.len() == zs.len(),
            "x/y/z slices must have equal length"
        );
        self.counting_sort_soa(xs, ys, zs, bx);
    }

    /// Counting sort particles by cell index (interleaved `Array2` input).
    ///
    /// Thin adapter over [`counting_sort_impl`](Self::counting_sort_impl).
    fn counting_sort(&mut self, points: FNx3View<'_>, bx: &SimBox) {
        let n_points = points.nrows();
        self.counting_sort_impl(
            n_points,
            |i| [points[[i, 0]], points[[i, 1]], points[[i, 2]]],
            bx,
        );
    }

    /// Counting sort particles by cell index (SoA `x`/`y`/`z` slices).
    ///
    /// SoA sibling of [`counting_sort`](Self::counting_sort); shares the same
    /// [`counting_sort_impl`](Self::counting_sort_impl) core, so the resulting
    /// `sorted_pos` / `cell_start` / `occupied_cells` are byte-identical to the
    /// `Array2` path for the same coordinates.
    fn counting_sort_soa(&mut self, xs: &[F], ys: &[F], zs: &[F], bx: &SimBox) {
        let n_points = xs.len();
        self.counting_sort_impl(n_points, |i| [xs[i], ys[i], zs[i]], bx);
    }

    /// Shared counting-sort core: cell assignment + prefix sum + scatter.
    ///
    /// Reads each particle position via `get_pt(i) -> [x, y, z]`, so both the
    /// interleaved (`Array2`) and column-major (SoA) entry points funnel
    /// through identical arithmetic. Sets up `grid`, `bx`, `cell_start`,
    /// `sorted_idx`, `sorted_pos`, and `occupied_cells`. Does NOT compute pairs.
    fn counting_sort_impl<G>(&mut self, n_points: usize, get_pt: G, bx: &SimBox)
    where
        G: Fn(usize) -> [F; 3],
    {
        let grid = CellGrid::for_cutoff(bx, self.cutoff);
        let n_cells = grid.n_cells();

        self.grid = grid;
        self.bx = bx.clone();

        // 1) Compute cell per particle, count per cell.
        //    `cell_of[i]` holds the final cell index for particle i; this buffer
        //    is never aliased with `sorted_idx` (which gets the reordering).
        self.cell_start.clear();
        self.cell_start.resize(n_cells + 1, 0);
        self.cell_of.resize(n_points, 0);
        for i in 0..n_points {
            let cell = grid.cell_of(bx, get_pt(i));
            self.cell_of[i] = cell as u32;
            self.cell_start[cell] += 1;
        }

        // 2) Prefix sum -> cell_start[c] = offset where cell c begins.
        //    Collect occupied cells while scanning.
        self.occupied_cells.clear();
        let mut acc = 0u32;
        for c in 0..n_cells {
            let count = self.cell_start[c];
            if count > 0 {
                self.occupied_cells.push(c as u32);
            }
            self.cell_start[c] = acc;
            acc += count;
        }
        self.cell_start[n_cells] = acc;
        debug_assert_eq!(acc as usize, n_points);

        // 3) Scatter particles into sorted order.
        //    `cursor[c]` starts at `cell_start[c]`; incremented as each particle
        //    in cell c is placed. Flat `sorted_pos` is 3N floats.
        self.cursor.resize(n_cells, 0);
        self.cursor.copy_from_slice(&self.cell_start[..n_cells]);

        self.sorted_idx.resize(n_points, 0);
        self.sorted_pos.resize(n_points * 3, 0.0);

        for i in 0..n_points {
            let cell = self.cell_of[i] as usize;
            let dst = self.cursor[cell] as usize;
            self.cursor[cell] += 1;
            self.sorted_idx[dst] = i as u32;
            let base = dst * 3;
            let p = get_pt(i);
            self.sorted_pos[base] = p[0];
            self.sorted_pos[base + 1] = p[1];
            self.sorted_pos[base + 2] = p[2];
        }
    }

    /// Serial half-shell pair search over occupied cells only.
    #[cfg(not(feature = "rayon"))]
    fn compute_pairs_serial(&mut self) {
        let cutoff2 = self.cutoff * self.cutoff;
        self.result.clear();

        let mut fwd_buf = [0usize; 27];

        for &cell in &self.occupied_cells {
            let cell = cell as usize;
            let start = self.cell_start[cell] as usize;
            let end = self.cell_start[cell + 1] as usize;

            // Self-cell pairs
            for si in start..end {
                let pi = pos_at(&self.sorted_pos, si);
                let oi = self.sorted_idx[si];
                for sj in (si + 1)..end {
                    let pj = pos_at(&self.sorted_pos, sj);
                    let dr = self.bx.shortest_vector_impl(pi, pj);
                    let d2 = dr[0] * dr[0] + dr[1] * dr[1] + dr[2] * dr[2];
                    if d2 <= cutoff2 {
                        self.result.push(oi, self.sorted_idx[sj], d2, dr);
                    }
                }
            }

            // Forward neighbor cells (stack buffer — no alloc)
            let n_fwd = self.grid.stencil_forward(cell, &mut fwd_buf);
            let fwd = &fwd_buf[..n_fwd];

            for si in start..end {
                let pi = pos_at(&self.sorted_pos, si);
                let oi = self.sorted_idx[si];

                for &nc in fwd {
                    let nc_start = self.cell_start[nc] as usize;
                    let nc_end = self.cell_start[nc + 1] as usize;

                    for sj in nc_start..nc_end {
                        let oj = self.sorted_idx[sj];
                        let pj = pos_at(&self.sorted_pos, sj);
                        let dr = self.bx.shortest_vector_impl(pi, pj);
                        let d2 = dr[0] * dr[0] + dr[1] * dr[1] + dr[2] * dr[2];
                        if d2 <= cutoff2 {
                            if oi < oj {
                                self.result.push(oi, oj, d2, dr);
                            } else {
                                self.result.push(oj, oi, d2, [-dr[0], -dr[1], -dr[2]]);
                            }
                        }
                    }
                }
            }
        }
    }

    /// Parallel half-shell pair search over occupied cells.
    ///
    /// For very small systems (few occupied cells) we fall back to the serial
    /// path — rayon's split/merge overhead dwarfs the pair work itself below
    /// ~64 cells. Empirically this wins ~2× at N=1k.
    #[cfg(feature = "rayon")]
    #[allow(clippy::needless_range_loop)]
    fn compute_pairs_parallel(&mut self) {
        use rayon::prelude::*;

        // Small-system fallback: rayon dispatch overhead > pair work.
        // Threshold chosen empirically: at N=1k with ρ=0.8 cutoff=2.5 there
        // are ~10 cells; the serial path is 30-50% faster.
        if self.occupied_cells.len() < 64 {
            self.compute_pairs_serial_inner();
            return;
        }

        let cutoff2 = self.cutoff * self.cutoff;

        let cell_start = &self.cell_start;
        let sorted_idx = &self.sorted_idx;
        let sorted_pos = &self.sorted_pos;
        let bx = &self.bx;
        let grid = self.grid;

        let merged = self
            .occupied_cells
            .par_iter()
            .fold(NeighborList::empty, |mut acc, &cell_u32| {
                let cell = cell_u32 as usize;
                let start = cell_start[cell] as usize;
                let end = cell_start[cell + 1] as usize;

                // Self-cell pairs.
                for si in start..end {
                    let pi = pos_at(sorted_pos, si);
                    let oi = sorted_idx[si];
                    for sj in (si + 1)..end {
                        let pj = pos_at(sorted_pos, sj);
                        let dr = bx.shortest_vector_impl(pi, pj);
                        let d2 = dr[0] * dr[0] + dr[1] * dr[1] + dr[2] * dr[2];
                        if d2 <= cutoff2 {
                            acc.push(oi, sorted_idx[sj], d2, dr);
                        }
                    }
                }

                // Forward neighbor cells (stack buffer — no alloc).
                let mut fwd_buf = [0usize; 27];
                let n_fwd = grid.stencil_forward(cell, &mut fwd_buf);
                let fwd = &fwd_buf[..n_fwd];

                for si in start..end {
                    let pi = pos_at(sorted_pos, si);
                    let oi = sorted_idx[si];

                    for &nc in fwd {
                        let nc_start = cell_start[nc] as usize;
                        let nc_end = cell_start[nc + 1] as usize;

                        for sj in nc_start..nc_end {
                            let oj = sorted_idx[sj];
                            let pj = pos_at(sorted_pos, sj);
                            let dr = bx.shortest_vector_impl(pi, pj);
                            let d2 = dr[0] * dr[0] + dr[1] * dr[1] + dr[2] * dr[2];
                            if d2 <= cutoff2 {
                                if oi < oj {
                                    acc.push(oi, oj, d2, dr);
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

    /// Serial pair search, used both by the no-rayon build and as the
    /// small-system fallback inside the rayon build.
    #[cfg(feature = "rayon")]
    fn compute_pairs_serial_inner(&mut self) {
        let cutoff2 = self.cutoff * self.cutoff;
        self.result.clear();
        let mut fwd_buf = [0usize; 27];

        for &cell in &self.occupied_cells {
            let cell = cell as usize;
            let start = self.cell_start[cell] as usize;
            let end = self.cell_start[cell + 1] as usize;

            for si in start..end {
                let pi = pos_at(&self.sorted_pos, si);
                let oi = self.sorted_idx[si];
                for sj in (si + 1)..end {
                    let pj = pos_at(&self.sorted_pos, sj);
                    let dr = self.bx.shortest_vector_impl(pi, pj);
                    let d2 = dr[0] * dr[0] + dr[1] * dr[1] + dr[2] * dr[2];
                    if d2 <= cutoff2 {
                        self.result.push(oi, self.sorted_idx[sj], d2, dr);
                    }
                }
            }

            let n_fwd = self.grid.stencil_forward(cell, &mut fwd_buf);
            let fwd = &fwd_buf[..n_fwd];

            for si in start..end {
                let pi = pos_at(&self.sorted_pos, si);
                let oi = self.sorted_idx[si];
                for &nc in fwd {
                    let nc_start = self.cell_start[nc] as usize;
                    let nc_end = self.cell_start[nc + 1] as usize;
                    for sj in nc_start..nc_end {
                        let oj = self.sorted_idx[sj];
                        let pj = pos_at(&self.sorted_pos, sj);
                        let dr = self.bx.shortest_vector_impl(pi, pj);
                        let d2 = dr[0] * dr[0] + dr[1] * dr[1] + dr[2] * dr[2];
                        if d2 <= cutoff2 {
                            if oi < oj {
                                self.result.push(oi, oj, d2, dr);
                            } else {
                                self.result.push(oj, oi, d2, [-dr[0], -dr[1], -dr[2]]);
                            }
                        }
                    }
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Inline stencil computation (zero-alloc: writes into caller-provided buffer)
// ---------------------------------------------------------------------------

/// Flat-slab position accessor: `[x, y, z]` of particle at sorted slot `si`.
#[inline(always)]
fn pos_at(sorted_pos: &[F], si: usize) -> [F; 3] {
    let base = si * 3;
    [sorted_pos[base], sorted_pos[base + 1], sorted_pos[base + 2]]
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::spatial::neighbors::NbList;
    use crate::spatial::region::simbox::SimBox;
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

    /// A point past the top face of a non-periodic axis belongs in the top
    /// cell, not the bottom one.
    ///
    /// Before `CellGrid` the fractional coordinate was folded into `[0, 1)`
    /// regardless of periodicity, so `z = 10.5` in a 10 Å box binned as if it
    /// were `z = 0.5` — the far end of the grid from its actual neighbour at
    /// `z = 9.5`, whose cell the stencil then never reached. Optimisation
    /// workloads leave the box routinely, so this is a missed pair, not an
    /// exotic edge case.
    #[test]
    fn out_of_box_point_on_a_non_periodic_axis_keeps_its_neighbours() {
        let bx = SimBox::cube(10.0, array![0.0, 0.0, 0.0], [false, false, false])
            .expect("invalid box length");
        // cutoff 2 => 5 cells per axis; 9.5 sits in the last cell, 10.5 outside.
        let pts = array![[5.0, 5.0, 10.5], [5.0, 5.0, 9.5]];
        let mut nl = NbList(LinkCell::new().cutoff(2.0));
        nl.build(pts.view(), &bx);
        assert_eq!(nl.query().n_pairs(), 1, "clamped point lost its neighbour");

        // And clamping must not invent a pair across the box: these two are
        // 11.0 apart in reality, only adjacent if the axis wrapped.
        let pts = array![[5.0, 5.0, 10.5], [5.0, 5.0, -0.5]];
        let mut nl = NbList(LinkCell::new().cutoff(2.0));
        nl.build(pts.view(), &bx);
        assert_eq!(nl.query().n_pairs(), 0, "non-periodic axis wrapped");
    }

    /// With two cells on a non-periodic axis, a query point in the upper cell
    /// must still see the lower one.
    ///
    /// The replaced stencil used `{0, +1}` offsets when an axis held two cells,
    /// so cell 1's only neighbour was out of range and its full stencil came
    /// back empty. Invisible on the pair path — the `nc > cell` forward filter
    /// covers the pair from the other side — and therefore only reachable
    /// through a query.
    #[test]
    fn query_across_two_cells_on_a_non_periodic_axis() {
        use crate::spatial::neighbors::NeighborQuery;

        // 10 Å box, cutoff 4 => floor(10/4) = 2 cells per axis.
        let bx = SimBox::cube(10.0, array![0.0, 0.0, 0.0], [false, false, false])
            .expect("invalid box length");
        let refs = array![[5.0, 5.0, 4.5]]; // cell 0 along z
        let query = array![[5.0, 5.0, 7.5]]; // cell 1 along z, 3.0 away

        let nq = NeighborQuery::new(&bx, refs.view(), 4.0);
        let res = nq.query(query.view());
        assert_eq!(res.n_pairs(), 1, "upper cell reported an empty stencil");
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

        let mut lc_index = LinkCell::new().cutoff(0.6);
        lc_index.build_index(pts.view(), &bx);
        let mut visit_pairs: Vec<(u32, u32, F, [F; 3])> = Vec::new();
        lc_index.visit_pairs(&mut |i, j, d2, diff| {
            visit_pairs.push((i, j, d2, diff));
        });
        visit_pairs.sort_by_key(|(i, j, _, _)| (*i, *j));

        assert_eq!(full_pairs.len(), visit_pairs.len());
        for (a, b) in full_pairs.iter().zip(visit_pairs.iter()) {
            assert_eq!(a.0, b.0);
            assert_eq!(a.1, b.1);
            assert!((a.2 - b.2).abs() < 1e-6);
            for d in 0..3 {
                assert!((a.3[d] - b.3[d]).abs() < 1e-6);
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

        let mut bf = NbList(crate::spatial::neighbors::bruteforce::BruteForce::new(0.6));
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

    #[test]
    fn non_periodic_no_wrap() {
        let bx =
            SimBox::cube(10.0, array![0.0, 0.0, 0.0], [false, false, false]).expect("invalid box");
        let pts = array![[0.1, 0.1, 0.1], [9.9, 9.9, 9.9]];

        let mut lc = LinkCell::new().cutoff(1.0);
        lc.build(pts.view(), &bx);
        let result = lc.query();

        assert_eq!(result.n_pairs(), 0, "non-periodic box should not wrap");
    }

    #[test]
    fn periodic_does_wrap() {
        let bx =
            SimBox::cube(10.0, array![0.0, 0.0, 0.0], [true, true, true]).expect("invalid box");
        let pts = array![[0.1, 0.1, 0.1], [9.9, 9.9, 9.9]];

        let mut lc = LinkCell::new().cutoff(1.0);
        lc.build(pts.view(), &bx);
        let result = lc.query();

        assert_eq!(result.n_pairs(), 1, "periodic box should wrap");
    }

    #[test]
    fn non_periodic_matches_brute_force() {
        let bx =
            SimBox::cube(20.0, array![0.0, 0.0, 0.0], [false, false, false]).expect("invalid box");
        let pts = array![
            [5.0, 5.0, 5.0],
            [5.5, 5.0, 5.0],
            [5.0, 5.5, 5.0],
            [10.0, 10.0, 10.0],
            [10.3, 10.0, 10.0],
        ];
        let cutoff = 1.0;

        let mut lc = LinkCell::new().cutoff(cutoff);
        lc.build(pts.view(), &bx);
        let lc_result = lc.query();

        let mut bf = crate::spatial::neighbors::bruteforce::BruteForce::new(cutoff);
        bf.build(pts.view(), &bx);
        let bf_result = bf.query();

        assert_eq!(
            lc_result.n_pairs(),
            bf_result.n_pairs(),
            "LinkCell and BruteForce should agree for non-periodic"
        );

        let mut lc_pairs: Vec<(u32, u32)> = lc_result
            .query_point_indices()
            .iter()
            .zip(lc_result.point_indices().iter())
            .map(|(&i, &j)| if i < j { (i, j) } else { (j, i) })
            .collect();
        lc_pairs.sort();

        let mut bf_pairs: Vec<(u32, u32)> = bf_result
            .query_point_indices()
            .iter()
            .zip(bf_result.point_indices().iter())
            .map(|(&i, &j)| if i < j { (i, j) } else { (j, i) })
            .collect();
        bf_pairs.sort();

        assert_eq!(lc_pairs, bf_pairs);
    }

    // --- sparse system: 3 particles in large box with small cutoff ---

    #[test]
    fn sparse_system_fast() {
        // This previously timed out: box=20, cutoff=0.5 → 64K cells, 3 particles.
        // With occupied_cells optimization, only 3 cells are visited.
        let bx =
            SimBox::cube(20.0, array![0.0, 0.0, 0.0], [false, false, false]).expect("invalid box");
        let pts = array![[1.0, 1.0, 1.0], [5.0, 5.0, 5.0], [9.0, 9.0, 9.0]];
        let mut lc = LinkCell::new().cutoff(0.5);
        lc.build(pts.view(), &bx);
        assert_eq!(lc.occupied_cells.len(), 3);
        assert_eq!(lc.query().n_pairs(), 0);
    }

    // --- freud: 4 collinear, count pairs ---

    #[test]
    fn collinear_pair_counts() {
        let bx = SimBox::cube(10.0, array![0.0, 0.0, 0.0], [false, false, false]).unwrap();
        let pts = array![
            [1.0, 5.0, 5.0],
            [2.0, 5.0, 5.0],
            [4.0, 5.0, 5.0],
            [3.0, 5.0, 5.0],
        ];
        let mut lc = LinkCell::new().cutoff(2.01);
        lc.build(pts.view(), &bx);
        assert_eq!(lc.query().n_pairs(), 5);
    }

    // --- all pairs within cutoff ---

    #[test]
    fn all_pairs_within_cutoff() {
        let bx = SimBox::cube(10.0, array![0.0, 0.0, 0.0], [false, false, false]).unwrap();
        let pts = array![
            [5.0, 5.0, 5.0],
            [5.5, 5.0, 5.0],
            [5.0, 5.5, 5.0],
            [5.5, 5.5, 5.0],
        ];
        let mut lc = LinkCell::new().cutoff(1.5);
        lc.build(pts.view(), &bx);
        assert_eq!(lc.query().n_pairs(), 6);
    }

    // --- brute force vs linkcell with PBC ---

    #[test]
    fn exhaustive_pbc() {
        let bx = SimBox::cube(10.0, array![0.0, 0.0, 0.0], [true, true, true]).unwrap();
        let pts = array![
            [1.0, 1.0, 1.0],
            [1.5, 1.0, 1.0],
            [9.5, 1.0, 1.0],
            [5.0, 5.0, 5.0],
            [5.3, 5.0, 5.0],
        ];
        let cutoff = 2.0;

        let mut lc = LinkCell::new().cutoff(cutoff);
        lc.build(pts.view(), &bx);
        let lc_result = lc.query();

        let mut bf = crate::spatial::neighbors::bruteforce::BruteForce::new(cutoff);
        bf.build(pts.view(), &bx);
        let bf_result = bf.query();

        assert_eq!(lc_result.n_pairs(), bf_result.n_pairs());
    }

    // --- edge cases ---

    #[test]
    fn no_pairs_large_separation() {
        let bx = SimBox::cube(10.0, array![0.0, 0.0, 0.0], [false, false, false]).unwrap();
        let pts = array![[1.0, 1.0, 1.0], [8.0, 8.0, 8.0],];
        let mut lc = LinkCell::new().cutoff(1.0);
        lc.build(pts.view(), &bx);
        assert_eq!(lc.query().n_pairs(), 0);
    }

    #[test]
    fn single_particle_no_pairs() {
        let bx = SimBox::cube(10.0, array![0.0, 0.0, 0.0], [false, false, false]).unwrap();
        let pts = array![[5.0, 5.0, 5.0],];
        let mut lc = LinkCell::new().cutoff(3.0);
        lc.build(pts.view(), &bx);
        assert_eq!(lc.query().n_pairs(), 0);
    }

    #[test]
    fn distances_correct() {
        let bx = SimBox::cube(10.0, array![0.0, 0.0, 0.0], [false, false, false]).unwrap();
        let pts = array![[3.0, 5.0, 5.0], [5.0, 5.0, 5.0],];
        let mut lc = LinkCell::new().cutoff(3.0);
        lc.build(pts.view(), &bx);
        let nlist = lc.query();
        assert_eq!(nlist.n_pairs(), 1);
        let dists = nlist.distances();
        assert!((dists[0] - 2.0).abs() < 1e-5);
    }

    #[test]
    fn pbc_distance_correct() {
        let bx = SimBox::cube(10.0, array![0.0, 0.0, 0.0], [true, true, true]).unwrap();
        let pts = array![[0.5, 5.0, 5.0], [9.5, 5.0, 5.0],]; // MIC dist = 1.0
        let mut lc = LinkCell::new().cutoff(2.0);
        lc.build(pts.view(), &bx);
        let nlist = lc.query();
        assert_eq!(nlist.n_pairs(), 1);
        let dists = nlist.distances();
        assert!((dists[0] - 1.0).abs() < 1e-5);
    }

    // --- SoA path is bit-identical to the Array2 path ---

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
    fn assert_bitwise_equal(a: &NeighborList, b: &NeighborList) {
        assert_eq!(a.n_pairs(), b.n_pairs(), "n_pairs differ");
        let da = a.vectors();
        let db = b.vectors();
        for k in 0..a.n_pairs() {
            assert_eq!(
                a.query_point_indices()[k],
                b.query_point_indices()[k],
                "idx_i"
            );
            assert_eq!(a.point_indices()[k], b.point_indices()[k], "idx_j");
            // Bitwise f64 equality — the arithmetic is identical, so it must match.
            assert_eq!(a.dist_sq()[k], b.dist_sq()[k], "dist_sq bitwise");
            for d in 0..3 {
                assert_eq!(da[[k, d]], db[[k, d]], "diff[{}] bitwise", d);
            }
        }
    }

    #[test]
    fn build_soa_matches_build_bitwise() {
        // Periodic cube fixture: several pairs, including a PBC-wrap pair.
        let bx = SimBox::cube(10.0, array![0.0, 0.0, 0.0], [true, true, true]).unwrap();
        let pts = array![
            [1.0, 1.0, 1.0],
            [1.5, 1.0, 1.0],
            [9.5, 1.0, 1.0],
            [5.0, 5.0, 5.0],
            [5.3, 5.0, 5.0],
            [2.2, 8.1, 3.3],
            [7.7, 2.4, 9.6],
        ];
        let (xs, ys, zs) = columns(&pts);
        let mut lc_a = LinkCell::new().cutoff(2.0);
        lc_a.build(pts.view(), &bx);
        let mut lc_s = LinkCell::new().cutoff(2.0);
        lc_s.build_soa(&xs, &ys, &zs, &bx);
        assert!(lc_a.query().n_pairs() > 0, "fixture should produce pairs");
        assert_bitwise_equal(lc_a.query(), lc_s.query());

        // Free / non-periodic fixture.
        let bxf = SimBox::cube(20.0, array![0.0, 0.0, 0.0], [false, false, false]).unwrap();
        let ptsf = array![
            [5.0, 5.0, 5.0],
            [5.5, 5.0, 5.0],
            [5.0, 5.5, 5.0],
            [10.0, 10.0, 10.0],
            [10.3, 10.0, 10.0],
            [1.0, 18.0, 3.0],
        ];
        let (fxs, fys, fzs) = columns(&ptsf);
        let mut lc_af = LinkCell::new().cutoff(1.0);
        lc_af.build(ptsf.view(), &bxf);
        let mut lc_sf = LinkCell::new().cutoff(1.0);
        lc_sf.build_soa(&fxs, &fys, &fzs, &bxf);
        assert!(lc_af.query().n_pairs() > 0, "fixture should produce pairs");
        assert_bitwise_equal(lc_af.query(), lc_sf.query());
    }
}
