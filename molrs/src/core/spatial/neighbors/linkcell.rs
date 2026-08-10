//! Cell-list neighbor search — O(N) with sorted-particle layout and half-shell
//! iteration.
//!
//! A *cell list* divides the simulation box into a grid of cells at least one
//! cutoff wide. A particle can then only have neighbors in its own cell and the
//! 26 cells touching it, which turns the O(N²) all-pairs scan into an O(N) one.
//!
//! Particles are *counting-sorted* by cell index — a linear-time sort that
//! works by tallying how many particles fall in each cell and then scattering
//! them into the resulting offsets — so that all particles in the same cell
//! occupy a contiguous slice of `sorted_idx` / `sorted_pos`. This gives
//! excellent cache locality during the pair search compared to a linked-list
//! layout.
//!
//! Only **occupied cells** are visited during pair search, so sparse systems
//! (few particles, many cells) pay O(N), not O(n_cells).
//!
//! *Half-shell* iteration means each cell only looks at a forward half of its
//! neighboring cells, so every unordered pair is discovered exactly once and
//! the resulting table satisfies `i < j`.

use crate::spatial::neighbors::{
    Backend, CellGrid, Neighbors, NeighborsStorage, PairVisitor, QueryMode,
};
use crate::spatial::simbox::SimBox;
use crate::types::{F, FNx3View};
use ndarray::array;

/// Cell-list neighbor search algorithm — the default
/// [`NeighborList`](crate::spatial::neighbors::NeighborList) backend.
///
/// Partitions space into a regular grid of cells whose width >= cutoff, then
/// searches only neighboring cells for pair interactions.  Uses half-shell
/// iteration so each pair is found exactly once.
///
/// The materialized table is a self-query: `mode()` is
/// `QueryMode::SelfQuery { num_points }` and every pair satisfies `i < j`. Which
/// physical columns it keeps is configurable through
/// [`with_storage`](Self::with_storage) and defaults to
/// [`NeighborsStorage::FULL`] — squared distances in Å² and minimum-image
/// displacements in Å.
///
/// With the `rayon` feature (default), the pair search is parallelized across
/// occupied cells via `rayon::par_iter`.
///
/// Search from outside this crate goes through
/// [`NeighborList::new`](crate::spatial::neighbors::NeighborList::new), which
/// drives this type.
#[derive(Debug, Clone)]
pub struct LinkCell {
    /// Interaction cutoff distance (Å). Also sets the minimum cell width, so
    /// changing it invalidates any previously built index.
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
    result: Neighbors,
    /// Reusable cursor buffer for counting-sort scatter (avoids allocation).
    cursor: Vec<u32>,
    /// Reusable cell-assignment buffer (avoids cloning sorted_idx).
    cell_of: Vec<u32>,
    /// Which optional columns the materialized [`Neighbors`] table keeps.
    storage: NeighborsStorage,
}

impl Default for LinkCell {
    fn default() -> Self {
        Self::new()
    }
}

impl LinkCell {
    /// Create a new `LinkCell` with zero cutoff (must be set via [`cutoff`](Self::cutoff)).
    ///
    /// The storage policy starts at [`NeighborsStorage::FULL`] and the cached
    /// table starts empty, tagged `SelfQuery { num_points: 0 }`.
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
            result: Neighbors::empty(
                QueryMode::SelfQuery { num_points: 0 },
                NeighborsStorage::FULL,
            ),
            cursor: Vec::new(),
            cell_of: Vec::new(),
            storage: NeighborsStorage::FULL,
        }
    }

    /// Set the cutoff distance in Å (builder pattern; consumes and returns
    /// `self`).
    ///
    /// Must end up positive before any build, which asserts on it.
    pub fn cutoff(mut self, cutoff: F) -> Self {
        self.cutoff = cutoff;
        self
    }

    /// Configure which optional pair fields are stored when materializing
    /// a [`Neighbors`] table via `build` / `update`.
    ///
    /// A column left out is reported as `None` by the table's accessor, not as
    /// zeros, and cannot be added back afterwards — pick the policy before
    /// building. Does not affect index-only building and pair streaming (those
    /// never store pairs, and always hand the visitor both quantities). Calling
    /// this discards any table already built.
    pub fn with_storage(mut self, storage: NeighborsStorage) -> Self {
        self.storage = storage;
        self.result = Neighbors::empty(QueryMode::SelfQuery { num_points: 0 }, storage);
        self
    }

    /// Storage policy used by the next materializing build.
    #[inline]
    pub fn storage(&self) -> NeighborsStorage {
        self.storage
    }

    /// Visit the reference points in the cell neighborhood of an arbitrary
    /// query point.
    ///
    /// Used by [`NeighborQuery::query`](super::NeighborQuery::query) for
    /// cross-query. Calls `callback(ref_index, dist_sq, [dx, dy, dz])` with
    /// `dist_sq` in Å² and the minimum-image displacement `r_ref - r_query` in
    /// Å.
    ///
    /// The callback fires for **every** point in the query point's own cell and
    /// in its surrounding cells (up to 26 distinct ones, fewer when an axis has
    /// only one or two cells), including points beyond the cutoff: a cell is at
    /// least one cutoff wide, so its corners reach farther than that. Applying
    /// the cutoff test is the caller's job.
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
    /// Both entry points funnel into the same
    /// [`CellGrid::cell_of`](crate::spatial::neighbors::CellGrid::cell_of), so
    /// they agree bit-for-bit on the same coordinate. Used by
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

    /// Index `points` in `bx` **and** materialize the half-shell pair table
    /// read back by [`query`](Self::query).
    ///
    /// The index-only path — build once, then stream pairs without storing them
    /// — is [`NeighborList`](crate::spatial::neighbors::NeighborList).
    ///
    /// # Panics
    /// Panics if the cutoff is not positive or `points` does not have exactly
    /// 3 columns.
    pub fn build(&mut self, points: FNx3View<'_>, bx: &SimBox) {
        assert!(self.cutoff > 0.0, "cutoff must be positive");
        self.update(points, bx);
    }

    /// Re-index new positions and re-materialize the pair table, reusing the
    /// internal buffers.
    ///
    /// There is no skin/Verlet shortcut: this recomputes the pairs, it does not
    /// decide for you whether a rebuild was necessary.
    ///
    /// # Panics
    /// Panics if the cutoff is not positive or `points` does not have exactly
    /// 3 columns.
    pub fn update(&mut self, points: FNx3View<'_>, bx: &SimBox) {
        assert!(self.cutoff > 0.0, "cutoff must be positive");
        assert!(points.ncols() == 3, "points must have shape (N, 3)");

        self.counting_sort(points, bx);
        let n_points = points.nrows();

        #[cfg(feature = "rayon")]
        self.compute_pairs_parallel();
        #[cfg(not(feature = "rayon"))]
        self.compute_pairs_serial();

        self.result.mode = QueryMode::SelfQuery {
            num_points: n_points,
        };
    }

    /// Reference to the pair table materialized by the last
    /// [`build`](Self::build) / [`update`](Self::update) / [`build_soa`](Self::build_soa).
    ///
    /// Before the first one this is an empty table tagged
    /// `SelfQuery { num_points: 0 }`, not an error.
    #[inline]
    pub fn query(&self) -> &Neighbors {
        &self.result
    }
}

impl Backend for LinkCell {
    fn build_index(&mut self, points: FNx3View<'_>, bx: &SimBox) {
        assert!(self.cutoff > 0.0, "cutoff must be positive");
        self.update_index(points, bx);
    }

    fn update_index(&mut self, points: FNx3View<'_>, bx: &SimBox) {
        assert!(self.cutoff > 0.0, "cutoff must be positive");
        assert!(points.ncols() == 3, "points must have shape (N, 3)");
        self.counting_sort(points, bx);
    }

    /// Sort the columns into cells directly — no interleaved copy.
    ///
    /// Overrides the interleaving default with [`build_index_soa`](Self::build_index_soa),
    /// which shares the counting-sort core with `build_index`, so the index is
    /// identical to the one the `N × 3` path produces.
    fn build_index_columns(&mut self, xs: &[F], ys: &[F], zs: &[F], bx: &SimBox) {
        self.build_index_soa(xs, ys, zs, bx);
    }

    /// On-demand pair traversal — zero allocation.
    ///
    /// Same half-shell iteration as `compute_pairs_serial` but calls the
    /// visitor instead of building a [`Neighbors`] table, so the storage policy
    /// is irrelevant here: the visitor always receives `dist_sq` (Å²) and the
    /// minimum-image displacement `r_j - r_i` (Å) for each pair, with `i < j`.
    ///
    /// Requires a prior `build_index` / `update_index` (or `build`, which also
    /// sorts the particles). Without one there are no occupied cells and the
    /// visitor is simply never called.
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
                            // Half-shell output contract: the pair is emitted
                            // with the smaller original index first. `dr` was
                            // computed as MIC(r_sj - r_si) for the *sorted*
                            // slots; when the roles swap, the sign must swap
                            // too, so that the visitor still receives
                            // r_j - r_i for the emitted (i, j).
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
    /// SoA sibling of `build`: build the self-query pair list from
    /// column-major `x`/`y`/`z` slices.
    ///
    /// "SoA" is *structure of arrays*: instead of one `N × 3` array of points,
    /// the coordinates arrive as three length-`N` slices (Å), one per axis.
    ///
    /// Shares the same counting-sort + pair-search core as `build`, so the
    /// resulting [`Neighbors`] table is byte-identical to `build` on the same
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

        self.result.mode = QueryMode::SelfQuery {
            num_points: n_points,
        };
    }

    /// SoA sibling of `build_index`: build the spatial index only (no pair
    /// pre-computation) from column-major slices.
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
                            // Half-shell output contract: the pair is emitted
                            // with the smaller original index first. `dr` was
                            // computed as MIC(r_sj - r_si) for the *sorted*
                            // slots; when the roles swap, the sign must swap
                            // too, so that the stored displacement is still
                            // r_j - r_i for the emitted (i, j).
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

        let storage = self.storage;
        let merged = self
            .occupied_cells
            .par_iter()
            .fold(
                || Neighbors::empty(QueryMode::SelfQuery { num_points: 0 }, storage),
                |mut acc, &cell_u32| {
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
                                    // Half-shell output contract: the pair is
                                    // emitted with the smaller original index
                                    // first. `dr` was computed as
                                    // MIC(r_sj - r_si) for the *sorted* slots;
                                    // when the roles swap, the sign must swap
                                    // too, so that the stored displacement is
                                    // still r_j - r_i for the emitted (i, j).
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
                },
            )
            .reduce(
                || Neighbors::empty(QueryMode::SelfQuery { num_points: 0 }, storage),
                |mut a, b| {
                    a.idx_i.extend_from_slice(&b.idx_i);
                    a.idx_j.extend_from_slice(&b.idx_j);
                    a.dist_sq.extend_from_slice(&b.dist_sq);
                    a.disp_flat.extend_from_slice(&b.disp_flat);
                    a
                },
            );

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
                            // Half-shell output contract: the pair is emitted
                            // with the smaller original index first. `dr` was
                            // computed as MIC(r_sj - r_si) for the *sorted*
                            // slots; when the roles swap, the sign must swap
                            // too, so that the stored displacement is still
                            // r_j - r_i for the emitted (i, j).
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
    use crate::spatial::simbox::SimBox;
    use ndarray::array;

    #[test]
    fn linked_cell_basic_pairs() {
        let bx = SimBox::cube(4.0, array![0.0, 0.0, 0.0], [true, true, true])
            .expect("invalid box length");
        let pts = array![[0.1, 0.2, 0.3], [0.3, 0.2, 0.1], [3.9, 3.8, 3.7]];
        let mut nl = LinkCell::new().cutoff(0.5);
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
        let mut nl = LinkCell::new().cutoff(0.5);
        nl.build(pts.view(), &bx);
        let res = nl.query();
        assert_eq!(res.n_pairs(), 1);
    }

    #[test]
    fn linked_cell_no_duplicates() {
        let bx = SimBox::cube(3.0, array![0.0, 0.0, 0.0], [true, true, true])
            .expect("invalid box length");
        let pts = array![[0.1, 0.1, 0.1], [0.2, 0.2, 0.2], [0.3, 0.3, 0.3]];
        let mut nl = LinkCell::new().cutoff(1.0);
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
        let mut nl = LinkCell::new().cutoff(1.0);
        nl.build(pts.view(), &bx);
        let res = nl.query();
        assert_eq!(res.n_pairs(), 1);
    }

    #[test]
    fn linked_cell_deterministic_order() {
        let bx = SimBox::cube(4.0, array![0.0, 0.0, 0.0], [true, true, true])
            .expect("invalid box length");
        let pts = array![[0.1, 0.2, 0.3], [0.4, 0.2, 0.3], [1.1, 1.2, 1.3]];
        let mut nl = LinkCell::new().cutoff(0.5);
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
        let d2 = res.dist_sq().expect("LinkCell materializes dist_sq");
        let disp = res.disp().expect("LinkCell materializes disp");
        let mut full_pairs: Vec<(u32, u32, F, [F; 3])> = (0..res.n_pairs())
            .map(|k| {
                (
                    res.query_point_indices()[k],
                    res.point_indices()[k],
                    d2[k],
                    [disp[[k, 0]], disp[[k, 1]], disp[[k, 2]]],
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

        let mut lc = LinkCell::new().cutoff(0.6);
        lc.build(pts.view(), &bx);
        let res_lc = lc.query();

        let mut bf = crate::spatial::neighbors::bruteforce::BruteForce::new(0.6);
        bf.build(pts.view(), &bx);
        let res_bf = bf.query();

        let d2_lc = res_lc.dist_sq().expect("LinkCell materializes dist_sq");
        let d2_bf = res_bf.dist_sq().expect("BruteForce materializes dist_sq");
        let disp_lc = res_lc.disp().expect("LinkCell materializes disp");
        let disp_bf = res_bf.disp().expect("BruteForce materializes disp");

        let mut lc_pairs: Vec<(u32, u32, F, [F; 3])> = (0..res_lc.n_pairs())
            .map(|k| {
                (
                    res_lc.query_point_indices()[k],
                    res_lc.point_indices()[k],
                    d2_lc[k],
                    [disp_lc[[k, 0]], disp_lc[[k, 1]], disp_lc[[k, 2]]],
                )
            })
            .collect();
        lc_pairs.sort_by_key(|(i, j, _, _)| (*i, *j));

        let mut bf_pairs: Vec<(u32, u32, F, [F; 3])> = (0..res_bf.n_pairs())
            .map(|k| {
                (
                    res_bf.query_point_indices()[k],
                    res_bf.point_indices()[k],
                    d2_bf[k],
                    [disp_bf[[k, 0]], disp_bf[[k, 1]], disp_bf[[k, 2]]],
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
        let d2 = nlist.dist_sq().expect("LinkCell materializes dist_sq");
        assert!((d2[0].sqrt() - 2.0).abs() < 1e-5);
    }

    #[test]
    fn pbc_distance_correct() {
        let bx = SimBox::cube(10.0, array![0.0, 0.0, 0.0], [true, true, true]).unwrap();
        let pts = array![[0.5, 5.0, 5.0], [9.5, 5.0, 5.0],]; // MIC dist = 1.0
        let mut lc = LinkCell::new().cutoff(2.0);
        lc.build(pts.view(), &bx);
        let nlist = lc.query();
        assert_eq!(nlist.n_pairs(), 1);
        let d2 = nlist.dist_sq().expect("LinkCell materializes dist_sq");
        assert!((d2[0].sqrt() - 1.0).abs() < 1e-5);
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
    fn assert_bitwise_equal(a: &Neighbors, b: &Neighbors) {
        assert_eq!(a.n_pairs(), b.n_pairs(), "n_pairs differ");
        let d2a = a.dist_sq().expect("LinkCell materializes dist_sq");
        let d2b = b.dist_sq().expect("LinkCell materializes dist_sq");
        let da = a.disp().expect("LinkCell materializes disp");
        let db = b.disp().expect("LinkCell materializes disp");
        for k in 0..a.n_pairs() {
            assert_eq!(
                a.query_point_indices()[k],
                b.query_point_indices()[k],
                "idx_i"
            );
            assert_eq!(a.point_indices()[k], b.point_indices()[k], "idx_j");
            // Bitwise f64 equality — the arithmetic is identical, so it must match.
            assert_eq!(d2a[k], d2b[k], "dist_sq bitwise");
            for d in 0..3 {
                assert_eq!(da[[k, d]], db[[k, d]], "disp[{}] bitwise", d);
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

// ---------------------------------------------------------------------------
// Equivalence matrix
// ---------------------------------------------------------------------------

/// `LinkCell` against an independent O(N²) oracle, over the full configuration
/// space the cell partition has to survive.
///
/// The oracle is a direct double loop over
/// [`SimBox::shortest_vector_impl`] — it shares no cell-assignment or stencil
/// code with the algorithm under test, so a defect in either cannot cancel out.
/// [`BruteForce`] is checked against the same oracle, which keeps the oracle
/// itself honest.
///
/// Both traversal modes are exercised. They fail differently: the pair path's
/// forward filter covers an unordered cell pair from whichever side is cheaper,
/// so a stencil that loses a neighbour in one direction still produces the
/// right pair set, while a query rooted at the cell with the broken stencil
/// silently returns nothing. Testing pairs alone leaves that entire class
/// unobserved.
#[cfg(test)]
mod equivalence {
    use super::*;
    use crate::spatial::neighbors::CellGrid;
    use crate::spatial::neighbors::NeighborQuery;
    use crate::spatial::neighbors::bruteforce::BruteForce;
    use crate::spatial::simbox::SimBox;
    use crate::types::F3x3;
    use ndarray::{Array2, array};
    use std::collections::{BTreeMap, BTreeSet};

    /// Deterministic point source — no RNG crate, no version drift, same
    /// stream on every platform.
    struct Lcg(u64);

    impl Lcg {
        fn unit(&mut self) -> F {
            self.0 = self
                .0
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            ((self.0 >> 11) as F) / ((1u64 << 53) as F)
        }
    }

    fn cells() -> Vec<(&'static str, F3x3)> {
        vec![
            (
                "ortho",
                SimBox::matrix_from_lengths_angles([10.0, 11.0, 12.0], [90.0, 90.0, 90.0]).unwrap(),
            ),
            (
                "hex",
                SimBox::matrix_from_lengths_angles([10.0, 10.0, 12.0], [90.0, 90.0, 120.0])
                    .unwrap(),
            ),
            (
                "tilted",
                SimBox::matrix_from_lengths_angles([10.0, 11.0, 12.0], [70.0, 80.0, 65.0]).unwrap(),
            ),
        ]
    }

    const PBCS: [[bool; 3]; 4] = [
        [true, true, true],
        [true, true, false],
        [true, false, false],
        [false, false, false],
    ];

    /// Points in Cartesian space: bulk sampling inside the cell, plus the two
    /// populations that break a naive partition — coordinates outside the cell
    /// (an optimiser's intermediate iterates leave the box routinely) and
    /// coordinates sitting exactly on a cell boundary.
    fn points(bx: &SimBox, seed: u64) -> Array2<F> {
        let mut rng = Lcg(seed);
        let mut frac: Vec<[F; 3]> = (0..40)
            .map(|_| [rng.unit(), rng.unit(), rng.unit()])
            .collect();

        for k in 0..3 {
            for &out in &[-1.7 as F, -0.15, 1.05, 2.4] {
                let mut p = [rng.unit(), rng.unit(), rng.unit()];
                p[k] = out;
                frac.push(p);
            }
        }
        for k in 0..3 {
            for &edge in &[0.0 as F, 1.0] {
                let mut p = [0.5 as F, 0.5, 0.5];
                p[k] = edge;
                frac.push(p);
            }
        }

        let mut pts = Array2::<F>::zeros((frac.len(), 3));
        for (i, f) in frac.iter().enumerate() {
            let cart = bx.make_cartesian(array![f[0], f[1], f[2]].view());
            for d in 0..3 {
                pts[[i, d]] = cart[d];
            }
        }
        pts
    }

    fn oracle_self(pts: &Array2<F>, bx: &SimBox, cutoff: F) -> BTreeMap<(u32, u32), F> {
        let c2 = cutoff * cutoff;
        let mut out = BTreeMap::new();
        for i in 0..pts.nrows() {
            let pi = [pts[[i, 0]], pts[[i, 1]], pts[[i, 2]]];
            for j in (i + 1)..pts.nrows() {
                let pj = [pts[[j, 0]], pts[[j, 1]], pts[[j, 2]]];
                let dr = bx.shortest_vector_impl(pi, pj);
                let d2 = dr[0] * dr[0] + dr[1] * dr[1] + dr[2] * dr[2];
                if d2 <= c2 {
                    out.insert((i as u32, j as u32), d2);
                }
            }
        }
        out
    }

    fn oracle_cross(
        query: &Array2<F>,
        refs: &Array2<F>,
        bx: &SimBox,
        cutoff: F,
    ) -> BTreeSet<(u32, u32)> {
        let c2 = cutoff * cutoff;
        let mut out = BTreeSet::new();
        for i in 0..query.nrows() {
            let qi = [query[[i, 0]], query[[i, 1]], query[[i, 2]]];
            for j in 0..refs.nrows() {
                let rj = [refs[[j, 0]], refs[[j, 1]], refs[[j, 2]]];
                let dr = bx.shortest_vector_impl(qi, rj);
                let d2 = dr[0] * dr[0] + dr[1] * dr[1] + dr[2] * dr[2];
                if d2 <= c2 {
                    out.insert((i as u32, j as u32));
                }
            }
        }
        out
    }

    /// Unordered pair multiset from a [`Neighbors`] table. A `BTreeMap` keyed on
    /// the ordered pair would silently swallow a duplicate emission, so
    /// duplicates are counted explicitly.
    ///
    /// The two physical columns are also checked against each other here rather
    /// than in a test of their own. `dist_sq` and `disp` are produced by the
    /// same minimum-image evaluation, so `dist_sq == ‖disp‖²` must hold for
    /// every pair — and collecting inside the matrix loop is what makes that
    /// claim executable across orthorhombic, hexagonal and tilted cells and all
    /// four periodicity combinations, i.e. through the triclinic MIC branch as
    /// well as the orthorhombic one. A cell partition that folded the two
    /// columns to different periodic images would fail here.
    fn collect(res: &Neighbors) -> (BTreeMap<(u32, u32), F>, usize) {
        let mut map = BTreeMap::new();
        let n = res.n_pairs();
        let d2 = res.dist_sq().expect("search materializes dist_sq");
        let disp = res.disp().expect("search materializes disp");
        for k in 0..n {
            let (i, j) = (res.query_point_indices()[k], res.point_indices()[k]);
            let norm_sq = disp[[k, 0]] * disp[[k, 0]]
                + disp[[k, 1]] * disp[[k, 1]]
                + disp[[k, 2]] * disp[[k, 2]];
            assert!(
                (d2[k] - norm_sq).abs() <= 1e-12,
                "pair ({i},{j}): dist_sq {} != ||disp||^2 {norm_sq}",
                d2[k]
            );
            let key = if i < j { (i, j) } else { (j, i) };
            map.insert(key, d2[k]);
        }
        (map, n)
    }

    /// Cutoffs chosen to drive `celldim` through 1, 2, 3 and 5 cells per axis.
    fn cutoffs(bx: &SimBox) -> Vec<F> {
        let npd = bx.nearest_plane_distance();
        let min = npd[0].min(npd[1]).min(npd[2]);
        vec![min / 1.0, min / 2.0, min / 3.0, min / 5.0]
    }

    #[test]
    fn pair_traversal_matches_the_oracle_over_the_matrix() {
        let mut seen_dims: BTreeSet<u32> = BTreeSet::new();
        let mut configs = 0usize;

        for (name, h) in cells() {
            for pbc in PBCS {
                let bx = SimBox::new(h.clone(), array![0.0, 0.0, 0.0], pbc).expect("box");
                let pts = points(&bx, 0x5EED);

                for cutoff in cutoffs(&bx) {
                    let tag = format!("{name} pbc={pbc:?} cutoff={cutoff:.3}");
                    seen_dims.extend(CellGrid::for_cutoff(&bx, cutoff).celldim());
                    configs += 1;

                    let want = oracle_self(&pts, &bx, cutoff);

                    let mut lc = LinkCell::new().cutoff(cutoff);
                    lc.build(pts.view(), &bx);
                    let (got, emitted) = collect(lc.query());

                    assert_eq!(emitted, want.len(), "{tag}: pair count (duplicate or gap)");
                    assert_eq!(
                        got.keys().collect::<Vec<_>>(),
                        want.keys().collect::<Vec<_>>(),
                        "{tag}: pair set"
                    );
                    for (key, d2) in &got {
                        assert!(
                            (d2 - want[key]).abs() <= 1e-12,
                            "{tag}: dist_sq for {key:?}: {d2} vs {}",
                            want[key]
                        );
                    }

                    // Keep the oracle honest against the reference algorithm.
                    let mut bf = BruteForce::new(cutoff);
                    bf.build(pts.view(), &bx);
                    let (bf_pairs, _) = collect(bf.query());
                    assert_eq!(
                        bf_pairs.keys().collect::<Vec<_>>(),
                        want.keys().collect::<Vec<_>>(),
                        "{tag}: oracle disagrees with BruteForce"
                    );
                }
            }
        }

        assert!(configs >= 48, "matrix shrank to {configs} configurations");
        for target in [1u32, 2, 3, 5] {
            assert!(
                seen_dims.contains(&target),
                "matrix never produced celldim {target}; saw {seen_dims:?}"
            );
        }
    }

    #[test]
    fn cross_query_matches_the_oracle_over_the_matrix() {
        for (name, h) in cells() {
            for pbc in PBCS {
                let bx = SimBox::new(h.clone(), array![0.0, 0.0, 0.0], pbc).expect("box");
                let refs = points(&bx, 0x5EED);
                let query = points(&bx, 0xC0FFEE);

                for cutoff in cutoffs(&bx) {
                    let tag = format!("{name} pbc={pbc:?} cutoff={cutoff:.3}");
                    let want = oracle_cross(&query, &refs, &bx, cutoff);

                    let nq = NeighborQuery::new(&bx, refs.view(), cutoff);
                    let res = nq.query(query.view());
                    let got: BTreeSet<(u32, u32)> = (0..res.n_pairs())
                        .map(|k| (res.query_point_indices()[k], res.point_indices()[k]))
                        .collect();

                    assert_eq!(got.len(), res.n_pairs(), "{tag}: duplicate query hit");
                    assert_eq!(got, want, "{tag}: cross-query set");
                }
            }
        }
    }
}
