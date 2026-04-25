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

use crate::neighbors::{NbListAlgo, NeighborList, PairVisitor, QueryMode};
use crate::region::simbox::SimBox;
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
    /// Number of cells along each axis `[nx, ny, nz]`.
    celldim: [u32; 3],
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
            celldim: [0; 3],
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
        mut callback: C,
    ) where
        C: FnMut(u32, F, [F; 3]),
    {
        let n_cells = (self.celldim[0] * self.celldim[1] * self.celldim[2]) as usize;
        if n_cells == 0 {
            return;
        }

        let query_cell = get_cell(bx, query_point, self.celldim);
        let qp = [query_point[0], query_point[1], query_point[2]];

        // Check the query cell itself + all 26 neighbor cells
        let pbc = bx.pbc();
        let mut buf = [0usize; 27];
        let n_all = stencil_all_into(query_cell, self.celldim, pbc, &mut buf);
        let all_cells = &buf[..n_all];
        for nc in std::iter::once(query_cell).chain(all_cells.iter().copied()) {
            let start = self.cell_start[nc] as usize;
            let end = self.cell_start[nc + 1] as usize;
            for si in start..end {
                let oj = self.sorted_idx[si];
                let pj = pos_at(&self.sorted_pos, si);
                let dr = bx.shortest_vector_raw(qp, pj);
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
        let pbc = self.bx.pbc();
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
                    let dr = self.bx.shortest_vector_raw(pi, pj);
                    let d2 = dr[0] * dr[0] + dr[1] * dr[1] + dr[2] * dr[2];
                    if d2 <= cutoff2 {
                        visitor.visit_pair(oi, self.sorted_idx[sj], d2, dr);
                    }
                }
            }

            // Forward neighbor cells (stack buffer, no alloc)
            let n_fwd = stencil_fwd_into(cell, self.celldim, pbc, &mut fwd_buf);
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
                        let dr = self.bx.shortest_vector_raw(pi, pj);
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
    /// Counting sort particles by cell index.
    ///
    /// Sets up `celldim`, `bx`, `cell_start`, `sorted_idx`, `sorted_pos`,
    /// and `occupied_cells`. Does NOT compute pairs.
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

        self.celldim = celldim;
        self.bx = bx.clone();

        // 1) Compute cell per particle, count per cell.
        //    `cell_of[i]` holds the final cell index for particle i; this buffer
        //    is never aliased with `sorted_idx` (which gets the reordering).
        self.cell_start.clear();
        self.cell_start.resize(n_cells + 1, 0);
        self.cell_of.resize(n_points, 0);
        for i in 0..n_points {
            let cell = get_cell(bx, points.row(i), celldim);
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
            self.sorted_pos[base] = points[[i, 0]];
            self.sorted_pos[base + 1] = points[[i, 1]];
            self.sorted_pos[base + 2] = points[[i, 2]];
        }
    }

    /// Serial half-shell pair search over occupied cells only.
    #[cfg(not(feature = "rayon"))]
    fn compute_pairs_serial(&mut self) {
        let cutoff2 = self.cutoff * self.cutoff;
        let pbc = self.bx.pbc();
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
                    let dr = self.bx.shortest_vector_raw(pi, pj);
                    let d2 = dr[0] * dr[0] + dr[1] * dr[1] + dr[2] * dr[2];
                    if d2 <= cutoff2 {
                        self.result.push(oi, self.sorted_idx[sj], d2, dr);
                    }
                }
            }

            // Forward neighbor cells (stack buffer — no alloc)
            let n_fwd = stencil_fwd_into(cell, self.celldim, pbc, &mut fwd_buf);
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
                        let dr = self.bx.shortest_vector_raw(pi, pj);
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
        let pbc = self.bx.pbc();

        let cell_start = &self.cell_start;
        let sorted_idx = &self.sorted_idx;
        let sorted_pos = &self.sorted_pos;
        let bx = &self.bx;
        let celldim = self.celldim;

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
                        let dr = bx.shortest_vector_raw(pi, pj);
                        let d2 = dr[0] * dr[0] + dr[1] * dr[1] + dr[2] * dr[2];
                        if d2 <= cutoff2 {
                            acc.push(oi, sorted_idx[sj], d2, dr);
                        }
                    }
                }

                // Forward neighbor cells (stack buffer — no alloc).
                let mut fwd_buf = [0usize; 27];
                let n_fwd = stencil_fwd_into(cell, celldim, pbc, &mut fwd_buf);
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
                            let dr = bx.shortest_vector_raw(pi, pj);
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
        let pbc = self.bx.pbc();
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
                    let dr = self.bx.shortest_vector_raw(pi, pj);
                    let d2 = dr[0] * dr[0] + dr[1] * dr[1] + dr[2] * dr[2];
                    if d2 <= cutoff2 {
                        self.result.push(oi, self.sorted_idx[sj], d2, dr);
                    }
                }
            }

            let n_fwd = stencil_fwd_into(cell, self.celldim, pbc, &mut fwd_buf);
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
                        let dr = self.bx.shortest_vector_raw(pi, pj);
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

/// Collect unique neighbor cell indices into a caller-owned buffer, applying
/// the given filter. Returns the number of entries written.
///
/// With small grids + PBC, multiple stencil offsets can wrap to the same cell,
/// so we sort + dedup after collection. The buffer is sized 13 (half-shell) or
/// 27 (full shell) by callers.
#[inline]
fn collect_stencil_into(
    cell: usize,
    celldim: [u32; 3],
    pbc: [bool; 3],
    filter: impl Fn(usize) -> bool,
    out: &mut [usize],
) -> usize {
    let idx = cell as u32;
    let nxy = celldim[0] * celldim[1];
    let cx = (idx % nxy) % celldim[0];
    let cy = (idx % nxy) / celldim[0];
    let cz = idx / nxy;

    let (si, ei) = stencil_range(cx, celldim[0]);
    let (sj, ej) = stencil_range(cy, celldim[1]);
    let (sk, ek) = stencil_range(cz, celldim[2]);

    let mut len = 0usize;
    for nk in sk..=ek {
        if !pbc[2] && (nk < 0 || nk >= celldim[2] as i32) {
            continue;
        }
        for nj in sj..=ej {
            if !pbc[1] && (nj < 0 || nj >= celldim[1] as i32) {
                continue;
            }
            for ni in si..=ei {
                if !pbc[0] && (ni < 0 || ni >= celldim[0] as i32) {
                    continue;
                }
                let wi = wrap(ni, celldim[0]);
                let wj = wrap(nj, celldim[1]);
                let wk = wrap(nk, celldim[2]);
                let nc = (wk * nxy + wj * celldim[0] + wi) as usize;
                if filter(nc) {
                    out[len] = nc;
                    len += 1;
                }
            }
        }
    }
    out[..len].sort_unstable();
    // In-place dedup (like Vec::dedup but on a slice prefix).
    let mut w = 0usize;
    for r in 0..len {
        if w == 0 || out[r] != out[w - 1] {
            out[w] = out[r];
            w += 1;
        }
    }
    w
}

/// Forward-neighbor cells (half-shell) — `nc > cell`. Writes into `out`,
/// returns deduped count. Buffer sized at 27 because tiny grids with PBC
/// can produce duplicates from multiple wraps before dedup.
#[inline]
fn stencil_fwd_into(
    cell: usize,
    celldim: [u32; 3],
    pbc: [bool; 3],
    out: &mut [usize; 27],
) -> usize {
    collect_stencil_into(cell, celldim, pbc, |nc| nc > cell, out)
}

/// Full-shell neighbor cells — `nc != cell`. Writes into `out`, returns
/// deduped count.
#[inline]
fn stencil_all_into(
    cell: usize,
    celldim: [u32; 3],
    pbc: [bool; 3],
    out: &mut [usize; 27],
) -> usize {
    collect_stencil_into(cell, celldim, pbc, |nc| nc != cell, out)
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Map a position to its cell index using fractional coordinates.
#[inline(always)]
fn get_cell(bx: &SimBox, r: ndarray::ArrayView1<'_, F>, celldim: [u32; 3]) -> usize {
    let frac = bx.make_fractional_fast_arr(r);
    let cx = (frac[0] * celldim[0] as F).floor() as u32 % celldim[0];
    let cy = (frac[1] * celldim[1] as F).floor() as u32 % celldim[1];
    let cz = (frac[2] * celldim[2] as F).floor() as u32 % celldim[2];
    (cz * celldim[1] * celldim[0] + cy * celldim[0] + cx) as usize
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

        let mut bf = crate::neighbors::bruteforce::BruteForce::new(cutoff);
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

        let mut bf = crate::neighbors::bruteforce::BruteForce::new(cutoff);
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
}
