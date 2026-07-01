//! Native periodic radical (Laguerre) Voronoi tessellation.
//!
//! Builds one [`VoronoiCells`] entry per generator by clipping a box-sized
//! convex polyhedron against the radical plane of nearby periodic neighbours
//! ([`super::cell::Poly::clip`]). This is the cell-by-cell strategy of voro++
//! (`container_periodic::compute_cell`, `src/v_container_prd.cpp`, driven as in
//! TRAVIS `vorowrapper.cpp`), with the radical/power plane offset from
//! `radius_poly` (`src/v_rad_option.h`).
//!
//! # Candidate search — O(N) at constant density
//! voro++ only clips a cell against particles near enough that their cut plane
//! can still reach the cell; it walks its container blocks outward and stops
//! once no closer plane can cut. molrs mirrors this: a one-pass cell-list grid
//! (built once, O(N)) yields each generator's candidate neighbours — including
//! periodic images — in increasing distance, and the cell is clipped in that
//! order until the **termination bound** is reached:
//!
//! A neighbour `j` at centre distance `d` cuts the cell along the radical plane
//! at signed distance `p_j = (d² + Rᵢ² − Rⱼ²) / (2d)` from the generator. If
//! `p_j` exceeds the cell's farthest vertex radius `R_max`, the plane lies
//! wholly outside the cell and cannot cut it. Taking the largest radius in the
//! system `R_⋆` as the conservative worst case (`Rⱼ = R_⋆`), no neighbour past
//! `d_stop = R_max + √(R_max² + R_⋆² − Rᵢ²)` can ever cut the cell, so the
//! distance-ordered walk stops there. For equal radii this is the familiar
//! `d > 2·R_max` Voronoi cutoff. `R_max` only shrinks as clipping proceeds, so
//! the bound is monotone and safe.
//!
//! If the grid cutoff turns out too small to reach `d_stop` for a cell (sparse
//! regions, large voids, a lone atom whose cell is the whole box), that cell
//! falls back to an exact exhaustive growing-shell search — the same
//! plane-cut semantics, so the result is identical; only the *candidate set*
//! differs, and the fast path is accepted only once it provably contains every
//! cutting plane. Correctness is pinned by `Σ volume == box volume` and the
//! analytic radical-plane / single-atom tests.
//!
//! # Radical plane
//! Between generators i, j at displacement `r = x_j − x_i` with radii `Rᵢ`,
//! `Rⱼ`, the radical plane is `|x|² − Rᵢ² = |x − r|² − Rⱼ²`, i.e.
//! `x·r = (|r|² + Rᵢ² − Rⱼ²)/2`; the cell keeps the generator side
//! (`x·r ≤ off`). Equal radii reduce to the plain Voronoi bisector at `r/2`.

use molrs::spatial::region::simbox::SimBox;
use molrs::types::F;
use ndarray::ArrayView2;

use super::cell::{BOUNDARY, Face, Poly, VoronoiCells};
use crate::compute::error::ComputeError;

/// Builder for the periodic radical-Voronoi tessellation. Orthorhombic boxes
/// only (triclinic is out of scope; see the spec).
#[derive(Debug, Clone, Copy, Default)]
pub struct RadicalVoronoi;

/// Largest periodic shell searched by the exhaustive fallback before accepting
/// a cell (a residual box face at shell 3 means the configuration is
/// pathologically sparse).
const MAX_SHELL: i32 = 3;
/// Area below which a residual boundary face is treated as numerical dust.
const FACE_EPS: F = 1e-9;

/// Conservative termination radius: no neighbour farther than this can still
/// cut a cell whose farthest vertex is at `rmax`. `r_star2` is the largest
/// squared radius in the system (worst-case neighbour), `ri2` the generator's.
#[inline]
fn d_stop(rmax: F, ri2: F, r_star2: F) -> F {
    let disc = rmax * rmax + r_star2 - ri2;
    rmax + disc.max(0.0).sqrt()
}

impl RadicalVoronoi {
    /// Tessellate `positions` (N×3, Å) with per-atom `radii` (length N; pass
    /// all-zero for a plain Voronoi diagram) in the periodic orthorhombic
    /// `simbox`.
    pub fn build(
        &self,
        positions: ArrayView2<F>,
        radii: &[F],
        simbox: &SimBox,
    ) -> Result<VoronoiCells, ComputeError> {
        let n = positions.nrows();
        if positions.ncols() != 3 {
            return Err(ComputeError::DimensionMismatch {
                expected: 3,
                got: positions.ncols(),
                what: "positions columns",
            });
        }
        if radii.len() != n {
            return Err(ComputeError::DimensionMismatch {
                expected: n,
                got: radii.len(),
                what: "radii length",
            });
        }
        let lv = simbox.lengths();
        let l = [lv[0], lv[1], lv[2]];
        if l[0] <= 0.0 || l[1] <= 0.0 || l[2] <= 0.0 {
            return Err(ComputeError::OutOfRange {
                field: "RadicalVoronoi::box_lengths",
                value: format!("{},{},{}", l[0], l[1], l[2]),
            });
        }

        if n == 0 {
            return Ok(VoronoiCells {
                volumes: Vec::new(),
                faces: Vec::new(),
            });
        }

        let r_star2 = radii.iter().map(|&r| r * r).fold(0.0, F::max);

        // Grid cutoff: a few mean spacings plus the radius span comfortably
        // bounds a liquid-like Voronoi cell; cells that need more fall back.
        let mean_spacing = (l[0] * l[1] * l[2] / n as F).cbrt();
        let r_cut = 4.0 * mean_spacing + 2.0 * r_star2.sqrt();
        let grid = CellGrid::new(positions, &l, r_cut);

        let mut volumes = Vec::with_capacity(n);
        let mut faces = Vec::with_capacity(n);
        let mut cand: Vec<(F, [F; 3], usize)> = Vec::new();

        for i in 0..n {
            let gi = [positions[[i, 0]], positions[[i, 1]], positions[[i, 2]]];
            let ri2 = radii[i] * radii[i];

            match self.build_cell_fast(&grid, &gi, ri2, r_star2, radii, &l, r_cut, &mut cand) {
                Some((vol, cf)) => {
                    volumes.push(vol);
                    faces.push(cf);
                }
                None => {
                    let (vol, cf) = self.build_cell_exhaustive(positions, radii, i, &gi, ri2, &l);
                    volumes.push(vol);
                    faces.push(cf);
                }
            }
        }

        Ok(VoronoiCells { volumes, faces })
    }

    /// Fast path: clip against grid candidates in increasing distance, stopping
    /// at the termination bound. Returns `None` (→ exhaustive fallback) if the
    /// grid cutoff `r_cut` cannot be proven to contain every cutting plane.
    #[allow(clippy::too_many_arguments)]
    fn build_cell_fast(
        &self,
        grid: &CellGrid,
        gi: &[F; 3],
        ri2: F,
        r_star2: F,
        radii: &[F],
        l: &[F; 3],
        r_cut: F,
        cand: &mut Vec<(F, [F; 3], usize)>,
    ) -> Option<(F, Vec<Face>)> {
        grid.collect(gi, r_cut, l, cand);
        cand.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

        let mut poly = Poly::box_cell(l[0], l[1], l[2]);
        for &(d2, dr, j) in cand.iter() {
            let d = d2.sqrt();
            if d > d_stop(poly.max_vertex_dist(), ri2, r_star2) {
                break;
            }
            let rj2 = radii[j] * radii[j];
            let off = 0.5 * (d2 + ri2 - rj2);
            poly.clip(dr, off, j as i64);
        }

        // Accept only if r_cut provably reached the termination bound, i.e. no
        // unseen neighbour beyond r_cut could still cut this cell.
        if r_cut + 1e-12 >= d_stop(poly.max_vertex_dist(), ri2, r_star2) {
            Some((poly.volume(), poly.cell_faces()))
        } else {
            None
        }
    }

    /// Exact exhaustive fallback: clip against every atom across a growing
    /// periodic shell until the cell is bounded by real neighbours. Used only
    /// for the rare cells the grid cutoff cannot prove complete.
    fn build_cell_exhaustive(
        &self,
        positions: ArrayView2<F>,
        radii: &[F],
        i: usize,
        gi: &[F; 3],
        ri2: F,
        l: &[F; 3],
    ) -> (F, Vec<Face>) {
        let n = positions.nrows();
        let (lx, ly, lz) = (l[0], l[1], l[2]);
        let mut shell = 1;
        loop {
            let mut poly = Poly::box_cell(lx, ly, lz);
            for j in 0..n {
                let rj2 = radii[j] * radii[j];
                for sx in -shell..=shell {
                    for sy in -shell..=shell {
                        for sz in -shell..=shell {
                            if i == j && sx == 0 && sy == 0 && sz == 0 {
                                continue;
                            }
                            let dr = [
                                positions[[j, 0]] + sx as F * lx - gi[0],
                                positions[[j, 1]] + sy as F * ly - gi[1],
                                positions[[j, 2]] + sz as F * lz - gi[2],
                            ];
                            let rr = dr[0] * dr[0] + dr[1] * dr[1] + dr[2] * dr[2];
                            if rr < 1e-18 {
                                continue;
                            }
                            let off = 0.5 * (rr + ri2 - rj2);
                            poly.clip(dr, off, j as i64);
                        }
                    }
                }
            }
            let cf = poly.cell_faces();
            let unbounded = cf
                .iter()
                .any(|f| f.neighbor == BOUNDARY && f.area > FACE_EPS);
            if !unbounded || shell >= MAX_SHELL {
                return (poly.volume(), cf);
            }
            shell += 1;
        }
    }
}

/// A uniform cell-list over the orthorhombic periodic box: each generator's
/// candidate neighbours (with periodic images) are gathered by walking the few
/// surrounding grid cells, giving O(1) lookup at constant density.
struct CellGrid {
    /// Cells per axis (≥ 1).
    nb: [usize; 3],
    /// Box lengths.
    l: [F; 3],
    /// Wrapped atom positions in `[0, L)`.
    wrapped: Vec<[F; 3]>,
    /// Atom indices per grid bin, row-major over `nb`.
    bins: Vec<Vec<u32>>,
}

impl CellGrid {
    fn new(positions: ArrayView2<F>, l: &[F; 3], r_cut: F) -> Self {
        let n = positions.nrows();
        let nb = [
            ((l[0] / r_cut).floor() as usize).max(1),
            ((l[1] / r_cut).floor() as usize).max(1),
            ((l[2] / r_cut).floor() as usize).max(1),
        ];
        let mut bins = vec![Vec::new(); nb[0] * nb[1] * nb[2]];
        let mut wrapped = Vec::with_capacity(n);
        for j in 0..n {
            let mut w = [0.0; 3];
            let mut cell = [0usize; 3];
            for (a, &la) in l.iter().enumerate() {
                let mut p = positions[[j, a]].rem_euclid(la);
                // guard the rem_euclid == L edge from rounding
                if p >= la {
                    p -= la;
                }
                w[a] = p;
                let c = ((p / la) * nb[a] as F).floor() as usize;
                cell[a] = c.min(nb[a] - 1);
            }
            wrapped.push(w);
            let idx = (cell[0] * nb[1] + cell[1]) * nb[2] + cell[2];
            bins[idx].push(j as u32);
        }
        CellGrid {
            nb,
            l: *l,
            wrapped,
            bins,
        }
    }

    #[inline]
    fn bin_of(&self, gi_wrapped: &[F; 3]) -> [i64; 3] {
        let mut c = [0i64; 3];
        for a in 0..3 {
            let ci = ((gi_wrapped[a] / self.l[a]) * self.nb[a] as F).floor() as i64;
            c[a] = ci.clamp(0, self.nb[a] as i64 - 1);
        }
        c
    }

    /// Collect every atom image within `r_cut` of generator position `gi`
    /// (wrapped internally) as `(d², displacement, atom_index)`, excluding the
    /// generator's own zero image. `out` is cleared first.
    fn collect(&self, gi: &[F; 3], r_cut: F, l: &[F; 3], out: &mut Vec<(F, [F; 3], usize)>) {
        out.clear();
        let giw = [
            gi[0].rem_euclid(l[0]),
            gi[1].rem_euclid(l[1]),
            gi[2].rem_euclid(l[2]),
        ];
        let base = self.bin_of(&giw);
        let reach = [
            (r_cut / (self.l[0] / self.nb[0] as F)).ceil() as i64,
            (r_cut / (self.l[1] / self.nb[1] as F)).ceil() as i64,
            (r_cut / (self.l[2] / self.nb[2] as F)).ceil() as i64,
        ];
        let rc2 = r_cut * r_cut;
        for dcx in -reach[0]..=reach[0] {
            let (cx, ix) = wrap_cell(base[0] + dcx, self.nb[0]);
            let ox = ix as F * l[0];
            for dcy in -reach[1]..=reach[1] {
                let (cy, iy) = wrap_cell(base[1] + dcy, self.nb[1]);
                let oy = iy as F * l[1];
                for dcz in -reach[2]..=reach[2] {
                    let (cz, iz) = wrap_cell(base[2] + dcz, self.nb[2]);
                    let oz = iz as F * l[2];
                    let idx = (cx * self.nb[1] + cy) * self.nb[2] + cz;
                    for &ju in &self.bins[idx] {
                        let j = ju as usize;
                        let w = &self.wrapped[j];
                        let dr = [w[0] + ox - giw[0], w[1] + oy - giw[1], w[2] + oz - giw[2]];
                        let d2 = dr[0] * dr[0] + dr[1] * dr[1] + dr[2] * dr[2];
                        if d2 < 1e-18 || d2 > rc2 {
                            continue;
                        }
                        out.push((d2, dr, j));
                    }
                }
            }
        }
    }
}

/// Map a possibly-out-of-range cell index to `(wrapped_index, image_count)`
/// where `image_count` is the number of whole boxes wrapped (the periodic
/// image offset along that axis).
#[inline]
fn wrap_cell(raw: i64, nb: usize) -> (usize, i64) {
    let nb_i = nb as i64;
    let wrapped = raw.rem_euclid(nb_i);
    let image = (raw - wrapped) / nb_i;
    (wrapped as usize, image)
}
