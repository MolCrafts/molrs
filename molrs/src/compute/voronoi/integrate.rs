//! Voronoi integration of a volumetric electron density into per-molecule
//! electromagnetic moments (charge + dipole).
//!
//! Ported from the reference implementation's Voronoi charge/dipole gathering (`CalcVoronoiCharges` /
//! dipole accumulation in `src/gather.cpp`), which assigns each density grid
//! point to its enclosing radical-Voronoi cell and sums the electronic charge
//! `q = −∫ρ dV` and dipole `μ = −∫ρ (r − r_ref) dV` per cell, then per molecule.
//! Cube-frame Bohr/atomic-unit conventions follow `src/bqb_cubeframe.cpp`.
//!
//! # Definitions (Thomas, Brehm, Kirchner, *PCCP* 2015, 17, 3207)
//!
//! With electron (number) density `ρ ≥ 0`:
//! - cell electronic population `Nᵃ = ∫_cell ρ dV` (electrons, ≥ 0);
//! - molecular charge `Q_m = Σ_{a∈m} (Z_a − Nᵃ)`;
//! - molecular dipole `μ_m = Σ_a Z_a (r_a − r_ref) − Σ_{a∈m} ∫_cell ρ (r − r_ref) dV`,
//!   with `r_ref` the molecule's centre of nuclear charge (documented;
//!   origin-dependent for `Q_m ≠ 0`).
//!
//! # Cell assignment
//!
//! A point `x` belongs to the radical cell of generator `i` iff `i` minimises
//! the power distance `|x − x_i|² − R_i²` — this argmin **is** the radical
//! Voronoi partition (same cells [`RadicalVoronoi`](super::RadicalVoronoi)
//! builds geometrically), so the integrator reuses the generators + radii
//! directly. Exact ties break to the lowest index (deterministic). All
//! displacements use the orthorhombic minimum image, so a molecule straddling
//! the periodic boundary integrates correctly.
//!
//! # Units
//!
//! Gaussian-cube volumetric values are atomic units (`e/Bohr³`); positions and
//! voxel vectors are normalised to Å by the cube reader. [`DensityGrid::from_cube_frame`]
//! converts the density `e/Bohr³ → e/Å³` (divide by `a³`, `a = 0.529177… Å/Bohr`)
//! so `∫ρ dV` is a pure electron count and `μ` is in `e·Å`.

use molrs::spatial::region::simbox::SimBox;
use molrs::store::frame::Frame;
use molrs::types::F;
use ndarray::{Array2, ArrayView2};

use crate::compute::error::ComputeError;
use crate::compute::result::ComputeResult;

/// Bohr → Å (CODATA, matches the cube reader's constant).
pub const BOHR_TO_ANG: F = 0.529_177_210_67;

/// A volumetric scalar density on a regular (possibly sheared) grid, in molrs
/// units: positions Å, density `e/Å³`.
#[derive(Debug, Clone)]
pub struct DensityGrid {
    /// Grid origin (Å).
    pub origin: [F; 3],
    /// Voxel basis vectors `a0,a1,a2` (Å); grid point `(i,j,k)` sits at
    /// `origin + i·a0 + j·a1 + k·a2` (point-centred, cube convention).
    pub basis: [[F; 3]; 3],
    /// Grid dimensions `(nx,ny,nz)`.
    pub dims: [usize; 3],
    /// Density per voxel, `e/Å³`, row-major `((i·ny)+j)·nz + k` (cube order,
    /// z fastest).
    pub density: Vec<F>,
    /// Voxel volume `|det[a0 a1 a2]|` (Å³).
    pub dv: F,
}

impl DensityGrid {
    /// Build a grid directly from in-Å density values (`e/Å³`).
    pub fn new(origin: [F; 3], basis: [[F; 3]; 3], dims: [usize; 3], density: Vec<F>) -> Self {
        let dv = det3(basis).abs();
        DensityGrid {
            origin,
            basis,
            dims,
            density,
            dv,
        }
    }

    /// Extract a [`DensityGrid`] from a cube [`Frame`](molrs::Frame) (the output
    /// of `io::data::cube::read_cube`): grid block `"grid"`, density column
    /// `"density"`. The cube reader leaves the density in its native `e/Bohr³`;
    /// this converts it to `e/Å³` (÷ `a³`) so downstream integration yields
    /// electrons / `e·Å`.
    pub fn from_cube_frame(frame: &Frame) -> Result<Self, ComputeError> {
        let grid_block = frame
            .get("grid")
            .ok_or(ComputeError::MissingBlock { name: "grid" })?;
        // Grid dims come from the block's structural shape ([nx,ny,nz] set by
        // the cube reader); the density column itself is stored flat row-major.
        let shape = grid_block.shape();
        if shape.len() != 3 {
            return Err(ComputeError::BadShape {
                expected: "3-D grid [nx,ny,nz]".into(),
                got: format!("{shape:?}"),
            });
        }
        let dims = [shape[0], shape[1], shape[2]];
        let grid = grid_block
            .get_float("density")
            .ok_or(ComputeError::MissingColumn {
                block: "grid",
                col: "density",
            })?;
        let simbox = frame.simbox.as_ref().ok_or(ComputeError::MissingSimBox)?;
        // h column j = voxel_axis_j × dim_j (cube reader); recover the per-voxel
        // basis by dividing the cell column by the dimension.
        let h = simbox.h_view();
        let o = simbox.origin_view();
        let mut basis = [[0.0; 3]; 3];
        for j in 0..3 {
            let nj = dims[j].max(1) as F;
            for i in 0..3 {
                basis[j][i] = h[[i, j]] / nj;
            }
        }
        let origin = [o[0], o[1], o[2]];

        // Density: e/Bohr³ (cube native) → e/Å³.
        let bohr3 = BOHR_TO_ANG * BOHR_TO_ANG * BOHR_TO_ANG;
        let is_ang = frame
            .meta
            .get("cube_units")
            .map(|u| u == "angstrom")
            .unwrap_or(false);
        // The Gaussian-cube volumetric block is atomic-unit by convention even
        // when the geometry flag is Å; only skip the divide if a producer has
        // explicitly stored e/Å³ (signalled here by an `angstrom` density tag).
        let scale = if is_ang { 1.0 } else { 1.0 / bohr3 };
        let density: Vec<F> = grid.iter().map(|&v| v * scale).collect();

        Ok(DensityGrid::new(origin, basis, dims, density))
    }

    /// Voxel point position (Å) for flat index `m`.
    #[inline]
    fn point(&self, m: usize) -> [F; 3] {
        let (nx, ny, nz) = (self.dims[0], self.dims[1], self.dims[2]);
        debug_assert_eq!(self.density.len(), nx * ny * nz);
        let i = m / (ny * nz);
        let rem = m % (ny * nz);
        let j = rem / nz;
        let k = rem % nz;
        let (fi, fj, fk) = (i as F, j as F, k as F);
        [
            self.origin[0] + fi * self.basis[0][0] + fj * self.basis[1][0] + fk * self.basis[2][0],
            self.origin[1] + fi * self.basis[0][1] + fj * self.basis[1][1] + fk * self.basis[2][1],
            self.origin[2] + fi * self.basis[0][2] + fj * self.basis[1][2] + fk * self.basis[2][2],
        ]
    }
}

/// Per-molecule electromagnetic moments for one frame.
#[derive(Debug, Clone)]
pub struct MolecularMoments {
    /// Molecular charge `Q_m` (e), length `n_mol`.
    pub charges: Vec<F>,
    /// Molecular dipole `μ_m` (e·Å), shape `(n_mol, 3)`.
    pub dipoles: Array2<F>,
    /// Reference point used per molecule (centre of nuclear charge), `(n_mol, 3)`.
    pub references: Array2<F>,
}

impl ComputeResult for MolecularMoments {}

/// Voronoi electron-density integrator.
#[derive(Debug, Clone, Copy, Default)]
pub struct VoronoiIntegration;

impl VoronoiIntegration {
    /// Integrate `grid` over the radical-Voronoi cells of the generators
    /// (`positions`, `radii`) and combine with nuclear charges into per-molecule
    /// charge + dipole.
    ///
    /// `atomic_numbers[a]` is the nuclear charge `Z_a`; `atom_to_mol[a]` maps
    /// atom `a` to a molecule index in `0..n_mol`. The reference point per
    /// molecule is its centre of nuclear charge (min-image unwrapped).
    #[allow(clippy::too_many_arguments)]
    pub fn integrate(
        &self,
        positions: ArrayView2<F>,
        radii: &[F],
        atomic_numbers: &[i32],
        atom_to_mol: &[usize],
        n_mol: usize,
        grid: &DensityGrid,
        simbox: &SimBox,
    ) -> Result<MolecularMoments, ComputeError> {
        let n = positions.nrows();
        if positions.ncols() != 3 {
            return Err(ComputeError::DimensionMismatch {
                expected: 3,
                got: positions.ncols(),
                what: "positions columns",
            });
        }
        for (name, len) in [
            ("radii", radii.len()),
            ("atomic_numbers", atomic_numbers.len()),
            ("atom_to_mol", atom_to_mol.len()),
        ] {
            if len != n {
                return Err(ComputeError::DimensionMismatch {
                    expected: n,
                    got: len,
                    what: leak(name),
                });
            }
        }
        if atom_to_mol.iter().any(|&m| m >= n_mol) {
            return Err(ComputeError::OutOfRange {
                field: "atom_to_mol",
                value: "molecule index >= n_mol".into(),
            });
        }
        let l = simbox.lengths();
        let lbox = [l[0], l[1], l[2]];

        // Hoist the generator positions into a contiguous `[F; 3]` buffer and
        // precompute the squared radii once — Pass B's voxel loop reads both
        // millions of times, so paying the ndarray indexing / `r·r` per voxel is
        // pure overhead.
        let gens: Vec<[F; 3]> = (0..n)
            .map(|a| [positions[[a, 0]], positions[[a, 1]], positions[[a, 2]]])
            .collect();
        let radii_sq: Vec<F> = radii.iter().map(|&r| r * r).collect();
        let genpos = |a: usize| gens[a];

        // --- Pass A: per-molecule reference = centre of nuclear charge,
        // built on atoms unwrapped to each molecule's first-seen atom. ---
        let mut anchor: Vec<Option<[F; 3]>> = vec![None; n_mol];
        let mut ref_num = vec![[0.0_f64; 3]; n_mol]; // Σ Z (r unwrapped)
        let mut ref_den = vec![0.0_f64; n_mol]; // Σ Z
        for a in 0..n {
            let m = atom_to_mol[a];
            let ra = genpos(a);
            let anc = *anchor[m].get_or_insert(ra);
            let ru = unwrap(ra, anc, lbox);
            let z = atomic_numbers[a] as F;
            for d in 0..3 {
                ref_num[m][d] += z * ru[d];
            }
            ref_den[m] += z;
        }
        let mut references = Array2::<F>::zeros((n_mol, 3));
        for m in 0..n_mol {
            let anc = anchor[m].unwrap_or([0.0; 3]);
            for d in 0..3 {
                references[[m, d]] = if ref_den[m].abs() > 0.0 {
                    ref_num[m][d] / ref_den[m]
                } else {
                    anc[d] // chargeless (all Z=0) molecule: fall back to anchor
                };
            }
        }

        // --- Nuclear contributions to charge + dipole. ---
        let mut charges = vec![0.0_f64; n_mol];
        let mut dipoles = Array2::<F>::zeros((n_mol, 3));
        for a in 0..n {
            let m = atom_to_mol[a];
            let z = atomic_numbers[a] as F;
            charges[m] += z;
            let rref = [references[[m, 0]], references[[m, 1]], references[[m, 2]]];
            let d = min_image(sub(genpos(a), rref), lbox);
            for c in 0..3 {
                dipoles[[m, c]] += z * d[c];
            }
        }

        // --- Pass B: assign each voxel to its radical cell (power-distance
        // argmin), accumulate electronic population + dipole on the owning
        // molecule. ---
        let n_voxels = grid.dims[0] * grid.dims[1] * grid.dims[2];
        if grid.density.len() != n_voxels {
            return Err(ComputeError::DimensionMismatch {
                expected: n_voxels,
                got: grid.density.len(),
                what: "density length vs grid dims",
            });
        }

        // Build a uniform periodic cell list over the generators so Pass B gathers
        // only nearby candidates instead of scanning all `n` generators per voxel.
        // `w_max` bounds every generator's radius² and drives the pruning guard.
        let w_max = radii_sq.iter().copied().fold(0.0_f64, f64::max);
        let r_max = radii.iter().copied().fold(0.0_f64, |m, r| m.max(r.abs()));
        let vol = lbox[0] * lbox[1] * lbox[2];
        let mean_spacing = if vol > 0.0 && n > 0 {
            (vol / n as F).cbrt()
        } else {
            0.0
        };
        // Cutoff generous enough that the 27-cell gather almost always already
        // contains the winning generator (guard passes → no fallback): a few mean
        // spacings plus the radius spread. Correctness never depends on this value
        // — the guard + brute-force fallback keep every result bit-identical to the
        // exhaustive search regardless of `r_cut`.
        let r_cut = {
            let base = 3.0 * mean_spacing + 2.0 * r_max;
            if base.is_finite() && base > 0.0 {
                base
            } else {
                lbox.iter().copied().fold(0.0_f64, f64::max).max(1.0)
            }
        };
        let cell_list = CellList::build(&gens, lbox, r_cut);

        // Electronic contribution of one voxel, deposited into per-molecule
        // charge / dipole accumulators. Each voxel is independent, so this is the
        // body of both the serial loop and the rayon fold below. The nearest
        // radical site is the power-distance argmin with a low-index tie-break —
        // resolved by a spatial-index gather (`CellList::nearest`) that is proven
        // bit-identical to the original exhaustive search via the pruning guard,
        // falling back to brute force when the guard cannot certify the winner.
        let deposit = |m: usize, charge: &mut [F], dip: &mut Array2<F>| {
            let rho = grid.density[m];
            if rho == 0.0 {
                return;
            }
            let x = grid.point(m);
            let best = cell_list.nearest(x, &gens, &radii_sq, w_max, r_cut, n);
            let mol = atom_to_mol[best];
            let n_elec = rho * grid.dv; // electrons in this voxel
            charge[mol] -= n_elec;
            let rref = [
                references[[mol, 0]],
                references[[mol, 1]],
                references[[mol, 2]],
            ];
            let disp = min_image(sub(x, rref), lbox);
            for c in 0..3 {
                dip[[mol, c]] -= n_elec * disp[c];
            }
        };

        // Pass B: accumulate the electronic charge/dipole per molecule. Nuclear
        // terms are already in `charges`/`dipoles`; the electronic sum lands in
        // its own accumulators (fanned out over voxels with rayon, then merged)
        // and is added on afterwards, so nuclear-then-electronic ordering is
        // preserved and only the electronic sum is reassociated.
        let zeros = || (vec![0.0_f64; n_mol], Array2::<F>::zeros((n_mol, 3)));
        #[cfg(feature = "rayon")]
        let (elec_charge, elec_dipole) = {
            use rayon::prelude::*;
            (0..n_voxels)
                .into_par_iter()
                .fold(zeros, |(mut c, mut d), m| {
                    deposit(m, &mut c, &mut d);
                    (c, d)
                })
                .reduce(zeros, |(mut ca, mut da), (cb, db)| {
                    for (x, y) in ca.iter_mut().zip(cb.iter()) {
                        *x += *y;
                    }
                    da += &db;
                    (ca, da)
                })
        };
        #[cfg(not(feature = "rayon"))]
        let (elec_charge, elec_dipole) = {
            let (mut c, mut d) = zeros();
            for m in 0..n_voxels {
                deposit(m, &mut c, &mut d);
            }
            (c, d)
        };

        for m in 0..n_mol {
            charges[m] += elec_charge[m];
            for c in 0..3 {
                dipoles[[m, c]] += elec_dipole[[m, c]];
            }
        }

        Ok(MolecularMoments {
            charges,
            dipoles,
            references,
        })
    }
}

// --- small orthorhombic vector helpers (the voronoi module is ortho-only) ---
//
// ponytail: specialized orthorhombic MIC over a precomputed box-length array,
// called once per voxel (millions of times) in the hot Pass-B loop below;
// `compute::util::mic_disp` is the general (box-kind-resolving) path used
// elsewhere. Kept local on purpose to avoid per-iteration box dispatch.

#[inline]
fn sub(a: [F; 3], b: [F; 3]) -> [F; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

/// Minimum-image displacement for an orthorhombic box.
#[inline]
fn min_image(mut d: [F; 3], l: [F; 3]) -> [F; 3] {
    for c in 0..3 {
        if l[c] > 0.0 {
            d[c] -= l[c] * (d[c] / l[c]).round();
        }
    }
    d
}

/// Unwrap `r` to be the min-image-closest copy to `anchor`.
#[inline]
fn unwrap(r: [F; 3], anchor: [F; 3], l: [F; 3]) -> [F; 3] {
    let d = min_image(sub(r, anchor), l);
    [anchor[0] + d[0], anchor[1] + d[1], anchor[2] + d[2]]
}

/// Radical (power / Laguerre) distance of point `x` to generator `g` with
/// squared radius `r_sq`, using the exact same orthorhombic min-image arithmetic
/// as the original Pass-B loop. Keeping this the single source of the formula is
/// what makes the spatial-index gather and the brute-force fallback yield
/// bit-identical `pow` values.
#[inline]
fn power_dist(x: [F; 3], g: [F; 3], r_sq: F, l: [F; 3]) -> F {
    let d = min_image(sub(x, g), l);
    d[0] * d[0] + d[1] * d[1] + d[2] * d[2] - r_sq
}

// --- uniform periodic cell list over the generators (Pass-B acceleration) ---
//
// Replaces the O(n_voxels · n_generators) argmin with a gather over the voxel's
// own cell + the 26 periodic neighbours, plus an exact pruning guard. Any
// generator outside the gathered region has min-image distance ≥ `r_cut` (cell
// edge ≥ `r_cut`), hence power distance ≥ `r_cut² − w_max`; if a candidate
// already beats that bound the gathered argmin is provably the global argmin.
// When the guard cannot certify it (empty region, or a candidate not yet inside
// the bound) Pass B falls back to the exhaustive search, so results are
// bit-identical to brute force for every voxel, whatever `r_cut` is.

/// Cell index of coordinate `p` along one axis (periodic, wrapped to `[0, l)`).
#[inline]
fn cell_index_1d(p: F, l: F, edge: F, ncell: usize) -> usize {
    if l <= 0.0 || ncell <= 1 {
        return 0;
    }
    let w = p - l * (p / l).floor(); // wrap into [0, l)
    let c = (w / edge).floor();
    if !c.is_finite() || c < 0.0 {
        0
    } else {
        (c as usize).min(ncell - 1)
    }
}

/// The (deduplicated) periodic neighbour cell indices along one axis: the cell
/// itself plus its two neighbours, collapsing to fewer when `ncell < 3` so no
/// cell is ever visited twice.
#[inline]
fn axis_cells(ic: usize, ncell: usize) -> ([usize; 3], usize) {
    match ncell {
        0 | 1 => ([0, 0, 0], 1),
        2 => ([ic, 1 - ic, 0], 2),
        _ => {
            let lo = if ic == 0 { ncell - 1 } else { ic - 1 };
            let hi = if ic + 1 == ncell { 0 } else { ic + 1 };
            ([lo, ic, hi], 3)
        }
    }
}

/// Uniform periodic cell list over the generator positions, built once and
/// shared read-only across the rayon-over-voxels Pass B.
struct CellList {
    /// Cells per axis (`≥ 1`).
    ncell: [usize; 3],
    /// Cell edge length per axis (`= lbox / ncell ≥ r_cut`; `∞` for a degenerate
    /// axis with `lbox ≤ 0`).
    cell_edge: [F; 3],
    /// Orthorhombic box lengths (same array used by the min-image compare).
    lbox: [F; 3],
    /// CSR offsets: cell `c` owns `cell_gens[cell_start[c]..cell_start[c + 1]]`.
    cell_start: Vec<usize>,
    /// Generator indices grouped by owning cell, ascending within each cell.
    cell_gens: Vec<usize>,
}

impl CellList {
    /// Bin `gens` into a periodic cell list with edge `≥ r_cut` on each periodic
    /// axis (via a stable counting sort, so generator indices stay ascending
    /// within a cell).
    fn build(gens: &[[F; 3]], lbox: [F; 3], r_cut: F) -> Self {
        let mut ncell = [1usize; 3];
        let mut cell_edge = [F::INFINITY; 3];
        for d in 0..3 {
            if lbox[d] > 0.0 {
                let nd = (lbox[d] / r_cut).floor();
                let nd = if nd.is_finite() && nd >= 1.0 {
                    nd as usize
                } else {
                    1
                };
                ncell[d] = nd.max(1);
                cell_edge[d] = lbox[d] / ncell[d] as F;
            }
        }
        let ncells = ncell[0] * ncell[1] * ncell[2];
        let cell_of = |p: [F; 3]| -> usize {
            let ix = cell_index_1d(p[0], lbox[0], cell_edge[0], ncell[0]);
            let iy = cell_index_1d(p[1], lbox[1], cell_edge[1], ncell[1]);
            let iz = cell_index_1d(p[2], lbox[2], cell_edge[2], ncell[2]);
            (ix * ncell[1] + iy) * ncell[2] + iz
        };
        let mut cell_start = vec![0usize; ncells + 1];
        for g in gens {
            cell_start[cell_of(*g) + 1] += 1;
        }
        for c in 0..ncells {
            cell_start[c + 1] += cell_start[c];
        }
        let mut cursor = cell_start.clone();
        let mut cell_gens = vec![0usize; gens.len()];
        for (a, g) in gens.iter().enumerate() {
            let c = cell_of(*g);
            cell_gens[cursor[c]] = a;
            cursor[c] += 1;
        }
        CellList {
            ncell,
            cell_edge,
            lbox,
            cell_start,
            cell_gens,
        }
    }

    /// Index of the generator minimising the power distance to `x`, identical
    /// (value and lowest-index tie-break) to the exhaustive `0..n` argmin.
    fn nearest(
        &self,
        x: [F; 3],
        gens: &[[F; 3]],
        radii_sq: &[F],
        w_max: F,
        r_cut: F,
        n: usize,
    ) -> usize {
        let icx = cell_index_1d(x[0], self.lbox[0], self.cell_edge[0], self.ncell[0]);
        let icy = cell_index_1d(x[1], self.lbox[1], self.cell_edge[1], self.ncell[1]);
        let icz = cell_index_1d(x[2], self.lbox[2], self.cell_edge[2], self.ncell[2]);
        let (ax, nax) = axis_cells(icx, self.ncell[0]);
        let (ay, nay) = axis_cells(icy, self.ncell[1]);
        let (az, naz) = axis_cells(icz, self.ncell[2]);

        let mut best = usize::MAX;
        let mut best_pow = F::INFINITY;
        let mut any = false;
        for &cx in &ax[..nax] {
            for &cy in &ay[..nay] {
                for &cz in &az[..naz] {
                    let c = (cx * self.ncell[1] + cy) * self.ncell[2] + cz;
                    for &a in &self.cell_gens[self.cell_start[c]..self.cell_start[c + 1]] {
                        any = true;
                        let pow = power_dist(x, gens[a], radii_sq[a], self.lbox);
                        // argmin with a lowest-index tie-break, tolerant of the
                        // grouped (non-index-order) candidate visiting: strictly
                        // smaller wins, exact ties go to the lower index. Uses only
                        // `<`/`<=` on the power distance so it matches the brute
                        // force's exact selection without a float `==`.
                        if pow < best_pow || (pow <= best_pow && a < best) {
                            best_pow = pow;
                            best = a;
                        }
                    }
                }
            }
        }

        // Pruning guard: any generator not gathered has min-image distance ≥
        // r_cut, so power ≥ r_cut² − w_max. If a candidate already beats that
        // (strict), no outside generator can tie or win → the gathered argmin is
        // the global argmin.
        if any && r_cut * r_cut - w_max > best_pow {
            return best;
        }

        // Fallback: exhaustive argmin over all generators — byte-for-byte the
        // original single-threaded search (same arithmetic, same strict-`<`
        // lowest-index tie-break).
        let mut fb_best = 0usize;
        let mut fb_pow = F::INFINITY;
        for a in 0..n {
            let pow = power_dist(x, gens[a], radii_sq[a], self.lbox);
            if pow < fb_pow {
                fb_pow = pow;
                fb_best = a;
            }
        }
        fb_best
    }
}

#[inline]
fn det3(m: [[F; 3]; 3]) -> F {
    m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
}

/// `ComputeError::DimensionMismatch::what` wants a `&'static str`; the error is
/// a programming bug (caller passed mismatched slices), so leaking the few
/// possible labels is acceptable and keeps the message specific.
fn leak(s: &str) -> &'static str {
    match s {
        "radii" => "radii length",
        "atomic_numbers" => "atomic_numbers length",
        "atom_to_mol" => "atom_to_mol length",
        _ => "input length",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn det3_and_point_indexing() {
        let g = DensityGrid::new(
            [0.0, 0.0, 0.0],
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            [2, 2, 2],
            vec![0.0; 8],
        );
        assert!((g.dv - 1.0).abs() < 1e-12);
        // m = ((i*ny)+j)*nz + k ; m=5 → i=1,j=0,k=1
        assert_eq!(g.point(5), [1.0, 0.0, 1.0]);
    }

    #[test]
    fn min_image_wraps_half_box() {
        let d = min_image([0.9, 0.0, 0.0], [1.0, 1.0, 1.0]);
        assert!((d[0] - (-0.1)).abs() < 1e-12);
    }

    /// The spatial-index nearest-generator search must return **exactly** the
    /// same index (value + lowest-index tie-break) as the exhaustive argmin for
    /// every query point — with the pruning guard active (`ncell > 1`), when it
    /// must fall back (tiny `r_cut`), and in the single-cell case (`ncell == 1`).
    #[test]
    fn cell_list_nearest_matches_brute_force() {
        let lbox = [12.0_f64, 10.0, 8.0];
        let n = 200usize;
        // Deterministic LCG → f64 in [0, 1); no external rng dependency.
        let mut state = 0x2545_F491_4F6C_DD1Du64;
        let mut rng = || {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            ((state >> 11) as f64) / ((1u64 << 53) as f64)
        };
        let gens: Vec<[F; 3]> = (0..n)
            .map(|_| [rng() * lbox[0], rng() * lbox[1], rng() * lbox[2]])
            .collect();
        let radii_sq: Vec<F> = (0..n).map(|_| (rng() * 1.5).powi(2)).collect();
        let w_max = radii_sq.iter().copied().fold(0.0_f64, f64::max);

        // Reference: byte-for-byte the exhaustive Pass-B search.
        let brute = |x: [F; 3]| -> usize {
            let mut best = 0usize;
            let mut best_pow = F::INFINITY;
            for a in 0..n {
                let pow = power_dist(x, gens[a], radii_sq[a], lbox);
                if pow < best_pow {
                    best_pow = pow;
                    best = a;
                }
            }
            best
        };

        // Query points spanning (and spilling outside) the box to exercise wrap.
        let queries: Vec<[F; 3]> = (0..600)
            .map(|_| {
                [
                    rng() * lbox[0] * 1.3 - 1.0,
                    rng() * lbox[1] * 1.3 - 1.0,
                    rng() * lbox[2] * 1.3 - 1.0,
                ]
            })
            .collect();

        // r_cut = 0.6 → tiny cells, frequent fallback; 3.0/5.0 → guard prunes;
        // 50.0 → single cell (ncell == 1).
        for &r_cut in &[0.6_f64, 3.0, 5.0, 50.0] {
            let cl = CellList::build(&gens, lbox, r_cut);
            for &x in &queries {
                assert_eq!(
                    cl.nearest(x, &gens, &radii_sq, w_max, r_cut, n),
                    brute(x),
                    "spatial index diverged from brute force at r_cut={r_cut}, x={x:?}"
                );
            }
        }
    }
}
