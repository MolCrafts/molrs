//! Voronoi integration of a volumetric electron density into per-molecule
//! electromagnetic moments (charge + dipole).
//!
//! Ported from TRAVIS's Voronoi charge/dipole gathering (`CalcVoronoiCharges` /
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

        let genpos = |a: usize| [positions[[a, 0]], positions[[a, 1]], positions[[a, 2]]];

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
        for m in 0..n_voxels {
            let rho = grid.density[m];
            if rho == 0.0 {
                continue;
            }
            let x = grid.point(m);
            // nearest radical site (power distance), deterministic tie-break.
            let mut best = 0usize;
            let mut best_pow = F::INFINITY;
            #[allow(clippy::needless_range_loop)] // body indexes both positions and radii by `a`
            for a in 0..n {
                let d = min_image(sub(x, genpos(a)), lbox);
                let pow = d[0] * d[0] + d[1] * d[1] + d[2] * d[2] - radii[a] * radii[a];
                if pow < best_pow {
                    best_pow = pow;
                    best = a;
                }
            }
            let mol = atom_to_mol[best];
            let n_elec = rho * grid.dv; // electrons in this voxel
            charges[mol] -= n_elec;
            let rref = [
                references[[mol, 0]],
                references[[mol, 1]],
                references[[mol, 2]],
            ];
            let disp = min_image(sub(x, rref), lbox);
            for c in 0..3 {
                dipoles[[mol, c]] -= n_elec * disp[c];
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
}
