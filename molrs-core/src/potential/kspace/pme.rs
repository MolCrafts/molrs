//! Particle Mesh Ewald (PME) electrostatic potential.
//!
//! Implements the full PME algorithm: self-energy, direct-space, exclusion
//! correction, and reciprocal-space (via 3D FFT built from 1D `rustfft`).
//!
//! Registered in [`KernelRegistry`] as `("kspace", "pme")`.
//! The constructor reads charges from `frame["atoms"]["charge"]` (float),
//! box vectors from style_params (`box_xx`, `box_yy`, `box_zz`, etc.),
//! and exclusion pairs from `frame["exclusions"]` (i, j columns).

use std::sync::{Arc, Mutex};

use rustfft::num_complex::Complex;
use rustfft::{Fft, FftPlanner};

use crate::forcefield::Params;
use crate::frame::Frame;
use crate::potential::Potential;
use crate::types::F;

// ---------------------------------------------------------------------------
// Precision-dependent helpers
// ---------------------------------------------------------------------------

#[cfg(not(feature = "f64"))]
const PI: F = std::f32::consts::PI;
#[cfg(feature = "f64")]
const PI: F = std::f64::consts::PI;

#[inline]
fn erfc_f(x: F) -> F {
    #[cfg(not(feature = "f64"))]
    {
        libm::erfcf(x)
    }
    #[cfg(feature = "f64")]
    {
        libm::erfc(x)
    }
}

#[inline]
fn erf_f(x: F) -> F {
    #[cfg(not(feature = "f64"))]
    {
        libm::erff(x)
    }
    #[cfg(feature = "f64")]
    {
        libm::erf(x)
    }
}

// ---------------------------------------------------------------------------
// Parameters
// ---------------------------------------------------------------------------

/// PME configuration parameters.
#[derive(Debug, Clone)]
pub struct PmeParams {
    /// Ewald splitting parameter (1/length).
    pub alpha: F,
    /// Real-space cutoff distance.
    pub cutoff: F,
    /// FFT grid dimensions `[Kx, Ky, Kz]`.
    pub grid_size: [usize; 3],
    /// B-spline interpolation order (typically 4 or 5).
    pub order: usize,
    /// Coulomb constant (e.g. 332.0636 for kcal/mol units, or 1.0).
    pub coulomb: F,
}

// ---------------------------------------------------------------------------
// FFT plan cache
// ---------------------------------------------------------------------------

struct FftPlans {
    fwd: [Arc<dyn Fft<F>>; 3],
    inv: [Arc<dyn Fft<F>>; 3],
}

// ---------------------------------------------------------------------------
// Scratch buffers (interior mutability for `&self`)
// ---------------------------------------------------------------------------

struct PmeScratch {
    grid: Vec<Complex<F>>,        // Kx*Ky*Kz complex grid
    buf: Vec<Complex<F>>,         // temp for 1D FFT rows
    fft_scratch: Vec<Complex<F>>, // rustfft scratch
}

// ---------------------------------------------------------------------------
// PmePotential
// ---------------------------------------------------------------------------

/// Full PME electrostatic potential implementing [`Potential`].
pub struct PmePotential {
    params: PmeParams,
    n_atoms: usize,
    charges: Vec<F>,
    h: [[F; 3]; 3],       // box matrix (row-major, lower-triangular)
    recip_h: [[F; 3]; 3], // inverse box matrix
    volume: F,
    exclusions: Vec<[usize; 2]>,
    self_energy: F,
    bspline_moduli: [Vec<F>; 3],
    fft_plans: FftPlans,
    scratch: Mutex<PmeScratch>,
}

impl PmePotential {
    /// Construct a new PME potential.
    ///
    /// * `charges` — per-atom partial charges (length `n_atoms`).
    /// * `box_vectors` — 3×3 box matrix, row-major, lower-triangular.
    /// * `exclusions` — pairs `[i, j]` with `i < j` whose reciprocal-space
    ///   interaction must be subtracted.
    pub fn new(
        params: PmeParams,
        charges: Vec<F>,
        box_vectors: [[F; 3]; 3],
        exclusions: Vec<[usize; 2]>,
    ) -> Self {
        let n_atoms = charges.len();
        let h = box_vectors;
        let recip_h = invert_box_vectors(&h);
        let volume = h[0][0] * h[1][1] * h[2][2]; // lower-triangular determinant

        // Self energy: -α/√π * C * Σq²
        let sum_q2: F = charges.iter().map(|q| q * q).sum();
        let self_energy = -(params.alpha / PI.sqrt()) * params.coulomb * sum_q2;

        // B-spline moduli
        let bspline_moduli = [
            compute_bspline_moduli(params.grid_size[0], params.order),
            compute_bspline_moduli(params.grid_size[1], params.order),
            compute_bspline_moduli(params.grid_size[2], params.order),
        ];

        // FFT plans
        let [kx, ky, kz] = params.grid_size;
        let mut planner = FftPlanner::<F>::new();
        let fft_plans = FftPlans {
            fwd: [
                planner.plan_fft_forward(kx),
                planner.plan_fft_forward(ky),
                planner.plan_fft_forward(kz),
            ],
            inv: [
                planner.plan_fft_inverse(kx),
                planner.plan_fft_inverse(ky),
                planner.plan_fft_inverse(kz),
            ],
        };

        // Scratch
        let grid_len = kx * ky * kz;
        let max_dim = kx.max(ky).max(kz);
        let max_fft_scratch = fft_plans
            .fwd
            .iter()
            .chain(fft_plans.inv.iter())
            .map(|p| p.get_inplace_scratch_len())
            .max()
            .unwrap_or(0);
        let zero = Complex::new(0.0, 0.0);
        let scratch = Mutex::new(PmeScratch {
            grid: vec![zero; grid_len],
            buf: vec![zero; max_dim],
            fft_scratch: vec![zero; max_fft_scratch],
        });

        Self {
            params,
            n_atoms,
            charges,
            h,
            recip_h,
            volume,
            exclusions,
            self_energy,
            bspline_moduli,
            fft_plans,
            scratch,
        }
    }

    /// Compute total PME energy for a flat coordinate vector.
    pub fn energy(&self, coords: &[F]) -> F {
        self.self_energy
            + self.direct_energy(coords)
            + self.exclusion_energy(coords)
            + self.reciprocal_energy(coords)
    }

    /// Compute PME forces (= -gradient) for a flat coordinate vector.
    pub fn forces(&self, coords: &[F]) -> Vec<F> {
        let mut grad = vec![0.0; coords.len()];
        // Self energy has zero gradient (constant).
        self.direct_gradient(coords, &mut grad);
        self.exclusion_gradient(coords, &mut grad);
        self.reciprocal_gradient(coords, &mut grad);
        for g in &mut grad {
            *g = -*g;
        }
        grad
    }

    // -----------------------------------------------------------------------
    // Direct space energy + gradient
    // -----------------------------------------------------------------------

    fn direct_energy(&self, coords: &[F]) -> F {
        let alpha = self.params.alpha;
        let cutoff = self.params.cutoff;
        let coulomb = self.params.coulomb;
        let cutoff2 = cutoff * cutoff;
        let mut energy: F = 0.0;

        for i in 0..self.n_atoms {
            for j in (i + 1)..self.n_atoms {
                if self.is_excluded(i, j) {
                    continue;
                }
                let (dx, dy, dz) = self.min_image_delta(coords, i, j);
                let r2 = dx * dx + dy * dy + dz * dz;
                if r2 >= cutoff2 {
                    continue;
                }
                let r = r2.sqrt();
                let alpha_r = alpha * r;
                energy += coulomb * self.charges[i] * self.charges[j] * erfc_f(alpha_r) / r;
            }
        }
        energy
    }

    fn direct_gradient(&self, coords: &[F], grad: &mut [F]) {
        let alpha = self.params.alpha;
        let cutoff = self.params.cutoff;
        let coulomb = self.params.coulomb;
        let cutoff2 = cutoff * cutoff;
        let two_alpha_over_sqrt_pi = 2.0 * alpha / PI.sqrt();

        for i in 0..self.n_atoms {
            for j in (i + 1)..self.n_atoms {
                if self.is_excluded(i, j) {
                    continue;
                }
                let (dx, dy, dz) = self.min_image_delta(coords, i, j);
                let r2 = dx * dx + dy * dy + dz * dz;
                if r2 >= cutoff2 {
                    continue;
                }
                let r = r2.sqrt();
                let alpha_r = alpha * r;
                let qi_qj = self.charges[i] * self.charges[j];
                let factor = -coulomb
                    * qi_qj
                    * (erfc_f(alpha_r) + two_alpha_over_sqrt_pi * r * (-alpha_r * alpha_r).exp())
                    / (r2 * r);
                grad[j * 3] += factor * dx;
                grad[j * 3 + 1] += factor * dy;
                grad[j * 3 + 2] += factor * dz;
                grad[i * 3] -= factor * dx;
                grad[i * 3 + 1] -= factor * dy;
                grad[i * 3 + 2] -= factor * dz;
            }
        }
    }

    // -----------------------------------------------------------------------
    // Exclusion correction energy + gradient
    // -----------------------------------------------------------------------

    fn exclusion_energy(&self, coords: &[F]) -> F {
        let alpha = self.params.alpha;
        let coulomb = self.params.coulomb;
        let mut energy: F = 0.0;

        for &[i, j] in &self.exclusions {
            let (dx, dy, dz) = self.delta(coords, i, j);
            let r = (dx * dx + dy * dy + dz * dz).sqrt();
            if r < 1e-15 {
                continue;
            }
            let alpha_r = alpha * r;
            energy -= coulomb * self.charges[i] * self.charges[j] * erf_f(alpha_r) / r;
        }
        energy
    }

    fn exclusion_gradient(&self, coords: &[F], grad: &mut [F]) {
        let alpha = self.params.alpha;
        let coulomb = self.params.coulomb;
        let two_alpha_over_sqrt_pi = 2.0 * alpha / PI.sqrt();

        for &[i, j] in &self.exclusions {
            let (dx, dy, dz) = self.delta(coords, i, j);
            let r2 = dx * dx + dy * dy + dz * dz;
            if r2 < 1e-30 {
                continue;
            }
            let r = r2.sqrt();
            let alpha_r = alpha * r;
            let qi_qj = self.charges[i] * self.charges[j];
            let factor = coulomb
                * qi_qj
                * (erf_f(alpha_r) - two_alpha_over_sqrt_pi * r * (-alpha_r * alpha_r).exp())
                / (r2 * r);
            grad[j * 3] += factor * dx;
            grad[j * 3 + 1] += factor * dy;
            grad[j * 3 + 2] += factor * dz;
            grad[i * 3] -= factor * dx;
            grad[i * 3 + 1] -= factor * dy;
            grad[i * 3 + 2] -= factor * dz;
        }
    }

    // -----------------------------------------------------------------------
    // Reciprocal space energy + gradient
    // -----------------------------------------------------------------------

    fn reciprocal_energy(&self, coords: &[F]) -> F {
        let mut scratch = self.scratch.lock().unwrap();
        let sqrt_coulomb = self.params.coulomb.sqrt();

        // Zero the grid
        let zero = Complex::new(0.0, 0.0);
        for c in scratch.grid.iter_mut() {
            *c = zero;
        }

        // Spread charges onto grid
        self.spread_charges(coords, &mut scratch.grid, sqrt_coulomb);

        // Forward 3D FFT — destructure to satisfy borrow checker
        let PmeScratch {
            ref mut grid,
            ref mut buf,
            ref mut fft_scratch,
        } = *scratch;
        self.fft_3d_forward(grid, buf, fft_scratch);

        // Convolution + energy accumulation
        let energy = self.reciprocal_convolution(&mut scratch.grid);
        0.5 * energy
    }

    fn reciprocal_gradient(&self, coords: &[F], grad: &mut [F]) {
        let mut scratch = self.scratch.lock().unwrap();
        let sqrt_coulomb = self.params.coulomb.sqrt();

        // Zero the grid
        let zero = Complex::new(0.0, 0.0);
        for c in scratch.grid.iter_mut() {
            *c = zero;
        }

        // Spread charges
        self.spread_charges(coords, &mut scratch.grid, sqrt_coulomb);

        // Forward FFT
        {
            let PmeScratch {
                ref mut grid,
                ref mut buf,
                ref mut fft_scratch,
            } = *scratch;
            self.fft_3d_forward(grid, buf, fft_scratch);
        }

        // Convolution (modifies grid in-place)
        let _ = self.reciprocal_convolution(&mut scratch.grid);

        // Inverse FFT (unnormalized, matching C++ irfftn with norm="forward")
        {
            let PmeScratch {
                ref mut grid,
                ref mut buf,
                ref mut fft_scratch,
            } = *scratch;
            self.fft_3d_inverse(grid, buf, fft_scratch);
        }

        // Interpolate forces from grid
        self.interpolate_forces(coords, &scratch.grid, sqrt_coulomb, grad);
    }

    // -----------------------------------------------------------------------
    // Charge spreading
    // -----------------------------------------------------------------------

    fn spread_charges(&self, coords: &[F], grid: &mut [Complex<F>], sqrt_coulomb: F) {
        let [kx, ky, kz] = self.params.grid_size;
        let order = self.params.order;

        for atom in 0..self.n_atoms {
            let (grid_index, data) = self.compute_spline(coords, atom);

            for ix in 0..order {
                let xindex = (grid_index[0] + ix) % kx;
                let dx = self.charges[atom] * sqrt_coulomb * data[ix][0];
                for iy in 0..order {
                    let yindex = (grid_index[1] + iy) % ky;
                    let dxdy = dx * data[iy][1];
                    for (iz, spline_z) in data.iter().enumerate().take(order) {
                        let zindex = (grid_index[2] + iz) % kz;
                        let index = xindex * ky * kz + yindex * kz + zindex;
                        grid[index].re += dxdy * spline_z[2];
                    }
                }
            }
        }
    }

    // -----------------------------------------------------------------------
    // Force interpolation from reciprocal grid
    // -----------------------------------------------------------------------

    fn interpolate_forces(
        &self,
        coords: &[F],
        grid: &[Complex<F>],
        sqrt_coulomb: F,
        grad: &mut [F],
    ) {
        let [kx, ky, kz] = self.params.grid_size;
        let order = self.params.order;

        for atom in 0..self.n_atoms {
            let (grid_index, data, ddata) = self.compute_spline_with_deriv(coords, atom);

            let mut dpos = [0.0 as F; 3];
            for ix in 0..order {
                let xindex = (grid_index[0] + ix) % kx;
                let dx = data[ix][0];
                let ddx = ddata[ix][0];
                for iy in 0..order {
                    let yindex = (grid_index[1] + iy) % ky;
                    let dy = data[iy][1];
                    let ddy = ddata[iy][1];
                    for iz in 0..order {
                        let zindex = (grid_index[2] + iz) % kz;
                        let dz = data[iz][2];
                        let ddz = ddata[iz][2];
                        let g = grid[xindex * ky * kz + yindex * kz + zindex].re;
                        dpos[0] += ddx * dy * dz * g;
                        dpos[1] += dx * ddy * dz * g;
                        dpos[2] += dx * dy * ddz * g;
                    }
                }
            }

            let scale = self.charges[atom] * sqrt_coulomb;
            let rh = &self.recip_h;
            let gs = self.params.grid_size;
            // Chain rule: fractional → Cartesian
            grad[atom * 3] += scale * (dpos[0] * gs[0] as F * rh[0][0]);
            grad[atom * 3 + 1] +=
                scale * (dpos[0] * gs[0] as F * rh[1][0] + dpos[1] * gs[1] as F * rh[1][1]);
            grad[atom * 3 + 2] += scale
                * (dpos[0] * gs[0] as F * rh[2][0]
                    + dpos[1] * gs[1] as F * rh[2][1]
                    + dpos[2] * gs[2] as F * rh[2][2]);
        }
    }

    // -----------------------------------------------------------------------
    // Reciprocal convolution — returns raw energy (before ×0.5)
    // -----------------------------------------------------------------------

    fn reciprocal_convolution(&self, grid: &mut [Complex<F>]) -> F {
        let [kx, ky, kz] = self.params.grid_size;
        let alpha = self.params.alpha;
        let recip_exp_factor = PI * PI / (alpha * alpha);
        let rh = &self.recip_h;
        let scale_factor = PI * self.volume;
        let xmod = &self.bspline_moduli[0];
        let ymod = &self.bspline_moduli[1];
        let zmod = &self.bspline_moduli[2];

        let mut energy: F = 0.0;

        for (ikx, &xmod_k) in xmod.iter().enumerate().take(kx) {
            let mx = if ikx < kx.div_ceil(2) {
                ikx as i64
            } else {
                ikx as i64 - kx as i64
            };
            let mhx = mx as F * rh[0][0];
            let bx = scale_factor * xmod_k;

            for (iky, &ymod_k) in ymod.iter().enumerate().take(ky) {
                let my = if iky < ky.div_ceil(2) {
                    iky as i64
                } else {
                    iky as i64 - ky as i64
                };
                let mhy = mx as F * rh[1][0] + my as F * rh[1][1];
                let mhx2y2 = mhx * mhx + mhy * mhy;
                let bxby = bx * ymod_k;

                for (ikz, &bz) in zmod.iter().enumerate().take(kz) {
                    let index = ikx * ky * kz + iky * kz + ikz;
                    let mz = if ikz < kz.div_ceil(2) {
                        ikz as i64
                    } else {
                        ikz as i64 - kz as i64
                    };
                    let mhz = mx as F * rh[2][0] + my as F * rh[2][1] + mz as F * rh[2][2];
                    let m2 = mhx2y2 + mhz * mhz;
                    let denom = m2 * bxby * bz;
                    let eterm = if index == 0 {
                        0.0
                    } else {
                        (-recip_exp_factor * m2).exp() / denom
                    };

                    let g = grid[index];
                    energy += eterm * (g.re * g.re + g.im * g.im);
                    grid[index] = g * eterm;
                }
            }
        }
        energy
    }

    // -----------------------------------------------------------------------
    // 3D FFT via three rounds of 1D FFT
    // -----------------------------------------------------------------------

    fn fft_3d_forward(
        &self,
        grid: &mut [Complex<F>],
        buf: &mut [Complex<F>],
        fft_scratch: &mut [Complex<F>],
    ) {
        let [kx, ky, kz] = self.params.grid_size;

        // Round 1: along z (contiguous) — Kx*Ky batches of length Kz
        for i in 0..(kx * ky) {
            let start = i * kz;
            self.fft_plans.fwd[2].process_with_scratch(&mut grid[start..start + kz], fft_scratch);
        }

        // Round 2: along y (stride Kz) — Kx*Kz batches of length Ky
        for ix in 0..kx {
            for iz in 0..kz {
                // Gather
                for iy in 0..ky {
                    buf[iy] = grid[ix * ky * kz + iy * kz + iz];
                }
                self.fft_plans.fwd[1].process_with_scratch(&mut buf[..ky], fft_scratch);
                // Scatter
                for iy in 0..ky {
                    grid[ix * ky * kz + iy * kz + iz] = buf[iy];
                }
            }
        }

        // Round 3: along x (stride Ky*Kz) — Ky*Kz batches of length Kx
        for iy in 0..ky {
            for iz in 0..kz {
                // Gather
                for ix in 0..kx {
                    buf[ix] = grid[ix * ky * kz + iy * kz + iz];
                }
                self.fft_plans.fwd[0].process_with_scratch(&mut buf[..kx], fft_scratch);
                // Scatter
                for ix in 0..kx {
                    grid[ix * ky * kz + iy * kz + iz] = buf[ix];
                }
            }
        }
    }

    fn fft_3d_inverse(
        &self,
        grid: &mut [Complex<F>],
        buf: &mut [Complex<F>],
        fft_scratch: &mut [Complex<F>],
    ) {
        let [kx, ky, kz] = self.params.grid_size;

        // Round 1: along x
        for iy in 0..ky {
            for iz in 0..kz {
                for ix in 0..kx {
                    buf[ix] = grid[ix * ky * kz + iy * kz + iz];
                }
                self.fft_plans.inv[0].process_with_scratch(&mut buf[..kx], fft_scratch);
                for ix in 0..kx {
                    grid[ix * ky * kz + iy * kz + iz] = buf[ix];
                }
            }
        }

        // Round 2: along y
        for ix in 0..kx {
            for iz in 0..kz {
                for iy in 0..ky {
                    buf[iy] = grid[ix * ky * kz + iy * kz + iz];
                }
                self.fft_plans.inv[1].process_with_scratch(&mut buf[..ky], fft_scratch);
                for iy in 0..ky {
                    grid[ix * ky * kz + iy * kz + iz] = buf[iy];
                }
            }
        }

        // Round 3: along z (contiguous)
        for i in 0..(kx * ky) {
            let start = i * kz;
            self.fft_plans.inv[2].process_with_scratch(&mut grid[start..start + kz], fft_scratch);
        }
    }

    // -----------------------------------------------------------------------
    // B-spline computation
    // -----------------------------------------------------------------------

    /// Compute B-spline coefficients for an atom. Returns `(grid_index[3], data[order][3])`.
    fn compute_spline(&self, coords: &[F], atom: usize) -> ([usize; 3], Vec<[F; 3]>) {
        let order = self.params.order;
        let gs = self.params.grid_size;
        let pos = [coords[atom * 3], coords[atom * 3 + 1], coords[atom * 3 + 2]];

        // Wrap position into box
        let mut pos_in_box = pos;
        for i in (0..3).rev() {
            let s = (pos_in_box[i] * self.recip_h[i][i]).floor();
            for (j, pos_j) in pos_in_box.iter_mut().enumerate() {
                *pos_j -= s * self.h[i][j];
            }
        }

        // Fractional coordinates → grid coordinates
        let mut grid_index = [0usize; 3];
        let mut dr = [0.0 as F; 3];
        for i in 0..3 {
            let mut t = pos_in_box[0] * self.recip_h[0][i]
                + pos_in_box[1] * self.recip_h[1][i]
                + pos_in_box[2] * self.recip_h[2][i];
            t = (t - t.floor()) * gs[i] as F;
            let ti = t as usize;
            dr[i] = t - ti as F;
            grid_index[i] = ti % gs[i];
        }

        // B-spline coefficients
        let mut data = vec![[0.0 as F; 3]; order];
        bspline_fill(&mut data, &dr, order);

        (grid_index, data)
    }

    /// Compute B-spline coefficients AND derivatives for an atom.
    fn compute_spline_with_deriv(
        &self,
        coords: &[F],
        atom: usize,
    ) -> ([usize; 3], Vec<[F; 3]>, Vec<[F; 3]>) {
        let order = self.params.order;
        let gs = self.params.grid_size;
        let pos = [coords[atom * 3], coords[atom * 3 + 1], coords[atom * 3 + 2]];

        let mut pos_in_box = pos;
        for i in (0..3).rev() {
            let s = (pos_in_box[i] * self.recip_h[i][i]).floor();
            for (j, pos_j) in pos_in_box.iter_mut().enumerate() {
                *pos_j -= s * self.h[i][j];
            }
        }

        let mut grid_index = [0usize; 3];
        let mut dr = [0.0 as F; 3];
        for i in 0..3 {
            let mut t = pos_in_box[0] * self.recip_h[0][i]
                + pos_in_box[1] * self.recip_h[1][i]
                + pos_in_box[2] * self.recip_h[2][i];
            t = (t - t.floor()) * gs[i] as F;
            let ti = t as usize;
            dr[i] = t - ti as F;
            grid_index[i] = ti % gs[i];
        }

        let mut data = vec![[0.0 as F; 3]; order];
        let mut ddata = vec![[0.0 as F; 3]; order];
        bspline_fill_with_deriv(&mut data, &mut ddata, &dr, order);

        (grid_index, data, ddata)
    }

    // -----------------------------------------------------------------------
    // Helpers
    // -----------------------------------------------------------------------

    /// Minimum-image displacement vector from atom i to atom j.
    fn min_image_delta(&self, coords: &[F], i: usize, j: usize) -> (F, F, F) {
        let mut dx = coords[j * 3] - coords[i * 3];
        let mut dy = coords[j * 3 + 1] - coords[i * 3 + 1];
        let mut dz = coords[j * 3 + 2] - coords[i * 3 + 2];

        // Apply minimum image convention for lower-triangular box
        let sz = (dz / self.h[2][2]).round();
        dx -= sz * self.h[2][0];
        dy -= sz * self.h[2][1];
        dz -= sz * self.h[2][2];

        let sy = (dy / self.h[1][1]).round();
        dx -= sy * self.h[1][0];
        dy -= sy * self.h[1][1];

        let sx = (dx / self.h[0][0]).round();
        dx -= sx * self.h[0][0];

        (dx, dy, dz)
    }

    /// Raw displacement from atom i to atom j (no minimum image).
    fn delta(&self, coords: &[F], i: usize, j: usize) -> (F, F, F) {
        (
            coords[j * 3] - coords[i * 3],
            coords[j * 3 + 1] - coords[i * 3 + 1],
            coords[j * 3 + 2] - coords[i * 3 + 2],
        )
    }

    fn is_excluded(&self, i: usize, j: usize) -> bool {
        let (lo, hi) = if i < j { (i, j) } else { (j, i) };
        self.exclusions.iter().any(|&[a, b]| a == lo && b == hi)
    }
}

// ---------------------------------------------------------------------------
// Potential trait
// ---------------------------------------------------------------------------

impl Potential for PmePotential {
    fn eval(&self, coords: &[F]) -> (F, Vec<F>) {
        let e = PmePotential::energy(self, coords);
        let f = PmePotential::forces(self, coords);
        (e, f)
    }
}

// ---------------------------------------------------------------------------
// Free helper functions
// ---------------------------------------------------------------------------

/// Invert a lower-triangular 3×3 box matrix.
fn invert_box_vectors(h: &[[F; 3]; 3]) -> [[F; 3]; 3] {
    let det = h[0][0] * h[1][1] * h[2][2];
    let s = 1.0 / det;
    [
        [h[1][1] * h[2][2] * s, 0.0, 0.0],
        [-h[1][0] * h[2][2] * s, h[0][0] * h[2][2] * s, 0.0],
        [
            (h[1][0] * h[2][1] - h[1][1] * h[2][0]) * s,
            -h[0][0] * h[2][1] * s,
            h[0][0] * h[1][1] * s,
        ],
    ]
}

/// Fill B-spline coefficients (no derivatives). Matches the C++ `computeSpline`.
fn bspline_fill(data: &mut [[F; 3]], dr: &[F; 3], order: usize) {
    let scale = 1.0 / (order - 1) as F;
    for i in 0..3 {
        data[order - 1][i] = 0.0;
        data[1][i] = dr[i];
        data[0][i] = 1.0 - dr[i];
        for j in 3..order {
            let div = 1.0 / (j - 1) as F;
            data[j - 1][i] = div * dr[i] * data[j - 2][i];
            for k in 1..(j - 1) {
                data[j - k - 1][i] = div
                    * ((dr[i] + k as F) * data[j - k - 2][i]
                        + (j as F - k as F - dr[i]) * data[j - k - 1][i]);
            }
            data[0][i] *= div * (1.0 - dr[i]);
        }
        // Final scaling pass
        data[order - 1][i] = scale * dr[i] * data[order - 2][i];
        for j in 1..(order - 1) {
            data[order - j - 1][i] = scale
                * ((dr[i] + j as F) * data[order - j - 2][i]
                    + (order as F - j as F - dr[i]) * data[order - j - 1][i]);
        }
        data[0][i] *= scale * (1.0 - dr[i]);
    }
}

/// Fill B-spline coefficients AND derivatives. Derivatives are computed just
/// before the final scaling pass (matching the C++ reference).
fn bspline_fill_with_deriv(data: &mut [[F; 3]], ddata: &mut [[F; 3]], dr: &[F; 3], order: usize) {
    let scale = 1.0 / (order - 1) as F;
    for i in 0..3 {
        data[order - 1][i] = 0.0;
        data[1][i] = dr[i];
        data[0][i] = 1.0 - dr[i];
        for j in 3..order {
            let div = 1.0 / (j - 1) as F;
            data[j - 1][i] = div * dr[i] * data[j - 2][i];
            for k in 1..(j - 1) {
                data[j - k - 1][i] = div
                    * ((dr[i] + k as F) * data[j - k - 2][i]
                        + (j as F - k as F - dr[i]) * data[j - k - 1][i]);
            }
            data[0][i] *= div * (1.0 - dr[i]);
        }
        // Derivatives (before final scaling)
        ddata[0][i] = -data[0][i];
        for j in 1..order {
            ddata[j][i] = data[j - 1][i] - data[j][i];
        }
        // Final scaling pass
        data[order - 1][i] = scale * dr[i] * data[order - 2][i];
        for j in 1..(order - 1) {
            data[order - j - 1][i] = scale
                * ((dr[i] + j as F) * data[order - j - 2][i]
                    + (order as F - j as F - dr[i]) * data[order - j - 1][i]);
        }
        data[0][i] *= scale * (1.0 - dr[i]);
    }
}

/// Precompute B-spline moduli for reciprocal-space convolution.
fn compute_bspline_moduli(grid_size: usize, order: usize) -> Vec<F> {
    // Compute the B-spline values at uniform intervals
    let mut bspline = vec![0.0 as F; grid_size];
    let mut data = vec![[0.0 as F; 3]; order];
    let dr = [0.0 as F; 3];
    bspline_fill(&mut data, &dr, order);
    // data[j][0] gives the B-spline value at u=0 for the j-th support point
    for j in 0..order {
        bspline[j] = data[j][0];
    }

    // Compute moduli via DFT of bspline values
    let two_pi_over_n = 2.0 * PI / grid_size as F;
    let mut moduli = vec![0.0 as F; grid_size];
    for (k, moduli_k) in moduli.iter_mut().enumerate().take(grid_size) {
        let mut sum_cos: F = 0.0;
        let mut sum_sin: F = 0.0;
        for (j, bspline_j) in bspline.iter().enumerate().take(order) {
            let arg = two_pi_over_n * k as F * j as F;
            sum_cos += *bspline_j * arg.cos();
            sum_sin += *bspline_j * arg.sin();
        }
        *moduli_k = sum_cos * sum_cos + sum_sin * sum_sin;
    }
    // Avoid division by zero at k=0
    if moduli[0] < 1e-30 {
        moduli[0] = 1e-30;
    }
    moduli
}

// ---------------------------------------------------------------------------
// KernelRegistry constructor
// ---------------------------------------------------------------------------

/// Constructor for the kernel registry.
///
/// **`style_params`** keys: `alpha`, `cutoff`, `grid_x`, `grid_y`, `grid_z`,
/// `order`, `coulomb`, and box vectors `box_xx`, `box_yx`, ..., `box_zz`
/// (9 elements, row-major lower-triangular).
///
/// **`frame`** blocks:
/// - `"atoms"` with `"charge"` column (f64/f32) — per-atom charges.
/// - `"exclusions"` with `"i"`, `"j"` columns (u32) — exclusion pairs.
pub fn pme_ctor(
    style_params: &Params,
    _type_params: &[(&str, &Params)],
    frame: &Frame,
) -> Result<Box<dyn Potential>, String> {
    let alpha = style_params.get("alpha").ok_or("PME: missing 'alpha'")? as F;
    let cutoff = style_params.get("cutoff").ok_or("PME: missing 'cutoff'")? as F;
    let grid_x = style_params.get("grid_x").ok_or("PME: missing 'grid_x'")? as usize;
    let grid_y = style_params.get("grid_y").ok_or("PME: missing 'grid_y'")? as usize;
    let grid_z = style_params.get("grid_z").ok_or("PME: missing 'grid_z'")? as usize;
    let order = style_params.get("order").ok_or("PME: missing 'order'")? as usize;
    let coulomb = style_params
        .get("coulomb")
        .ok_or("PME: missing 'coulomb'")? as F;

    // Read charges from Frame's "atoms" block
    let atoms = frame
        .get("atoms")
        .ok_or("PME: Frame missing \"atoms\" block")?;
    let charges: Vec<F> = if let Some(charge_col) = atoms.get_float("charge") {
        charge_col.iter().copied().collect()
    } else {
        return Err("PME: atoms block missing \"charge\" float column".into());
    };

    // Read box vectors from style_params
    let box_xx = style_params.get("box_xx").ok_or("PME: missing 'box_xx'")? as F;
    let box_xy = style_params.get("box_xy").unwrap_or(0.0) as F;
    let box_xz = style_params.get("box_xz").unwrap_or(0.0) as F;
    let box_yx = style_params.get("box_yx").unwrap_or(0.0) as F;
    let box_yy = style_params.get("box_yy").ok_or("PME: missing 'box_yy'")? as F;
    let box_yz = style_params.get("box_yz").unwrap_or(0.0) as F;
    let box_zx = style_params.get("box_zx").unwrap_or(0.0) as F;
    let box_zy = style_params.get("box_zy").unwrap_or(0.0) as F;
    let box_zz = style_params.get("box_zz").ok_or("PME: missing 'box_zz'")? as F;
    let box_vectors = [
        [box_xx, box_xy, box_xz],
        [box_yx, box_yy, box_yz],
        [box_zx, box_zy, box_zz],
    ];

    // Read exclusions from Frame's "exclusions" block (optional)
    let mut exclusions = Vec::new();
    if let Some(block) = frame.get("exclusions")
        && let (Some(i_col), Some(j_col)) = (block.get_uint("atomi"), block.get_uint("atomj"))
    {
        for idx in 0..i_col.len() {
            exclusions.push([i_col[idx] as usize, j_col[idx] as usize]);
        }
    }

    let params = PmeParams {
        alpha,
        cutoff,
        grid_size: [grid_x, grid_y, grid_z],
        order,
        coulomb,
    };

    Ok(Box::new(PmePotential::new(
        params,
        charges,
        box_vectors,
        exclusions,
    )))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn cubic_box(l: F) -> [[F; 3]; 3] {
        [[l, 0.0, 0.0], [0.0, l, 0.0], [0.0, 0.0, l]]
    }

    // --- B-spline unit tests ---

    #[test]
    fn test_bspline_partition_of_unity() {
        // For any fractional offset, the B-spline values must sum to 1.
        let order = 4;
        for &u in &[0.0, 0.1, 0.25, 0.5, 0.75, 0.99] {
            let u: F = u as F;
            let dr = [u, u, u];
            let mut data = vec![[0.0 as F; 3]; order];
            bspline_fill(&mut data, &dr, order);
            let sum: F = data.iter().map(|d| d[0]).sum();
            assert!(
                (sum - 1.0).abs() < 1e-5,
                "partition of unity failed for u={}: sum={}",
                u,
                sum
            );
        }
    }

    #[test]
    fn test_bspline_deriv_sum_zero() {
        let order = 4;
        for &u in &[0.1, 0.25, 0.5, 0.75, 0.9] {
            let u: F = u as F;
            let dr = [u, u, u];
            let mut data = vec![[0.0 as F; 3]; order];
            let mut ddata = vec![[0.0 as F; 3]; order];
            bspline_fill_with_deriv(&mut data, &mut ddata, &dr, order);
            let sum: F = ddata.iter().map(|d| d[0]).sum();
            assert!(
                sum.abs() < 1e-5,
                "derivative sum should be zero for u={}: sum={}",
                u,
                sum
            );
        }
    }

    #[test]
    fn test_bspline_order5() {
        let order = 5;
        let dr: [F; 3] = [0.3, 0.7, 0.5];
        let mut data = vec![[0.0 as F; 3]; order];
        bspline_fill(&mut data, &dr, order);
        for dim in 0..3 {
            let sum: F = data.iter().map(|d| d[dim]).sum();
            assert!((sum - 1.0).abs() < 1e-5, "dim={}: sum={}", dim, sum);
        }
    }

    // --- FFT round-trip test ---

    #[test]
    fn test_fft_roundtrip() {
        let params = PmeParams {
            alpha: 0.3,
            cutoff: 5.0,
            grid_size: [4, 4, 4],
            order: 4,
            coulomb: 1.0,
        };
        let pme = PmePotential::new(params, vec![1.0], cubic_box(10.0), vec![]);
        let n = 4 * 4 * 4;
        let mut grid: Vec<Complex<F>> = (0..n).map(|i| Complex::new(i as F, 0.0)).collect();
        let original = grid.clone();
        let max_dim = 4;
        let zero = Complex::new(0.0 as F, 0.0);
        let mut buf = vec![zero; max_dim];
        let max_scratch = pme
            .fft_plans
            .fwd
            .iter()
            .chain(pme.fft_plans.inv.iter())
            .map(|p| p.get_inplace_scratch_len())
            .max()
            .unwrap_or(0);
        let mut fft_scratch = vec![zero; max_scratch];

        pme.fft_3d_forward(&mut grid, &mut buf, &mut fft_scratch);
        pme.fft_3d_inverse(&mut grid, &mut buf, &mut fft_scratch);

        // Normalize
        let inv_n: F = 1.0 / n as F;
        for c in grid.iter_mut() {
            *c *= inv_n;
        }

        for i in 0..n {
            assert!(
                (grid[i].re - original[i].re).abs() < 1e-3,
                "FFT round-trip failed at {}: got {}, expected {}",
                i,
                grid[i].re,
                original[i].re,
            );
            assert!(
                grid[i].im.abs() < 1e-3,
                "FFT round-trip imaginary part at {}: {}",
                i,
                grid[i].im,
            );
        }
    }

    // --- Two-ion test ---

    #[test]
    fn test_two_ions_energy() {
        let box_l: F = 20.0;
        let r: F = 3.0;
        let alpha: F = 0.3;
        let coulomb: F = 1.0;
        let params = PmeParams {
            alpha,
            cutoff: 9.0,
            grid_size: [32, 32, 32],
            order: 5,
            coulomb,
        };
        let charges = vec![1.0, -1.0];
        let exclusions = vec![];
        let pme = PmePotential::new(params, charges, cubic_box(box_l), exclusions);

        let coords: Vec<F> = vec![
            box_l / 2.0,
            box_l / 2.0,
            box_l / 2.0,
            box_l / 2.0 + r,
            box_l / 2.0,
            box_l / 2.0,
        ];
        let e = pme.energy(&coords);

        // Coulomb energy in vacuum: -1/r (with coulomb=1, q=+1,-1)
        let e_vacuum: F = -coulomb / r;
        // With periodic images the energy differs slightly, but should be close
        assert!(
            (e - e_vacuum).abs() < 0.05,
            "PME energy={}, vacuum={}, diff={}",
            e,
            e_vacuum,
            (e - e_vacuum).abs()
        );
    }

    // --- Numerical forces test ---

    #[test]
    fn test_numerical_forces() {
        let box_l: F = 10.0;
        let params = PmeParams {
            alpha: 0.4,
            cutoff: 4.5,
            grid_size: [16, 16, 16],
            order: 4,
            coulomb: 1.0,
        };
        let charges = vec![0.5, -0.3, 0.2];
        let exclusions = vec![[0, 1]]; // exclude the 0-1 pair
        let pme = PmePotential::new(params, charges, cubic_box(box_l), exclusions);

        let coords: Vec<F> = vec![2.0, 3.0, 4.0, 5.0, 3.5, 4.5, 7.0, 6.0, 5.0];

        let forces = pme.forces(&coords);

        let eps: F = 1e-3;
        for idx in 0..9 {
            let mut cp = coords.clone();
            let mut cm = coords.clone();
            cp[idx] += eps;
            cm[idx] -= eps;
            let numerical_force = -(pme.energy(&cp) - pme.energy(&cm)) / (2.0 * eps);
            assert!(
                (forces[idx] - numerical_force).abs() < 1.0,
                "idx={}: analytical={:.6}, numerical={:.6}, diff={:.2e}",
                idx,
                forces[idx],
                numerical_force,
                (forces[idx] - numerical_force).abs()
            );
        }
    }

    // --- Newton's third law ---

    #[test]
    fn test_newton_third_law() {
        let box_l: F = 10.0;
        let params = PmeParams {
            alpha: 0.35,
            cutoff: 4.5,
            grid_size: [32, 32, 32],
            order: 5,
            coulomb: 1.0,
        };
        let charges = vec![0.5, -0.3, 0.4, -0.6];
        let exclusions = vec![[0, 1], [2, 3]];
        let pme = PmePotential::new(params, charges, cubic_box(box_l), exclusions);

        let coords: Vec<F> = vec![1.0, 2.0, 3.0, 4.0, 2.5, 3.5, 6.0, 7.0, 2.0, 8.0, 7.5, 2.5];
        let forces = pme.forces(&coords);

        for dim in 0..3 {
            let sum: F = (0..4).map(|a| forces[a * 3 + dim]).sum();
            assert!(sum.abs() < 0.1, "dim={}: total force sum={:.2e}", dim, sum);
        }
    }

    // --- Integration test: PME + PairLJ126 in Potentials ---

    #[test]
    fn test_pme_in_potentials_collection() {
        use crate::potential::Potentials;
        use crate::potential::kernels::PairLJ126;

        let box_l: F = 10.0;
        let params = PmeParams {
            alpha: 0.3,
            cutoff: 4.5,
            grid_size: [16, 16, 16],
            order: 4,
            coulomb: 1.0,
        };
        let charges = vec![0.5, -0.5];
        let pme = PmePotential::new(params, charges, cubic_box(box_l), vec![]);

        let lj = PairLJ126::new(vec![0], vec![1], vec![1.0], vec![1.0]);

        let mut pots = Potentials::new();
        pots.push(Box::new(pme));
        pots.push(Box::new(lj));

        let coords: Vec<F> = vec![
            box_l / 2.0,
            box_l / 2.0,
            box_l / 2.0,
            box_l / 2.0 + 2.0,
            box_l / 2.0,
            box_l / 2.0,
        ];

        let e = pots.energy(&coords);
        assert!(e.is_finite(), "energy should be finite, got {}", e);
        assert!(e.abs() > 1e-10, "energy should be non-zero");

        let forces = pots.forces(&coords);
        for (i, &f) in forces.iter().enumerate() {
            assert!(F::is_finite(f), "forces[{}] should be finite", i);
        }
    }

    // --- Box inversion test ---

    #[test]
    fn test_invert_box_vectors() {
        let h: [[F; 3]; 3] = [[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]];
        let inv = invert_box_vectors(&h);
        assert!((inv[0][0] - 0.1).abs() < 1e-5);
        assert!((inv[1][1] - 0.1).abs() < 1e-5);
        assert!((inv[2][2] - 0.1).abs() < 1e-5);
        assert!(inv[0][1].abs() < 1e-5);
        assert!(inv[1][0].abs() < 1e-5);
    }

    #[test]
    fn test_invert_box_vectors_triclinic() {
        let h: [[F; 3]; 3] = [[10.0, 0.0, 0.0], [2.0, 8.0, 0.0], [1.0, 3.0, 6.0]];
        let inv = invert_box_vectors(&h);
        // Verify H * H^{-1} = I (row-by-row dot product)
        for (row, h_row) in h.iter().enumerate() {
            for (col, _) in inv[0].iter().enumerate() {
                let mut dot: F = 0.0;
                for (k, h_row_k) in h_row.iter().enumerate() {
                    dot += h_row_k * inv[k][col];
                }
                let expected: F = if row == col { 1.0 } else { 0.0 };
                assert!(
                    (dot - expected).abs() < 1e-4,
                    "H*Hinv[{}][{}]={}, expected {}",
                    row,
                    col,
                    dot,
                    expected
                );
            }
        }
    }
}
