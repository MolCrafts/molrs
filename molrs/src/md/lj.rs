//! Pair potentials for in-process MD.
//!
//! A [`Potential`] is a **pair kernel**: `calc_energy` / `calc_force` /
//! `eval → (energy, force)` on already-reduced geometry (`r²`, MIC `disp`).
//! System-level overloads take a [`VerletSkin`] neighbour list + positions.
//!
//! Distinct from [`crate::ff::potential::Potential`] (molecule-bound, flat 3N,
//! kcal/mol).

use ndarray::{Array2, ArrayView2};

use molrs::spatial::neighbors::{Neighbors, SkinError, VerletSkin};
use molrs::types::F;

use super::error::MdError;

/// Pair kernel: geometry in → energy / force on `j` out.
pub trait Potential: Send + Sync {
    /// Pair energy from `r2 = |disp|²` and MIC `disp = r_j − r_i`.
    fn calc_energy(&self, r2: F, disp: [F; 3]) -> Option<F>;

    /// Pair force on `j` from `r2` and MIC `disp` (force on `i` is the negation).
    fn calc_force(&self, r2: F, disp: [F; 3]) -> Option<[F; 3]>;

    /// Pair `(energy, force_on_j)` from `r2` and MIC `disp`.
    fn eval(&self, r2: F, disp: [F; 3]) -> Option<(F, [F; 3])> {
        match (self.calc_energy(r2, disp), self.calc_force(r2, disp)) {
            (Some(e), Some(f)) => Some((e, f)),
            _ => None,
        }
    }
}

/// Mie prefactor `C` so 12-6 recovers `4ε`.
fn mie_c(n: i32, m: i32) -> F {
    let n = n as F;
    let m = m as F;
    (n / (n - m)) * (n / m).powf(m / (n - m))
}

/// Lennard-Jones / Mie pair [`Potential`].
#[derive(Clone, Copy, Debug)]
pub struct LJ {
    epsilon: F,
    sigma: F,
    cutoff: F,
    n: i32,
    m: i32,
    shifted: bool,
    smeared: bool,
    cutoff2: F,
    ceps: F,
    e0: F,
    f_rc: F,
}

impl LJ {
    pub fn new(
        epsilon: F,
        sigma: F,
        cutoff: F,
        n: i32,
        m: i32,
        shifted: bool,
        smeared: bool,
    ) -> Result<Self, MdError> {
        if epsilon <= 0.0 || sigma <= 0.0 || cutoff <= 0.0 {
            return Err(MdError::Invalid(
                "LJ requires epsilon, sigma, cutoff > 0".into(),
            ));
        }
        if m <= 0 || n <= m {
            return Err(MdError::Invalid(format!(
                "LJ exponents must satisfy n > m > 0, got n={n}, m={m}"
            )));
        }
        let cutoff2 = cutoff * cutoff;
        let ceps = mie_c(n, m) * epsilon;
        let sr_c = sigma / cutoff;
        let sr_n_c = sr_c.powi(n);
        let sr_m_c = sr_c.powi(m);
        let u_c = ceps * (sr_n_c - sr_m_c);
        let fac_c = ceps * ((n as F) * sr_n_c - (m as F) * sr_m_c) / cutoff2;
        let shift_energy = shifted || smeared;
        Ok(Self {
            epsilon,
            sigma,
            cutoff,
            n,
            m,
            shifted: shift_energy,
            smeared,
            cutoff2,
            ceps,
            e0: if shift_energy { u_c } else { 0.0 },
            f_rc: if smeared { fac_c * cutoff } else { 0.0 },
        })
    }

    pub fn lj126(epsilon: F, sigma: F, cutoff: F) -> Result<Self, MdError> {
        Self::new(epsilon, sigma, cutoff, 12, 6, true, false)
    }

    pub fn epsilon(&self) -> F {
        self.epsilon
    }
    pub fn sigma(&self) -> F {
        self.sigma
    }
    pub fn cutoff(&self) -> F {
        self.cutoff
    }
    pub fn n(&self) -> i32 {
        self.n
    }
    pub fn m(&self) -> i32 {
        self.m
    }
    pub fn shifted(&self) -> bool {
        self.shifted
    }
    pub fn smeared(&self) -> bool {
        self.smeared
    }

    fn pair_kernel(&self, r2: F, disp: [F; 3]) -> Option<(F, [F; 3])> {
        if r2 <= 0.0 || r2 > self.cutoff2 {
            return None;
        }
        // Fast 12-6 path: no `sqrt` / `powi` (smeared still needs `r`).
        if self.n == 12 && self.m == 6 {
            let inv_r2 = 1.0 / r2;
            let sr2 = self.sigma * self.sigma * inv_r2;
            let sr6 = sr2 * sr2 * sr2;
            let sr12 = sr6 * sr6;
            let mut energy = self.ceps * (sr12 - sr6) - self.e0;
            let mut fac = self.ceps * (12.0 * sr12 - 6.0 * sr6) * inv_r2;
            if self.f_rc != 0.0 {
                let r = r2.sqrt();
                energy += (r - self.cutoff) * self.f_rc;
                fac -= self.f_rc / r;
            }
            return Some((energy, [fac * disp[0], fac * disp[1], fac * disp[2]]));
        }
        let sr = self.sigma / r2.sqrt();
        let sr_n = sr.powi(self.n);
        let sr_m = sr.powi(self.m);
        let mut energy = self.ceps * (sr_n - sr_m) - self.e0;
        let mut fac = self.ceps * ((self.n as F) * sr_n - (self.m as F) * sr_m) / r2;
        if self.f_rc != 0.0 {
            let r = r2.sqrt();
            energy += (r - self.cutoff) * self.f_rc;
            fac -= self.f_rc / r;
        }
        Some((energy, [fac * disp[0], fac * disp[1], fac * disp[2]]))
    }

    /// Scatter from already-reduced neighbour columns.
    pub fn eval_pairs(
        &self,
        n_atoms: usize,
        i: &[u32],
        j: &[u32],
        disp: ArrayView2<'_, F>,
        dist_sq: Option<&[F]>,
    ) -> Result<(F, Array2<F>), MdError> {
        let n_pairs = i.len();
        if j.len() != n_pairs || disp.nrows() != n_pairs || disp.ncols() != 3 {
            return Err(MdError::Invalid(format!(
                "pair columns must share n_pairs={n_pairs}: j={}, disp={:?}",
                j.len(),
                disp.shape()
            )));
        }
        if let Some(d2) = dist_sq
            && d2.len() != n_pairs
        {
            return Err(MdError::Invalid(format!(
                "dist_sq length {} disagrees with n_pairs={n_pairs}",
                d2.len()
            )));
        }
        let mut forces = Array2::<F>::zeros((n_atoms, 3));
        let mut energy = 0.0;
        for p in 0..n_pairs {
            let ia = i[p] as usize;
            let ja = j[p] as usize;
            if ia >= n_atoms || ja >= n_atoms {
                return Err(MdError::Invalid(format!(
                    "pair ({ia}, {ja}) is outside n_atoms={n_atoms}"
                )));
            }
            let d = [disp[[p, 0]], disp[[p, 1]], disp[[p, 2]]];
            let r2 = match dist_sq {
                Some(col) => col[p],
                None => d[0] * d[0] + d[1] * d[1] + d[2] * d[2],
            };
            let Some((e, f)) = self.pair_kernel(r2, d) else {
                continue;
            };
            energy += e;
            forces[[ja, 0]] += f[0];
            forces[[ja, 1]] += f[1];
            forces[[ja, 2]] += f[2];
            forces[[ia, 0]] -= f[0];
            forces[[ia, 1]] -= f[1];
            forces[[ia, 2]] -= f[2];
        }
        Ok((energy, forces))
    }

    /// Update a [`VerletSkin`] and return total energy.
    pub fn calc_energy(
        &self,
        neighbors: &mut VerletSkin,
        pos: ArrayView2<'_, F>,
    ) -> Result<F, MdError> {
        Ok(self.eval(neighbors, pos)?.0)
    }

    /// Update a [`VerletSkin`] and return forces `(N, 3)`.
    pub fn calc_force(
        &self,
        neighbors: &mut VerletSkin,
        pos: ArrayView2<'_, F>,
    ) -> Result<Array2<F>, MdError> {
        Ok(self.eval(neighbors, pos)?.1)
    }

    /// Update a [`VerletSkin`], then `(energy, forces)` with **current** MIC.
    pub fn eval(
        &self,
        neighbors: &mut VerletSkin,
        pos: ArrayView2<'_, F>,
    ) -> Result<(F, Array2<F>), MdError> {
        let mut forces = Array2::<F>::zeros((pos.nrows(), 3));
        let energy = self.eval_into(neighbors, pos, &mut forces)?;
        Ok((energy, forces))
    }

    /// Like [`eval`](Self::eval) but writes forces into `forces` (must be `(N, 3)`).
    pub fn eval_into(
        &self,
        neighbors: &mut VerletSkin,
        pos: ArrayView2<'_, F>,
        forces: &mut Array2<F>,
    ) -> Result<F, MdError> {
        neighbors.update(pos).map_err(skin_err)?;
        let n_atoms = pos.nrows();
        if forces.shape() != [n_atoms, 3] {
            return Err(MdError::Invalid(format!(
                "forces must have shape ({n_atoms}, 3), got {:?}",
                forces.shape()
            )));
        }
        forces.fill(0.0);
        let mut energy = 0.0;
        neighbors.for_each_pair_at(pos, |i, j, r2, disp| {
            let Some((e, f)) = self.pair_kernel(r2, disp) else {
                return;
            };
            let ia = i as usize;
            let ja = j as usize;
            debug_assert!(ia < n_atoms && ja < n_atoms);
            energy += e;
            forces[[ja, 0]] += f[0];
            forces[[ja, 1]] += f[1];
            forces[[ja, 2]] += f[2];
            forces[[ia, 0]] -= f[0];
            forces[[ia, 1]] -= f[1];
            forces[[ia, 2]] -= f[2];
        });
        Ok(energy)
    }

    /// Evaluate against a materialized neighbour table.
    pub fn eval_table(
        &self,
        n_atoms: usize,
        neighbors: &Neighbors,
    ) -> Result<(F, Array2<F>), MdError> {
        let disp = neighbors
            .disp()
            .ok_or_else(|| MdError::Invalid("LJ needs the Neighbors disp column".into()))?;
        self.eval_pairs(
            n_atoms,
            neighbors.query_point_indices(),
            neighbors.point_indices(),
            disp,
            neighbors.dist_sq(),
        )
    }
}

impl Potential for LJ {
    fn calc_energy(&self, r2: F, disp: [F; 3]) -> Option<F> {
        self.pair_kernel(r2, disp).map(|(e, _)| e)
    }

    fn calc_force(&self, r2: F, disp: [F; 3]) -> Option<[F; 3]> {
        self.pair_kernel(r2, disp).map(|(_, f)| f)
    }

    fn eval(&self, r2: F, disp: [F; 3]) -> Option<(F, [F; 3])> {
        self.pair_kernel(r2, disp)
    }
}

fn skin_err(err: SkinError) -> MdError {
    match err {
        SkinError::Invalid(msg) => MdError::Invalid(msg),
        SkinError::Guard(msg) => MdError::Neighbor(msg),
    }
}

#[cfg(test)]
mod tests {
    use ndarray::array;

    use molrs::spatial::neighbors::{NeighborList, NeighborPolicy, NeighborsStorage, VerletSkin};
    use molrs::spatial::simbox::SimBox;

    use super::*;

    fn cube(a: F) -> SimBox {
        SimBox::cube(a, ndarray::array![0.0, 0.0, 0.0], [true, true, true]).unwrap()
    }

    fn make_skin(cutoff: F, skin: F, pos: ndarray::ArrayView2<'_, F>) -> VerletSkin {
        VerletSkin::new(
            NeighborList::new(cutoff + skin),
            cutoff,
            NeighborPolicy {
                skin,
                ..NeighborPolicy::default()
            },
            pos,
            cube(20.0),
        )
        .unwrap()
    }

    #[test]
    fn mie_c_is_four_for_12_6() {
        assert!((mie_c(12, 6) - 4.0).abs() < 1e-12);
    }

    #[test]
    fn pair_beyond_cutoff_is_none() {
        let lj = LJ::lj126(1.0, 1.0, 2.0).unwrap();
        assert!(Potential::eval(&lj, 4.01, [2.0025, 0.0, 0.0]).is_none());
    }

    #[test]
    fn unshifted_energy_at_sigma_is_zero() {
        let lj = LJ::new(1.5, 2.0, 5.0, 12, 6, false, false).unwrap();
        let (e, f) = Potential::eval(&lj, 4.0, [2.0, 0.0, 0.0]).unwrap();
        assert!(e.abs() < 1e-12);
        assert!(f[0] > 0.0);
    }

    #[test]
    fn shifted_energy_at_sigma_is_minus_uc() {
        let eps = 1.5;
        let sigma = 2.0;
        let cutoff = 5.0;
        let lj = LJ::lj126(eps, sigma, cutoff).unwrap();
        let e = Potential::calc_energy(&lj, sigma * sigma, [sigma, 0.0, 0.0]).unwrap();
        let sr6_c = (sigma / cutoff).powi(6);
        let e0 = 4.0 * eps * (sr6_c * sr6_c - sr6_c);
        assert!((e + e0).abs() < 1e-12);
    }

    #[test]
    fn smeared_force_vanishes_at_cutoff() {
        let lj = LJ::new(1.0, 1.0, 2.5, 12, 6, true, true).unwrap();
        let rc = 2.5;
        let (e, f) = Potential::eval(&lj, rc * rc, [rc, 0.0, 0.0]).unwrap();
        assert!(e.abs() < 1e-12);
        assert!(f[0].abs() < 1e-12);
    }

    #[test]
    fn eval_list_tracks_live_geometry_inside_half_skin() {
        let pos0 = array![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]];
        let mut skin = make_skin(2.0, 1.0, pos0.view());
        let lj = LJ::lj126(1.0, 1.0, 2.0).unwrap();
        let (_, f0) = lj.eval(&mut skin, pos0.view()).unwrap();
        assert_eq!(skin.rebuild_count, 0);
        let pos1 = array![[0.0, 0.0, 0.0], [1.4, 0.0, 0.0]];
        let (_, f1) = lj.eval(&mut skin, pos1.view()).unwrap();
        assert_eq!(skin.rebuild_count, 0);
        assert!((f1[[0, 0]] - f0[[0, 0]]).abs() > 1e-12);
        let pos2 = array![[0.0, 0.0, 0.0], [1.6, 0.0, 0.0]];
        let _ = lj.eval(&mut skin, pos2.view()).unwrap();
        assert_eq!(skin.rebuild_count, 1);
    }

    #[test]
    fn eval_table_matches_pair_sum() {
        let pos = array![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]];
        let mut nl = NeighborList::new(2.5);
        nl.build(pos.view(), &cube(20.0));
        let table = nl.neighbors(NeighborsStorage::FULL);
        let lj = LJ::lj126(1.0, 1.0, 2.5).unwrap();
        let (e, f) = lj.eval_table(2, &table).unwrap();
        assert!((f[[0, 0]] + f[[1, 0]]).abs() < 1e-12);
        assert!(e.is_finite());
    }
}
