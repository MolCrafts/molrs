//! Unique LJ / Mie pair kernel (`lj/cut`).
//!
//! Pair source is fixed at construction:
//! - [`LJCut::new`] / [`LJCut::lj126`] — uniform ε/σ, loop-fed pairs (MD)
//! - [`LJCut::compiled`] — per-pair ε/σ from a ForceField `pairs` block
//!
//! Arithmetic uses `inv_r2 = 1/r2`. Degenerate pairs `r2 < 1e-24` are skipped.

use std::collections::HashMap;

use crate::ff::forcefield::Params;
use crate::ff::potential::Potential;
use crate::ff::potential::geometry::validate_coords;
use crate::ff::potential::pair::PairPotential;
use molrs::spatial::neighbors::{Neighbors, VerletSkin};
use molrs::store::frame::Frame;
use molrs::types::F;
use ndarray::{Array2, ArrayView2};

const MIN_R2: F = 1e-24;

#[derive(Clone, Debug)]
enum PairSource {
    Loop,
    Compiled {
        atom_i: Vec<usize>,
        atom_j: Vec<usize>,
        epsilon: Vec<F>,
        sigma: Vec<F>,
    },
}

/// LAMMPS `pair_style lj/cut`.
#[derive(Clone, Debug)]
pub struct LJCut {
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
    source: PairSource,
}

fn mie_c(n: i32, m: i32) -> F {
    let n = n as F;
    let m = m as F;
    (n / (n - m)) * (n / m).powf(m / (n - m))
}

impl LJCut {
    pub fn new(
        epsilon: F,
        sigma: F,
        cutoff: F,
        n: i32,
        m: i32,
        shifted: bool,
        smeared: bool,
    ) -> Result<Self, String> {
        if epsilon <= 0.0 || sigma <= 0.0 || cutoff <= 0.0 {
            return Err("LJCut requires epsilon, sigma, cutoff > 0".into());
        }
        if m <= 0 || n <= m {
            return Err(format!(
                "LJCut exponents must satisfy n > m > 0, got n={n}, m={m}"
            ));
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
            source: PairSource::Loop,
        })
    }

    pub fn lj126(epsilon: F, sigma: F, cutoff: F) -> Result<Self, String> {
        Self::new(epsilon, sigma, cutoff, 12, 6, true, false)
    }

    pub fn compiled(
        atom_i: Vec<usize>,
        atom_j: Vec<usize>,
        epsilon: Vec<F>,
        sigma: Vec<F>,
    ) -> Self {
        assert_eq!(atom_i.len(), atom_j.len());
        assert_eq!(atom_i.len(), epsilon.len());
        assert_eq!(atom_i.len(), sigma.len());
        Self {
            epsilon: 1.0,
            sigma: 1.0,
            cutoff: F::INFINITY,
            n: 12,
            m: 6,
            shifted: false,
            smeared: false,
            cutoff2: F::INFINITY,
            ceps: 4.0,
            e0: 0.0,
            f_rc: 0.0,
            source: PairSource::Compiled {
                atom_i,
                atom_j,
                epsilon,
                sigma,
            },
        }
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

    #[allow(clippy::too_many_arguments)]
    fn pair_kernel_params(
        &self,
        r2: F,
        disp: [F; 3],
        sigma: F,
        ceps: F,
        e0: F,
        f_rc: F,
        cutoff2: F,
        n: i32,
        m: i32,
    ) -> Option<(F, [F; 3])> {
        if r2 < MIN_R2 || r2 > cutoff2 {
            return None;
        }
        if n == 12 && m == 6 {
            let inv_r2 = 1.0 / r2;
            let sr2 = sigma * sigma * inv_r2;
            let sr6 = sr2 * sr2 * sr2;
            let sr12 = sr6 * sr6;
            let mut energy = ceps * (sr12 - sr6) - e0;
            let mut fac = ceps * (12.0 * sr12 - 6.0 * sr6) * inv_r2;
            if f_rc != 0.0 {
                let r = r2.sqrt();
                energy += (r - self.cutoff) * f_rc;
                fac -= f_rc / r;
            }
            return Some((energy, [fac * disp[0], fac * disp[1], fac * disp[2]]));
        }
        let sr = sigma / r2.sqrt();
        let sr_n = sr.powi(n);
        let sr_m = sr.powi(m);
        let mut energy = ceps * (sr_n - sr_m) - e0;
        let mut fac = ceps * ((n as F) * sr_n - (m as F) * sr_m) / r2;
        if f_rc != 0.0 {
            let r = r2.sqrt();
            energy += (r - self.cutoff) * f_rc;
            fac -= f_rc / r;
        }
        Some((energy, [fac * disp[0], fac * disp[1], fac * disp[2]]))
    }

    fn pair_kernel(&self, r2: F, disp: [F; 3]) -> Option<(F, [F; 3])> {
        self.pair_kernel_params(
            r2,
            disp,
            self.sigma,
            self.ceps,
            self.e0,
            self.f_rc,
            self.cutoff2,
            self.n,
            self.m,
        )
    }

    fn fold_compiled(&self, coords: &[F]) -> (F, Vec<F>) {
        let PairSource::Compiled {
            atom_i,
            atom_j,
            epsilon,
            sigma,
        } = &self.source
        else {
            return (0.0, vec![0.0; coords.len()]);
        };
        let n_atoms = validate_coords(coords);
        let mut energy = 0.0;
        let mut forces = vec![0.0; coords.len()];
        for idx in 0..atom_i.len() {
            let i = atom_i[idx];
            let j = atom_j[idx];
            debug_assert!(i < n_atoms && j < n_atoms);
            let dx = coords[j * 3] - coords[i * 3];
            let dy = coords[j * 3 + 1] - coords[i * 3 + 1];
            let dz = coords[j * 3 + 2] - coords[i * 3 + 2];
            let r2 = dx * dx + dy * dy + dz * dz;
            let ceps = 4.0 * epsilon[idx];
            let Some((e, f)) = self.pair_kernel_params(
                r2,
                [dx, dy, dz],
                sigma[idx],
                ceps,
                0.0,
                0.0,
                F::INFINITY,
                12,
                6,
            ) else {
                continue;
            };
            energy += e;
            forces[j * 3] += f[0];
            forces[j * 3 + 1] += f[1];
            forces[j * 3 + 2] += f[2];
            forces[i * 3] -= f[0];
            forces[i * 3 + 1] -= f[1];
            forces[i * 3 + 2] -= f[2];
        }
        (energy, forces)
    }

    fn fold_neighbors(&self, coords: &[F], pairs: &Neighbors) -> (F, Vec<F>) {
        let mut forces = vec![0.0; coords.len()];
        let Some(disp) = pairs.disp() else {
            return (0.0, forces);
        };
        let i = pairs.query_point_indices();
        let j = pairs.point_indices();
        let d2 = pairs.dist_sq();
        let mut energy = 0.0;
        for p in 0..i.len() {
            let ia = i[p] as usize;
            let ja = j[p] as usize;
            let d = [disp[[p, 0]], disp[[p, 1]], disp[[p, 2]]];
            let r2 = match d2 {
                Some(col) => col[p],
                None => d[0] * d[0] + d[1] * d[1] + d[2] * d[2],
            };
            let Some((e, f)) = self.pair_kernel(r2, d) else {
                continue;
            };
            energy += e;
            let bj = 3 * ja;
            let bi = 3 * ia;
            forces[bj] += f[0];
            forces[bj + 1] += f[1];
            forces[bj + 2] += f[2];
            forces[bi] -= f[0];
            forces[bi + 1] -= f[1];
            forces[bi + 2] -= f[2];
        }
        (energy, forces)
    }

    /// Compose `pairs_at` + table fold. Neighbour search stays the caller's.
    pub fn eval(
        &self,
        neighbors: &mut VerletSkin,
        pos: ArrayView2<'_, F>,
    ) -> Result<(F, Array2<F>), String> {
        let table = neighbors.pairs_at(pos).map_err(|e| e.to_string())?;
        let n = pos.nrows();
        let coords: Vec<F> = match pos.as_slice() {
            Some(s) => s.to_vec(),
            None => pos.iter().copied().collect(),
        };
        let (e, f) = self.calc_energy_forces_with_pairs(&coords, table);
        let forces = Array2::from_shape_vec((n, 3), f).map_err(|err| err.to_string())?;
        Ok((e, forces))
    }
}

impl PairPotential for LJCut {
    fn pair_energy(&self, r2: F, disp: [F; 3]) -> Option<F> {
        self.pair_kernel(r2, disp).map(|(e, _)| e)
    }
    fn pair_force(&self, r2: F, disp: [F; 3]) -> Option<[F; 3]> {
        self.pair_kernel(r2, disp).map(|(_, f)| f)
    }
    fn pair_eval(&self, r2: F, disp: [F; 3]) -> Option<(F, [F; 3])> {
        self.pair_kernel(r2, disp)
    }
}

impl Potential for LJCut {
    fn calc_energy_forces(&self, coords: &[F]) -> (F, Vec<F>) {
        match &self.source {
            PairSource::Compiled { .. } => self.fold_compiled(coords),
            PairSource::Loop => (0.0, vec![0.0; coords.len()]),
        }
    }

    fn calc_energy_forces_with_pairs(&self, coords: &[F], pairs: &Neighbors) -> (F, Vec<F>) {
        match &self.source {
            PairSource::Compiled { .. } => self.fold_compiled(coords),
            PairSource::Loop => self.fold_neighbors(coords, pairs),
        }
    }
}

/// Construct a compiled [`LJCut`] from per-atom-type params + a neighbour list.
pub fn pair_lj_cut_ctor(
    style_params: &Params,
    type_params: &[(&str, &Params)],
    frame: &Frame,
) -> Result<Box<dyn Potential>, String> {
    let type_map: HashMap<&str, &Params> = type_params.iter().copied().collect();
    let scale_14 = style_params.get("lj14scale").unwrap_or(1.0) as F;

    let atoms = frame
        .get("atoms")
        .ok_or_else(|| "LJCut: frame missing \"atoms\" block".to_string())?;
    let atom_types = atoms
        .get_string("type")
        .ok_or_else(|| "LJCut: atoms block missing \"type\" column".to_string())?;
    let block = frame
        .get("pairs")
        .ok_or_else(|| "LJCut: frame missing \"pairs\" block".to_string())?;
    let i_col = block
        .get_uint("atomi")
        .ok_or_else(|| "LJCut: pairs block missing \"atomi\" column".to_string())?;
    let j_col = block
        .get_uint("atomj")
        .ok_or_else(|| "LJCut: pairs block missing \"atomj\" column".to_string())?;
    let is_14 = block.get_bool("is_14");

    let n = i_col.len();
    let mut atom_i = Vec::with_capacity(n);
    let mut atom_j = Vec::with_capacity(n);
    let mut eps_vec = Vec::with_capacity(n);
    let mut sig_vec = Vec::with_capacity(n);

    let per_atom = |t: &str| -> Result<(F, F), String> {
        let p = type_map
            .get(t)
            .ok_or_else(|| format!("LJCut: unknown atom type '{t}'"))?;
        let eps = p
            .get("epsilon")
            .ok_or_else(|| format!("LJCut type '{t}': missing 'epsilon'"))? as F;
        let sigma = p
            .get("sigma")
            .ok_or_else(|| format!("LJCut type '{t}': missing 'sigma'"))? as F;
        Ok((eps, sigma))
    };

    for idx in 0..n {
        let (eps_i, sig_i) = per_atom(&atom_types[i_col[idx] as usize])?;
        let (eps_j, sig_j) = per_atom(&atom_types[j_col[idx] as usize])?;
        let mut eps = (eps_i * eps_j).sqrt();
        let sigma = 0.5 * (sig_i + sig_j);
        if is_14.is_some_and(|b| b[idx]) {
            eps *= scale_14;
        }
        atom_i.push(i_col[idx] as usize);
        atom_j.push(j_col[idx] as usize);
        eps_vec.push(eps);
        sig_vec.push(sigma);
    }

    Ok(Box::new(LJCut::compiled(atom_i, atom_j, eps_vec, sig_vec)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mie_c_is_four_for_12_6() {
        assert!((mie_c(12, 6) - 4.0).abs() < 1e-12);
    }

    #[test]
    fn loop_without_pairs_is_explicit_zero() {
        let lj = LJCut::lj126(1.0, 1.0, 2.5).unwrap();
        let coords: Vec<F> = vec![0.0, 0.0, 0.0, 1.1, 0.0, 0.0];
        let (e, f) = Potential::calc_energy_forces(&lj, &coords);
        assert_eq!(e, 0.0);
        assert!(f.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn compiled_newton_third_law() {
        let pot = LJCut::compiled(vec![0], vec![1], vec![0.5], vec![1.0]);
        let coords: Vec<F> = vec![0.0, 0.0, 0.0, 1.5, 0.3, 0.1];
        let (e, forces) = pot.calc_energy_forces(&coords);
        assert!(e.is_finite());
        for dim in 0..3 {
            assert!((forces[dim] + forces[3 + dim]).abs() < 1e-12);
        }
    }

    #[test]
    fn compiled_ignores_loop_pairs() {
        let pot = LJCut::compiled(vec![0], vec![1], vec![1.0], vec![1.0]);
        let coords: Vec<F> = vec![0.0, 0.0, 0.0, 2.0, 0.0, 0.0];
        let (e0, _) = pot.calc_energy_forces(&coords);
        let extra = Neighbors::from_pairs(
            [molrs::spatial::neighbors::NeighborPair {
                i: 0,
                j: 1,
                dist_sq: 1.0,
                disp: [1.0, 0.0, 0.0],
            }],
            molrs::spatial::neighbors::NeighborsStorage::FULL,
            molrs::spatial::neighbors::QueryMode::SelfQuery { num_points: 2 },
        );
        let (e1, _) = pot.calc_energy_forces_with_pairs(&coords, &extra);
        assert_eq!(e0, e1);
    }

    #[test]
    fn unshifted_energy_at_sigma_is_zero() {
        let lj = LJCut::new(1.5, 2.0, 5.0, 12, 6, false, false).unwrap();
        let (e, f) = lj.pair_eval(4.0, [2.0, 0.0, 0.0]).unwrap();
        assert!(e.abs() < 1e-12);
        assert!(f[0] > 0.0);
    }
}
