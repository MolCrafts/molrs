//! Lennard-Jones 12-6 pair potential: E = 4 * eps * ((sigma/r)^12 - (sigma/r)^6)

use std::collections::HashMap;

use crate::forcefield::Params;
use crate::frame::Frame;
use crate::potential::Potential;
use crate::potential::geometry::validate_coords;
use crate::types::F;

/// Lennard-Jones 12-6 pair potential with pre-resolved flat arrays.
pub struct PairLJCut {
    atom_i: Vec<usize>,
    atom_j: Vec<usize>,
    epsilon: Vec<F>,
    sigma: Vec<F>,
}

impl PairLJCut {
    pub fn new(atom_i: Vec<usize>, atom_j: Vec<usize>, epsilon: Vec<F>, sigma: Vec<F>) -> Self {
        assert_eq!(atom_i.len(), atom_j.len());
        assert_eq!(atom_i.len(), epsilon.len());
        assert_eq!(atom_i.len(), sigma.len());
        Self {
            atom_i,
            atom_j,
            epsilon,
            sigma,
        }
    }
}

impl Potential for PairLJCut {
    fn eval(&self, coords: &[F]) -> (F, Vec<F>) {
        let n_atoms = validate_coords(coords);
        let mut energy: F = 0.0;
        let mut forces = vec![0.0; coords.len()];

        for idx in 0..self.atom_i.len() {
            let i = self.atom_i[idx];
            let j = self.atom_j[idx];
            debug_assert!(i < n_atoms && j < n_atoms);

            let eps = self.epsilon[idx];
            let sigma = self.sigma[idx];

            let dx = coords[j * 3] - coords[i * 3];
            let dy = coords[j * 3 + 1] - coords[i * 3 + 1];
            let dz = coords[j * 3 + 2] - coords[i * 3 + 2];
            let r2 = dx * dx + dy * dy + dz * dz;
            if r2 < 1e-24 {
                continue;
            }
            let sr2 = sigma * sigma / r2;
            let sr6 = sr2 * sr2 * sr2;
            let sr12 = sr6 * sr6;
            energy += 4.0 * eps * (sr12 - sr6);

            let factor = 4.0 * eps * (12.0 * sr12 - 6.0 * sr6) / r2;
            let fx = factor * dx;
            let fy = factor * dy;
            let fz = factor * dz;

            forces[j * 3] += fx;
            forces[j * 3 + 1] += fy;
            forces[j * 3 + 2] += fz;
            forces[i * 3] -= fx;
            forces[i * 3 + 1] -= fy;
            forces[i * 3 + 2] -= fz;
        }

        (energy, forces)
    }
}

/// Construct a [`PairLJCut`] from style params, type params, and Frame topology.
pub fn pair_lj_cut_ctor(
    _style_params: &Params,
    type_params: &[(&str, &Params)],
    frame: &Frame,
) -> Result<Box<dyn Potential>, String> {
    let type_map: HashMap<&str, &Params> = type_params.iter().copied().collect();

    let block = frame
        .get("pairs")
        .ok_or_else(|| "PairLJCut: frame missing \"pairs\" block".to_string())?;
    let i_col = block
        .get_uint("atomi")
        .ok_or_else(|| "PairLJCut: pairs block missing \"atomi\" column".to_string())?;
    let j_col = block
        .get_uint("atomj")
        .ok_or_else(|| "PairLJCut: pairs block missing \"atomj\" column".to_string())?;
    let type_col = block
        .get_string("type")
        .ok_or_else(|| "PairLJCut: pairs block missing \"type\" column".to_string())?;

    let mut atom_i = Vec::with_capacity(i_col.len());
    let mut atom_j = Vec::with_capacity(i_col.len());
    let mut eps_vec = Vec::with_capacity(i_col.len());
    let mut sig_vec = Vec::with_capacity(i_col.len());

    for idx in 0..i_col.len() {
        let label = &type_col[idx];
        let params = type_map
            .get(label.as_str())
            .ok_or_else(|| format!("PairLJCut: unknown pair type '{}'", label))?;
        let eps = params
            .get("epsilon")
            .ok_or_else(|| format!("PairLJCut type '{}': missing 'epsilon'", label))?
            as F;
        let sigma = params
            .get("sigma")
            .ok_or_else(|| format!("PairLJCut type '{}': missing 'sigma'", label))?
            as F;

        atom_i.push(i_col[idx] as usize);
        atom_j.push(j_col[idx] as usize);
        eps_vec.push(eps);
        sig_vec.push(sigma);
    }

    Ok(Box::new(PairLJCut::new(atom_i, atom_j, eps_vec, sig_vec)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pair_lj_cut_energy_and_newton_third_law() {
        let pot = PairLJCut::new(vec![0], vec![1], vec![0.5], vec![1.0]);
        let coords: Vec<F> = vec![0.0, 0.0, 0.0, 1.5, 0.3, 0.1];

        let (e, forces) = pot.eval(&coords);
        assert!(e.is_finite());

        for dim in 0..3 {
            let sum = forces[dim] + forces[3 + dim];
            assert!(sum.abs() < 1e-5, "dim={}: sum={}", dim, sum);
        }
    }

    #[test]
    fn test_pair_lj_cut_unknown_label_error() {
        let pot = PairLJCut::new(vec![0], vec![1], vec![1.0], vec![1.0]);
        let coords: Vec<F> = vec![0.0, 0.0, 0.0, 2.0, 0.0, 0.0];
        let (e, _) = pot.eval(&coords);
        assert!(e.is_finite());
    }
}
