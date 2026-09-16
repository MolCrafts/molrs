//! Harmonic angle potential: E = 0.5 * k * (theta - theta0)^2

use std::collections::HashMap;

use crate::ff::forcefield::Params;
use crate::ff::potential::Potential;
use crate::ff::potential::geometry::{compute_angle, validate_coords};
use molrs::store::frame::Frame;
use molrs::types::F;

/// Harmonic angle potential with pre-resolved flat arrays.
/// `theta0` is stored in radians.
pub struct AngleHarmonic {
    atom_i: Vec<usize>,
    atom_j: Vec<usize>,
    atom_k: Vec<usize>,
    k: Vec<F>,
    theta0: Vec<F>,
}

impl AngleHarmonic {
    pub fn new(
        atom_i: Vec<usize>,
        atom_j: Vec<usize>,
        atom_k: Vec<usize>,
        k: Vec<F>,
        theta0: Vec<F>,
    ) -> Self {
        let n = atom_i.len();
        assert_eq!(atom_j.len(), n);
        assert_eq!(atom_k.len(), n);
        assert_eq!(k.len(), n);
        assert_eq!(theta0.len(), n);
        Self {
            atom_i,
            atom_j,
            atom_k,
            k,
            theta0,
        }
    }
}

impl Potential for AngleHarmonic {
    fn calc_energy_forces(&self, coords: &[F]) -> (F, Vec<F>) {
        let _n_atoms = validate_coords(coords);
        let mut energy: F = 0.0;
        let mut forces = vec![0.0; coords.len()];

        for idx in 0..self.atom_i.len() {
            let i = self.atom_i[idx];
            let j = self.atom_j[idx];
            let k = self.atom_k[idx];
            let k_spring = self.k[idx];
            let theta0 = self.theta0[idx];

            let theta = compute_angle(coords, i, j, k);
            let dtheta = theta - theta0;
            energy += 0.5 * k_spring * dtheta * dtheta;

            // dE/dtheta = k (theta - theta0)
            super::accumulate_angle_forces(coords, i, j, k, k_spring * dtheta, &mut forces);
        }

        (energy, forces)
    }
}

/// Construct an [`AngleHarmonic`] from style params, type params, and Frame topology.
pub fn angle_harmonic_ctor(
    _style_params: &Params,
    type_params: &[(&str, &Params)],
    frame: &Frame,
) -> Result<Box<dyn Potential>, String> {
    let type_map: HashMap<&str, &Params> = type_params.iter().copied().collect();

    let block = frame
        .get("angles")
        .ok_or_else(|| "AngleHarmonic: frame missing \"angles\" block".to_string())?;
    let i_col = block
        .get_uint("atomi")
        .ok_or_else(|| "AngleHarmonic: angles block missing \"atomi\" column".to_string())?;
    let j_col = block
        .get_uint("atomj")
        .ok_or_else(|| "AngleHarmonic: angles block missing \"atomj\" column".to_string())?;
    let k_col = block
        .get_uint("atomk")
        .ok_or_else(|| "AngleHarmonic: angles block missing \"atomk\" column".to_string())?;
    let type_col = block
        .get_string("type")
        .ok_or_else(|| "AngleHarmonic: angles block missing \"type\" column".to_string())?;

    let mut atom_i = Vec::with_capacity(i_col.len());
    let mut atom_j = Vec::with_capacity(i_col.len());
    let mut atom_k = Vec::with_capacity(i_col.len());
    let mut k_vec = Vec::with_capacity(i_col.len());
    let mut theta0_vec = Vec::with_capacity(i_col.len());

    for idx in 0..i_col.len() {
        let label = &type_col[idx];
        let params = type_map
            .get(label.as_str())
            .ok_or_else(|| format!("AngleHarmonic: unknown angle type '{}'", label))?;
        // `k` is the one spelling (spec ff-params-01); the `k0` alias is gone.
        let k = params
            .get("k")
            .ok_or_else(|| format!("AngleHarmonic type '{}': missing 'k'", label))?
            as F;
        // theta0 is consumed in radians; readers normalize at their boundary.
        let theta0_rad = params
            .get("theta0")
            .ok_or_else(|| format!("AngleHarmonic type '{}': missing 'theta0'", label))?
            as F;

        atom_i.push(i_col[idx] as usize);
        atom_j.push(j_col[idx] as usize);
        atom_k.push(k_col[idx] as usize);
        k_vec.push(k);
        theta0_vec.push(theta0_rad);
    }

    Ok(Box::new(AngleHarmonic::new(
        atom_i, atom_j, atom_k, k_vec, theta0_vec,
    )))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_angle_harmonic_energy() {
        let theta0: F = std::f64::consts::FRAC_PI_2 as F;
        let pot = AngleHarmonic::new(vec![0], vec![1], vec![2], vec![50.0], vec![theta0]);
        let coords: Vec<F> = vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0];

        let (e, _) = pot.calc_energy_forces(&coords);
        assert!(e.abs() < 1e-4);
    }
}
