//! Harmonic angle potential: E = 0.5 * k0 * (theta - theta0)^2

use std::collections::HashMap;

use crate::forcefield::Params;
use crate::frame::Frame;
use crate::potential::Potential;
use crate::potential::geometry::{compute_angle, validate_coords};
use crate::types::F;

/// Harmonic angle potential with pre-resolved flat arrays.
/// `theta0` is stored in radians.
pub struct AngleHarmonic {
    atom_i: Vec<usize>,
    atom_j: Vec<usize>,
    atom_k: Vec<usize>,
    k0: Vec<F>,
    theta0: Vec<F>,
}

impl AngleHarmonic {
    pub fn new(
        atom_i: Vec<usize>,
        atom_j: Vec<usize>,
        atom_k: Vec<usize>,
        k0: Vec<F>,
        theta0: Vec<F>,
    ) -> Self {
        let n = atom_i.len();
        assert_eq!(atom_j.len(), n);
        assert_eq!(atom_k.len(), n);
        assert_eq!(k0.len(), n);
        assert_eq!(theta0.len(), n);
        Self {
            atom_i,
            atom_j,
            atom_k,
            k0,
            theta0,
        }
    }
}

impl Potential for AngleHarmonic {
    fn eval(&self, coords: &[F]) -> (F, Vec<F>) {
        let _n_atoms = validate_coords(coords);
        let mut energy: F = 0.0;
        let mut forces = vec![0.0; coords.len()];

        for idx in 0..self.atom_i.len() {
            let i = self.atom_i[idx];
            let j = self.atom_j[idx];
            let k = self.atom_k[idx];
            let k_spring = self.k0[idx];
            let theta0 = self.theta0[idx];

            let theta = compute_angle(coords, i, j, k);
            let dtheta = theta - theta0;
            energy += 0.5 * k_spring * dtheta * dtheta;

            angle_forces(coords, i, j, k, k_spring, theta0, &mut forces);
        }

        (energy, forces)
    }
}

/// Accumulate angle forces for angle i-j-k with harmonic potential.
fn angle_forces(
    coords: &[F],
    i: usize,
    j: usize,
    k: usize,
    k_spring: F,
    theta0: F,
    forces: &mut [F],
) {
    let rji = [
        coords[i * 3] - coords[j * 3],
        coords[i * 3 + 1] - coords[j * 3 + 1],
        coords[i * 3 + 2] - coords[j * 3 + 2],
    ];
    let rjk = [
        coords[k * 3] - coords[j * 3],
        coords[k * 3 + 1] - coords[j * 3 + 1],
        coords[k * 3 + 2] - coords[j * 3 + 2],
    ];

    let d_ji_sq = rji[0] * rji[0] + rji[1] * rji[1] + rji[2] * rji[2];
    let d_jk_sq = rjk[0] * rjk[0] + rjk[1] * rjk[1] + rjk[2] * rjk[2];
    let d_ji = d_ji_sq.sqrt();
    let d_jk = d_jk_sq.sqrt();

    if d_ji < 1e-12 || d_jk < 1e-12 {
        return;
    }

    let dot = rji[0] * rjk[0] + rji[1] * rjk[1] + rji[2] * rjk[2];
    let cos_theta = (dot / (d_ji * d_jk)).clamp(-1.0, 1.0);
    let theta = cos_theta.acos();
    let sin_theta = (1.0 - cos_theta * cos_theta).sqrt().max(1e-12);

    let de_dtheta = k_spring * (theta - theta0);
    let prefactor = de_dtheta / sin_theta;

    for dim in 0..3 {
        let dcos_dri = rjk[dim] / (d_ji * d_jk) - cos_theta * rji[dim] / d_ji_sq;
        let dcos_drk = rji[dim] / (d_ji * d_jk) - cos_theta * rjk[dim] / d_jk_sq;
        let dcos_drj = -dcos_dri - dcos_drk;

        forces[i * 3 + dim] += prefactor * dcos_dri;
        forces[k * 3 + dim] += prefactor * dcos_drk;
        forces[j * 3 + dim] += prefactor * dcos_drj;
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
    let mut k0_vec = Vec::with_capacity(i_col.len());
    let mut theta0_vec = Vec::with_capacity(i_col.len());

    for idx in 0..i_col.len() {
        let label = &type_col[idx];
        let params = type_map
            .get(label.as_str())
            .ok_or_else(|| format!("AngleHarmonic: unknown angle type '{}'", label))?;
        let k0 = params
            .get("k0")
            .ok_or_else(|| format!("AngleHarmonic type '{}': missing 'k0'", label))?
            as F;
        let theta0_rad = params
            .get("theta0")
            .ok_or_else(|| format!("AngleHarmonic type '{}': missing 'theta0'", label))?
            .to_radians() as F;

        atom_i.push(i_col[idx] as usize);
        atom_j.push(j_col[idx] as usize);
        atom_k.push(k_col[idx] as usize);
        k0_vec.push(k0);
        theta0_vec.push(theta0_rad);
    }

    Ok(Box::new(AngleHarmonic::new(
        atom_i, atom_j, atom_k, k0_vec, theta0_vec,
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

        let (e, _) = pot.eval(&coords);
        assert!(e.abs() < 1e-4);
    }
}
