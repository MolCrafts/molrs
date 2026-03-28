//! MMFF94 angle bending and stretch-bend coupling kernels.

use std::collections::HashMap;

use crate::forcefield::Params;
use crate::frame::Frame;
use crate::potential::Potential;
use crate::potential::geometry::{
    accumulate_angle_forces, compute_angle, mag3, sub3, validate_coords,
};
use crate::types::F;

/// md/A -> kcal/mol conversion.
const MDYNE_A_TO_KCAL: f64 = 143.9325;
/// Cubic bend constant (rad^-1), = -0.007 * 180/pi.
const CB_RAD: f64 = -0.40107;

// ---------------------------------------------------------------------------
// MMFFAngleBend: E = (1/2)*143.9325*ka*dth^2*(1 + cb*dth)
// ---------------------------------------------------------------------------

pub struct MMFFAngleBend {
    atom_i: Vec<usize>,
    atom_j: Vec<usize>,
    atom_k: Vec<usize>,
    ka: Vec<F>,
    theta0: Vec<F>,
}

impl Potential for MMFFAngleBend {
    fn eval(&self, coords: &[F]) -> (F, Vec<F>) {
        let _n = validate_coords(coords);
        let mut energy: F = 0.0;
        let mut forces = vec![0.0 as F; coords.len()];
        let conv = MDYNE_A_TO_KCAL as F;
        let cb = CB_RAD as F;

        for idx in 0..self.atom_i.len() {
            let (i, j, k) = (self.atom_i[idx], self.atom_j[idx], self.atom_k[idx]);
            let theta = compute_angle(coords, i, j, k);
            let dth = theta - self.theta0[idx];
            energy += 0.5 * conv * self.ka[idx] * dth * dth * (1.0 + cb * dth);

            let de_dth = conv * self.ka[idx] * dth * (1.0 + 1.5 * cb * dth);
            accumulate_angle_forces(coords, i, j, k, de_dth, &mut forces);
        }
        (energy, forces)
    }
}

pub fn mmff_angle_ctor(
    _sp: &Params,
    tp: &[(&str, &Params)],
    frame: &Frame,
) -> Result<Box<dyn Potential>, String> {
    let type_map: HashMap<&str, &Params> = tp.iter().copied().collect();
    let block = frame
        .get("angles")
        .ok_or("mmff_angle: missing \"angles\" block")?;
    let ic = block.get_uint("atomi").ok_or("missing atomi")?;
    let jc = block.get_uint("atomj").ok_or("missing atomj")?;
    let kc = block.get_uint("atomk").ok_or("missing atomk")?;
    let tc = block.get_string("type").ok_or("missing type")?;

    let n = ic.len();
    let (mut ai, mut aj, mut ak, mut ka, mut th0) = (
        Vec::with_capacity(n),
        Vec::with_capacity(n),
        Vec::with_capacity(n),
        Vec::with_capacity(n),
        Vec::with_capacity(n),
    );

    for idx in 0..n {
        let p = type_map
            .get(tc[idx].as_str())
            .ok_or_else(|| format!("mmff_angle: unknown '{}'", tc[idx]))?;
        ai.push(ic[idx] as usize);
        aj.push(jc[idx] as usize);
        ak.push(kc[idx] as usize);
        ka.push(p.get("ka").ok_or("missing 'ka'")? as F);
        th0.push((p.get("theta0").ok_or("missing 'theta0'")? as F).to_radians());
    }
    Ok(Box::new(MMFFAngleBend {
        atom_i: ai,
        atom_j: aj,
        atom_k: ak,
        ka,
        theta0: th0,
    }))
}

// ---------------------------------------------------------------------------
// MMFFStretchBend: E = 143.9325*(kba_ijk*dr_ij + kba_kji*dr_kj)*dth
// ---------------------------------------------------------------------------

pub struct MMFFStretchBend {
    atom_i: Vec<usize>,
    atom_j: Vec<usize>,
    atom_k: Vec<usize>,
    kba_ijk: Vec<F>,
    kba_kji: Vec<F>,
    r0_ij: Vec<F>,
    r0_kj: Vec<F>,
    theta0: Vec<F>,
}

impl Potential for MMFFStretchBend {
    fn eval(&self, coords: &[F]) -> (F, Vec<F>) {
        let _n = validate_coords(coords);
        let mut energy: F = 0.0;
        let mut forces = vec![0.0 as F; coords.len()];
        let conv = MDYNE_A_TO_KCAL as F;

        for idx in 0..self.atom_i.len() {
            let (i, j, k) = (self.atom_i[idx], self.atom_j[idx], self.atom_k[idx]);
            let rij_vec = sub3(coords, i, coords, j);
            let rkj_vec = sub3(coords, k, coords, j);
            let rij = mag3(rij_vec);
            let rkj = mag3(rkj_vec);
            let theta = compute_angle(coords, i, j, k);
            let dr_ij = rij - self.r0_ij[idx];
            let dr_kj = rkj - self.r0_kj[idx];
            let dth = theta - self.theta0[idx];

            let term = self.kba_ijk[idx] * dr_ij + self.kba_kji[idx] * dr_kj;
            energy += conv * term * dth;

            // dE/ddth = conv * term
            accumulate_angle_forces(coords, i, j, k, conv * term, &mut forces);
            // dE/dr_ij = conv * kba_ijk * dth
            if rij > 1e-12 as F {
                let f_r = -conv * self.kba_ijk[idx] * dth / rij;
                for dim in 0..3 {
                    forces[i * 3 + dim] += f_r * rij_vec[dim];
                    forces[j * 3 + dim] -= f_r * rij_vec[dim];
                }
            }
            // dE/dr_kj = conv * kba_kji * dth
            if rkj > 1e-12 as F {
                let f_r = -conv * self.kba_kji[idx] * dth / rkj;
                for dim in 0..3 {
                    forces[k * 3 + dim] += f_r * rkj_vec[dim];
                    forces[j * 3 + dim] -= f_r * rkj_vec[dim];
                }
            }
        }
        (energy, forces)
    }
}

pub fn mmff_stbn_ctor(
    _sp: &Params,
    tp: &[(&str, &Params)],
    frame: &Frame,
) -> Result<Box<dyn Potential>, String> {
    let type_map: HashMap<&str, &Params> = tp.iter().copied().collect();
    let block = frame.get("angles").ok_or("mmff_stbn: missing \"angles\"")?;
    let ic = block.get_uint("atomi").ok_or("missing atomi")?;
    let jc = block.get_uint("atomj").ok_or("missing atomj")?;
    let kc = block.get_uint("atomk").ok_or("missing atomk")?;
    let tc = block.get_string("stbn_type").ok_or("missing stbn_type")?;

    let n = ic.len();
    let mut pot = MMFFStretchBend {
        atom_i: Vec::with_capacity(n),
        atom_j: Vec::with_capacity(n),
        atom_k: Vec::with_capacity(n),
        kba_ijk: Vec::with_capacity(n),
        kba_kji: Vec::with_capacity(n),
        r0_ij: Vec::with_capacity(n),
        r0_kj: Vec::with_capacity(n),
        theta0: Vec::with_capacity(n),
    };
    for idx in 0..n {
        let p = type_map
            .get(tc[idx].as_str())
            .ok_or_else(|| format!("mmff_stbn: unknown '{}'", tc[idx]))?;
        pot.atom_i.push(ic[idx] as usize);
        pot.atom_j.push(jc[idx] as usize);
        pot.atom_k.push(kc[idx] as usize);
        pot.kba_ijk
            .push(p.get("kba_ijk").ok_or("missing kba_ijk")? as F);
        pot.kba_kji
            .push(p.get("kba_kji").ok_or("missing kba_kji")? as F);
        pot.r0_ij.push(p.get("r0_ij").ok_or("missing r0_ij")? as F);
        pot.r0_kj.push(p.get("r0_kj").ok_or("missing r0_kj")? as F);
        pot.theta0
            .push((p.get("theta0").ok_or("missing theta0")? as F).to_radians());
    }
    Ok(Box::new(pot))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mmff_angle_at_equilibrium() {
        let theta0: F = (109.5 as F).to_radians();
        let pot = MMFFAngleBend {
            atom_i: vec![0],
            atom_j: vec![1],
            atom_k: vec![2],
            ka: vec![0.608],
            theta0: vec![theta0],
        };
        let r = 1.5 as F;
        let half = theta0 / 2.0;
        let coords: Vec<F> = vec![
            r * half.cos(),
            r * half.sin(),
            0.0,
            0.0,
            0.0,
            0.0,
            r * half.cos(),
            -r * half.sin(),
            0.0,
        ];
        let (e, _) = pot.eval(&coords);
        assert!(e.abs() < 1e-4, "angle energy at eq should be ~0, got {}", e);
    }
}
