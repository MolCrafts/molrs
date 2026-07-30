//! UFF angle bend (RDKit `AngleBendContrib`).
//!
//! Per-instance columns: `ka`, `order` (0–4), and for `order==0` the Fourier
//! coefficients `c0`/`c1`/`c2` derived from θ₀.

use crate::ff::forcefield::Params;
use crate::ff::potential::Potential;
use crate::ff::potential::angle::accumulate_angle_forces;
use crate::ff::potential::geometry::{dot3, mag3, sub3, validate_coords};
use molrs::store::frame::Frame;
use molrs::types::F;

pub struct UffAngle {
    atom_i: Vec<usize>,
    atom_j: Vec<usize>,
    atom_k: Vec<usize>,
    ka: Vec<F>,
    order: Vec<u8>,
    c0: Vec<F>,
    c1: Vec<F>,
    c2: Vec<F>,
}

impl Potential for UffAngle {
    fn calc_energy_forces(&self, coords: &[F]) -> (F, Vec<F>) {
        let _n = validate_coords(coords);
        let mut energy = 0.0 as F;
        let mut forces = vec![0.0 as F; coords.len()];

        for idx in 0..self.atom_i.len() {
            let (i, j, k) = (self.atom_i[idx], self.atom_j[idx], self.atom_k[idx]);
            let rji = sub3(coords, i, coords, j);
            let rjk = sub3(coords, k, coords, j);
            let d1 = mag3(rji);
            let d2 = mag3(rjk);
            if d1 < 1e-12 as F || d2 < 1e-12 as F {
                continue;
            }
            let cos_t = (dot3(rji, rjk) / (d1 * d2)).clamp(-1.0, 1.0);
            let sin_sq = (1.0 - cos_t * cos_t).max(0.0);
            let sin_t = sin_sq.sqrt().max(1e-8 as F);
            let cos2 = cos_t * cos_t - sin_sq;

            let order = self.order[idx];
            let ka = self.ka[idx];
            let (term, d_e_d_theta) = if order == 0 {
                let term = self.c0[idx] + self.c1[idx] * cos_t + self.c2[idx] * cos2;
                let d_e = -ka * (self.c1[idx] * sin_t + 2.0 * self.c2[idx] * (2.0 * sin_t * cos_t));
                (term, d_e)
            } else {
                let n = order as F;
                let cos_n = match order {
                    1 => cos_t,
                    2 => cos2,
                    3 => cos_t * (cos_t * cos_t - 3.0 * sin_sq),
                    4 => cos_t.powi(4) - 6.0 * cos_t * cos_t * sin_sq + sin_sq * sin_sq,
                    _ => cos_t,
                };
                let term = (1.0 - cos_n) / (n * n);
                // dE/dθ = (ka/n) * sin(nθ)  with sign from d cos(nθ)/dθ = -n sin(nθ)
                // so d(1-cos(nθ))/dθ = n sin(nθ), and /n² gives sin(nθ)/n
                let sin_n = match order {
                    1 => sin_t,
                    2 => 2.0 * sin_t * cos_t,
                    3 => sin_t * (3.0 - 4.0 * sin_t * sin_t),
                    4 => cos_t * sin_t * (4.0 - 8.0 * sin_t * sin_t),
                    _ => sin_t,
                };
                let d_e = ka * sin_n / n;
                (term, d_e)
            };

            energy += ka * term;
            accumulate_angle_forces(coords, i, j, k, d_e_d_theta, &mut forces);
        }
        (energy, forces)
    }
}

pub fn uff_angle_ctor(
    _sp: &Params,
    _tp: &[(&str, &Params)],
    frame: &Frame,
) -> Result<Box<dyn Potential>, String> {
    let block = frame
        .get("angles")
        .ok_or("uff_angle: missing \"angles\" block")?;
    let i = block.get_uint("atomi").ok_or("uff_angle: missing atomi")?;
    let j = block.get_uint("atomj").ok_or("uff_angle: missing atomj")?;
    let k = block.get_uint("atomk").ok_or("uff_angle: missing atomk")?;
    let ka = block.get_float("ka").ok_or("uff_angle: missing ka")?;
    let order = block.get_float("order").ok_or("uff_angle: missing order")?;
    let c0 = block.get_float("c0").ok_or("uff_angle: missing c0")?;
    let c1 = block.get_float("c1").ok_or("uff_angle: missing c1")?;
    let c2 = block.get_float("c2").ok_or("uff_angle: missing c2")?;
    let n = i.len();
    Ok(Box::new(UffAngle {
        atom_i: (0..n).map(|t| i[t] as usize).collect(),
        atom_j: (0..n).map(|t| j[t] as usize).collect(),
        atom_k: (0..n).map(|t| k[t] as usize).collect(),
        ka: ka.iter().map(|&v| v as F).collect(),
        order: order.iter().map(|&v| v as u8).collect(),
        c0: c0.iter().map(|&v| v as F).collect(),
        c1: c1.iter().map(|&v| v as F).collect(),
        c2: c2.iter().map(|&v| v as F).collect(),
    }))
}
