//! MMFF94 Van der Waals (buffered 14-7) and electrostatic (buffered Coulomb) kernels.

use std::collections::HashMap;

use crate::forcefield::Params;
use crate::frame::Frame;
use crate::potential::Potential;
use crate::potential::geometry::{mag3, sub3, validate_coords};
use crate::types::F;

/// e^2/(4*pi*eps0) in kcal*A/(mol*e^2).
const COULOMB_CONST: f64 = 332.0716;
/// Electrostatic buffering distance (A).
const ELE_DELTA: f64 = 0.05;

// ---------------------------------------------------------------------------
// MMFFVdW: Buffered 14-7 potential
// ---------------------------------------------------------------------------

pub struct MMFFVdW {
    atom_i: Vec<usize>,
    atom_j: Vec<usize>,
    r_star: Vec<F>,
    epsilon: Vec<F>,
}

impl Potential for MMFFVdW {
    fn eval(&self, coords: &[F]) -> (F, Vec<F>) {
        let _n = validate_coords(coords);
        let mut energy: F = 0.0;
        let mut forces = vec![0.0 as F; coords.len()];

        for idx in 0..self.atom_i.len() {
            let (i, j) = (self.atom_i[idx], self.atom_j[idx]);
            let d = sub3(coords, j, coords, i);
            let r = mag3(d);
            if r < 1e-12 as F {
                continue;
            }
            let rs = self.r_star[idx];
            let eps = self.epsilon[idx];

            let rho = r + 0.07 * rs;
            let u = 1.07 * rs / rho;
            let u7 = u * u * u * u * u * u * u;
            let r7 = r * r * r * r * r * r * r;
            let rs7 = rs * rs * rs * rs * rs * rs * rs;
            let v = 1.12 * rs7 / (r7 + 0.12 * rs7);

            energy += eps * u7 * (v - 2.0);

            let du_dr = -u / rho;
            let dv_dr = -7.0 * r * r * r * r * r * r * v / (r7 + 0.12 * rs7);
            let de_dr = eps * (7.0 * u7 / u * du_dr * (v - 2.0) + u7 * dv_dr);

            let factor = -de_dr / r;
            for dim in 0..3 {
                forces[j * 3 + dim] += factor * d[dim];
                forces[i * 3 + dim] -= factor * d[dim];
            }
        }
        (energy, forces)
    }
}

/// Per-atom VdW parameters for combining rules.
struct VdwAtomParams {
    alpha: f64,
    n_eff: f64,
    a_i: f64,
    g_i: f64,
}

/// Compute VdW combining rules from per-atom parameters.
fn vdw_combining(pi: &VdwAtomParams, pj: &VdwAtomParams) -> (F, F) {
    let rs_i = pi.a_i * pi.alpha.powf(0.25);
    let rs_j = pj.a_i * pj.alpha.powf(0.25);
    let gamma = (rs_i - rs_j) / (rs_i + rs_j);
    let r_star = 0.5 * (rs_i + rs_j) * (1.0 + 0.2 * (1.0 - (-12.0 * gamma * gamma).exp()));
    let denom = (pi.alpha / pi.n_eff).sqrt() + (pj.alpha / pj.n_eff).sqrt();
    let epsilon = 181.16 * pi.g_i * pj.g_i * pi.alpha * pj.alpha / (denom * r_star.powi(6));
    (r_star as F, epsilon as F)
}

pub fn mmff_vdw_ctor(
    _sp: &Params,
    tp: &[(&str, &Params)],
    frame: &Frame,
) -> Result<Box<dyn Potential>, String> {
    let type_map: HashMap<&str, &Params> = tp.iter().copied().collect();
    let atoms = frame.get("atoms").ok_or("mmff_vdw: missing \"atoms\"")?;
    let atom_types = atoms
        .get_string("type")
        .ok_or("mmff_vdw: missing atom \"type\"")?;
    let pairs = frame.get("pairs").ok_or("mmff_vdw: missing \"pairs\"")?;
    let ic = pairs.get_uint("atomi").ok_or("missing atomi")?;
    let jc = pairs.get_uint("atomj").ok_or("missing atomj")?;

    let n = ic.len();
    let (mut ai, mut aj, mut rs_vec, mut eps_vec) = (
        Vec::with_capacity(n),
        Vec::with_capacity(n),
        Vec::with_capacity(n),
        Vec::with_capacity(n),
    );

    for idx in 0..n {
        let ti = &atom_types[ic[idx] as usize];
        let tj = &atom_types[jc[idx] as usize];
        let pi = type_map
            .get(ti.as_str())
            .ok_or_else(|| format!("mmff_vdw: unknown atom type '{}'", ti))?;
        let pj = type_map
            .get(tj.as_str())
            .ok_or_else(|| format!("mmff_vdw: unknown atom type '{}'", tj))?;

        let to_vdw = |p: &Params, label: &str| -> Result<VdwAtomParams, String> {
            let get = |k: &str| {
                p.get(k)
                    .ok_or_else(|| format!("mmff_vdw type '{}': missing '{}'", label, k))
            };
            Ok(VdwAtomParams {
                alpha: get("alpha")?,
                n_eff: get("n_eff")?,
                a_i: get("a_i")?,
                g_i: get("g_i")?,
            })
        };
        let (rs, eps) = vdw_combining(&to_vdw(pi, ti)?, &to_vdw(pj, tj)?);
        ai.push(ic[idx] as usize);
        aj.push(jc[idx] as usize);
        rs_vec.push(rs);
        eps_vec.push(eps);
    }
    Ok(Box::new(MMFFVdW {
        atom_i: ai,
        atom_j: aj,
        r_star: rs_vec,
        epsilon: eps_vec,
    }))
}

// ---------------------------------------------------------------------------
// MMFFElectrostatic: E = 332.0716 * qi * qj / (D * (r + delta))
// ---------------------------------------------------------------------------

pub struct MMFFElectrostatic {
    atom_i: Vec<usize>,
    atom_j: Vec<usize>,
    qi_qj: Vec<F>,
    dielectric: F,
    scale_14: Vec<F>,
}

impl Potential for MMFFElectrostatic {
    fn eval(&self, coords: &[F]) -> (F, Vec<F>) {
        let _n = validate_coords(coords);
        let mut energy: F = 0.0;
        let mut forces = vec![0.0 as F; coords.len()];
        let conv = COULOMB_CONST as F;
        let delta = ELE_DELTA as F;

        for idx in 0..self.atom_i.len() {
            let (i, j) = (self.atom_i[idx], self.atom_j[idx]);
            let d = sub3(coords, j, coords, i);
            let r = mag3(d);
            let r_buf = r + delta;
            let qq = self.qi_qj[idx] * self.scale_14[idx];
            energy += conv * qq / (self.dielectric * r_buf);

            if r < 1e-12 as F {
                continue;
            }
            let de_dr = -conv * qq / (self.dielectric * r_buf * r_buf);
            let factor = -de_dr / r;
            for dim in 0..3 {
                forces[j * 3 + dim] += factor * d[dim];
                forces[i * 3 + dim] -= factor * d[dim];
            }
        }
        (energy, forces)
    }
}

pub fn mmff_ele_ctor(
    sp: &Params,
    _tp: &[(&str, &Params)],
    frame: &Frame,
) -> Result<Box<dyn Potential>, String> {
    let dielectric = sp.get("dielectric").unwrap_or(1.0) as F;
    let atoms = frame.get("atoms").ok_or("mmff_ele: missing \"atoms\"")?;
    let charges = atoms
        .get_float("charge")
        .ok_or("mmff_ele: missing atom \"charge\" column")?;
    let pairs = frame.get("pairs").ok_or("mmff_ele: missing \"pairs\"")?;
    let ic = pairs.get_uint("atomi").ok_or("missing atomi")?;
    let jc = pairs.get_uint("atomj").ok_or("missing atomj")?;
    let is_14 = pairs.get_bool("is_14");

    let n = ic.len();
    let (mut ai, mut aj, mut qq, mut s14) = (
        Vec::with_capacity(n),
        Vec::with_capacity(n),
        Vec::with_capacity(n),
        Vec::with_capacity(n),
    );

    for idx in 0..n {
        let qi = charges[ic[idx] as usize] as F;
        let qj = charges[jc[idx] as usize] as F;
        ai.push(ic[idx] as usize);
        aj.push(jc[idx] as usize);
        qq.push(qi * qj);
        s14.push(if is_14.is_some_and(|b| b[idx]) {
            0.75
        } else {
            1.0
        });
    }
    Ok(Box::new(MMFFElectrostatic {
        atom_i: ai,
        atom_j: aj,
        qi_qj: qq,
        dielectric,
        scale_14: s14,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mmff_vdw_combining() {
        let p = VdwAtomParams {
            alpha: 1.050,
            n_eff: 2.490,
            a_i: 3.890,
            g_i: 1.282,
        };
        let (rs, eps) = vdw_combining(&p, &p);
        assert!(rs > 0.0, "r_star should be positive");
        assert!(eps > 0.0, "epsilon should be positive");
    }

    #[test]
    fn test_mmff_vdw_energy() {
        let pot = MMFFVdW {
            atom_i: vec![0],
            atom_j: vec![1],
            r_star: vec![1.94],
            epsilon: vec![0.02],
        };
        let coords: Vec<F> = vec![0.0, 0.0, 0.0, 3.0, 0.0, 0.0];
        let (e, forces) = pot.eval(&coords);
        assert!(e.is_finite());
        for dim in 0..3 {
            let sum = forces[dim] + forces[3 + dim];
            assert!(sum.abs() < 1e-4, "dim {}: sum = {}", dim, sum);
        }
    }

    #[test]
    fn test_mmff_electrostatic() {
        let pot = MMFFElectrostatic {
            atom_i: vec![0],
            atom_j: vec![1],
            qi_qj: vec![0.5 * -0.5],
            dielectric: 1.0,
            scale_14: vec![1.0],
        };
        let coords: Vec<F> = vec![0.0, 0.0, 0.0, 2.0, 0.0, 0.0];
        let (e, forces) = pot.eval(&coords);
        assert!(e < 0.0, "opposite charges should give negative energy");
        let sum = forces[0] + forces[3];
        assert!(sum.abs() < 1e-3, "force sum = {}", sum);
    }
}
