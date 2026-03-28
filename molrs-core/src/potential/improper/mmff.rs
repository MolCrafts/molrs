//! MMFF94 out-of-plane bending: E = 0.5*143.9325*koop*chi^2 (Wilson angle)

use std::collections::HashMap;

use crate::forcefield::Params;
use crate::frame::Frame;
use crate::potential::Potential;
use crate::potential::geometry::{cross3, dot3, mag3, sub3, validate_coords};
use crate::types::F;

/// md/A -> kcal/mol conversion.
const MDYNE_A_TO_KCAL: f64 = 143.9325;

pub struct MMFFOutOfPlane {
    atom_i: Vec<usize>,
    atom_j: Vec<usize>,
    atom_k: Vec<usize>,
    atom_l: Vec<usize>,
    koop: Vec<F>,
}

impl Potential for MMFFOutOfPlane {
    fn eval(&self, coords: &[F]) -> (F, Vec<F>) {
        let _n = validate_coords(coords);
        let mut energy: F = 0.0;
        let mut forces = vec![0.0 as F; coords.len()];
        let conv = MDYNE_A_TO_KCAL as F;

        for idx in 0..self.atom_i.len() {
            let (i, j, k, l) = (
                self.atom_i[idx],
                self.atom_j[idx],
                self.atom_k[idx],
                self.atom_l[idx],
            );
            let a = sub3(coords, i, coords, j); // ji
            let b = sub3(coords, k, coords, j); // jk
            let c = sub3(coords, l, coords, j); // jl
            let n = cross3(a, b);
            let n_mag = mag3(n);
            let c_mag = mag3(c);
            if n_mag < 1e-12 as F || c_mag < 1e-12 as F {
                continue;
            }

            let sin_chi = (dot3(c, n) / (c_mag * n_mag)).clamp(-1.0, 1.0);
            let chi = sin_chi.asin();
            let cos_chi = chi.cos();
            energy += 0.5 * conv * self.koop[idx] * chi * chi;

            if cos_chi.abs() < 1e-12 as F {
                continue;
            }
            let prefactor = conv * self.koop[idx] * chi / cos_chi;

            // d(sin chi)/d(r_l)
            let inv_cn = 1.0 / (c_mag * n_mag);
            for dim in 0..3 {
                let dsdc = n[dim] * inv_cn - sin_chi * c[dim] / (c_mag * c_mag);
                forces[l * 3 + dim] += -prefactor * dsdc;
                forces[j * 3 + dim] -= -prefactor * dsdc;
            }
            // d(sin chi)/d(r_i)  (a = r_i - r_j, affects n = a x b)
            let bxc = cross3(b, c);
            let bxn = cross3(b, n);
            for dim in 0..3 {
                let dsda = bxc[dim] * inv_cn - sin_chi * bxn[dim] / (n_mag * n_mag);
                forces[i * 3 + dim] += -prefactor * dsda;
                forces[j * 3 + dim] -= -prefactor * dsda;
            }
            // d(sin chi)/d(r_k)  (b = r_k - r_j, affects n = a x b)
            let axc = cross3(a, c);
            let axn = cross3(a, n);
            for dim in 0..3 {
                let dsdb = -axc[dim] * inv_cn + sin_chi * axn[dim] / (n_mag * n_mag);
                forces[k * 3 + dim] += -prefactor * dsdb;
                forces[j * 3 + dim] -= -prefactor * dsdb;
            }
        }
        (energy, forces)
    }
}

pub fn mmff_oop_ctor(
    _sp: &Params,
    tp: &[(&str, &Params)],
    frame: &Frame,
) -> Result<Box<dyn Potential>, String> {
    let type_map: HashMap<&str, &Params> = tp.iter().copied().collect();
    let block = frame
        .get("impropers")
        .ok_or("mmff_oop: missing \"impropers\"")?;
    let ic = block.get_uint("atomi").ok_or("missing atomi")?;
    let jc = block.get_uint("atomj").ok_or("missing atomj")?;
    let kc = block.get_uint("atomk").ok_or("missing atomk")?;
    let lc = block.get_uint("atoml").ok_or("missing atoml")?;
    let tc = block.get_string("type").ok_or("missing type")?;

    let n = ic.len();
    let (mut ai, mut aj, mut ak, mut al, mut koop) = (
        Vec::with_capacity(n),
        Vec::with_capacity(n),
        Vec::with_capacity(n),
        Vec::with_capacity(n),
        Vec::with_capacity(n),
    );

    for idx in 0..n {
        let p = type_map
            .get(tc[idx].as_str())
            .ok_or_else(|| format!("mmff_oop: unknown '{}'", tc[idx]))?;
        ai.push(ic[idx] as usize);
        aj.push(jc[idx] as usize);
        ak.push(kc[idx] as usize);
        al.push(lc[idx] as usize);
        koop.push(p.get("koop").ok_or("missing koop")? as F);
    }
    Ok(Box::new(MMFFOutOfPlane {
        atom_i: ai,
        atom_j: aj,
        atom_k: ak,
        atom_l: al,
        koop,
    }))
}
