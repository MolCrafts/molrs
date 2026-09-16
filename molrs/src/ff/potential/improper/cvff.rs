//! CVFF improper (LAMMPS `improper_style cvff`):
//!
//! E(χ) = k · [1 + s · cos(n·χ)]
//!
//! `sign` is s = ±1 (a sign, **not** a phase — hence its own canonical name)
//! and `periodicity` is the integer multiplicity n. The improper angle χ is the
//! dihedral angle defined by the quadruple I-J-K-L, so the geometry reuses the
//! shared dihedral routines.

use std::collections::HashMap;

use crate::ff::forcefield::Params;
use crate::ff::potential::Potential;
use crate::ff::potential::geometry::{
    accumulate_dihedral_forces, compute_dihedral, validate_coords,
};
use molrs::store::frame::Frame;
use molrs::types::F;

/// CVFF improper with pre-resolved flat arrays.
pub struct ImproperCvff {
    atom_i: Vec<usize>,
    atom_j: Vec<usize>,
    atom_k: Vec<usize>,
    atom_l: Vec<usize>,
    k: Vec<F>,
    d: Vec<F>,
    n: Vec<F>,
}

impl Potential for ImproperCvff {
    fn calc_energy_forces(&self, coords: &[F]) -> (F, Vec<F>) {
        let _n = validate_coords(coords);
        let mut energy: F = 0.0;
        let mut forces = vec![0.0 as F; coords.len()];

        for idx in 0..self.atom_i.len() {
            let (i, j, k, l) = (
                self.atom_i[idx],
                self.atom_j[idx],
                self.atom_k[idx],
                self.atom_l[idx],
            );
            let chi = compute_dihedral(coords, i, j, k, l);
            let (ki, di, ni) = (self.k[idx], self.d[idx], self.n[idx]);
            energy += ki * (1.0 + di * (ni * chi).cos());
            // dE/dχ = −K·d·n·sin(n·χ)
            let de_dchi = -ki * di * ni * (ni * chi).sin();
            accumulate_dihedral_forces(coords, i, j, k, l, de_dchi, &mut forces);
        }
        (energy, forces)
    }
}

/// Construct an [`ImproperCvff`] from per-type params (`k`, `d`, `n`) and a
/// Frame's `"impropers"` block (`atomi/atomj/atomk/atoml/type`).
pub fn improper_cvff_ctor(
    _sp: &Params,
    tp: &[(&str, &Params)],
    frame: &Frame,
) -> Result<Box<dyn Potential>, String> {
    let type_map: HashMap<&str, &Params> = tp.iter().copied().collect();
    let block = frame
        .get("impropers")
        .ok_or("improper_cvff: missing \"impropers\" block")?;
    let ic = block.get_uint("atomi").ok_or("missing atomi")?;
    let jc = block.get_uint("atomj").ok_or("missing atomj")?;
    let kc = block.get_uint("atomk").ok_or("missing atomk")?;
    let lc = block.get_uint("atoml").ok_or("missing atoml")?;
    let tc = block.get_string("type").ok_or("missing type")?;

    let n = ic.len();
    let (mut ai, mut aj, mut ak, mut al) = (
        Vec::with_capacity(n),
        Vec::with_capacity(n),
        Vec::with_capacity(n),
        Vec::with_capacity(n),
    );
    let (mut kk, mut dd, mut nn) = (
        Vec::with_capacity(n),
        Vec::with_capacity(n),
        Vec::with_capacity(n),
    );

    for idx in 0..n {
        let p = type_map
            .get(tc[idx].as_str())
            .ok_or_else(|| format!("improper_cvff: unknown type '{}'", tc[idx]))?;
        ai.push(ic[idx] as usize);
        aj.push(jc[idx] as usize);
        ak.push(kc[idx] as usize);
        al.push(lc[idx] as usize);
        kk.push(p.get("k").ok_or("improper_cvff: missing k")? as F);
        dd.push(p.get("sign").ok_or("improper_cvff: missing sign")? as F);
        nn.push(
            p.get("periodicity")
                .ok_or("improper_cvff: missing periodicity")? as F,
        );
    }
    Ok(Box::new(ImproperCvff {
        atom_i: ai,
        atom_j: aj,
        atom_k: ak,
        atom_l: al,
        k: kk,
        d: dd,
        n: nn,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn quad(phi: F) -> Vec<F> {
        let (s, c) = phi.sin_cos();
        vec![0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, c, s]
    }

    fn single(k: F, d: F, n: F) -> ImproperCvff {
        ImproperCvff {
            atom_i: vec![0],
            atom_j: vec![1],
            atom_k: vec![2],
            atom_l: vec![3],
            k: vec![k],
            d: vec![d],
            n: vec![n],
        }
    }

    #[test]
    fn energy_phase() {
        // K[1 + d cos(nχ)], k=2,d=1,n=1: E(0)=4, E(π)=0.
        let p = single(2.0, 1.0, 1.0);
        assert!((p.calc_energy_forces(&quad(0.0)).0 - 4.0).abs() < 1e-9);
        assert!(p.calc_energy_forces(&quad(std::f64::consts::PI)).0.abs() < 1e-9);
    }

    #[test]
    fn numerical_gradient() {
        let pot = single(2.3, -1.0, 2.0);
        let coords: Vec<F> = vec![0.1, 1.0, 0.2, 0.0, 0.0, 0.0, 1.0, 0.0, -0.1, 1.2, -0.8, 0.5];
        let (_, forces) = pot.calc_energy_forces(&coords);
        let h = 1e-6;
        for d in 0..coords.len() {
            let mut cp = coords.clone();
            let mut cm = coords.clone();
            cp[d] += h;
            cm[d] -= h;
            let ep = pot.calc_energy_forces(&cp).0;
            let em = pot.calc_energy_forces(&cm).0;
            let fd = -(ep - em) / (2.0 * h);
            assert!(
                (forces[d] - fd).abs() < 1e-5,
                "comp {d}: analytic {} vs fd {fd}",
                forces[d]
            );
        }
    }

    #[test]
    fn newtons_third_law() {
        let pot = single(1.0, 1.0, 3.0);
        let coords: Vec<F> = vec![0.1, 1.0, 0.2, 0.0, 0.0, 0.0, 1.0, 0.0, -0.1, 1.2, -0.8, 0.5];
        let (_, f) = pot.calc_energy_forces(&coords);
        for dim in 0..3 {
            let s: F = (0..4).map(|a| f[a * 3 + dim]).sum();
            assert!(s.abs() < 1e-9, "dim {dim} force sum {s}");
        }
    }
}
