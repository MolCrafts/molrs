//! UFF inversion (out-of-plane), RDKit `InversionContrib`.
//!
//! Per-instance columns on `impropers`: `K`, `c0`, `c1`, `c2`.
//! Centre atom is `atomj` (Wilson / RDKit convention: i–j–k with j central,
//! fourth atom `atoml`).

use ndarray::{Array2, ArrayView2};

use crate::ff::forcefield::Params;
use crate::ff::potential::geometry::{cross3, dot3, mag3, sub3, term_table, validate_coords};
use crate::ff::potential::{IndexedTerms, Member, Potential};
use molrs::store::frame::Frame;
use molrs::types::F;

pub struct UffInversion {
    atom_i: Vec<usize>,
    atom_j: Vec<usize>,
    atom_k: Vec<usize>,
    atom_l: Vec<usize>,
    k: Vec<F>,
    c0: Vec<F>,
    c1: Vec<F>,
    c2: Vec<F>,
}

impl UffInversion {
    /// The physics, once. Which atoms a term names is the only thing
    /// that differs between the two entry points, so it is the only thing
    /// passed in — a second copy of the loop would be a second place for
    /// the force expression to drift.
    fn fold(
        &self,
        coords: &[F],
        n_terms: usize,
        atoms: impl Fn(usize) -> (usize, usize, usize, usize),
    ) -> (F, Vec<F>) {
        let _n = validate_coords(coords);
        let mut energy = 0.0 as F;
        let mut forces = vec![0.0 as F; coords.len()];

        for idx in 0..n_terms {
            let (i, j, k, l) = atoms(idx);
            // cosY from RDKit Utils::calculateCosY
            let r_ji = sub3(coords, i, coords, j);
            let r_jk = sub3(coords, k, coords, j);
            let r_jl = sub3(coords, l, coords, j);
            let d_ji = mag3(r_ji);
            let d_jk = mag3(r_jk);
            let d_jl = mag3(r_jl);
            if d_ji < 1e-12 || d_jk < 1e-12 || d_jl < 1e-12 {
                continue;
            }
            // The gradient block below (t1 = r̂_JL × r̂_JK, t2 = r̂_JI × r̂_JL,
            // t3 = r̂_JK × r̂_JI) is written for the normal (−r_JI) × r_JK.
            // E is even in cosY so the wrong orientation is invisible in the
            // energy; the force is odd in it and comes out inverted.
            let mut n = cross3([-r_ji[0], -r_ji[1], -r_ji[2]], r_jk);
            let ln = mag3(n);
            if ln < 1e-12 {
                continue;
            }
            n = [n[0] / ln, n[1] / ln, n[2] / ln];
            let r_jl_u = [r_jl[0] / d_jl, r_jl[1] / d_jl, r_jl[2] / d_jl];
            let cos_y = dot3(n, r_jl_u).clamp(-1.0, 1.0);
            let sin_y_sq = (1.0 - cos_y * cos_y).max(0.0);
            let sin_y = sin_y_sq.sqrt();
            let cos2_w = 2.0 * sin_y * sin_y - 1.0;
            let kk = self.k[idx];
            energy += kk * (self.c0[idx] + self.c1[idx] * sin_y + self.c2[idx] * cos2_w);

            // Gradient (RDKit getGrad) — forces = −grad
            let r_ji_u = [r_ji[0] / d_ji, r_ji[1] / d_ji, r_ji[2] / d_ji];
            let r_jk_u = [r_jk[0] / d_jk, r_jk[1] / d_jk, r_jk[2] / d_jk];
            let cos_theta = dot3(r_ji_u, r_jk_u).clamp(-1.0, 1.0);
            let sin_theta = (1.0 - cos_theta * cos_theta).max(0.0).sqrt().max(1e-8);
            let sin_y_s = sin_y.max(1e-8);
            // dE/dW = −K (C1 cosY + 4 C2 cosY sinY)
            let d_e_d_w = -kk * (self.c1[idx] * cos_y + 4.0 * self.c2[idx] * cos_y * sin_y);

            let t1 = cross3(r_jl_u, r_jk_u);
            let t2 = cross3(r_ji_u, r_jl_u);
            let t3 = cross3(r_jk_u, r_ji_u);
            let term1 = sin_y_s * sin_theta;
            let term2 = cos_y / (sin_y_s * sin_theta * sin_theta);

            for dim in 0..3 {
                let tg1 =
                    (t1[dim] / term1 - (r_ji_u[dim] - r_jk_u[dim] * cos_theta) * term2) / d_ji;
                let tg3 =
                    (t2[dim] / term1 - (r_jk_u[dim] - r_ji_u[dim] * cos_theta) * term2) / d_jk;
                let tg4 = (t3[dim] / term1 - r_jl_u[dim] * cos_y / sin_y_s) / d_jl;
                // RDKit writes to grad; force = −grad
                forces[i * 3 + dim] -= d_e_d_w * tg1;
                forces[j * 3 + dim] -= -d_e_d_w * (tg1 + tg3 + tg4);
                forces[k * 3 + dim] -= d_e_d_w * tg3;
                forces[l * 3 + dim] -= d_e_d_w * tg4;
            }
        }
        (energy, forces)
    }
}

impl Potential for UffInversion {
    fn calc_energy_forces(&self, coords: &[F]) -> (F, Vec<F>) {
        self.fold(coords, self.atom_i.len(), |t| {
            (
                self.atom_i[t],
                self.atom_j[t],
                self.atom_k[t],
                self.atom_l[t],
            )
        })
    }
}

impl IndexedTerms for UffInversion {
    fn terms(&self) -> Array2<u32> {
        term_table(&[&self.atom_i, &self.atom_j, &self.atom_k, &self.atom_l])
    }
    fn calc_energy_forces_with_terms(
        &self,
        coords: &[F],
        terms: ArrayView2<'_, u32>,
    ) -> (F, Vec<F>) {
        debug_assert_eq!(
            terms.nrows(),
            self.atom_i.len(),
            "the row set is the force field's; only the atoms a row names may be rebound"
        );
        self.fold(coords, terms.nrows(), |t| {
            (
                terms[[t, 0]] as usize,
                terms[[t, 1]] as usize,
                terms[[t, 2]] as usize,
                terms[[t, 3]] as usize,
            )
        })
    }
}

pub fn uff_inversion_ctor(
    _sp: &Params,
    _tp: &[(&str, &Params)],
    frame: &Frame,
) -> Result<Member, String> {
    let Some(block) = frame.get("impropers") else {
        return Ok(Member::indexed(UffInversion {
            atom_i: vec![],
            atom_j: vec![],
            atom_k: vec![],
            atom_l: vec![],
            k: vec![],
            c0: vec![],
            c1: vec![],
            c2: vec![],
        }));
    };
    if block.nrows().unwrap_or(0) == 0 {
        return Ok(Member::indexed(UffInversion {
            atom_i: vec![],
            atom_j: vec![],
            atom_k: vec![],
            atom_l: vec![],
            k: vec![],
            c0: vec![],
            c1: vec![],
            c2: vec![],
        }));
    }
    let i = block
        .get_uint("atomi")
        .ok_or("uff_inversion: missing atomi")?;
    let j = block
        .get_uint("atomj")
        .ok_or("uff_inversion: missing atomj")?;
    let k = block
        .get_uint("atomk")
        .ok_or("uff_inversion: missing atomk")?;
    let l = block
        .get_uint("atoml")
        .ok_or("uff_inversion: missing atoml")?;
    let kk = block.get_float("K").ok_or("uff_inversion: missing K")?;
    let c0 = block.get_float("c0").ok_or("uff_inversion: missing c0")?;
    let c1 = block.get_float("c1").ok_or("uff_inversion: missing c1")?;
    let c2 = block.get_float("c2").ok_or("uff_inversion: missing c2")?;
    let n = i.len();
    Ok(Member::indexed(UffInversion {
        atom_i: (0..n).map(|t| i[t] as usize).collect(),
        atom_j: (0..n).map(|t| j[t] as usize).collect(),
        atom_k: (0..n).map(|t| k[t] as usize).collect(),
        atom_l: (0..n).map(|t| l[t] as usize).collect(),
        k: kk.iter().map(|&v| v as F).collect(),
        c0: c0.iter().map(|&v| v as F).collect(),
        c1: c1.iter().map(|&v| v as F).collect(),
        c2: c2.iter().map(|&v| v as F).collect(),
    }))
}
