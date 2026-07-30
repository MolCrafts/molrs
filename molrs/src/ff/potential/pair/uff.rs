//! UFF van der Waals: E = D · [(x/r)¹² − 2·(x/r)⁶] (RDKit `vdWContrib`).
//!
//! Per-instance columns on the `pairs` block: `xij`, `Dij` (baked by the
//! typifier after the neighbour list is built — but the typifier only sets
//! atom `x1`/`D1`; this ctor combines them geometrically like RDKit).

use crate::ff::forcefield::Params;
use crate::ff::potential::Potential;
use crate::ff::potential::geometry::validate_coords;
use molrs::store::frame::Frame;
use molrs::types::F;

pub struct UffVdW {
    atom_i: Vec<usize>,
    atom_j: Vec<usize>,
    xij: Vec<F>,
    dij: Vec<F>,
}

impl Potential for UffVdW {
    fn calc_energy_forces(&self, coords: &[F]) -> (F, Vec<F>) {
        let _n = validate_coords(coords);
        let mut energy = 0.0 as F;
        let mut forces = vec![0.0 as F; coords.len()];

        for idx in 0..self.atom_i.len() {
            let (i, j) = (self.atom_i[idx], self.atom_j[idx]);
            let dx = coords[j * 3] - coords[i * 3];
            let dy = coords[j * 3 + 1] - coords[i * 3 + 1];
            let dz = coords[j * 3 + 2] - coords[i * 3 + 2];
            let r2 = dx * dx + dy * dy + dz * dz;
            if r2 < 1e-24 as F {
                continue;
            }
            let r = r2.sqrt();
            let xij = self.xij[idx];
            let dij = self.dij[idx];
            // cutoff at 2 * xij (RDKit default threshMultiplier ≈ 2 when used that way)
            if r > 2.0 * xij {
                continue;
            }
            let inv_r = 1.0 / r;
            let rr = xij * inv_r;
            let r2u = rr * rr;
            let r6 = r2u * r2u * r2u;
            let r12 = r6 * r6;
            energy += dij * (r12 - 2.0 * r6);

            // RDKit: preFactor = 12·D/x · ((x/r)⁷ − (x/r)¹³);
            // dGrad on atom1 (i) = preFactor · (i−j)/r  (gradient contrib);
            // force = −gradient, so F_i = −preFactor · (i−j)/r = preFactor · (j−i)/r
            let r7 = r6 * rr;
            let r13 = r12 * rr;
            let pref = 12.0 * dij / xij * (r7 - r13);
            let factor = pref / r;
            // (j−i) = (dx,dy,dz)
            forces[i * 3] += factor * dx;
            forces[i * 3 + 1] += factor * dy;
            forces[i * 3 + 2] += factor * dz;
            forces[j * 3] -= factor * dx;
            forces[j * 3 + 1] -= factor * dy;
            forces[j * 3 + 2] -= factor * dz;
        }
        (energy, forces)
    }
}

pub fn uff_lj_ctor(
    _sp: &Params,
    _tp: &[(&str, &Params)],
    frame: &Frame,
) -> Result<Box<dyn Potential>, String> {
    let atoms = frame.get("atoms").ok_or("uff_lj: missing atoms")?;
    let x1 = atoms.get_float("x1").ok_or("uff_lj: missing atoms.x1")?;
    let d1 = atoms.get_float("D1").ok_or("uff_lj: missing atoms.D1")?;

    let pairs = frame
        .get("pairs")
        .ok_or("uff_lj: missing pairs (call intramolecular_pairs first)")?;
    if pairs.nrows().unwrap_or(0) == 0 {
        return Ok(Box::new(UffVdW {
            atom_i: vec![],
            atom_j: vec![],
            xij: vec![],
            dij: vec![],
        }));
    }
    let pi = pairs
        .get_uint("atomi")
        .ok_or("uff_lj: pairs missing atomi")?;
    let pj = pairs
        .get_uint("atomj")
        .ok_or("uff_lj: pairs missing atomj")?;
    let n = pi.len();
    let mut atom_i = Vec::with_capacity(n);
    let mut atom_j = Vec::with_capacity(n);
    let mut xij = Vec::with_capacity(n);
    let mut dij = Vec::with_capacity(n);
    for t in 0..n {
        let i = pi[t] as usize;
        let j = pj[t] as usize;
        atom_i.push(i);
        atom_j.push(j);
        xij.push(((x1[i] * x1[j]) as F).sqrt());
        dij.push(((d1[i] * d1[j]) as F).sqrt());
    }
    Ok(Box::new(UffVdW {
        atom_i,
        atom_j,
        xij,
        dij,
    }))
}
