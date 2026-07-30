//! UFF torsion: E = V/2 · (1 − cosTerm · cos(n·φ)) (RDKit `TorsionAngleContrib`).

use crate::ff::forcefield::Params;
use crate::ff::potential::Potential;
use crate::ff::potential::geometry::{cross3, dot3, mag3, sub3, validate_coords};
use molrs::store::frame::Frame;
use molrs::types::F;

pub struct UffTorsion {
    atom_i: Vec<usize>,
    atom_j: Vec<usize>,
    atom_k: Vec<usize>,
    atom_l: Vec<usize>,
    v: Vec<F>,
    order: Vec<u8>,
    cos_term: Vec<F>,
}

impl Potential for UffTorsion {
    fn calc_energy_forces(&self, coords: &[F]) -> (F, Vec<F>) {
        let _n = validate_coords(coords);
        let mut energy = 0.0 as F;
        let mut forces = vec![0.0 as F; coords.len()];

        for idx in 0..self.atom_i.len() {
            let (a, b, c, d) = (
                self.atom_i[idx],
                self.atom_j[idx],
                self.atom_k[idx],
                self.atom_l[idx],
            );
            // RDKit: r1=p1-p2, r2=p3-p2, r3=p2-p3, r4=p4-p3
            let r1 = sub3(coords, a, coords, b);
            let r2 = sub3(coords, c, coords, b);
            let r3 = sub3(coords, b, coords, c);
            let r4 = sub3(coords, d, coords, c);
            let t1 = cross3(r1, r2);
            let t2 = cross3(r3, r4);
            let d1 = mag3(t1);
            let d2 = mag3(t2);
            if d1 < 1e-12 as F || d2 < 1e-12 as F {
                continue;
            }
            let cos_phi = (dot3(t1, t2) / (d1 * d2)).clamp(-1.0, 1.0);
            let sin_sq = (1.0 - cos_phi * cos_phi).max(0.0);
            let sin_phi = sin_sq.sqrt();

            let order = self.order[idx];
            let cos_n = match order {
                2 => 1.0 - 2.0 * sin_sq,
                3 => cos_phi * (cos_phi * cos_phi - 3.0 * sin_sq),
                6 => 1.0 + sin_sq * (-32.0 * sin_sq * sin_sq + 48.0 * sin_sq - 18.0),
                _ => cos_phi,
            };
            let v = self.v[idx];
            let cos_term = self.cos_term[idx];
            energy += v / 2.0 * (1.0 - cos_term * cos_n);

            // RDKit `getThetaDeriv`: dE/dφ with E = V/2 · (1 − cosTerm · cos(nφ)).
            let mut d_e_d_phi = match order {
                2 => 2.0 * sin_phi * cos_phi,
                3 => sin_phi * (3.0 - 4.0 * sin_phi * sin_phi),
                6 => cos_phi * sin_phi * (32.0 * sin_sq * (sin_sq - 1.0) + 6.0),
                _ => 0.0,
            };
            d_e_d_phi *= -(v / 2.0 * cos_term) * (order as F);
            let sin_term = if sin_phi.abs() < 1e-8 {
                d_e_d_phi / cos_phi.signum().max(1e-8).abs().copysign(cos_phi).max(1e-8)
            } else {
                d_e_d_phi / sin_phi
            };

            // Project using RDKit calcTorsionGrad (simplified via force on planes)
            // Use standard dihedral force accumulation.
            accumulate_torsion_forces(
                coords,
                a,
                b,
                c,
                d,
                sin_term,
                cos_phi,
                &t1,
                &t2,
                d1,
                d2,
                &r1,
                &r2,
                &r3,
                &r4,
                &mut forces,
            );
        }
        (energy, forces)
    }
}

/// RDKit-style torsion gradient (adapted from `calcTorsionGrad`).
#[allow(clippy::too_many_arguments)]
fn accumulate_torsion_forces(
    _coords: &[F],
    a: usize,
    b: usize,
    c: usize,
    d: usize,
    sin_term: F,
    cos_phi: F,
    t1: &[F; 3],
    t2: &[F; 3],
    d1: F,
    d2: F,
    r1: &[F; 3],
    r2: &[F; 3],
    r3: &[F; 3],
    r4: &[F; 3],
    forces: &mut [F],
) {
    // r vectors as RDKit: r[0]=r1, r[1]=r2, r[2]=r3, r[3]=r4
    // t[0]=t1, t[1]=t2, d[0]=d1, d[1]=d2
    let d_cos_dt = [
        1.0 / d1 * (t2[0] - cos_phi * t1[0]),
        1.0 / d1 * (t2[1] - cos_phi * t1[1]),
        1.0 / d1 * (t2[2] - cos_phi * t1[2]),
        1.0 / d2 * (t1[0] - cos_phi * t2[0]),
        1.0 / d2 * (t1[1] - cos_phi * t2[1]),
        1.0 / d2 * (t1[2] - cos_phi * t2[2]),
    ];

    // g[0] (atom a)
    forces[a * 3] += sin_term * (d_cos_dt[2] * r2[1] - d_cos_dt[1] * r2[2]);
    forces[a * 3 + 1] += sin_term * (d_cos_dt[0] * r2[2] - d_cos_dt[2] * r2[0]);
    forces[a * 3 + 2] += sin_term * (d_cos_dt[1] * r2[0] - d_cos_dt[0] * r2[1]);

    // g[1] (atom b)
    forces[b * 3] += sin_term
        * (d_cos_dt[1] * (r2[2] - r1[2])
            + d_cos_dt[2] * (r1[1] - r2[1])
            + d_cos_dt[4] * (-r4[2])
            + d_cos_dt[5] * r4[1]);
    forces[b * 3 + 1] += sin_term
        * (d_cos_dt[0] * (r1[2] - r2[2])
            + d_cos_dt[2] * (r2[0] - r1[0])
            + d_cos_dt[3] * r4[2]
            + d_cos_dt[5] * (-r4[0]));
    forces[b * 3 + 2] += sin_term
        * (d_cos_dt[0] * (r2[1] - r1[1])
            + d_cos_dt[1] * (r1[0] - r2[0])
            + d_cos_dt[3] * (-r4[1])
            + d_cos_dt[4] * r4[0]);

    // g[2] (atom c)
    forces[c * 3] += sin_term
        * (d_cos_dt[1] * r1[2]
            + d_cos_dt[2] * (-r1[1])
            + d_cos_dt[4] * (r4[2] - r3[2])
            + d_cos_dt[5] * (r3[1] - r4[1]));
    forces[c * 3 + 1] += sin_term
        * (d_cos_dt[0] * (-r1[2])
            + d_cos_dt[2] * r1[0]
            + d_cos_dt[3] * (r3[2] - r4[2])
            + d_cos_dt[5] * (r4[0] - r3[0]));
    forces[c * 3 + 2] += sin_term
        * (d_cos_dt[0] * r1[1]
            + d_cos_dt[1] * (-r1[0])
            + d_cos_dt[3] * (r4[1] - r3[1])
            + d_cos_dt[4] * (r3[0] - r4[0]));

    // g[3] (atom d)
    forces[d * 3] += sin_term * (d_cos_dt[4] * r3[2] - d_cos_dt[5] * r3[1]);
    forces[d * 3 + 1] += sin_term * (d_cos_dt[5] * r3[0] - d_cos_dt[3] * r3[2]);
    forces[d * 3 + 2] += sin_term * (d_cos_dt[3] * r3[1] - d_cos_dt[4] * r3[0]);
}

pub fn uff_torsion_ctor(
    _sp: &Params,
    _tp: &[(&str, &Params)],
    frame: &Frame,
) -> Result<Box<dyn Potential>, String> {
    let block = frame
        .get("dihedrals")
        .ok_or("uff_torsion: missing \"dihedrals\" block")?;
    let i = block
        .get_uint("atomi")
        .ok_or("uff_torsion: missing atomi")?;
    let j = block
        .get_uint("atomj")
        .ok_or("uff_torsion: missing atomj")?;
    let k = block
        .get_uint("atomk")
        .ok_or("uff_torsion: missing atomk")?;
    let l = block
        .get_uint("atoml")
        .ok_or("uff_torsion: missing atoml")?;
    let v = block.get_float("V").ok_or("uff_torsion: missing V")?;
    let order = block
        .get_float("order")
        .ok_or("uff_torsion: missing order")?;
    let cos_term = block
        .get_float("cosTerm")
        .ok_or("uff_torsion: missing cosTerm")?;
    let n = i.len();
    Ok(Box::new(UffTorsion {
        atom_i: (0..n).map(|t| i[t] as usize).collect(),
        atom_j: (0..n).map(|t| j[t] as usize).collect(),
        atom_k: (0..n).map(|t| k[t] as usize).collect(),
        atom_l: (0..n).map(|t| l[t] as usize).collect(),
        v: v.iter().map(|&x| x as F).collect(),
        order: order.iter().map(|&x| x as u8).collect(),
        cos_term: cos_term.iter().map(|&x| x as F).collect(),
    }))
}
