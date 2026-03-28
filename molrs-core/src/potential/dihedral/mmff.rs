//! MMFF94 torsional rotation: E = 0.5*(V1*(1+cos phi) + V2*(1-cos 2phi) + V3*(1+cos 3phi))

use std::collections::HashMap;

use crate::forcefield::Params;
use crate::frame::Frame;
use crate::potential::Potential;
use crate::potential::geometry::{accumulate_dihedral_forces, compute_dihedral, validate_coords};
use crate::types::F;

pub struct MMFFTorsion {
    atom_i: Vec<usize>,
    atom_j: Vec<usize>,
    atom_k: Vec<usize>,
    atom_l: Vec<usize>,
    v1: Vec<F>,
    v2: Vec<F>,
    v3: Vec<F>,
}

impl Potential for MMFFTorsion {
    fn eval(&self, coords: &[F]) -> (F, Vec<F>) {
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
            let phi = compute_dihedral(coords, i, j, k, l);

            let (s1, c1) = phi.sin_cos();
            let (s2, c2) = (2.0 * phi).sin_cos();
            let (s3, c3) = (3.0 * phi).sin_cos();

            energy += 0.5
                * (self.v1[idx] * (1.0 + c1)
                    + self.v2[idx] * (1.0 - c2)
                    + self.v3[idx] * (1.0 + c3));

            let de_dphi =
                0.5 * (-self.v1[idx] * s1 + 2.0 * self.v2[idx] * s2 - 3.0 * self.v3[idx] * s3);
            accumulate_dihedral_forces(coords, i, j, k, l, de_dphi, &mut forces);
        }
        (energy, forces)
    }
}

pub fn mmff_torsion_ctor(
    _sp: &Params,
    tp: &[(&str, &Params)],
    frame: &Frame,
) -> Result<Box<dyn Potential>, String> {
    let type_map: HashMap<&str, &Params> = tp.iter().copied().collect();
    let block = frame
        .get("dihedrals")
        .ok_or("mmff_torsion: missing \"dihedrals\"")?;
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
    let (mut v1, mut v2, mut v3) = (
        Vec::with_capacity(n),
        Vec::with_capacity(n),
        Vec::with_capacity(n),
    );

    for idx in 0..n {
        let p = type_map
            .get(tc[idx].as_str())
            .ok_or_else(|| format!("mmff_torsion: unknown '{}'", tc[idx]))?;
        ai.push(ic[idx] as usize);
        aj.push(jc[idx] as usize);
        ak.push(kc[idx] as usize);
        al.push(lc[idx] as usize);
        v1.push(p.get("v1").ok_or("missing v1")? as F);
        v2.push(p.get("v2").ok_or("missing v2")? as F);
        v3.push(p.get("v3").ok_or("missing v3")? as F);
    }
    Ok(Box::new(MMFFTorsion {
        atom_i: ai,
        atom_j: aj,
        atom_k: ak,
        atom_l: al,
        v1,
        v2,
        v3,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mmff_torsion() {
        let pot = MMFFTorsion {
            atom_i: vec![0],
            atom_j: vec![1],
            atom_k: vec![2],
            atom_l: vec![3],
            v1: vec![0.0],
            v2: vec![0.0],
            v3: vec![0.3],
        };
        let coords: Vec<F> = vec![0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, -1.0, 0.0];
        let (e, forces) = pot.eval(&coords);
        assert!(e.is_finite());
        let fx: F = forces.iter().step_by(3).sum();
        let fy: F = forces.iter().skip(1).step_by(3).sum();
        let fz: F = forces.iter().skip(2).step_by(3).sum();
        assert!(
            (fx.abs() + fy.abs() + fz.abs()) < 0.1,
            "force sum too large"
        );
    }
}
