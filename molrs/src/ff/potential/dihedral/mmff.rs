//! MMFF94 torsional rotation: E = 0.5*(V1*(1+cos phi) + V2*(1-cos 2phi) + V3*(1+cos 3phi))

use ndarray::{Array2, ArrayView2};

use crate::ff::forcefield::Params;
use crate::ff::potential::geometry::{
    accumulate_dihedral_forces, compute_dihedral, term_table, validate_coords,
};
use crate::ff::potential::{IndexedTerms, Member, Potential};
use molrs::store::frame::Frame;
use molrs::types::F;

pub struct MMFFTorsion {
    atom_i: Vec<usize>,
    atom_j: Vec<usize>,
    atom_k: Vec<usize>,
    atom_l: Vec<usize>,
    v1: Vec<F>,
    v2: Vec<F>,
    v3: Vec<F>,
}

impl MMFFTorsion {
    /// The physics, once. Which atoms a term names is the only thing
    /// that differs between the two entry points, so it is the only thing
    /// passed in — a second copy of the loop would be a second place for
    /// the force expression to drift.
    fn fold(
        &self,
        coords: &[F],
        out: &mut [F],
        n_terms: usize,
        atoms: impl Fn(usize) -> (usize, usize, usize, usize),
    ) -> F {
        let _n = validate_coords(coords);
        let mut energy: F = 0.0;
        let forces = out;

        for idx in 0..n_terms {
            let (i, j, k, l) = atoms(idx);
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
            accumulate_dihedral_forces(coords, i, j, k, l, de_dphi, forces);
        }
        energy
    }
}

impl Potential for MMFFTorsion {
    fn calc_energy_forces(&self, coords: &[F]) -> (F, Vec<F>) {
        let mut out = vec![0.0; coords.len()];
        let energy = self.accumulate(coords, &mut out);
        (energy, out)
    }

    fn accumulate(&self, coords: &[F], out: &mut [F]) -> F {
        self.fold(coords, out, self.atom_i.len(), |t| {
            (
                self.atom_i[t],
                self.atom_j[t],
                self.atom_k[t],
                self.atom_l[t],
            )
        })
    }
}

impl IndexedTerms for MMFFTorsion {
    fn terms(&self) -> Array2<u32> {
        term_table(&[&self.atom_i, &self.atom_j, &self.atom_k, &self.atom_l])
    }
    fn calc_energy_forces_with_terms(
        &self,
        coords: &[F],
        terms: ArrayView2<'_, u32>,
    ) -> (F, Vec<F>) {
        let mut out = vec![0.0; coords.len()];
        let energy = self.accumulate_with_terms(coords, terms, &mut out);
        (energy, out)
    }

    fn accumulate_with_terms(&self, coords: &[F], terms: ArrayView2<'_, u32>, out: &mut [F]) -> F {
        debug_assert_eq!(
            terms.nrows(),
            self.atom_i.len(),
            "the row set is the force field's; only the atoms a row names may be rebound"
        );
        self.fold(coords, out, terms.nrows(), |t| {
            (
                terms[[t, 0]] as usize,
                terms[[t, 1]] as usize,
                terms[[t, 2]] as usize,
                terms[[t, 3]] as usize,
            )
        })
    }
}

pub fn mmff_torsion_ctor(
    _sp: &Params,
    _tp: &[(&str, &Params)],
    frame: &Frame,
) -> Result<Member, String> {
    // Per-instance parameters: the MMFF typifier baked v1/v2/v3 onto each
    // dihedral (table → empirical). This kernel only reads the columns and
    // evaluates — no force-field-specific resolution lives here.
    let block = frame
        .get("dihedrals")
        .ok_or("mmff_torsion: missing \"dihedrals\"")?;
    let ic = block.get_uint("atomi").ok_or("missing atomi")?;
    let jc = block.get_uint("atomj").ok_or("missing atomj")?;
    let kc = block.get_uint("atomk").ok_or("missing atomk")?;
    let lc = block.get_uint("atoml").ok_or("missing atoml")?;
    let v1c = block
        .get_float("v1")
        .ok_or("mmff_torsion: missing \"v1\" column (typifier did not bake torsion params)")?;
    let v2c = block
        .get_float("v2")
        .ok_or("mmff_torsion: missing \"v2\" column (typifier did not bake torsion params)")?;
    let v3c = block
        .get_float("v3")
        .ok_or("mmff_torsion: missing \"v3\" column (typifier did not bake torsion params)")?;

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
        ai.push(ic[idx] as usize);
        aj.push(jc[idx] as usize);
        ak.push(kc[idx] as usize);
        al.push(lc[idx] as usize);
        v1.push(v1c[idx] as F);
        v2.push(v2c[idx] as F);
        v3.push(v3c[idx] as F);
    }
    Ok(Member::indexed(MMFFTorsion {
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
        let (e, forces) = pot.calc_energy_forces(&coords);
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
