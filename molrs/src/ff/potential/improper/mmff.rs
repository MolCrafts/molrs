//! MMFF94 out-of-plane bending: E = 0.5*143.9325*koop*chi^2 (Wilson angle)

use ndarray::{Array2, ArrayView2};

use crate::ff::forcefield::Params;
use crate::ff::potential::geometry::{cross3, dot3, mag3, sub3, term_table, validate_coords};
use crate::ff::potential::{IndexedTerms, Member, Potential};
use molrs::store::frame::Frame;
use molrs::types::F;

use crate::ff::constants::MDYNE_A_TO_KCAL;

pub struct MMFFOutOfPlane {
    atom_i: Vec<usize>,
    atom_j: Vec<usize>,
    atom_k: Vec<usize>,
    atom_l: Vec<usize>,
    koop: Vec<F>,
}

impl MMFFOutOfPlane {
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
        let conv = MDYNE_A_TO_KCAL as F;

        for idx in 0..n_terms {
            let (i, j, k, l) = atoms(idx);
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
        energy
    }
}

impl Potential for MMFFOutOfPlane {
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

impl IndexedTerms for MMFFOutOfPlane {
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

pub fn mmff_oop_ctor(
    _sp: &Params,
    _tp: &[(&str, &Params)],
    frame: &Frame,
) -> Result<Member, String> {
    // Per-instance parameters: the MMFF typifier baked koop onto each improper.
    // This kernel only reads the column and evaluates.
    let block = frame
        .get("impropers")
        .ok_or("mmff_oop: missing \"impropers\"")?;
    let ic = block.get_uint("atomi").ok_or("missing atomi")?;
    let jc = block.get_uint("atomj").ok_or("missing atomj")?;
    let kc = block.get_uint("atomk").ok_or("missing atomk")?;
    let lc = block.get_uint("atoml").ok_or("missing atoml")?;
    let koopc = block
        .get_float("koop")
        .ok_or("mmff_oop: missing \"koop\" column (typifier did not bake oop params)")?;

    let n = ic.len();
    let (mut ai, mut aj, mut ak, mut al, mut koop) = (
        Vec::with_capacity(n),
        Vec::with_capacity(n),
        Vec::with_capacity(n),
        Vec::with_capacity(n),
        Vec::with_capacity(n),
    );

    for idx in 0..n {
        ai.push(ic[idx] as usize);
        aj.push(jc[idx] as usize);
        ak.push(kc[idx] as usize);
        al.push(lc[idx] as usize);
        koop.push(koopc[idx] as F);
    }
    Ok(Member::indexed(MMFFOutOfPlane {
        atom_i: ai,
        atom_j: aj,
        atom_k: ak,
        atom_l: al,
        koop,
    }))
}
