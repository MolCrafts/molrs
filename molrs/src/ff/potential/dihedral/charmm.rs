//! CHARMM proper dihedral:
//!
//! E(φ) = k·[1 + cos(n·φ − γ)]
//!
//! `k` is the force constant (kcal/mol), `periodicity` the integer multiplicity,
//! and `phase` the phase γ in radians (readers normalize at their boundary; the LAMMPS
//! `dihedral_style charmm` degree value is converted at read). The 1-4
//! pair weight `w` is a non-bonded scaling factor handled by the pair term, not
//! the torsion energy, so it is read but does not enter this kernel.

use std::collections::HashMap;

use ndarray::{Array2, ArrayView2};

use crate::ff::forcefield::Params;
use crate::ff::potential::Potential;
use crate::ff::potential::geometry::{
    accumulate_dihedral_forces, compute_dihedral, term_table, validate_coords,
};
use molrs::store::frame::Frame;
use molrs::types::F;

/// CHARMM proper dihedral with pre-resolved flat arrays.
pub struct DihedralCharmm {
    atom_i: Vec<usize>,
    atom_j: Vec<usize>,
    atom_k: Vec<usize>,
    atom_l: Vec<usize>,
    k: Vec<F>,
    n: Vec<F>,
    /// phase in radians
    d: Vec<F>,
}

impl DihedralCharmm {
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
        let mut energy: F = 0.0;
        let mut forces = vec![0.0 as F; coords.len()];

        for idx in 0..n_terms {
            let (i, j, k, l) = atoms(idx);
            let phi = compute_dihedral(coords, i, j, k, l);
            let (ki, ni, di) = (self.k[idx], self.n[idx], self.d[idx]);
            let arg = ni * phi - di;
            energy += ki * (1.0 + arg.cos());
            // dE/dφ = −K·n·sin(n·φ − d)
            let de_dphi = -ki * ni * arg.sin();
            accumulate_dihedral_forces(coords, i, j, k, l, de_dphi, &mut forces);
        }
        (energy, forces)
    }
}

impl Potential for DihedralCharmm {
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

    fn terms(&self) -> Option<Array2<u32>> {
        Some(term_table(&[
            &self.atom_i,
            &self.atom_j,
            &self.atom_k,
            &self.atom_l,
        ]))
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

/// Construct a [`DihedralCharmm`] from per-type params (`k`, `n`, `d` radians)
/// and a Frame's `"dihedrals"` block (`atomi/atomj/atomk/atoml/type`).
pub fn dihedral_charmm_ctor(
    _sp: &Params,
    tp: &[(&str, &Params)],
    frame: &Frame,
) -> Result<Box<dyn Potential>, String> {
    let type_map: HashMap<&str, &Params> = tp.iter().copied().collect();
    let block = frame
        .get("dihedrals")
        .ok_or("dihedral_charmm: missing \"dihedrals\" block")?;
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
    let (mut kk, mut nn, mut dd) = (
        Vec::with_capacity(n),
        Vec::with_capacity(n),
        Vec::with_capacity(n),
    );

    for idx in 0..n {
        let p = type_map
            .get(tc[idx].as_str())
            .ok_or_else(|| format!("dihedral_charmm: unknown type '{}'", tc[idx]))?;
        ai.push(ic[idx] as usize);
        aj.push(jc[idx] as usize);
        ak.push(kc[idx] as usize);
        al.push(lc[idx] as usize);
        kk.push(p.get("k").ok_or("dihedral_charmm: missing k")? as F);
        nn.push(
            p.get("periodicity")
                .ok_or("dihedral_charmm: missing periodicity")? as F,
        );
        dd.push(p.get("phase").unwrap_or(0.0) as F); // radians (normalized at read)
    }
    Ok(Box::new(DihedralCharmm {
        atom_i: ai,
        atom_j: aj,
        atom_k: ak,
        atom_l: al,
        k: kk,
        n: nn,
        d: dd,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn quad(phi: F) -> Vec<F> {
        let (s, c) = phi.sin_cos();
        vec![0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, c, s]
    }

    fn single(k: F, n: F, d_deg: F) -> DihedralCharmm {
        DihedralCharmm {
            atom_i: vec![0],
            atom_j: vec![1],
            atom_k: vec![2],
            atom_l: vec![3],
            k: vec![k],
            n: vec![n],
            d: vec![d_deg.to_radians()],
        }
    }

    #[test]
    fn energy_phase() {
        // E = K[1 + cos(nφ − d)]. With n=1, d=0: E(0)=2K, E(π)=0.
        let e0 = single(1.5, 1.0, 0.0).calc_energy_forces(&quad(0.0)).0;
        let epi = single(1.5, 1.0, 0.0)
            .calc_energy_forces(&quad(std::f64::consts::PI))
            .0;
        assert!((e0 - 3.0).abs() < 1e-9, "E(0) got {e0}");
        assert!(epi.abs() < 1e-9, "E(pi) got {epi}");
    }

    #[test]
    fn numerical_gradient() {
        let pot = single(2.3, 2.0, 30.0);
        let coords: Vec<F> = vec![0.1, 1.0, 0.2, 0.0, 0.0, 0.0, 1.0, 0.0, -0.1, 1.2, -0.8, 0.5];
        let (_, forces) = pot.calc_energy_forces(&coords);
        let h = 1e-6;
        for dd in 0..coords.len() {
            let mut cp = coords.clone();
            let mut cm = coords.clone();
            cp[dd] += h;
            cm[dd] -= h;
            let ep = pot.calc_energy_forces(&cp).0;
            let em = pot.calc_energy_forces(&cm).0;
            let fd = -(ep - em) / (2.0 * h);
            assert!(
                (forces[dd] - fd).abs() < 1e-5,
                "comp {dd}: analytic {} vs fd {fd}",
                forces[dd]
            );
        }
    }

    #[test]
    fn newtons_third_law() {
        let pot = single(1.0, 3.0, 45.0);
        let coords: Vec<F> = vec![0.1, 1.0, 0.2, 0.0, 0.0, 0.0, 1.0, 0.0, -0.1, 1.2, -0.8, 0.5];
        let (_, f) = pot.calc_energy_forces(&coords);
        for dim in 0..3 {
            let s: F = (0..4).map(|a| f[a * 3 + dim]).sum();
            assert!(s.abs() < 1e-9, "dim {dim} force sum {s}");
        }
    }
}
