//! Multi/harmonic proper dihedral (LAMMPS `dihedral_style multi/harmonic`):
//!
//! E(φ) = Σ_{n=1..5} A_n · cos^(n−1)(φ)
//!       = A₁ + A₂cosφ + A₃cos²φ + A₄cos³φ + A₅cos⁴φ
//!
//! Coefficients A₁..A₅ are in energy units (kcal/mol).

use std::collections::HashMap;

use ndarray::{Array2, ArrayView2};

use crate::ff::forcefield::Params;
use crate::ff::potential::geometry::{
    accumulate_dihedral_forces, compute_dihedral, term_table, validate_coords,
};
use crate::ff::potential::{IndexedTerms, Member, Potential};
use molrs::store::frame::Frame;
use molrs::types::F;

/// Multi/harmonic proper dihedral with pre-resolved flat arrays.
pub struct DihedralMultiHarmonic {
    atom_i: Vec<usize>,
    atom_j: Vec<usize>,
    atom_k: Vec<usize>,
    atom_l: Vec<usize>,
    /// A₁..A₅ per dihedral instance.
    a: Vec<[F; 5]>,
}

impl DihedralMultiHarmonic {
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
            let c = phi.cos();
            let a = &self.a[idx];

            // E = Σ A_n c^(n−1)  (Horner)
            energy += a[0] + c * (a[1] + c * (a[2] + c * (a[3] + c * a[4])));

            // dE/dc = A₂ + 2A₃c + 3A₄c² + 4A₅c³
            let de_dc = a[1] + c * (2.0 * a[2] + c * (3.0 * a[3] + c * 4.0 * a[4]));
            // dE/dφ = dE/dc · (−sinφ)
            let de_dphi = -phi.sin() * de_dc;
            accumulate_dihedral_forces(coords, i, j, k, l, de_dphi, &mut forces);
        }
        (energy, forces)
    }
}

impl Potential for DihedralMultiHarmonic {
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

impl IndexedTerms for DihedralMultiHarmonic {
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

/// Construct a [`DihedralMultiHarmonic`] from per-type params (`a1`..`a5`) and a
/// Frame's `"dihedrals"` block (`atomi/atomj/atomk/atoml/type`).
pub fn dihedral_multi_harmonic_ctor(
    _sp: &Params,
    tp: &[(&str, &Params)],
    frame: &Frame,
) -> Result<Member, String> {
    let type_map: HashMap<&str, &Params> = tp.iter().copied().collect();
    let block = frame
        .get("dihedrals")
        .ok_or("dihedral_multi_harmonic: missing \"dihedrals\" block")?;
    let ic = block.get_uint("atomi").ok_or("missing atomi")?;
    let jc = block.get_uint("atomj").ok_or("missing atomj")?;
    let kc = block.get_uint("atomk").ok_or("missing atomk")?;
    let lc = block.get_uint("atoml").ok_or("missing atoml")?;
    let tc = block.get_string("type").ok_or("missing type")?;

    let n = ic.len();
    let (mut ai, mut aj, mut ak, mut al, mut a) = (
        Vec::with_capacity(n),
        Vec::with_capacity(n),
        Vec::with_capacity(n),
        Vec::with_capacity(n),
        Vec::with_capacity(n),
    );

    for idx in 0..n {
        let p = type_map
            .get(tc[idx].as_str())
            .ok_or_else(|| format!("dihedral_multi_harmonic: unknown type '{}'", tc[idx]))?;
        ai.push(ic[idx] as usize);
        aj.push(jc[idx] as usize);
        ak.push(kc[idx] as usize);
        al.push(lc[idx] as usize);
        a.push([
            p.get("a1").unwrap_or(0.0) as F,
            p.get("a2").unwrap_or(0.0) as F,
            p.get("a3").unwrap_or(0.0) as F,
            p.get("a4").unwrap_or(0.0) as F,
            p.get("a5").unwrap_or(0.0) as F,
        ]);
    }
    Ok(Member::indexed(DihedralMultiHarmonic {
        atom_i: ai,
        atom_j: aj,
        atom_k: ak,
        atom_l: al,
        a,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn quad(phi: F) -> Vec<F> {
        let (s, c) = phi.sin_cos();
        vec![0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, c, s]
    }

    fn single(a: [F; 5]) -> DihedralMultiHarmonic {
        DihedralMultiHarmonic {
            atom_i: vec![0],
            atom_j: vec![1],
            atom_k: vec![2],
            atom_l: vec![3],
            a: vec![a],
        }
    }

    #[test]
    fn energy_matches_series() {
        // At φ=0, cos=1 → E = ΣA_n. At φ=π/2, cos=0 → E = A₁.
        let a = [0.5, 1.0, -0.3, 0.2, 0.1];
        let e0 = single(a).calc_energy_forces(&quad(0.0)).0;
        assert!((e0 - a.iter().sum::<F>()).abs() < 1e-9, "E(0) got {e0}");
        let e90 = single(a)
            .calc_energy_forces(&quad(std::f64::consts::FRAC_PI_2))
            .0;
        assert!((e90 - a[0]).abs() < 1e-9, "E(pi/2) got {e90}");
    }

    #[test]
    fn numerical_gradient() {
        let pot = single([0.5, 1.3, -0.7, 0.9, 0.4]);
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
        let pot = single([0.5, 1.0, 0.5, 0.3, 0.2]);
        let coords: Vec<F> = vec![0.1, 1.0, 0.2, 0.0, 0.0, 0.0, 1.0, 0.0, -0.1, 1.2, -0.8, 0.5];
        let (_, f) = pot.calc_energy_forces(&coords);
        for dim in 0..3 {
            let s: F = (0..4).map(|a| f[a * 3 + dim]).sum();
            assert!(s.abs() < 1e-9, "dim {dim} force sum {s}");
        }
    }
}
