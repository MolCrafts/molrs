//! MMFF94 bond stretching: E = (1/2)*143.9325*kb*dr^2*(1 + cs*dr + 7/12*cs^2*dr^2)

use ndarray::{Array2, ArrayView2};

use crate::ff::forcefield::Params;
use crate::ff::potential::Potential;
use crate::ff::potential::geometry::{mag3, sub3, term_table, validate_coords};
use molrs::store::frame::Frame;
use molrs::types::F;

use crate::ff::constants::MDYNE_A_TO_KCAL;
/// Cubic stretch constant (A^-1).
const CS: f64 = -2.0;

pub struct MMFFBondStretch {
    atom_i: Vec<usize>,
    atom_j: Vec<usize>,
    kb: Vec<F>,
    r0: Vec<F>,
}

impl MMFFBondStretch {
    /// The physics, once. Which atoms a term names is the only thing
    /// that differs between the two entry points, so it is the only thing
    /// passed in — a second copy of the loop would be a second place for
    /// the force expression to drift.
    fn fold(
        &self,
        coords: &[F],
        n_terms: usize,
        atoms: impl Fn(usize) -> (usize, usize),
    ) -> (F, Vec<F>) {
        let _n = validate_coords(coords);
        let mut energy: F = 0.0;
        let mut forces = vec![0.0 as F; coords.len()];
        let cs = CS as F;
        let conv = MDYNE_A_TO_KCAL as F;

        for idx in 0..n_terms {
            let (i, j) = atoms(idx);
            let d = sub3(coords, j, coords, i);
            let r = mag3(d);
            let dr = r - self.r0[idx];
            let cs2 = cs * cs;

            energy += 0.5
                * conv
                * self.kb[idx]
                * dr
                * dr
                * (1.0 + cs * dr + (7.0 / 12.0) * cs2 * dr * dr);

            if r < 1e-12 as F {
                continue;
            }
            let de_dr =
                conv * self.kb[idx] * dr * (1.0 + 1.5 * cs * dr + (7.0 / 6.0) * cs2 * dr * dr);
            let factor = -de_dr / r;
            for dim in 0..3 {
                forces[j * 3 + dim] += factor * d[dim];
                forces[i * 3 + dim] -= factor * d[dim];
            }
        }
        (energy, forces)
    }
}

impl Potential for MMFFBondStretch {
    fn calc_energy_forces(&self, coords: &[F]) -> (F, Vec<F>) {
        self.fold(coords, self.atom_i.len(), |t| {
            (self.atom_i[t], self.atom_j[t])
        })
    }

    fn terms(&self) -> Option<Array2<u32>> {
        Some(term_table(&[&self.atom_i, &self.atom_j]))
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
            (terms[[t, 0]] as usize, terms[[t, 1]] as usize)
        })
    }
}

pub fn mmff_bond_ctor(
    _sp: &Params,
    _tp: &[(&str, &Params)],
    frame: &Frame,
) -> Result<Box<dyn Potential>, String> {
    // Per-instance parameters: the MMFF typifier baked `kb`/`r0` onto each bond
    // (table → equivalence fallback → empirical rules). This kernel only reads the
    // columns and evaluates — no force-field-specific resolution lives here.
    let block = frame
        .get("bonds")
        .ok_or("mmff_bond: missing \"bonds\" block")?;
    let i_col = block
        .get_uint("atomi")
        .ok_or("mmff_bond: missing \"atomi\"")?;
    let j_col = block
        .get_uint("atomj")
        .ok_or("mmff_bond: missing \"atomj\"")?;
    let kb_col = block
        .get_float("kb")
        .ok_or("mmff_bond: missing \"kb\" column (typifier did not bake bond params)")?;
    let r0_col = block
        .get_float("r0")
        .ok_or("mmff_bond: missing \"r0\" column (typifier did not bake bond params)")?;

    let n = i_col.len();
    let (mut ai, mut aj, mut kb, mut r0) = (
        Vec::with_capacity(n),
        Vec::with_capacity(n),
        Vec::with_capacity(n),
        Vec::with_capacity(n),
    );
    for idx in 0..n {
        ai.push(i_col[idx] as usize);
        aj.push(j_col[idx] as usize);
        kb.push(kb_col[idx] as F);
        r0.push(r0_col[idx] as F);
    }
    Ok(Box::new(MMFFBondStretch {
        atom_i: ai,
        atom_j: aj,
        kb,
        r0,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mmff_bond_at_equilibrium() {
        let pot = MMFFBondStretch {
            atom_i: vec![0],
            atom_j: vec![1],
            kb: vec![4.258],
            r0: vec![1.508],
        };
        let coords: Vec<F> = vec![0.0, 0.0, 0.0, 1.508, 0.0, 0.0];
        let (e, _) = pot.calc_energy_forces(&coords);
        assert!(
            e.abs() < 1e-6,
            "energy at equilibrium should be ~0, got {}",
            e
        );
    }

    #[test]
    fn test_mmff_bond_stretched() {
        let pot = MMFFBondStretch {
            atom_i: vec![0],
            atom_j: vec![1],
            kb: vec![4.258],
            r0: vec![1.508],
        };
        let coords: Vec<F> = vec![0.0, 0.0, 0.0, 1.608, 0.0, 0.0];
        let (e, forces) = pot.calc_energy_forces(&coords);
        assert!(e > 0.0, "stretched bond should have positive energy");
        for dim in 0..3 {
            let sum = forces[dim] + forces[3 + dim];
            assert!(sum.abs() < 1e-4, "force sum should be ~0, got {}", sum);
        }
    }
}
