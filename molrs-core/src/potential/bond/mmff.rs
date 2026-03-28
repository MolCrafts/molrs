//! MMFF94 bond stretching: E = (1/2)*143.9325*kb*dr^2*(1 + cs*dr + 7/12*cs^2*dr^2)

use std::collections::HashMap;

use crate::forcefield::Params;
use crate::frame::Frame;
use crate::potential::Potential;
use crate::potential::geometry::{mag3, sub3, validate_coords};
use crate::types::F;

/// md/A -> kcal/mol conversion.
const MDYNE_A_TO_KCAL: f64 = 143.9325;
/// Cubic stretch constant (A^-1).
const CS: f64 = -2.0;

pub struct MMFFBondStretch {
    atom_i: Vec<usize>,
    atom_j: Vec<usize>,
    kb: Vec<F>,
    r0: Vec<F>,
}

impl Potential for MMFFBondStretch {
    fn eval(&self, coords: &[F]) -> (F, Vec<F>) {
        let _n = validate_coords(coords);
        let mut energy: F = 0.0;
        let mut forces = vec![0.0 as F; coords.len()];
        let cs = CS as F;
        let conv = MDYNE_A_TO_KCAL as F;

        for idx in 0..self.atom_i.len() {
            let (i, j) = (self.atom_i[idx], self.atom_j[idx]);
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

pub fn mmff_bond_ctor(
    _sp: &Params,
    tp: &[(&str, &Params)],
    frame: &Frame,
) -> Result<Box<dyn Potential>, String> {
    let type_map: HashMap<&str, &Params> = tp.iter().copied().collect();
    let block = frame
        .get("bonds")
        .ok_or("mmff_bond: missing \"bonds\" block")?;
    let i_col = block
        .get_uint("atomi")
        .ok_or("mmff_bond: missing \"atomi\"")?;
    let j_col = block
        .get_uint("atomj")
        .ok_or("mmff_bond: missing \"atomj\"")?;
    let ty_col = block
        .get_string("type")
        .ok_or("mmff_bond: missing \"type\"")?;

    let n = i_col.len();
    let (mut ai, mut aj, mut kb, mut r0) = (
        Vec::with_capacity(n),
        Vec::with_capacity(n),
        Vec::with_capacity(n),
        Vec::with_capacity(n),
    );

    for idx in 0..n {
        let p = type_map
            .get(ty_col[idx].as_str())
            .ok_or_else(|| format!("mmff_bond: unknown type '{}'", ty_col[idx]))?;
        ai.push(i_col[idx] as usize);
        aj.push(j_col[idx] as usize);
        kb.push(p.get("kb").ok_or("mmff_bond: missing 'kb'")? as F);
        r0.push(p.get("r0").ok_or("mmff_bond: missing 'r0'")? as F);
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
        let (e, _) = pot.eval(&coords);
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
        let (e, forces) = pot.eval(&coords);
        assert!(e > 0.0, "stretched bond should have positive energy");
        for dim in 0..3 {
            let sum = forces[dim] + forces[3 + dim];
            assert!(sum.abs() < 1e-4, "force sum should be ~0, got {}", sum);
        }
    }
}
