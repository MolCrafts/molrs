//! UFF bond stretch: E = ½ · kb · (r − r0)² (RDKit `BondStretchContrib`).

use ndarray::{Array2, ArrayView2};

use crate::ff::forcefield::Params;
use crate::ff::potential::geometry::term_table;
use crate::ff::potential::geometry::validate_coords;
use crate::ff::potential::{IndexedTerms, Member, Potential};
use molrs::store::frame::Frame;
use molrs::types::F;

/// Harmonic UFF bond stretch with per-instance `kb` / `r0`.
pub struct UffBond {
    atom_i: Vec<usize>,
    atom_j: Vec<usize>,
    kb: Vec<F>,
    r0: Vec<F>,
}

impl UffBond {
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
        let mut energy = 0.0 as F;
        let mut forces = vec![0.0 as F; coords.len()];
        for idx in 0..n_terms {
            let (i, j) = atoms(idx);
            let dx = coords[j * 3] - coords[i * 3];
            let dy = coords[j * 3 + 1] - coords[i * 3 + 1];
            let dz = coords[j * 3 + 2] - coords[i * 3 + 2];
            let r = (dx * dx + dy * dy + dz * dz).sqrt();
            let dr = r - self.r0[idx];
            energy += 0.5 * self.kb[idx] * dr * dr;
            if r < 1e-12 as F {
                continue;
            }
            // F = −∇E; ∇_j r = (j−i)/r, so F_j = −(kb·dr)·(j−i)/r
            let pref = -self.kb[idx] * dr / r;
            for dim in 0..3 {
                let d = [dx, dy, dz][dim];
                forces[j * 3 + dim] += pref * d;
                forces[i * 3 + dim] -= pref * d;
            }
        }
        (energy, forces)
    }
}

impl Potential for UffBond {
    fn calc_energy_forces(&self, coords: &[F]) -> (F, Vec<F>) {
        self.fold(coords, self.atom_i.len(), |t| {
            (self.atom_i[t], self.atom_j[t])
        })
    }
}

impl IndexedTerms for UffBond {
    fn terms(&self) -> Array2<u32> {
        term_table(&[&self.atom_i, &self.atom_j])
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

pub fn uff_bond_ctor(
    _sp: &Params,
    _tp: &[(&str, &Params)],
    frame: &Frame,
) -> Result<Member, String> {
    let block = frame
        .get("bonds")
        .ok_or("uff_bond: missing \"bonds\" block")?;
    let i = block.get_uint("atomi").ok_or("uff_bond: missing atomi")?;
    let j = block.get_uint("atomj").ok_or("uff_bond: missing atomj")?;
    let kb = block
        .get_float("kb")
        .ok_or("uff_bond: missing kb (typifier must bake)")?;
    let r0 = block
        .get_float("r0")
        .ok_or("uff_bond: missing r0 (typifier must bake)")?;
    let n = i.len();
    Ok(Member::indexed(UffBond {
        atom_i: (0..n).map(|t| i[t] as usize).collect(),
        atom_j: (0..n).map(|t| j[t] as usize).collect(),
        kb: kb.iter().map(|&v| v as F).collect(),
        r0: r0.iter().map(|&v| v as F).collect(),
    }))
}
