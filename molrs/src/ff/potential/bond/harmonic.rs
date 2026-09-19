//! Harmonic bond potential: E = 0.5 * k * (r - r0)^2

use std::collections::HashMap;

use ndarray::{Array2, ArrayView2};

use crate::ff::forcefield::Params;
use crate::ff::potential::Potential;
use crate::ff::potential::geometry::{term_table, validate_coords};
use molrs::store::frame::Frame;
use molrs::types::F;

/// Harmonic bond potential with pre-resolved flat arrays.
pub struct BondHarmonic {
    atom_i: Vec<usize>,
    atom_j: Vec<usize>,
    k: Vec<F>,
    r0: Vec<F>,
}

impl BondHarmonic {
    pub fn new(atom_i: Vec<usize>, atom_j: Vec<usize>, k: Vec<F>, r0: Vec<F>) -> Self {
        assert_eq!(atom_i.len(), atom_j.len());
        assert_eq!(atom_i.len(), k.len());
        assert_eq!(atom_i.len(), r0.len());
        Self {
            atom_i,
            atom_j,
            k,
            r0,
        }
    }
}

impl BondHarmonic {
    /// The physics, once. Which atoms a term names is the only thing that
    /// differs between the two entry points, so it is the only thing passed
    /// in — a second copy of the loop would be a second place for the force
    /// expression to drift.
    fn fold(
        &self,
        coords: &[F],
        n_terms: usize,
        atoms: impl Fn(usize) -> (usize, usize),
    ) -> (F, Vec<F>) {
        let n_atoms = validate_coords(coords);
        let mut energy: F = 0.0;
        let mut forces = vec![0.0; coords.len()];

        for idx in 0..n_terms {
            let (i, j) = atoms(idx);
            debug_assert!(i < n_atoms && j < n_atoms);

            let k = self.k[idx];
            let r0 = self.r0[idx];

            let dx = coords[j * 3] - coords[i * 3];
            let dy = coords[j * 3 + 1] - coords[i * 3 + 1];
            let dz = coords[j * 3 + 2] - coords[i * 3 + 2];
            let r = (dx * dx + dy * dy + dz * dz).sqrt();
            let dr = r - r0;
            energy += 0.5 * k * dr * dr;

            if r < 1e-12 {
                continue;
            }

            let factor = -k * dr / r;
            let fx = factor * dx;
            let fy = factor * dy;
            let fz = factor * dz;

            forces[j * 3] += fx;
            forces[j * 3 + 1] += fy;
            forces[j * 3 + 2] += fz;
            forces[i * 3] -= fx;
            forces[i * 3 + 1] -= fy;
            forces[i * 3 + 2] -= fz;
        }

        (energy, forces)
    }
}

impl Potential for BondHarmonic {
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

/// Construct a [`BondHarmonic`] from style params, type params, and Frame topology.
pub fn bond_harmonic_ctor(
    _style_params: &Params,
    type_params: &[(&str, &Params)],
    frame: &Frame,
) -> Result<Box<dyn Potential>, String> {
    let type_map: HashMap<&str, &Params> = type_params.iter().copied().collect();

    let block = frame
        .get("bonds")
        .ok_or_else(|| "BondHarmonic: frame missing \"bonds\" block".to_string())?;
    let i_col = block
        .get_uint("atomi")
        .ok_or_else(|| "BondHarmonic: bonds block missing \"atomi\" column".to_string())?;
    let j_col = block
        .get_uint("atomj")
        .ok_or_else(|| "BondHarmonic: bonds block missing \"atomj\" column".to_string())?;
    let type_col = block
        .get_string("type")
        .ok_or_else(|| "BondHarmonic: bonds block missing \"type\" column".to_string())?;

    let mut atom_i = Vec::with_capacity(i_col.len());
    let mut atom_j = Vec::with_capacity(i_col.len());
    let mut k_vec = Vec::with_capacity(i_col.len());
    let mut r0_vec = Vec::with_capacity(i_col.len());

    for idx in 0..i_col.len() {
        let label = &type_col[idx];
        let params = type_map
            .get(label.as_str())
            .ok_or_else(|| format!("BondHarmonic: unknown bond type '{}'", label))?;
        // `k` is the one spelling (spec ff-params-01). The `k0` alias is gone:
        // it was the same key in two conventions, half of the tree meaning
        // `E = ½k(r−r₀)²` and half meaning AMBER's un-halved `K`.
        let k = params
            .get("k")
            .ok_or_else(|| format!("BondHarmonic type '{}': missing 'k'", label))?
            as F;
        let r0 = params
            .get("r0")
            .ok_or_else(|| format!("BondHarmonic type '{}': missing 'r0'", label))?
            as F;

        atom_i.push(i_col[idx] as usize);
        atom_j.push(j_col[idx] as usize);
        k_vec.push(k);
        r0_vec.push(r0);
    }

    Ok(Box::new(BondHarmonic::new(atom_i, atom_j, k_vec, r0_vec)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ff::forcefield::ForceField;
    use crate::ff::potential::extract_coords;
    use molrs::store::block::Block;
    use molrs::types::Idx;
    use ndarray::Array1;

    fn make_atoms(coords: &[[F; 3]]) -> Block {
        let mut atoms = Block::new();
        atoms
            .insert(
                "x",
                Array1::from_vec(coords.iter().map(|p| p[0]).collect()).into_dyn(),
            )
            .unwrap();
        atoms
            .insert(
                "y",
                Array1::from_vec(coords.iter().map(|p| p[1]).collect()).into_dyn(),
            )
            .unwrap();
        atoms
            .insert(
                "z",
                Array1::from_vec(coords.iter().map(|p| p[2]).collect()).into_dyn(),
            )
            .unwrap();
        atoms
    }

    #[test]
    fn test_bond_harmonic_energy_and_force() {
        let pot = BondHarmonic::new(vec![0], vec![1], vec![300.0], vec![1.5]);
        let coords: Vec<F> = vec![0.0, 0.0, 0.0, 2.0, 0.0, 0.0];

        let (e, forces) = pot.calc_energy_forces(&coords);
        assert!((e - 37.5).abs() < 1e-3);
        assert!((forces[0] - 150.0).abs() < 1e-3);
        assert!((forces[3] + 150.0).abs() < 1e-3);
    }

    #[test]
    fn test_forcefield_compile_integration() {
        let mut ff = ForceField::new("test");
        ff.def_bondstyle("harmonic")
            .def_type("CT-CT", &[("k", 300.0), ("r0", 1.5)]);

        let mut frame = Frame::new();
        frame.insert("atoms", make_atoms(&[[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]));

        let mut bonds = Block::new();
        bonds
            .insert("atomi", Array1::from_vec(vec![0 as Idx]).into_dyn())
            .unwrap();
        bonds
            .insert("atomj", Array1::from_vec(vec![1 as Idx]).into_dyn())
            .unwrap();
        bonds
            .insert(
                "type",
                Array1::from_vec(vec!["CT-CT".to_string()]).into_dyn(),
            )
            .unwrap();
        frame.insert("bonds", bonds);

        let pots = ff.to_potentials(&frame).unwrap();
        let coords = extract_coords(&frame).unwrap();
        let e = pots.calc_energy(&coords);
        assert!((e - 37.5).abs() < 1e-3);
    }
}
