//! Harmonic bond potential: E = 0.5 * k0 * (r - r0)^2

use std::collections::HashMap;

use crate::forcefield::Params;
use crate::frame::Frame;
use crate::potential::Potential;
use crate::potential::geometry::validate_coords;
use crate::types::F;

/// Harmonic bond potential with pre-resolved flat arrays.
pub struct BondHarmonic {
    atom_i: Vec<usize>,
    atom_j: Vec<usize>,
    k0: Vec<F>,
    r0: Vec<F>,
}

impl BondHarmonic {
    pub fn new(atom_i: Vec<usize>, atom_j: Vec<usize>, k0: Vec<F>, r0: Vec<F>) -> Self {
        assert_eq!(atom_i.len(), atom_j.len());
        assert_eq!(atom_i.len(), k0.len());
        assert_eq!(atom_i.len(), r0.len());
        Self {
            atom_i,
            atom_j,
            k0,
            r0,
        }
    }
}

impl Potential for BondHarmonic {
    fn eval(&self, coords: &[F]) -> (F, Vec<F>) {
        let n_atoms = validate_coords(coords);
        let mut energy: F = 0.0;
        let mut forces = vec![0.0; coords.len()];

        for idx in 0..self.atom_i.len() {
            let i = self.atom_i[idx];
            let j = self.atom_j[idx];
            debug_assert!(i < n_atoms && j < n_atoms);

            let k0 = self.k0[idx];
            let r0 = self.r0[idx];

            let dx = coords[j * 3] - coords[i * 3];
            let dy = coords[j * 3 + 1] - coords[i * 3 + 1];
            let dz = coords[j * 3 + 2] - coords[i * 3 + 2];
            let r = (dx * dx + dy * dy + dz * dz).sqrt();
            let dr = r - r0;
            energy += 0.5 * k0 * dr * dr;

            if r < 1e-12 {
                continue;
            }

            let factor = -k0 * dr / r;
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
    let mut k0_vec = Vec::with_capacity(i_col.len());
    let mut r0_vec = Vec::with_capacity(i_col.len());

    for idx in 0..i_col.len() {
        let label = &type_col[idx];
        let params = type_map
            .get(label.as_str())
            .ok_or_else(|| format!("BondHarmonic: unknown bond type '{}'", label))?;
        let k0 = params
            .get("k0")
            .ok_or_else(|| format!("BondHarmonic type '{}': missing 'k0'", label))?
            as F;
        let r0 = params
            .get("r0")
            .ok_or_else(|| format!("BondHarmonic type '{}': missing 'r0'", label))?
            as F;

        atom_i.push(i_col[idx] as usize);
        atom_j.push(j_col[idx] as usize);
        k0_vec.push(k0);
        r0_vec.push(r0);
    }

    Ok(Box::new(BondHarmonic::new(atom_i, atom_j, k0_vec, r0_vec)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::block::Block;
    use crate::forcefield::ForceField;
    use crate::potential::extract_coords;
    use crate::types::U;
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

        let (e, forces) = pot.eval(&coords);
        assert!((e - 37.5).abs() < 1e-3);
        assert!((forces[0] - 150.0).abs() < 1e-3);
        assert!((forces[3] + 150.0).abs() < 1e-3);
    }

    #[test]
    fn test_forcefield_compile_integration() {
        let mut ff = ForceField::new("test");
        ff.def_bondstyle("harmonic")
            .def_type("CT-CT", &[("k0", 300.0), ("r0", 1.5)]);

        let mut frame = Frame::new();
        frame.insert("atoms", make_atoms(&[[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]));

        let mut bonds = Block::new();
        bonds
            .insert("atomi", Array1::from_vec(vec![0 as U]).into_dyn())
            .unwrap();
        bonds
            .insert("atomj", Array1::from_vec(vec![1 as U]).into_dyn())
            .unwrap();
        bonds
            .insert(
                "type",
                Array1::from_vec(vec!["CT-CT".to_string()]).into_dyn(),
            )
            .unwrap();
        frame.insert("bonds", bonds);

        let pots = ff.compile(&frame).unwrap();
        let coords = extract_coords(&frame).unwrap();
        let e = pots.energy(&coords);
        assert!((e - 37.5).abs() < 1e-3);
    }
}
