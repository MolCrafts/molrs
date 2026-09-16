//! Buckingham pair potential: E = A * exp(-r/rho) - C / r^6
//!
//! The exp-6 form used for repulsion/dispersion (e.g. CL&Pol non-bonded cores).
//! Parameters per pair type: `a` (energy), `rho` (length), `c` (energy*length^6).
//! Lowercase is the canonical spelling (spec ff-params-01) and matches molpy;
//! GROMACS spells the middle one `B = 1/rho`, normalized at that reader.

use std::collections::HashMap;

use crate::ff::forcefield::Params;
use crate::ff::potential::Potential;
use crate::ff::potential::geometry::validate_coords;
use molrs::store::frame::Frame;
use molrs::types::F;

/// Buckingham pair potential with pre-resolved flat arrays.
pub struct PairBuck {
    atom_i: Vec<usize>,
    atom_j: Vec<usize>,
    a: Vec<F>,
    rho: Vec<F>,
    c: Vec<F>,
}

impl PairBuck {
    pub fn new(atom_i: Vec<usize>, atom_j: Vec<usize>, a: Vec<F>, rho: Vec<F>, c: Vec<F>) -> Self {
        assert_eq!(atom_i.len(), atom_j.len());
        assert_eq!(atom_i.len(), a.len());
        assert_eq!(atom_i.len(), rho.len());
        assert_eq!(atom_i.len(), c.len());
        Self {
            atom_i,
            atom_j,
            a,
            rho,
            c,
        }
    }
}

impl Potential for PairBuck {
    fn calc_energy_forces(&self, coords: &[F]) -> (F, Vec<F>) {
        let n_atoms = validate_coords(coords);
        let mut energy: F = 0.0;
        let mut forces = vec![0.0; coords.len()];

        for idx in 0..self.atom_i.len() {
            let i = self.atom_i[idx];
            let j = self.atom_j[idx];
            debug_assert!(i < n_atoms && j < n_atoms);

            let a = self.a[idx];
            let rho = self.rho[idx];
            let c = self.c[idx];

            let dx = coords[j * 3] - coords[i * 3];
            let dy = coords[j * 3 + 1] - coords[i * 3 + 1];
            let dz = coords[j * 3 + 2] - coords[i * 3 + 2];
            let r2 = dx * dx + dy * dy + dz * dz;
            if r2 < 1e-24 {
                continue;
            }
            let r = r2.sqrt();
            let exp_term = a * (-r / rho).exp();
            let r6 = r2 * r2 * r2;
            energy += exp_term - c / r6;

            // E = A exp(-r/rho) - C r^-6
            // dE/dr = -(A/rho) exp(-r/rho) + 6 C r^-7
            // factor = -(1/r) dE/dr = (A/(rho r)) exp(-r/rho) - 6 C r^-8
            let factor = exp_term / (rho * r) - 6.0 * c / (r6 * r2);
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

/// Construct a [`PairBuck`] from style params, type params, and Frame topology.
pub fn pair_buck_ctor(
    _style_params: &Params,
    type_params: &[(&str, &Params)],
    frame: &Frame,
) -> Result<Box<dyn Potential>, String> {
    let type_map: HashMap<&str, &Params> = type_params.iter().copied().collect();

    let block = frame
        .get("pairs")
        .ok_or_else(|| "PairBuck: frame missing \"pairs\" block".to_string())?;
    let i_col = block
        .get_uint("atomi")
        .ok_or_else(|| "PairBuck: pairs block missing \"atomi\" column".to_string())?;
    let j_col = block
        .get_uint("atomj")
        .ok_or_else(|| "PairBuck: pairs block missing \"atomj\" column".to_string())?;
    let type_col = block
        .get_string("type")
        .ok_or_else(|| "PairBuck: pairs block missing \"type\" column".to_string())?;

    let mut atom_i = Vec::with_capacity(i_col.len());
    let mut atom_j = Vec::with_capacity(i_col.len());
    let mut a_vec = Vec::with_capacity(i_col.len());
    let mut rho_vec = Vec::with_capacity(i_col.len());
    let mut c_vec = Vec::with_capacity(i_col.len());

    for idx in 0..i_col.len() {
        let label = &type_col[idx];
        let params = type_map
            .get(label.as_str())
            .ok_or_else(|| format!("PairBuck: unknown pair type '{}'", label))?;
        let a = params
            .get("a")
            .ok_or_else(|| format!("PairBuck type '{}': missing 'a'", label))? as F;
        let rho = params
            .get("rho")
            .ok_or_else(|| format!("PairBuck type '{}': missing 'rho'", label))?
            as F;
        let c = params
            .get("c")
            .ok_or_else(|| format!("PairBuck type '{}': missing 'c'", label))? as F;

        atom_i.push(i_col[idx] as usize);
        atom_j.push(j_col[idx] as usize);
        a_vec.push(a);
        rho_vec.push(rho);
        c_vec.push(c);
    }

    Ok(Box::new(PairBuck::new(
        atom_i, atom_j, a_vec, rho_vec, c_vec,
    )))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn numerical_forces(pot: &PairBuck, coords: &[F]) -> Vec<F> {
        let h = 1e-6;
        let mut num = vec![0.0; coords.len()];
        for k in 0..coords.len() {
            let mut cp = coords.to_vec();
            let mut cm = coords.to_vec();
            cp[k] += h;
            cm[k] -= h;
            let ep = pot.calc_energy(&cp);
            let em = pot.calc_energy(&cm);
            num[k] = -(ep - em) / (2.0 * h); // force = -dE/dx
        }
        num
    }

    #[test]
    fn energy_matches_closed_form() {
        // A=1000, rho=0.3, C=100 at r=2.0:
        // E = 1000*exp(-2/0.3) - 100/64
        let pot = PairBuck::new(vec![0], vec![1], vec![1000.0], vec![0.3], vec![100.0]);
        let coords: Vec<F> = vec![0.0, 0.0, 0.0, 2.0, 0.0, 0.0];
        let e = pot.calc_energy(&coords);
        let expected = 1000.0 * (-2.0f64 / 0.3).exp() - 100.0 / 64.0;
        assert!((e - expected).abs() < 1e-9, "energy {e} vs {expected}");
    }

    #[test]
    fn forces_match_finite_difference() {
        let pot = PairBuck::new(vec![0], vec![1], vec![1000.0], vec![0.3], vec![100.0]);
        // off-axis geometry exercising all three components
        let coords: Vec<F> = vec![0.1, -0.2, 0.05, 1.3, 0.6, -0.3];
        let (_, analytical) = pot.calc_energy_forces(&coords);
        let numerical = numerical_forces(&pot, &coords);
        for k in 0..coords.len() {
            assert!(
                (analytical[k] - numerical[k]).abs() < 1e-5,
                "k={k} analytical={} numerical={}",
                analytical[k],
                numerical[k]
            );
        }
    }

    #[test]
    fn newtons_third_law() {
        let pot = PairBuck::new(vec![0], vec![1], vec![1000.0], vec![0.3], vec![100.0]);
        let coords: Vec<F> = vec![0.0, 0.0, 0.0, 1.2, 0.7, -0.4];
        let (_, f) = pot.calc_energy_forces(&coords);
        for dim in 0..3 {
            assert!((f[dim] + f[3 + dim]).abs() < 1e-9, "dim {dim}");
        }
    }
}
