//! Tang-Toennies charge / induced-dipole damping (CL&Pol short-range damping).
//!
//! Damps the Coulomb interaction between a charge and an induced dipole (a Drude
//! shell) at short range, preventing the polarization catastrophe:
//!
//! ```text
//! f_n(r) = 1 - c exp(-b r) sum_{k=0}^{n} (b r)^k / k!
//! ```
//!
//! so the damped pair energy is `f_n(r) * q_i q_j / r`. The derivative collapses
//! to a single term: `f'_n(r) = c b exp(-b r) (b r)^n / n!`. CL&Pol canonical
//! settings: `order = 4`, `b = 4.5` (1/A), `c = 1.0` — taken from the pair style's
//! params; the per-atom-type `charge` comes from the atoms block.
//!
//! Reference: Tang & Toennies, J. Chem. Phys. 80 (1984) 3726,
//! DOI 10.1063/1.447150; as emitted by paduagroup/clandpol `coul_tt`.

use std::collections::HashMap;

use crate::ff::forcefield::Params;
use crate::ff::potential::Potential;
use crate::ff::potential::geometry::validate_coords;
use molrs::store::frame::Frame;
use molrs::types::F;

/// Tang-Toennies damped Coulomb pair potential. `b`/`n`/`c` are style-level;
/// `qq[idx]` is the charge product `q_i q_j` of each pair.
pub struct PairTangToennies {
    atom_i: Vec<usize>,
    atom_j: Vec<usize>,
    qq: Vec<F>,
    b: F,
    n: usize,
    c: F,
}

impl PairTangToennies {
    pub fn new(atom_i: Vec<usize>, atom_j: Vec<usize>, qq: Vec<F>, b: F, n: usize, c: F) -> Self {
        assert_eq!(atom_i.len(), atom_j.len());
        assert_eq!(atom_i.len(), qq.len());
        Self {
            atom_i,
            atom_j,
            qq,
            b,
            n,
            c,
        }
    }

    /// `(f_n(r), f'_n(r))` — damping factor and its radial derivative.
    fn damping(&self, r: F) -> (F, F) {
        let br = self.b * r;
        // series = sum_{k=0}^n (br)^k / k!, accumulated term-by-term.
        let mut term = 1.0; // k = 0
        let mut series = term;
        for k in 1..=self.n {
            term *= br / (k as F);
            series += term;
        }
        // term is now (br)^n / n! after the loop's final iteration (k = n).
        let e = (-br).exp();
        let f = 1.0 - self.c * e * series;
        let fp = self.c * self.b * e * term; // c b e^{-br} (br)^n / n!
        (f, fp)
    }
}

impl Potential for PairTangToennies {
    fn calc_energy_forces(&self, coords: &[F]) -> (F, Vec<F>) {
        let n_atoms = validate_coords(coords);
        let mut energy: F = 0.0;
        let mut forces = vec![0.0; coords.len()];

        for idx in 0..self.atom_i.len() {
            let i = self.atom_i[idx];
            let j = self.atom_j[idx];
            debug_assert!(i < n_atoms && j < n_atoms);

            let qq = self.qq[idx];

            let dx = coords[j * 3] - coords[i * 3];
            let dy = coords[j * 3 + 1] - coords[i * 3 + 1];
            let dz = coords[j * 3 + 2] - coords[i * 3 + 2];
            let r2 = dx * dx + dy * dy + dz * dz;
            if r2 < 1e-24 {
                continue;
            }
            let r = r2.sqrt();
            let (f, fp) = self.damping(r);
            energy += f * qq / r;

            // V = f qq / r ; dV/dr = qq (f'/r - f/r^2)
            // factor = -(1/r) dV/dr = qq (f/r^3 - f'/r^2)
            let dvdr = qq * (fp / r - f / r2);
            let factor = -dvdr / r;
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

/// Construct a [`PairTangToennies`] from style params, per-atom-type charge, and topology.
///
/// Style params: `b` (default 4.5), `order` (the damping order n, default 4),
/// `c` (default 1.0). The
/// thole-like per-atom-type `charge` is read from the atoms block.
pub fn pair_tang_toennies_ctor(
    style_params: &Params,
    type_params: &[(&str, &Params)],
    frame: &Frame,
) -> Result<Box<dyn Potential>, String> {
    let type_map: HashMap<&str, &Params> = type_params.iter().copied().collect();
    let b = style_params.get("b").unwrap_or(4.5) as F;
    let n = style_params.get("order").unwrap_or(4.0).round() as usize;
    let c = style_params.get("c").unwrap_or(1.0) as F;

    let atoms = frame
        .get("atoms")
        .ok_or_else(|| "PairTangToennies: frame missing \"atoms\" block".to_string())?;
    let atom_types = atoms
        .get_string("type")
        .ok_or_else(|| "PairTangToennies: atoms block missing \"type\" column".to_string())?;

    let block = frame
        .get("pairs")
        .ok_or_else(|| "PairTangToennies: frame missing \"pairs\" block".to_string())?;
    let i_col = block
        .get_uint("atomi")
        .ok_or_else(|| "PairTangToennies: pairs block missing \"atomi\" column".to_string())?;
    let j_col = block
        .get_uint("atomj")
        .ok_or_else(|| "PairTangToennies: pairs block missing \"atomj\" column".to_string())?;

    let charge = |type_name: &str| -> Result<F, String> {
        type_map
            .get(type_name)
            .ok_or_else(|| format!("PairTangToennies: unknown atom type '{}'", type_name))?
            .get("charge")
            .ok_or_else(|| format!("PairTangToennies type '{}': missing 'charge'", type_name))
            .map(|v| v as F)
    };

    let mut atom_i = Vec::with_capacity(i_col.len());
    let mut atom_j = Vec::with_capacity(i_col.len());
    let mut qq = Vec::with_capacity(i_col.len());
    for idx in 0..i_col.len() {
        let i = i_col[idx] as usize;
        let j = j_col[idx] as usize;
        let qi = charge(&atom_types[i])?;
        let qj = charge(&atom_types[j])?;
        atom_i.push(i);
        atom_j.push(j);
        qq.push(qi * qj);
    }

    Ok(Box::new(PairTangToennies::new(atom_i, atom_j, qq, b, n, c)))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn numerical_forces(pot: &PairTangToennies, coords: &[F]) -> Vec<F> {
        let h = 1e-6;
        let mut num = vec![0.0; coords.len()];
        for k in 0..coords.len() {
            let mut cp = coords.to_vec();
            let mut cm = coords.to_vec();
            cp[k] += h;
            cm[k] -= h;
            num[k] = -(pot.calc_energy(&cp) - pot.calc_energy(&cm)) / (2.0 * h);
        }
        num
    }

    #[test]
    fn damping_matches_closed_form() {
        // n=2, b=1, c=1, r=2: series = 1 + 2 + 2 = 5; f = 1 - e^-2 * 5
        let pot = PairTangToennies::new(vec![0], vec![1], vec![1.0], 1.0, 2, 1.0);
        let coords: Vec<F> = vec![0.0, 0.0, 0.0, 2.0, 0.0, 0.0];
        let f = 1.0 - (-2.0f64).exp() * 5.0;
        let expected = f * 1.0 / 2.0;
        assert!((pot.calc_energy(&coords) - expected).abs() < 1e-12);
    }

    #[test]
    fn forces_match_finite_difference() {
        let pot = PairTangToennies::new(vec![0], vec![1], vec![-0.7], 4.5, 4, 1.0);
        let coords: Vec<F> = vec![0.1, -0.2, 0.05, 1.3, 0.6, -0.3];
        let (_, analytical) = pot.calc_energy_forces(&coords);
        let numerical = numerical_forces(&pot, &coords);
        for k in 0..coords.len() {
            assert!(
                (analytical[k] - numerical[k]).abs() < 1e-5,
                "k={k} a={} n={}",
                analytical[k],
                numerical[k]
            );
        }
    }
}
