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
use crate::ff::potential::gather_copies;
use crate::ff::potential::geometry::validate_coords;
use crate::ff::potential::pair::atom_type_index;
use crate::ff::potential::pair::energy_forces;
use molrs::math::Virial;
use molrs::spatial::neighbors::Neighbors;
use molrs::store::frame::Frame;
use molrs::types::F;

/// Tang-Toennies damped Coulomb pair potential. `b`/`n`/`c` are style-level;
/// `qq[idx]` is the charge product `q_i q_j` of each pair.
/// Where a pair's charge product comes from.
enum Charges {
    /// Products resolved against one fixed pair list at construction.
    Compiled {
        atom_i: Vec<usize>,
        atom_j: Vec<usize>,
        qq: Vec<F>,
    },
    /// Per-atom charges; the product is formed when a pair turns up.
    ///
    /// What a neighbour-driven evaluation needs: a neighbour table is a
    /// different list of pairs every rebuild. Under a ghost régime the vector
    /// covers the copies too, each carrying its owner's charge.
    PerAtom {
        q: Vec<F>,
        /// How many of the entries above are atoms; the rest are copies, and
        /// are rebuilt from their owners whenever the copy list is.
        n_owned: usize,
    },
}

pub struct PairTangToennies {
    charges: Charges,
    b: F,
    n: usize,
    c: F,
}

impl PairTangToennies {
    pub fn new(atom_i: Vec<usize>, atom_j: Vec<usize>, qq: Vec<F>, b: F, n: usize, c: F) -> Self {
        assert_eq!(atom_i.len(), atom_j.len());
        assert_eq!(atom_i.len(), qq.len());
        Self {
            charges: Charges::Compiled { atom_i, atom_j, qq },
            b,
            n,
            c,
        }
    }

    /// A kernel that forms `qᵢqⱼ` from the atoms a pair names.
    pub fn typed(q: Vec<F>, b: F, n: usize, c: F) -> Self {
        let n_owned = q.len();
        Self {
            charges: Charges::PerAtom { q, n_owned },
            b,
            n,
            c,
        }
    }

    /// The pair term for one already-reduced separation.
    fn pair_kernel(&self, r2: F, disp: [F; 3], qq: F) -> Option<(F, [F; 3])> {
        if r2 < 1e-24 {
            return None;
        }
        let r = r2.sqrt();
        let (f, fp) = self.damping(r);
        let energy = f * qq / r;
        // V = f qq / r ; dV/dr = qq (f'/r - f/r^2)
        // factor = -(1/r) dV/dr = qq (f/r^3 - f'/r^2)
        let dvdr = qq * (fp / r - f / r2);
        let factor = -dvdr / r;
        Some((
            energy,
            [factor * disp[0], factor * disp[1], factor * disp[2]],
        ))
    }

    /// The accumulation, once.
    fn fold(
        &self,
        n_components: usize,
        n_pairs: usize,
        pair: impl Fn(usize) -> (usize, usize, F, [F; 3], F),
    ) -> (F, Vec<F>, Virial) {
        let mut energy: F = 0.0;
        let mut forces = vec![0.0; n_components];
        let mut virial = Virial::ZERO;
        for idx in 0..n_pairs {
            let (i, j, qq, disp, r2) = pair(idx);
            let Some((e, f)) = self.pair_kernel(r2, disp, qq) else {
                continue;
            };
            energy += e;
            virial.add_outer(f, disp);
            forces[j * 3] += f[0];
            forces[j * 3 + 1] += f[1];
            forces[j * 3 + 2] += f[2];
            forces[i * 3] -= f[0];
            forces[i * 3 + 1] -= f[1];
            forces[i * 3 + 2] -= f[2];
        }
        (energy, forces, virial)
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
        let Charges::Compiled { atom_i, atom_j, qq } = &self.charges else {
            // Per-atom charges need a pair table, and nobody handed one over.
            return (0.0, vec![0.0; coords.len()]);
        };
        energy_forces(self.fold(coords.len(), atom_i.len(), |idx| {
            let i = atom_i[idx];
            let j = atom_j[idx];
            debug_assert!(i < n_atoms && j < n_atoms);
            let d = [
                coords[j * 3] - coords[i * 3],
                coords[j * 3 + 1] - coords[i * 3 + 1],
                coords[j * 3 + 2] - coords[i * 3 + 2],
            ];
            let r2 = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
            (i, j, qq[idx], d, r2)
        }))
    }

    fn calc_energy_forces_with_pairs(&self, coords: &[F], pairs: &Neighbors) -> (F, Vec<F>) {
        let (e, f, _) = self.calc_energy_forces_with_pairs_virial(coords, pairs);
        (e, f)
    }

    fn calc_energy_forces_with_pairs_virial(
        &self,
        coords: &[F],
        pairs: &Neighbors,
    ) -> (F, Vec<F>, Option<Virial>) {
        let Charges::PerAtom { q, .. } = &self.charges else {
            // A compiled kernel answers for its own list, not for this one.
            let (e, f) = self.calc_energy_forces(coords);
            return (e, f, None);
        };
        let (Some(disp), Some(d2)) = (pairs.disp(), pairs.dist_sq()) else {
            return (0.0, vec![0.0; coords.len()], None);
        };
        let i_col = pairs.query_point_indices();
        let j_col = pairs.point_indices();
        let (e, f, w) = self.fold(coords.len(), i_col.len(), |p| {
            let i = i_col[p] as usize;
            let j = j_col[p] as usize;
            debug_assert!(
                i < q.len() && j < q.len(),
                "a pair names an atom the charge vector does not cover"
            );
            (
                i,
                j,
                q[i] * q[j],
                [disp[[p, 0]], disp[[p, 1]], disp[[p, 2]]],
                d2[p],
            )
        });
        (e, f, Some(w))
    }

    fn gather_onto_copies(&mut self, owner: &[u32]) {
        let Charges::PerAtom { q, n_owned, .. } = &mut self.charges else {
            // Nothing per atom to extend.
            return;
        };
        gather_copies(q, *n_owned, owner);
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

/// Construct a neighbour-driven [`PairTangToennies`] from per-atom parameters.
///
/// The counterpart of [`pair_tang_toennies_ctor`]: the same force field, keyed on the atoms
/// instead of on a pair list, so it can answer for whatever pairs a neighbour
/// search turns up. It reads no `pairs` block — there is none to read when the
/// list is rebuilt every few steps.
pub fn pair_tang_toennies_typed_ctor(
    style_params: &Params,
    type_params: &[(&str, &Params)],
    frame: &Frame,
) -> Result<Box<dyn Potential>, String> {
    let type_map: HashMap<&str, &Params> = type_params.iter().copied().collect();
    let b = style_params.get("b").unwrap_or(4.5) as F;
    let n = style_params.get("order").unwrap_or(4.0).round() as usize;
    let c = style_params.get("c").unwrap_or(1.0) as F;

    let (type_id, labels) = atom_type_index(frame)?;
    let mut per_type = Vec::with_capacity(labels.len());
    for l in &labels {
        let p = type_map
            .get(l.as_str())
            .ok_or_else(|| format!("PairTangToennies: unknown atom type '{l}'"))?;
        per_type.push(
            p.get("charge")
                .ok_or_else(|| format!("PairTangToennies type '{l}': missing 'charge'"))?
                as F,
        );
    }
    let q: Vec<F> = type_id.iter().map(|&t| per_type[t as usize]).collect();
    Ok(Box::new(PairTangToennies::typed(q, b, n, c)))
}

#[cfg(test)]
mod tests {

    /// Forming `qᵢqⱼ` from the atoms is the same number as having formed it
    /// earlier against a fixed list — bit for bit, on the same pairs.
    #[test]
    fn per_atom_charges_score_a_pair_exactly_as_compiled_products() {
        use crate::ff::potential::pair::testing::{assert_same, table_over};

        let q = vec![0.4_f64, -0.7, 0.3, -0.2];
        let coords: Vec<F> = vec![
            0.0, 0.0, 0.0, //
            2.1, 0.4, 0.2, //
            1.2, 1.9, 0.7, //
            3.0, 2.3, 1.1,
        ];
        let links = [(0_usize, 1_usize), (0, 2), (1, 3), (2, 3)];
        let (b, n, c) = (4.5_f64, 4_usize, 1.0_f64);

        let (ai, aj): (Vec<usize>, Vec<usize>) = links.iter().copied().unzip();
        let qq: Vec<F> = links.iter().map(|&(i, j)| q[i] * q[j]).collect();
        let compiled = PairTangToennies::new(ai, aj, qq, b, n, c);
        let typed = PairTangToennies::typed(q, b, n, c);

        let table = table_over(&coords, &links);
        assert_same(
            "coul/tt",
            compiled.calc_energy_forces(&coords),
            typed.calc_energy_forces_with_pairs(&coords, &table),
        );
    }
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
