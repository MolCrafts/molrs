//! Class2 (9-6) Lennard-Jones pair potential:
//! E = epsilon * (2*(sigma/r)^9 - 3*(sigma/r)^6)
//!
//! The COMPASS/class2 non-bonded form. Parameters per pair type: `epsilon`
//! (energy), `sigma` (length).

use std::collections::HashMap;

use crate::ff::forcefield::Params;
use crate::ff::forcefield::pair_type_name;
use crate::ff::potential::Potential;
use crate::ff::potential::gather_copies;
use crate::ff::potential::geometry::validate_coords;
use crate::ff::potential::pair::atom_type_index;
use crate::ff::potential::pair::type_pair;
use molrs::spatial::neighbors::Neighbors;
use molrs::store::frame::Frame;
use molrs::types::F;

/// Where a pair's class2 `(ε, σ)` comes from.
enum Source {
    /// Resolved against one fixed pair list at construction, keyed by the
    /// `type` label on each row.
    Compiled {
        atom_i: Vec<usize>,
        atom_j: Vec<usize>,
        epsilon: Vec<F>,
        sigma: Vec<F>,
    },
    /// A type-pair table, keyed by the types of the two atoms.
    ///
    /// The row label a compiled kernel keys on belongs to a pair list that a
    /// neighbour engine rebuilds from scratch; the atoms' types survive it.
    /// This is LAMMPS's `pair_coeff i j` model. Under a ghost régime `type_id`
    /// covers the copies too, each carrying its owner's type.
    Typed {
        type_id: Vec<u32>,
        ntypes: usize,
        epsilon: Vec<F>,
        sigma: Vec<F>,
        /// How many of the entries above are atoms; the rest are copies, and
        /// are rebuilt from their owners whenever the copy list is.
        n_owned: usize,
    },
}

pub struct PairLJClass2 {
    source: Source,
}

impl PairLJClass2 {
    pub fn new(atom_i: Vec<usize>, atom_j: Vec<usize>, epsilon: Vec<F>, sigma: Vec<F>) -> Self {
        let n = atom_i.len();
        assert_eq!(atom_j.len(), n);
        assert_eq!(epsilon.len(), n);
        assert_eq!(sigma.len(), n);
        Self {
            source: Source::Compiled {
                atom_i,
                atom_j,
                epsilon,
                sigma,
            },
        }
    }

    /// A kernel that finds its parameters from the types of the two atoms.
    ///
    /// `type_id` is one type index per atom; each table is `ntypes × ntypes`
    /// laid out `ti * ntypes + tj`. This is the form a neighbour-driven
    /// evaluation needs.
    ///
    /// A neighbour table is a different list of pairs every rebuild, so a
    /// parameter resolved against an older one belongs to different atoms.
    /// Keyed on the atoms instead, it can be found for whatever pair turns up
    /// — including one that names a periodic copy.
    pub fn typed(type_id: Vec<u32>, ntypes: usize, epsilon: Vec<F>, sigma: Vec<F>) -> Self {
        let n_cells = ntypes * ntypes;
        assert_eq!(n_cells, epsilon.len(), "epsilon must cover every type pair");
        assert_eq!(n_cells, sigma.len(), "sigma must cover every type pair");
        debug_assert!(
            type_id.iter().all(|&t| (t as usize) < ntypes),
            "an atom has a type with no parameters"
        );
        let n_owned = type_id.len();
        Self {
            source: Source::Typed {
                type_id,
                ntypes,
                epsilon,
                sigma,
                n_owned,
            },
        }
    }

    /// The pair term for one already-reduced separation.
    fn pair_kernel(&self, r2: F, disp: [F; 3], eps: F, sigma: F) -> Option<(F, [F; 3])> {
        if r2 < 1e-24 {
            return None;
        }
        let r = r2.sqrt();
        let u = sigma / r; // sigma/r
        let u3 = u * u * u;
        let u6 = u3 * u3;
        let u9 = u6 * u3;
        let energy = eps * (2.0 * u9 - 3.0 * u6);

        // E = eps(2 u^9 - 3 u^6), u = sigma/r, du/dr = -u/r
        // dE/dr = eps(18 u^8 - 18 u^5)(-u/r) = -18 eps (u^9 - u^6)/r
        // factor = -(1/r) dE/dr = 18 eps (u^9 - u^6)/r^2
        let factor = 18.0 * eps * (u9 - u6) / r2;
        Some((
            energy,
            [factor * disp[0], factor * disp[1], factor * disp[2]],
        ))
    }

    /// The accumulation, once. Only where the pairs and the parameters come
    /// from differs between the two entry points.
    fn fold(
        &self,
        n_components: usize,
        n_pairs: usize,
        pair: impl Fn(usize) -> (usize, usize, (F, F), [F; 3], F),
    ) -> (F, Vec<F>) {
        let mut energy: F = 0.0;
        let mut forces = vec![0.0; n_components];
        for idx in 0..n_pairs {
            let (i, j, (eps, sigma), disp, r2) = pair(idx);
            let Some((e, f)) = self.pair_kernel(r2, disp, eps, sigma) else {
                continue;
            };
            energy += e;
            forces[j * 3] += f[0];
            forces[j * 3 + 1] += f[1];
            forces[j * 3 + 2] += f[2];
            forces[i * 3] -= f[0];
            forces[i * 3 + 1] -= f[1];
            forces[i * 3 + 2] -= f[2];
        }
        (energy, forces)
    }
}

impl Potential for PairLJClass2 {
    fn calc_energy_forces(&self, coords: &[F]) -> (F, Vec<F>) {
        let n_atoms = validate_coords(coords);
        let Source::Compiled {
            atom_i,
            atom_j,
            epsilon,
            sigma,
        } = &self.source
        else {
            // A type table needs a pair table, and nobody handed one over.
            return (0.0, vec![0.0; coords.len()]);
        };
        self.fold(coords.len(), atom_i.len(), |idx| {
            let i = atom_i[idx];
            let j = atom_j[idx];
            debug_assert!(i < n_atoms && j < n_atoms);
            let d = [
                coords[j * 3] - coords[i * 3],
                coords[j * 3 + 1] - coords[i * 3 + 1],
                coords[j * 3 + 2] - coords[i * 3 + 2],
            ];
            let r2 = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
            (i, j, (epsilon[idx], sigma[idx]), d, r2)
        })
    }

    fn calc_energy_forces_with_pairs(&self, coords: &[F], pairs: &Neighbors) -> (F, Vec<F>) {
        let Source::Typed {
            type_id,
            ntypes,
            epsilon,
            sigma,
            ..
        } = &self.source
        else {
            // A compiled kernel answers for its own list, not for this one.
            return self.calc_energy_forces(coords);
        };
        let (Some(disp), Some(d2)) = (pairs.disp(), pairs.dist_sq()) else {
            return (0.0, vec![0.0; coords.len()]);
        };
        let i_col = pairs.query_point_indices();
        let j_col = pairs.point_indices();
        self.fold(coords.len(), i_col.len(), |p| {
            let i = i_col[p] as usize;
            let j = j_col[p] as usize;
            debug_assert!(
                i < type_id.len() && j < type_id.len(),
                "a pair names an atom the type table does not cover"
            );
            let t = type_pair(type_id[i], type_id[j], *ntypes);
            (
                i,
                j,
                (epsilon[t], sigma[t]),
                [disp[[p, 0]], disp[[p, 1]], disp[[p, 2]]],
                d2[p],
            )
        })
    }

    fn gather_onto_copies(&mut self, owner: &[u32]) {
        let Source::Typed {
            type_id, n_owned, ..
        } = &mut self.source
        else {
            // Nothing per atom to extend.
            return;
        };
        gather_copies(type_id, *n_owned, owner);
    }
}

/// Construct a [`PairLJClass2`] from style params, type params, and Frame topology.
pub fn pair_lj_class2_ctor(
    _style_params: &Params,
    type_params: &[(&str, &Params)],
    frame: &Frame,
) -> Result<Box<dyn Potential>, String> {
    let type_map: HashMap<&str, &Params> = type_params.iter().copied().collect();

    let block = frame
        .get("pairs")
        .ok_or_else(|| "PairLJClass2: frame missing \"pairs\" block".to_string())?;
    let i_col = block
        .get_uint("atomi")
        .ok_or_else(|| "PairLJClass2: pairs block missing \"atomi\" column".to_string())?;
    let j_col = block
        .get_uint("atomj")
        .ok_or_else(|| "PairLJClass2: pairs block missing \"atomj\" column".to_string())?;
    let type_col = block
        .get_string("type")
        .ok_or_else(|| "PairLJClass2: pairs block missing \"type\" column".to_string())?;

    let mut atom_i = Vec::with_capacity(i_col.len());
    let mut atom_j = Vec::with_capacity(i_col.len());
    let mut eps_vec = Vec::with_capacity(i_col.len());
    let mut sig_vec = Vec::with_capacity(i_col.len());

    for idx in 0..i_col.len() {
        let label = &type_col[idx];
        let params = type_map
            .get(label.as_str())
            .ok_or_else(|| format!("PairLJClass2: unknown pair type '{}'", label))?;
        let eps = params
            .get("epsilon")
            .ok_or_else(|| format!("PairLJClass2 type '{}': missing 'epsilon'", label))?
            as F;
        let sigma = params
            .get("sigma")
            .ok_or_else(|| format!("PairLJClass2 type '{}': missing 'sigma'", label))?
            as F;

        atom_i.push(i_col[idx] as usize);
        atom_j.push(j_col[idx] as usize);
        eps_vec.push(eps);
        sig_vec.push(sigma);
    }

    Ok(Box::new(PairLJClass2::new(
        atom_i, atom_j, eps_vec, sig_vec,
    )))
}

/// Construct a neighbour-driven [`PairLJClass2`] from per-atom parameters.
///
/// The counterpart of [`pair_lj_class2_ctor`]: the same force field, keyed on the atoms
/// instead of on a pair list, so it can answer for whatever pairs a neighbour
/// search turns up. It reads no `pairs` block — there is none to read when the
/// list is rebuilt every few steps.
pub fn pair_lj_class2_typed_ctor(
    _style_params: &Params,
    type_params: &[(&str, &Params)],
    frame: &Frame,
) -> Result<Box<dyn Potential>, String> {
    let type_map: HashMap<&str, &Params> = type_params.iter().copied().collect();
    let (type_id, labels) = atom_type_index(frame)?;
    let ntypes = labels.len();
    let mut epsilon = vec![0.0 as F; ntypes * ntypes];
    let mut sigma = vec![0.0 as F; ntypes * ntypes];
    for ti in 0..ntypes {
        for tj in 0..ntypes {
            // A cross-pair may be declared either way round; a self-pair is
            // named by the atom type alone.
            let forward = pair_type_name(&labels[ti], &labels[tj]);
            let reverse = pair_type_name(&labels[tj], &labels[ti]);
            let p = type_map
                .get(forward.as_str())
                .or_else(|| type_map.get(reverse.as_str()))
                .ok_or_else(|| format!("PairLJClass2: unknown pair type '{forward}'"))?;
            let t = type_pair(ti as u32, tj as u32, ntypes);
            epsilon[t] = p
                .get("epsilon")
                .ok_or_else(|| format!("PairLJClass2 type '{forward}': missing 'epsilon'"))?
                as F;
            sigma[t] = p
                .get("sigma")
                .ok_or_else(|| format!("PairLJClass2 type '{forward}': missing 'sigma'"))?
                as F;
        }
    }
    Ok(Box::new(PairLJClass2::typed(
        type_id, ntypes, epsilon, sigma,
    )))
}

#[cfg(test)]
mod tests {

    /// A type-pair table gives the same number as a per-row label resolved
    /// earlier against a fixed list — bit for bit, on the same pairs.
    ///
    /// The row label belongs to a pair list a neighbour engine rebuilds from
    /// scratch; the atoms' types survive it. That is the whole difference.
    #[test]
    fn a_type_table_scores_a_pair_exactly_as_compiled_rows() {
        use crate::ff::potential::pair::testing::{assert_same, table_over};

        let ntypes = 2_usize;
        let type_id = vec![0_u32, 1, 0, 1];
        let epsilon: Vec<F> = vec![0.12, 0.09, 0.09, 0.17];
        let sigma: Vec<F> = vec![3.4, 3.1, 3.1, 2.9];
        let coords: Vec<F> = vec![
            0.0, 0.0, 0.0, //
            2.1, 0.4, 0.2, //
            1.2, 1.9, 0.7, //
            3.0, 2.3, 1.1,
        ];
        let links = [(0_usize, 1_usize), (0, 2), (1, 3), (2, 3)];

        let (ai, aj): (Vec<usize>, Vec<usize>) = links.iter().copied().unzip();
        let epsilon_c: Vec<F> = links
            .iter()
            .map(|&(i, j)| epsilon[type_pair(type_id[i], type_id[j], ntypes)])
            .collect();
        let sigma_c: Vec<F> = links
            .iter()
            .map(|&(i, j)| sigma[type_pair(type_id[i], type_id[j], ntypes)])
            .collect();
        let compiled = PairLJClass2::new(ai, aj, epsilon_c, sigma_c);
        let typed = PairLJClass2::typed(type_id, ntypes, epsilon, sigma);

        let table = table_over(&coords, &links);
        assert_same(
            "lj/class2",
            compiled.calc_energy_forces(&coords),
            typed.calc_energy_forces_with_pairs(&coords, &table),
        );
    }
    use super::*;

    fn numerical_forces(pot: &PairLJClass2, coords: &[F]) -> Vec<F> {
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
    fn energy_at_sigma_is_negative_eps() {
        // At r = sigma: E = eps(2 - 3) = -eps.
        let pot = PairLJClass2::new(vec![0], vec![1], vec![0.5], vec![1.0]);
        let coords: Vec<F> = vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0];
        assert!((pot.calc_energy(&coords) - (-0.5)).abs() < 1e-12);
    }

    #[test]
    fn force_vanishes_at_minimum() {
        // Minimum of 2u^9 - 3u^6 is at u=1 (r=sigma); force is zero there.
        let pot = PairLJClass2::new(vec![0], vec![1], vec![0.5], vec![1.0]);
        let coords: Vec<F> = vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0];
        let (_, f) = pot.calc_energy_forces(&coords);
        for fi in f {
            assert!(fi.abs() < 1e-9, "force {fi}");
        }
    }

    #[test]
    fn forces_match_finite_difference() {
        let pot = PairLJClass2::new(vec![0], vec![1], vec![0.5], vec![1.0]);
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
        let pot = PairLJClass2::new(vec![0], vec![1], vec![0.5], vec![1.0]);
        let coords: Vec<F> = vec![0.0, 0.0, 0.0, 1.2, 0.7, -0.4];
        let (_, f) = pot.calc_energy_forces(&coords);
        for dim in 0..3 {
            assert!((f[dim] + f[3 + dim]).abs() < 1e-9, "dim {dim}");
        }
    }
}
