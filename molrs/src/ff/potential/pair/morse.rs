//! Morse pair potential: E = D0 * ((1 - exp(-alpha*(r-r0)))^2 - 1)
//!
//! Morse non-bonded form (note the `-1` offset vs the Morse bond, so the well
//! minimum is `-D0` at `r = r0`). Parameters per pair type: `D0`, `alpha`, `r0`.

use std::collections::HashMap;

use crate::ff::forcefield::Params;
use crate::ff::forcefield::pair_type_name;
use crate::ff::potential::Potential;
use crate::ff::potential::gather_copies;
use crate::ff::potential::geometry::validate_coords;
use crate::ff::potential::pair::atom_type_index;
use crate::ff::potential::pair::energy_forces;
use crate::ff::potential::pair::type_pair;
use molrs::math::Virial;
use molrs::spatial::neighbors::Neighbors;
use molrs::store::frame::Frame;
use molrs::types::F;

/// Where a pair's Morse `(D₀, α, r₀)` comes from.
enum Source {
    /// Resolved against one fixed pair list at construction, keyed by the
    /// `type` label on each row.
    Compiled {
        atom_i: Vec<usize>,
        atom_j: Vec<usize>,
        d0: Vec<F>,
        alpha: Vec<F>,
        r0: Vec<F>,
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
        d0: Vec<F>,
        alpha: Vec<F>,
        r0: Vec<F>,
        /// How many of the entries above are atoms; the rest are copies, and
        /// are rebuilt from their owners whenever the copy list is.
        n_owned: usize,
    },
}

pub struct PairMorse {
    source: Source,
}

impl PairMorse {
    pub fn new(
        atom_i: Vec<usize>,
        atom_j: Vec<usize>,
        d0: Vec<F>,
        alpha: Vec<F>,
        r0: Vec<F>,
    ) -> Self {
        let n = atom_i.len();
        assert_eq!(atom_j.len(), n);
        assert_eq!(d0.len(), n);
        assert_eq!(alpha.len(), n);
        assert_eq!(r0.len(), n);
        Self {
            source: Source::Compiled {
                atom_i,
                atom_j,
                d0,
                alpha,
                r0,
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
    pub fn typed(type_id: Vec<u32>, ntypes: usize, d0: Vec<F>, alpha: Vec<F>, r0: Vec<F>) -> Self {
        let n_cells = ntypes * ntypes;
        assert_eq!(n_cells, d0.len(), "d0 must cover every type pair");
        assert_eq!(n_cells, alpha.len(), "alpha must cover every type pair");
        assert_eq!(n_cells, r0.len(), "r0 must cover every type pair");
        debug_assert!(
            type_id.iter().all(|&t| (t as usize) < ntypes),
            "an atom has a type with no parameters"
        );
        let n_owned = type_id.len();
        Self {
            source: Source::Typed {
                type_id,
                ntypes,
                d0,
                alpha,
                r0,
                n_owned,
            },
        }
    }

    /// The pair term for one already-reduced separation.
    fn pair_kernel(&self, r2: F, disp: [F; 3], d0: F, alpha: F, r0: F) -> Option<(F, [F; 3])> {
        if r2 < 1e-24 {
            return None;
        }
        let r = r2.sqrt();
        let y = (-alpha * (r - r0)).exp();
        let one_my = 1.0 - y;
        let energy = d0 * (one_my * one_my - 1.0);

        // d/dr[(1-y)^2 - 1] = 2 alpha y (1 - y)  -> dE/dr = 2 D0 alpha y (1-y)
        let dedr = 2.0 * d0 * alpha * y * one_my;
        let factor = -dedr / r;
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
        pair: impl Fn(usize) -> (usize, usize, (F, F, F), [F; 3], F),
    ) -> (F, Vec<F>, Virial) {
        let mut energy: F = 0.0;
        let mut forces = vec![0.0; n_components];
        let mut virial = Virial::ZERO;
        for idx in 0..n_pairs {
            let (i, j, (d0, alpha, r0), disp, r2) = pair(idx);
            let Some((e, f)) = self.pair_kernel(r2, disp, d0, alpha, r0) else {
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
}

impl Potential for PairMorse {
    fn calc_energy_forces(&self, coords: &[F]) -> (F, Vec<F>) {
        let n_atoms = validate_coords(coords);
        let Source::Compiled {
            atom_i,
            atom_j,
            d0,
            alpha,
            r0,
        } = &self.source
        else {
            // A type table needs a pair table, and nobody handed one over.
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
            (i, j, (d0[idx], alpha[idx], r0[idx]), d, r2)
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
        let Source::Typed {
            type_id,
            ntypes,
            d0,
            alpha,
            r0,
            ..
        } = &self.source
        else {
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
                i < type_id.len() && j < type_id.len(),
                "a pair names an atom the type table does not cover"
            );
            let t = type_pair(type_id[i], type_id[j], *ntypes);
            (
                i,
                j,
                (d0[t], alpha[t], r0[t]),
                [disp[[p, 0]], disp[[p, 1]], disp[[p, 2]]],
                d2[p],
            )
        });
        (e, f, Some(w))
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

/// Construct a [`PairMorse`] from style params, type params, and Frame topology.
pub fn pair_morse_ctor(
    _style_params: &Params,
    type_params: &[(&str, &Params)],
    frame: &Frame,
) -> Result<Box<dyn Potential>, String> {
    let type_map: HashMap<&str, &Params> = type_params.iter().copied().collect();

    let block = frame
        .get("pairs")
        .ok_or_else(|| "PairMorse: frame missing \"pairs\" block".to_string())?;
    let i_col = block
        .get_uint("atomi")
        .ok_or_else(|| "PairMorse: pairs block missing \"atomi\" column".to_string())?;
    let j_col = block
        .get_uint("atomj")
        .ok_or_else(|| "PairMorse: pairs block missing \"atomj\" column".to_string())?;
    let type_col = block
        .get_string("type")
        .ok_or_else(|| "PairMorse: pairs block missing \"type\" column".to_string())?;

    let (mut ai, mut aj) = (Vec::new(), Vec::new());
    let (mut dv, mut av, mut rv) = (Vec::new(), Vec::new(), Vec::new());
    let need = |p: &Params, key: &str, label: &str| -> Result<F, String> {
        p.get(key)
            .ok_or_else(|| format!("PairMorse type '{}': missing '{}'", label, key))
            .map(|v| v as F)
    };
    for idx in 0..i_col.len() {
        let label = &type_col[idx];
        let p = type_map
            .get(label.as_str())
            .ok_or_else(|| format!("PairMorse: unknown pair type '{}'", label))?;
        ai.push(i_col[idx] as usize);
        aj.push(j_col[idx] as usize);
        dv.push(need(p, "D0", label)?);
        av.push(need(p, "alpha", label)?);
        rv.push(need(p, "r0", label)?);
    }

    Ok(Box::new(PairMorse::new(ai, aj, dv, av, rv)))
}

/// Construct a neighbour-driven [`PairMorse`] from per-atom parameters.
///
/// The counterpart of [`pair_morse_ctor`]: the same force field, keyed on the atoms
/// instead of on a pair list, so it can answer for whatever pairs a neighbour
/// search turns up. It reads no `pairs` block — there is none to read when the
/// list is rebuilt every few steps.
pub fn pair_morse_typed_ctor(
    _style_params: &Params,
    type_params: &[(&str, &Params)],
    frame: &Frame,
) -> Result<Box<dyn Potential>, String> {
    let type_map: HashMap<&str, &Params> = type_params.iter().copied().collect();
    let (type_id, labels) = atom_type_index(frame)?;
    let ntypes = labels.len();
    let mut d0 = vec![0.0 as F; ntypes * ntypes];
    let mut alpha = vec![0.0 as F; ntypes * ntypes];
    let mut r0 = vec![0.0 as F; ntypes * ntypes];
    for ti in 0..ntypes {
        for tj in 0..ntypes {
            // A cross-pair may be declared either way round; a self-pair is
            // named by the atom type alone.
            let forward = pair_type_name(&labels[ti], &labels[tj]);
            let reverse = pair_type_name(&labels[tj], &labels[ti]);
            let p = type_map
                .get(forward.as_str())
                .or_else(|| type_map.get(reverse.as_str()))
                .ok_or_else(|| format!("PairMorse: unknown pair type '{forward}'"))?;
            let t = type_pair(ti as u32, tj as u32, ntypes);
            d0[t] = p
                .get("d0")
                .ok_or_else(|| format!("PairMorse type '{forward}': missing 'd0'"))?
                as F;
            alpha[t] = p
                .get("alpha")
                .ok_or_else(|| format!("PairMorse type '{forward}': missing 'alpha'"))?
                as F;
            r0[t] = p
                .get("r0")
                .ok_or_else(|| format!("PairMorse type '{forward}': missing 'r0'"))?
                as F;
        }
    }
    Ok(Box::new(PairMorse::typed(type_id, ntypes, d0, alpha, r0)))
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
        use crate::ff::potential::pair::testing::{
            assert_same, assert_virial_matches_forces, table_over,
        };

        let ntypes = 2_usize;
        let type_id = vec![0_u32, 1, 0, 1];
        let d0: Vec<F> = vec![0.25, 0.18, 0.18, 0.31];
        let alpha: Vec<F> = vec![1.9, 2.1, 2.1, 1.7];
        let r0: Vec<F> = vec![3.4, 3.1, 3.1, 2.9];
        let coords: Vec<F> = vec![
            0.0, 0.0, 0.0, //
            2.1, 0.4, 0.2, //
            1.2, 1.9, 0.7, //
            3.0, 2.3, 1.1,
        ];
        let links = [(0_usize, 1_usize), (0, 2), (1, 3), (2, 3)];

        let (ai, aj): (Vec<usize>, Vec<usize>) = links.iter().copied().unzip();
        let d0_c: Vec<F> = links
            .iter()
            .map(|&(i, j)| d0[type_pair(type_id[i], type_id[j], ntypes)])
            .collect();
        let alpha_c: Vec<F> = links
            .iter()
            .map(|&(i, j)| alpha[type_pair(type_id[i], type_id[j], ntypes)])
            .collect();
        let r0_c: Vec<F> = links
            .iter()
            .map(|&(i, j)| r0[type_pair(type_id[i], type_id[j], ntypes)])
            .collect();
        let compiled = PairMorse::new(ai, aj, d0_c, alpha_c, r0_c);
        let typed = PairMorse::typed(type_id, ntypes, d0, alpha, r0);

        let table = table_over(&coords, &links);
        assert_same(
            "morse",
            compiled.calc_energy_forces(&coords),
            typed.calc_energy_forces_with_pairs(&coords, &table),
        );
        assert_virial_matches_forces(
            "morse",
            &coords,
            typed.calc_energy_forces_with_pairs_virial(&coords, &table),
        );
    }
    use super::*;

    fn numerical_forces(pot: &PairMorse, coords: &[F]) -> Vec<F> {
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
    fn well_minimum_is_minus_d0() {
        // At r = r0: E = D0 (0 - 1) = -D0, force zero.
        let pot = PairMorse::new(vec![0], vec![1], vec![5.0], vec![1.5], vec![3.0]);
        let coords: Vec<F> = vec![0.0, 0.0, 0.0, 3.0, 0.0, 0.0];
        let (e, f) = pot.calc_energy_forces(&coords);
        assert!((e - (-5.0)).abs() < 1e-12, "energy {e}");
        for fi in f {
            assert!(fi.abs() < 1e-9);
        }
    }

    #[test]
    fn forces_match_finite_difference() {
        let pot = PairMorse::new(vec![0], vec![1], vec![5.0], vec![1.5], vec![3.0]);
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
