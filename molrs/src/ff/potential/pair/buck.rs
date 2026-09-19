//! Buckingham pair potential: E = A * exp(-r/rho) - C / r^6
//!
//! The exp-6 form used for repulsion/dispersion (e.g. CL&Pol non-bonded cores).
//! Parameters per pair type: `a` (energy), `rho` (length), `c` (energy*length^6).
//! Lowercase is the canonical spelling (spec ff-params-01) and matches molpy;
//! GROMACS spells the middle one `B = 1/rho`, normalized at that reader.

use std::collections::HashMap;

use crate::ff::forcefield::Params;
use crate::ff::forcefield::pair_type_name;
use crate::ff::potential::Potential;
use crate::ff::potential::gather_copies;
use crate::ff::potential::geometry::validate_coords;
use crate::ff::potential::pair::atom_type_index;
use crate::ff::potential::pair::energy_forces;
use crate::ff::potential::pair::fold_chunks;
use crate::ff::potential::pair::type_pair;
use molrs::math::Virial;
use molrs::spatial::neighbors::Neighbors;
use molrs::store::frame::Frame;
use molrs::types::F;

/// Where a pair's Buckingham `(A, ρ, C)` comes from.
enum Source {
    /// Resolved against one fixed pair list at construction, keyed by the
    /// `type` label on each row.
    Compiled {
        atom_i: Vec<usize>,
        atom_j: Vec<usize>,
        a: Vec<F>,
        rho: Vec<F>,
        c: Vec<F>,
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
        a: Vec<F>,
        rho: Vec<F>,
        c: Vec<F>,
        /// How many of the entries above are atoms; the rest are copies, and
        /// are rebuilt from their owners whenever the copy list is.
        n_owned: usize,
    },
}

pub struct PairBuck {
    source: Source,
}

impl PairBuck {
    pub fn new(atom_i: Vec<usize>, atom_j: Vec<usize>, a: Vec<F>, rho: Vec<F>, c: Vec<F>) -> Self {
        let n = atom_i.len();
        assert_eq!(atom_j.len(), n);
        assert_eq!(a.len(), n);
        assert_eq!(rho.len(), n);
        assert_eq!(c.len(), n);
        Self {
            source: Source::Compiled {
                atom_i,
                atom_j,
                a,
                rho,
                c,
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
    pub fn typed(type_id: Vec<u32>, ntypes: usize, a: Vec<F>, rho: Vec<F>, c: Vec<F>) -> Self {
        let n_cells = ntypes * ntypes;
        assert_eq!(n_cells, a.len(), "a must cover every type pair");
        assert_eq!(n_cells, rho.len(), "rho must cover every type pair");
        assert_eq!(n_cells, c.len(), "c must cover every type pair");
        debug_assert!(
            type_id.iter().all(|&t| (t as usize) < ntypes),
            "an atom has a type with no parameters"
        );
        let n_owned = type_id.len();
        Self {
            source: Source::Typed {
                type_id,
                ntypes,
                a,
                rho,
                c,
                n_owned,
            },
        }
    }

    /// The pair term for one already-reduced separation.
    fn pair_kernel(&self, r2: F, disp: [F; 3], a: F, rho: F, c: F) -> Option<(F, [F; 3])> {
        if r2 < 1e-24 {
            return None;
        }
        // A zero `rho` is what an unparameterised entry of a type-pair table
        // looks like. `A·exp(-r/0)` is zero and `0/(0·r)` is NaN, so the pair
        // would contribute no energy and an undefined force — the worst of the
        // two possible failures. It contributes nothing instead.
        if rho <= 0.0 {
            return None;
        }
        let r = r2.sqrt();
        let exp_term = a * (-r / rho).exp();
        let r6 = r2 * r2 * r2;
        let energy = exp_term - c / r6;

        // E = A exp(-r/rho) - C r^-6
        // dE/dr = -(A/rho) exp(-r/rho) + 6 C r^-7
        // factor = -(1/r) dE/dr = (A/(rho r)) exp(-r/rho) - 6 C r^-8
        let factor = exp_term / (rho * r) - 6.0 * c / (r6 * r2);
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
        pair: impl Fn(usize) -> (usize, usize, (F, F, F), [F; 3], F) + Sync,
    ) -> (F, Vec<F>, Virial) {
        let mut forces = vec![0.0; n_components];
        let (energy, virial) = self.fold_into(&mut forces, &[], n_pairs, pair);
        (energy, forces, virial)
    }

    /// The accumulation, adding into the caller's buffer and scaling each pair.
    fn fold_into(
        &self,
        out: &mut [F],
        factor: &[F],
        n_pairs: usize,
        pair: impl Fn(usize) -> (usize, usize, (F, F, F), [F; 3], F) + Sync,
    ) -> (F, Virial) {
        fold_chunks(out, n_pairs, |acc, rows| {
            self.fold_rows(acc, factor, rows, &pair)
        })
    }

    /// One contiguous range of pairs, into `out`. The whole fold when it runs
    /// serially; one chunk of it when it does not.
    fn fold_rows(
        &self,
        out: &mut [F],
        factor: &[F],
        rows: std::ops::Range<usize>,
        pair: &(impl Fn(usize) -> (usize, usize, (F, F, F), [F; 3], F) + Sync),
    ) -> (F, Virial) {
        let mut energy: F = 0.0;
        let mut virial = Virial::ZERO;
        for idx in rows {
            let w = if factor.is_empty() { 1.0 } else { factor[idx] };
            // Exactly zero *skips*: a bonded pair sits at bond length,
            // where a repulsive term is enormous, and scaling it by zero
            // would be arithmetic on a number that should never have been
            // computed.
            if w == 0.0 {
                continue;
            }
            let (i, j, (a, rho, c), disp, r2) = pair(idx);
            let Some((e, f)) = self.pair_kernel(r2, disp, a, rho, c) else {
                continue;
            };
            let f = [w * f[0], w * f[1], w * f[2]];
            energy += w * e;
            virial.add_outer(f, disp);
            out[j * 3] += f[0];
            out[j * 3 + 1] += f[1];
            out[j * 3 + 2] += f[2];
            out[i * 3] -= f[0];
            out[i * 3 + 1] -= f[1];
            out[i * 3 + 2] -= f[2];
        }
        (energy, virial)
    }
}

impl Potential for PairBuck {
    fn calc_energy_forces(&self, coords: &[F]) -> (F, Vec<F>) {
        let n_atoms = validate_coords(coords);
        let Source::Compiled {
            atom_i,
            atom_j,
            a,
            rho,
            c,
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
            (i, j, (a[idx], rho[idx], c[idx]), d, r2)
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
        let mut forces = vec![0.0; coords.len()];
        let (e, w) = self.accumulate_pairs(coords, pairs, &[], &mut forces);
        (e, forces, w)
    }

    fn accumulate_pairs(
        &self,
        coords: &[F],
        pairs: &Neighbors,
        factor: &[F],
        out: &mut [F],
    ) -> (F, Option<Virial>) {
        let Source::Typed {
            type_id,
            ntypes,
            a,
            rho,
            c,
            ..
        } = &self.source
        else {
            // A compiled kernel answers for its own list, not for this one.
            // A compiled kernel cannot read the table, so it cannot read a
            // per-pair weight either. Both providers refuse one, so this is
            // the free-boundary path and `factor` is empty.
            debug_assert!(factor.is_empty());
            let (e, f) = self.calc_energy_forces(coords);
            for (acc, v) in out.iter_mut().zip(&f) {
                *acc += v;
            }
            return (e, None);
        };
        let (Some(disp), Some(d2)) = (pairs.disp(), pairs.dist_sq()) else {
            return (0.0, None);
        };
        let i_col = pairs.query_point_indices();
        let j_col = pairs.point_indices();
        let (e, w) = self.fold_into(out, factor, i_col.len(), |p| {
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
                (a[t], rho[t], c[t]),
                [disp[[p, 0]], disp[[p, 1]], disp[[p, 2]]],
                d2[p],
            )
        });
        (e, Some(w))
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

    fn binds_a_fixed_pair_list(&self) -> bool {
        matches!(self.source, Source::Compiled { .. })
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

/// Construct a neighbour-driven [`PairBuck`] from per-atom parameters.
///
/// The counterpart of [`pair_buck_ctor`]: the same force field, keyed on the atoms
/// instead of on a pair list, so it can answer for whatever pairs a neighbour
/// search turns up. It reads no `pairs` block — there is none to read when the
/// list is rebuilt every few steps.
pub fn pair_buck_typed_ctor(
    _style_params: &Params,
    type_params: &[(&str, &Params)],
    frame: &Frame,
) -> Result<Box<dyn Potential>, String> {
    let type_map: HashMap<&str, &Params> = type_params.iter().copied().collect();
    let (type_id, labels) = atom_type_index(frame)?;
    let ntypes = labels.len();
    let mut a = vec![0.0 as F; ntypes * ntypes];
    let mut rho = vec![0.0 as F; ntypes * ntypes];
    let mut c = vec![0.0 as F; ntypes * ntypes];
    for ti in 0..ntypes {
        for tj in 0..ntypes {
            // A cross-pair may be declared either way round; a self-pair is
            // named by the atom type alone.
            let forward = pair_type_name(&labels[ti], &labels[tj]);
            let reverse = pair_type_name(&labels[tj], &labels[ti]);
            let p = type_map
                .get(forward.as_str())
                .or_else(|| type_map.get(reverse.as_str()))
                .ok_or_else(|| format!("PairBuck: unknown pair type '{forward}'"))?;
            let t = type_pair(ti as u32, tj as u32, ntypes);
            a[t] = p
                .get("a")
                .ok_or_else(|| format!("PairBuck type '{forward}': missing 'a'"))?
                as F;
            rho[t] = p
                .get("rho")
                .ok_or_else(|| format!("PairBuck type '{forward}': missing 'rho'"))?
                as F;
            c[t] = p
                .get("c")
                .ok_or_else(|| format!("PairBuck type '{forward}': missing 'c'"))?
                as F;
        }
    }
    Ok(Box::new(PairBuck::typed(type_id, ntypes, a, rho, c)))
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
        let a: Vec<F> = vec![12000.0, 9000.0, 9000.0, 7000.0];
        let rho: Vec<F> = vec![0.31, 0.29, 0.29, 0.27];
        let c: Vec<F> = vec![280.0, 190.0, 190.0, 140.0];
        let coords: Vec<F> = vec![
            0.0, 0.0, 0.0, //
            2.1, 0.4, 0.2, //
            1.2, 1.9, 0.7, //
            3.0, 2.3, 1.1,
        ];
        let links = [(0_usize, 1_usize), (0, 2), (1, 3), (2, 3)];

        let (ai, aj): (Vec<usize>, Vec<usize>) = links.iter().copied().unzip();
        let a_c: Vec<F> = links
            .iter()
            .map(|&(i, j)| a[type_pair(type_id[i], type_id[j], ntypes)])
            .collect();
        let rho_c: Vec<F> = links
            .iter()
            .map(|&(i, j)| rho[type_pair(type_id[i], type_id[j], ntypes)])
            .collect();
        let c_c: Vec<F> = links
            .iter()
            .map(|&(i, j)| c[type_pair(type_id[i], type_id[j], ntypes)])
            .collect();
        let compiled = PairBuck::new(ai, aj, a_c, rho_c, c_c);
        let typed = PairBuck::typed(type_id, ntypes, a, rho, c);

        let table = table_over(&coords, &links);
        assert_same(
            "buck",
            compiled.calc_energy_forces(&coords),
            typed.calc_energy_forces_with_pairs(&coords, &table),
        );
        assert_virial_matches_forces(
            "buck",
            &coords,
            typed.calc_energy_forces_with_pairs_virial(&coords, &table),
        );
    }
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
