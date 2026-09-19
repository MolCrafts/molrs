//! UFF van der Waals: E = D · [(x/r)¹² − 2·(x/r)⁶] (RDKit `vdWContrib`).
//!
//! Per-instance columns on the `pairs` block: `xij`, `Dij` (baked by the
//! typifier after the neighbour list is built — but the typifier only sets
//! atom `x1`/`D1`; this ctor combines them geometrically like RDKit).

use crate::ff::forcefield::Params;
use crate::ff::potential::Potential;
use crate::ff::potential::gather_copies;
use crate::ff::potential::geometry::validate_coords;
use crate::ff::potential::pair::energy_forces;
use molrs::math::Virial;
use molrs::spatial::neighbors::Neighbors;
use molrs::store::frame::Frame;
use molrs::types::F;

/// Where a pair's `(xᵢⱼ, Dᵢⱼ)` comes from.
enum Source {
    /// Combined against one fixed pair list at construction.
    Compiled {
        atom_i: Vec<usize>,
        atom_j: Vec<usize>,
        xij: Vec<F>,
        dij: Vec<F>,
    },
    /// Per-atom `x1`/`D1`, combined when a pair turns up.
    ///
    /// What a neighbour-driven evaluation needs: a neighbour table is a
    /// different list of pairs every rebuild, so parameters combined against
    /// an older one belong to different atoms. Under a ghost régime the
    /// vectors cover the copies too, each carrying its owner's values.
    PerAtom {
        x1: Vec<F>,
        d1: Vec<F>,
        /// How many of the entries above are atoms; the rest are copies, and
        /// are rebuilt from their owners whenever the copy list is.
        n_owned: usize,
    },
}

pub struct UffVdW {
    source: Source,
}

impl UffVdW {
    /// Parameters combined against a fixed pair list.
    pub fn compiled(atom_i: Vec<usize>, atom_j: Vec<usize>, xij: Vec<F>, dij: Vec<F>) -> Self {
        assert_eq!(atom_i.len(), atom_j.len());
        assert_eq!(atom_i.len(), xij.len());
        assert_eq!(atom_i.len(), dij.len());
        Self {
            source: Source::Compiled {
                atom_i,
                atom_j,
                xij,
                dij,
            },
        }
    }

    /// Per-atom `x1`/`D1`, combined geometrically when a pair turns up — the
    /// same rule [`uff_lj_ctor`] applies, and RDKit with it.
    pub fn typed(x1: Vec<F>, d1: Vec<F>) -> Self {
        assert_eq!(x1.len(), d1.len());
        let n_owned = x1.len();
        Self {
            source: Source::PerAtom { x1, d1, n_owned },
        }
    }

    /// The pair term for one already-reduced separation.
    ///
    /// Returns the force **on `i`** — UFF's sign convention, kept rather than
    /// normalised so this stays a transcription of RDKit's `vdWContrib`.
    fn pair_kernel(&self, r2: F, disp: [F; 3], xij: F, dij: F) -> Option<(F, [F; 3])> {
        if r2 < 1e-24 as F {
            return None;
        }
        let r = r2.sqrt();
        // cutoff at 2 * xij (RDKit default threshMultiplier ≈ 2 when used that way)
        if r > 2.0 * xij {
            return None;
        }
        let inv_r = 1.0 / r;
        let rr = xij * inv_r;
        let r2u = rr * rr;
        let r6 = r2u * r2u * r2u;
        let r12 = r6 * r6;
        let energy = dij * (r12 - 2.0 * r6);

        // RDKit: preFactor = 12·D/x · ((x/r)⁷ − (x/r)¹³);
        // dGrad on atom1 (i) = preFactor · (i−j)/r  (gradient contrib);
        // force = −gradient, so F_i = −preFactor · (i−j)/r = preFactor · (j−i)/r
        let r7 = r6 * rr;
        let r13 = r12 * rr;
        let pref = 12.0 * dij / xij * (r7 - r13);
        let factor = pref / r;
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
        pair: impl Fn(usize) -> (usize, usize, F, F, [F; 3], F),
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
        pair: impl Fn(usize) -> (usize, usize, F, F, [F; 3], F),
    ) -> (F, Virial) {
        let mut energy = 0.0 as F;
        let mut virial = Virial::ZERO;
        for idx in 0..n_pairs {
            let w = if factor.is_empty() { 1.0 } else { factor[idx] };
            // Exactly zero *skips*: a bonded pair sits at bond length,
            // where a repulsive term is enormous, and scaling it by zero
            // would be arithmetic on a number that should never have been
            // computed.
            if w == 0.0 {
                continue;
            }
            let (i, j, xij, dij, disp, r2) = pair(idx);
            let Some((e, f)) = self.pair_kernel(r2, disp, xij, dij) else {
                continue;
            };
            let f = [w * f[0], w * f[1], w * f[2]];
            energy += w * e;
            virial.add_outer(f, [-disp[0], -disp[1], -disp[2]]);
            out[i * 3] += f[0];
            out[i * 3 + 1] += f[1];
            out[i * 3 + 2] += f[2];
            out[j * 3] -= f[0];
            out[j * 3 + 1] -= f[1];
            out[j * 3 + 2] -= f[2];
        }
        (energy, virial)
    }
}

impl Potential for UffVdW {
    fn calc_energy_forces(&self, coords: &[F]) -> (F, Vec<F>) {
        let _n = validate_coords(coords);
        let Source::Compiled {
            atom_i,
            atom_j,
            xij,
            dij,
        } = &self.source
        else {
            // Per-atom parameters need a pair table, and nobody handed one over.
            return (0.0, vec![0.0 as F; coords.len()]);
        };
        energy_forces(self.fold(coords.len(), atom_i.len(), |idx| {
            let (i, j) = (atom_i[idx], atom_j[idx]);
            let d = [
                coords[j * 3] - coords[i * 3],
                coords[j * 3 + 1] - coords[i * 3 + 1],
                coords[j * 3 + 2] - coords[i * 3 + 2],
            ];
            let r2 = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
            (i, j, xij[idx], dij[idx], d, r2)
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
        let Source::PerAtom { x1, d1, .. } = &self.source else {
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
                i < x1.len() && j < x1.len(),
                "a pair names an atom the per-atom parameters do not cover"
            );
            (
                i,
                j,
                ((x1[i] * x1[j]) as F).sqrt(),
                ((d1[i] * d1[j]) as F).sqrt(),
                [disp[[p, 0]], disp[[p, 1]], disp[[p, 2]]],
                d2[p],
            )
        });
        (e, Some(w))
    }

    fn gather_onto_copies(&mut self, owner: &[u32]) {
        let Source::PerAtom {
            x1, d1, n_owned, ..
        } = &mut self.source
        else {
            // Nothing per atom to extend.
            return;
        };
        gather_copies(x1, *n_owned, owner);
        gather_copies(d1, *n_owned, owner);
    }

    fn binds_a_fixed_pair_list(&self) -> bool {
        matches!(self.source, Source::Compiled { .. })
    }
}

pub fn uff_lj_ctor(
    _sp: &Params,
    _tp: &[(&str, &Params)],
    frame: &Frame,
) -> Result<Box<dyn Potential>, String> {
    let atoms = frame.get("atoms").ok_or("uff_lj: missing atoms")?;
    let x1 = atoms.get_float("x1").ok_or("uff_lj: missing atoms.x1")?;
    let d1 = atoms.get_float("D1").ok_or("uff_lj: missing atoms.D1")?;

    let pairs = frame
        .get("pairs")
        .ok_or("uff_lj: missing pairs (call intramolecular_pairs first)")?;
    if pairs.nrows().unwrap_or(0) == 0 {
        return Ok(Box::new(UffVdW::compiled(vec![], vec![], vec![], vec![])));
    }
    let pi = pairs
        .get_uint("atomi")
        .ok_or("uff_lj: pairs missing atomi")?;
    let pj = pairs
        .get_uint("atomj")
        .ok_or("uff_lj: pairs missing atomj")?;
    let n = pi.len();
    let mut atom_i = Vec::with_capacity(n);
    let mut atom_j = Vec::with_capacity(n);
    let mut xij = Vec::with_capacity(n);
    let mut dij = Vec::with_capacity(n);
    for t in 0..n {
        let i = pi[t] as usize;
        let j = pj[t] as usize;
        atom_i.push(i);
        atom_j.push(j);
        xij.push(((x1[i] * x1[j]) as F).sqrt());
        dij.push(((d1[i] * d1[j]) as F).sqrt());
    }
    Ok(Box::new(UffVdW::compiled(atom_i, atom_j, xij, dij)))
}

/// Construct a neighbour-driven [`UffVdW`] from per-atom parameters.
///
/// The counterpart of [`uff_lj_ctor`]: the same force field, keyed on the atoms
/// instead of on a pair list, so it can answer for whatever pairs a neighbour
/// search turns up. It reads no `pairs` block — there is none to read when the
/// list is rebuilt every few steps.
pub fn uff_lj_typed_ctor(
    _style_params: &Params,
    _type_params: &[(&str, &Params)],
    frame: &Frame,
) -> Result<Box<dyn Potential>, String> {
    let atoms = frame.get("atoms").ok_or("uff_lj: missing atoms")?;
    let x1 = atoms.get_float("x1").ok_or("uff_lj: missing atoms.x1")?;
    let d1 = atoms.get_float("D1").ok_or("uff_lj: missing atoms.D1")?;
    Ok(Box::new(UffVdW::typed(
        x1.iter().map(|&v| v as F).collect(),
        d1.iter().map(|&v| v as F).collect(),
    )))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ff::potential::pair::testing::{
        assert_same, assert_virial_matches_forces, table_over,
    };

    /// Combining `x1`/`D1` when a pair turns up is the same number as having
    /// combined them earlier against a fixed list — bit for bit, on the same
    /// pairs.
    ///
    /// The difference is only *when*, and it matters because a neighbour table
    /// is a different list of pairs every rebuild: parameters combined against
    /// an older one belong to different atoms, silently.
    #[test]
    fn per_atom_parameters_score_a_pair_exactly_as_compiled_ones() {
        let x1 = vec![3.5_f64, 3.1, 3.8, 2.9];
        let d1 = vec![0.06_f64, 0.09, 0.05, 0.12];
        let coords: Vec<F> = vec![
            0.0, 0.0, 0.0, //
            2.1, 0.4, 0.2, //
            1.2, 1.9, 0.7, //
            3.0, 2.3, 1.1,
        ];
        let links = [(0_usize, 1_usize), (0, 2), (1, 3), (2, 3)];

        let (ai, aj): (Vec<usize>, Vec<usize>) = links.iter().copied().unzip();
        let xij: Vec<F> = links
            .iter()
            .map(|&(i, j)| ((x1[i] * x1[j]) as F).sqrt())
            .collect();
        let dij: Vec<F> = links
            .iter()
            .map(|&(i, j)| ((d1[i] * d1[j]) as F).sqrt())
            .collect();
        let compiled = UffVdW::compiled(ai, aj, xij, dij);
        let typed = UffVdW::typed(x1, d1);

        let table = table_over(&coords, &links);
        assert_same(
            "uff_lj",
            compiled.calc_energy_forces(&coords),
            typed.calc_energy_forces_with_pairs(&coords, &table),
        );
        assert_virial_matches_forces(
            "uff_lj",
            &coords,
            typed.calc_energy_forces_with_pairs_virial(&coords, &table),
        );
    }
}
