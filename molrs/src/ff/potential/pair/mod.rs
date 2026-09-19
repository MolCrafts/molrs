//! Pair potential kernels.

use ndarray::{Array2, ArrayView2};

use molrs::spatial::neighbors::Neighbors;
use molrs::store::frame::Frame;
use molrs::types::F;

/// Pair kernel: already-reduced geometry in, energy / force on `j` out.
pub trait PairPotential: Send + Sync {
    fn pair_energy(&self, r2: F, disp: [F; 3]) -> Option<F>;
    fn pair_force(&self, r2: F, disp: [F; 3]) -> Option<[F; 3]>;
    fn pair_eval(&self, r2: F, disp: [F; 3]) -> Option<(F, [F; 3])> {
        match (self.pair_energy(r2, disp), self.pair_force(r2, disp)) {
            (Some(e), Some(f)) => Some((e, f)),
            _ => None,
        }
    }

    fn eval_pairs(
        &self,
        n_atoms: usize,
        i: &[u32],
        j: &[u32],
        disp: ArrayView2<'_, F>,
        dist_sq: Option<&[F]>,
    ) -> Result<(F, Array2<F>), String> {
        let n_pairs = i.len();
        if j.len() != n_pairs || disp.nrows() != n_pairs || disp.ncols() != 3 {
            return Err(format!(
                "pair columns must share n_pairs={n_pairs}: j={}, disp={:?}",
                j.len(),
                disp.shape()
            ));
        }
        if let Some(d2) = dist_sq
            && d2.len() != n_pairs
        {
            return Err(format!(
                "dist_sq length {} disagrees with n_pairs={n_pairs}",
                d2.len()
            ));
        }
        let mut forces = Array2::<F>::zeros((n_atoms, 3));
        let mut energy = 0.0;
        for p in 0..n_pairs {
            let ia = i[p] as usize;
            let ja = j[p] as usize;
            if ia >= n_atoms || ja >= n_atoms {
                return Err(format!("pair ({ia}, {ja}) is outside n_atoms={n_atoms}"));
            }
            let d = [disp[[p, 0]], disp[[p, 1]], disp[[p, 2]]];
            let r2 = match dist_sq {
                Some(col) => col[p],
                None => d[0] * d[0] + d[1] * d[1] + d[2] * d[2],
            };
            let Some((e, f)) = self.pair_eval(r2, d) else {
                continue;
            };
            energy += e;
            forces[[ja, 0]] += f[0];
            forces[[ja, 1]] += f[1];
            forces[[ja, 2]] += f[2];
            forces[[ia, 0]] -= f[0];
            forces[[ia, 1]] -= f[1];
            forces[[ia, 2]] -= f[2];
        }
        Ok((energy, forces))
    }

    fn eval_table(&self, n_atoms: usize, neighbors: &Neighbors) -> Result<(F, Array2<F>), String> {
        let disp = neighbors
            .disp()
            .ok_or_else(|| "eval_table needs the Neighbors disp column".to_owned())?;
        self.eval_pairs(
            n_atoms,
            neighbors.query_point_indices(),
            neighbors.point_indices(),
            disp,
            neighbors.dist_sq(),
        )
    }
}

/// Drop the virial from a fold's result.
///
/// The accumulation loop always has both terms of a pair in hand, so tallying
/// `Σ f ⊗ r` there is free; the entry points that do not report one say so by
/// discarding it here rather than by keeping a second loop that does not.
#[inline]
pub(crate) fn energy_forces((e, f, _): (F, Vec<F>, molrs::math::Virial)) -> (F, Vec<F>) {
    (e, f)
}

/// Map each atom to a dense type index, and hand back the labels in that order.
///
/// A neighbour-driven kernel keys its parameters on the atoms, so it needs the
/// types as small integers it can index a table with — not as the strings the
/// frame carries. The labels come back so the caller can look each type's
/// parameters up once, rather than once per atom.
pub(crate) fn atom_type_index(frame: &Frame) -> Result<(Vec<u32>, Vec<String>), String> {
    let atoms = frame
        .get("atoms")
        .ok_or_else(|| "typed pair kernel: frame missing \"atoms\" block".to_string())?;
    let types = atoms
        .get_string("type")
        .ok_or_else(|| "typed pair kernel: atoms block missing \"type\" column".to_string())?;
    let mut labels: Vec<String> = Vec::new();
    let mut index = std::collections::HashMap::new();
    let mut type_id = Vec::with_capacity(types.len());
    for t in types.iter() {
        let next = labels.len() as u32;
        let id = *index.entry(t.clone()).or_insert_with(|| {
            labels.push(t.clone());
            next
        });
        type_id.push(id);
    }
    Ok((type_id, labels))
}

/// Index into a type-pair parameter table laid out `ti * ntypes + tj`.
///
/// This is LAMMPS's `pair_coeff i j` model, and it is what a neighbour-driven
/// evaluation needs: which pairs exist is re-decided at every rebuild, so a
/// pair's parameters have to be findable from the atoms it names rather than
/// from the row it used to occupy.
#[inline]
pub(crate) fn type_pair(ti: u32, tj: u32, ntypes: usize) -> usize {
    ti as usize * ntypes + tj as usize
}

pub mod buck;
pub mod coul_cut;
pub mod lj_class2;
pub mod lj_cut;
pub mod mmff;
pub mod morse;
pub mod tang_toennies;
pub mod thole;
pub mod uff;

pub use buck::{PairBuck, pair_buck_ctor};
pub use coul_cut::{PairCoulCut, pair_coul_cut_ctor};
pub use lj_class2::{PairLJClass2, pair_lj_class2_ctor};
pub use lj_cut::{LJCut, Mixing, pair_lj_cut_ctor};
pub use mmff::{MMFFVdW, mmff_vdw_ctor};
pub use morse::{PairMorse, pair_morse_ctor};
pub use tang_toennies::{PairTangToennies, pair_tang_toennies_ctor};
pub use thole::{PairThole, pair_thole_ctor};
pub use uff::{UffVdW, uff_lj_ctor};

#[cfg(test)]
pub(crate) mod testing {
    use molrs::spatial::neighbors::{NeighborPair, Neighbors, NeighborsStorage, QueryMode};
    use molrs::types::F;

    /// A neighbour table over exactly `links`, with the displacements a
    /// neighbour engine would have computed for them.
    ///
    /// Every typed kernel is checked against its compiled twin on the same
    /// pairs, and "the same pairs" has to mean the same arithmetic too: the
    /// compiled path takes plain coordinate differences, so the table must
    /// carry those and not a re-derived approximation of them.
    pub(crate) fn table_over(coords: &[F], links: &[(usize, usize)]) -> Neighbors {
        let n_points = coords.len() / 3;
        let pairs: Vec<NeighborPair> = links
            .iter()
            .map(|&(i, j)| {
                let d = [
                    coords[j * 3] - coords[i * 3],
                    coords[j * 3 + 1] - coords[i * 3 + 1],
                    coords[j * 3 + 2] - coords[i * 3 + 2],
                ];
                NeighborPair {
                    i: i as u32,
                    j: j as u32,
                    dist_sq: d[0] * d[0] + d[1] * d[1] + d[2] * d[2],
                    disp: d,
                }
            })
            .collect();
        Neighbors::from_pairs(
            pairs,
            NeighborsStorage::FULL,
            QueryMode::SelfQuery {
                num_points: n_points,
            },
        )
    }

    /// Assert two evaluations agree bit for bit, and that they said something.
    pub(crate) fn assert_same(label: &str, a: (F, Vec<F>), b: (F, Vec<F>)) {
        let (e_a, f_a) = a;
        let (e_b, f_b) = b;
        assert!(
            e_a.abs() > 1e-9,
            "{label}: the configuration must interact for this to assert anything; got {e_a}"
        );
        assert_eq!(
            e_a.to_bits(),
            e_b.to_bits(),
            "{label}: energy {e_a} vs {e_b}"
        );
        assert_eq!(f_a.len(), f_b.len(), "{label}: force length");
        for (c, (x, y)) in f_a.iter().zip(&f_b).enumerate() {
            assert_eq!(
                x.to_bits(),
                y.to_bits(),
                "{label}: force component {c}: {x} vs {y}"
            );
        }
    }
}
