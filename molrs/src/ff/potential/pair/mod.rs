//! Pair potential kernels.

use ndarray::{Array2, ArrayView2};

use molrs::spatial::neighbors::Neighbors;
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
pub use lj_cut::{LJCut, pair_lj_cut_ctor};
pub use mmff::{MMFFVdW, mmff_vdw_ctor};
pub use morse::{PairMorse, pair_morse_ctor};
pub use tang_toennies::{PairTangToennies, pair_tang_toennies_ctor};
pub use thole::{PairThole, pair_thole_ctor};
pub use uff::{UffVdW, uff_lj_ctor};
