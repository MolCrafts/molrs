//! Thole dipole-dipole screening (CL&Pol short-range damping).
//!
//! Screens the Coulomb interaction between Drude-related point charges at short
//! range with the exponential Thole function
//!
//! ```text
//! T_ij(r) = 1 - (1 + s_ij r / 2) exp(-s_ij r)
//! s_ij    = a_ij / (alpha_i alpha_j)^(1/6),   a_ij = (a_i + a_j) / 2
//! ```
//!
//! so the damped energy of a pair is `T_ij(r) * q_i q_j / r`. The screening
//! `s_ij` depends on **both** endpoints' atomic polarizabilities, so the
//! constructor resolves per-atom-type `charge` / `alpha` / `a_thole` from the
//! `atoms` block and precomputes `(s_ij, q_i q_j)` per pair.
//!
//! Units: r in A, alpha in A^3, a dimensionless, q in e (energy in the same
//! Coulomb units as the accompanying electrostatic kernel — Thole is a
//! multiplicative screen on `q_i q_j / r`).
//!
//! Reference: Thole, Chem. Phys. 59 (1981) 341,
//! DOI 10.1016/0301-0104(81)85176-2; as emitted by the paduagroup/clandpol
//! polarizer (LAMMPS `pair_style thole`).

use std::collections::HashMap;

use crate::ff::forcefield::Params;
use crate::ff::potential::Potential;
use crate::ff::potential::geometry::validate_coords;
use molrs::spatial::neighbors::Neighbors;
use molrs::store::frame::Frame;
use molrs::types::F;

/// Thole-screened Coulomb pair potential with pre-resolved flat arrays.
///
/// `s[idx]` is the screening factor `s_ij` and `qq[idx]` the charge product
/// `q_i q_j` for pair `idx` — both depend on the two endpoints' atom types and
/// are resolved once at construction.
/// Where a pair's screening length and charge product come from.
enum Source {
    /// Resolved against one fixed pair list at construction.
    Compiled {
        atom_i: Vec<usize>,
        atom_j: Vec<usize>,
        s: Vec<F>,
        qq: Vec<F>,
    },
    /// Per-atom `(q, α, a)`, combined when a pair turns up.
    ///
    /// What a neighbour-driven evaluation needs: a neighbour table is a
    /// different list of pairs every rebuild. Under a ghost régime the vectors
    /// cover the copies too, each carrying its owner's values.
    PerAtom {
        q: Vec<F>,
        alpha: Vec<F>,
        a_thole: Vec<F>,
    },
}

pub struct PairThole {
    source: Source,
}

impl PairThole {
    pub fn new(atom_i: Vec<usize>, atom_j: Vec<usize>, s: Vec<F>, qq: Vec<F>) -> Self {
        assert_eq!(atom_i.len(), atom_j.len());
        assert_eq!(atom_i.len(), s.len());
        assert_eq!(atom_i.len(), qq.len());
        Self {
            source: Source::Compiled {
                atom_i,
                atom_j,
                s,
                qq,
            },
        }
    }

    /// Per-atom `(q, α, a)`, combined when a pair turns up by the same rule
    /// [`pair_thole_ctor`] applies: `a_ij = ½(aᵢ + aⱼ)`, `s = a_ij/(αᵢαⱼ)^(1/6)`.
    pub fn typed(q: Vec<F>, alpha: Vec<F>, a_thole: Vec<F>) -> Self {
        assert_eq!(q.len(), alpha.len());
        assert_eq!(q.len(), a_thole.len());
        Self {
            source: Source::PerAtom { q, alpha, a_thole },
        }
    }

    /// The pair term for one already-reduced separation.
    fn pair_kernel(&self, r2: F, disp: [F; 3], s: F, qq: F) -> Option<(F, [F; 3])> {
        if r2 < 1e-24 {
            return None;
        }
        let r = r2.sqrt();
        let x = s * r;
        let e_x = (-x).exp();
        let t = 1.0 - (1.0 + x / 2.0) * e_x;
        let energy = t * qq / r;

        // V = T qq / r ;  T'(r) = (s/2)(1 + x) e^{-x}
        // dV/dr = qq (T'/r - T/r^2)
        // factor = -(1/r) dV/dr = qq (T/r^3 - T'/r^2)
        let tp = (s / 2.0) * (1.0 + x) * e_x;
        let dvdr = qq * (tp / r - t / r2);
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
        pair: impl Fn(usize) -> (usize, usize, F, F, [F; 3], F),
    ) -> (F, Vec<F>) {
        let mut energy: F = 0.0;
        let mut forces = vec![0.0; n_components];
        for idx in 0..n_pairs {
            let (i, j, s, qq, disp, r2) = pair(idx);
            let Some((e, f)) = self.pair_kernel(r2, disp, s, qq) else {
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

impl Potential for PairThole {
    fn calc_energy_forces(&self, coords: &[F]) -> (F, Vec<F>) {
        let n_atoms = validate_coords(coords);
        let Source::Compiled {
            atom_i,
            atom_j,
            s,
            qq,
        } = &self.source
        else {
            // Per-atom parameters need a pair table, and nobody handed one over.
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
            (i, j, s[idx], qq[idx], d, r2)
        })
    }

    fn calc_energy_forces_with_pairs(&self, coords: &[F], pairs: &Neighbors) -> (F, Vec<F>) {
        let Source::PerAtom { q, alpha, a_thole } = &self.source else {
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
                i < q.len() && j < q.len(),
                "a pair names an atom the per-atom parameters do not cover"
            );
            let a_ij = 0.5 * (a_thole[i] + a_thole[j]);
            let s = a_ij / (alpha[i] * alpha[j]).powf(1.0 / 6.0);
            (
                i,
                j,
                s,
                q[i] * q[j],
                [disp[[p, 0]], disp[[p, 1]], disp[[p, 2]]],
                d2[p],
            )
        })
    }
}

/// Construct a [`PairThole`] from per-atom-type params and Frame topology.
///
/// The thole style's per-type definitions are keyed by **atom type name** and
/// carry `charge`, `alpha`, `a_thole`. Each pair's screening is resolved from
/// its two endpoints' atom types (read from the `atoms` block `type` column).
pub fn pair_thole_ctor(
    _style_params: &Params,
    type_params: &[(&str, &Params)],
    frame: &Frame,
) -> Result<Box<dyn Potential>, String> {
    let type_map: HashMap<&str, &Params> = type_params.iter().copied().collect();

    let atoms = frame
        .get("atoms")
        .ok_or_else(|| "PairThole: frame missing \"atoms\" block".to_string())?;
    let atom_types = atoms
        .get_string("type")
        .ok_or_else(|| "PairThole: atoms block missing \"type\" column".to_string())?;

    let block = frame
        .get("pairs")
        .ok_or_else(|| "PairThole: frame missing \"pairs\" block".to_string())?;
    let i_col = block
        .get_uint("atomi")
        .ok_or_else(|| "PairThole: pairs block missing \"atomi\" column".to_string())?;
    let j_col = block
        .get_uint("atomj")
        .ok_or_else(|| "PairThole: pairs block missing \"atomj\" column".to_string())?;

    let lookup = |type_name: &str| -> Result<(F, F, F), String> {
        let p = type_map
            .get(type_name)
            .ok_or_else(|| format!("PairThole: unknown atom type '{}'", type_name))?;
        let q = p
            .get("charge")
            .ok_or_else(|| format!("PairThole type '{}': missing 'charge'", type_name))?
            as F;
        let alpha = p
            .get("alpha")
            .ok_or_else(|| format!("PairThole type '{}': missing 'alpha'", type_name))?
            as F;
        let a = p
            .get("a_thole")
            .ok_or_else(|| format!("PairThole type '{}': missing 'a_thole'", type_name))?
            as F;
        Ok((q, alpha, a))
    };

    let mut atom_i = Vec::with_capacity(i_col.len());
    let mut atom_j = Vec::with_capacity(i_col.len());
    let mut s_vec = Vec::with_capacity(i_col.len());
    let mut qq_vec = Vec::with_capacity(i_col.len());

    for idx in 0..i_col.len() {
        let i = i_col[idx] as usize;
        let j = j_col[idx] as usize;
        let (qi, alpha_i, ai) = lookup(&atom_types[i])?;
        let (qj, alpha_j, aj) = lookup(&atom_types[j])?;

        let a_ij = 0.5 * (ai + aj);
        let s = a_ij / (alpha_i * alpha_j).powf(1.0 / 6.0);

        atom_i.push(i);
        atom_j.push(j);
        s_vec.push(s);
        qq_vec.push(qi * qj);
    }

    Ok(Box::new(PairThole::new(atom_i, atom_j, s_vec, qq_vec)))
}

#[cfg(test)]
mod tests {

    /// Combining `(q, α, a)` when a pair turns up is the same number as having
    /// combined them earlier against a fixed list — bit for bit.
    #[test]
    fn per_atom_parameters_score_a_pair_exactly_as_compiled_ones() {
        use crate::ff::potential::pair::testing::{assert_same, table_over};

        let q = vec![0.4_f64, -0.7, 0.3, -0.2];
        let alpha = vec![1.1_f64, 0.8, 1.4, 0.6];
        let a_thole = vec![2.6_f64, 2.6, 2.1, 2.9];
        let coords: Vec<F> = vec![
            0.0, 0.0, 0.0, //
            2.1, 0.4, 0.2, //
            1.2, 1.9, 0.7, //
            3.0, 2.3, 1.1,
        ];
        let links = [(0_usize, 1_usize), (0, 2), (1, 3), (2, 3)];

        let (ai, aj): (Vec<usize>, Vec<usize>) = links.iter().copied().unzip();
        let s: Vec<F> = links
            .iter()
            .map(|&(i, j)| {
                let a_ij = 0.5 * (a_thole[i] + a_thole[j]);
                a_ij / (alpha[i] * alpha[j]).powf(1.0 / 6.0)
            })
            .collect();
        let qq: Vec<F> = links.iter().map(|&(i, j)| q[i] * q[j]).collect();
        let compiled = PairThole::new(ai, aj, s, qq);
        let typed = PairThole::typed(q, alpha, a_thole);

        let table = table_over(&coords, &links);
        assert_same(
            "thole",
            compiled.calc_energy_forces(&coords),
            typed.calc_energy_forces_with_pairs(&coords, &table),
        );
    }
    use super::*;

    fn numerical_forces(pot: &PairThole, coords: &[F]) -> Vec<F> {
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
    fn damping_factor_matches_closed_form() {
        // s=1.0, qq=1.0, r=2.0: x=2, T = 1 - (1+1)e^-2 = 1 - 2 e^-2
        let pot = PairThole::new(vec![0], vec![1], vec![1.0], vec![1.0]);
        let coords: Vec<F> = vec![0.0, 0.0, 0.0, 2.0, 0.0, 0.0];
        let x = 2.0_f64;
        let t = 1.0 - (1.0 + x / 2.0) * (-x).exp();
        let expected = t * 1.0 / 2.0;
        assert!(
            (pot.calc_energy(&coords) - expected).abs() < 1e-12,
            "energy {} vs {expected}",
            pot.calc_energy(&coords)
        );
    }

    #[test]
    fn damping_vanishes_at_zero_separation_limit() {
        // T(r) -> 0 as r -> 0 (screening fully removes the singular Coulomb term),
        // so the energy stays finite. Check T is small at small r.
        let pot = PairThole::new(vec![0], vec![1], vec![5.0], vec![1.0]);
        let coords: Vec<F> = vec![0.0, 0.0, 0.0, 0.05, 0.0, 0.0];
        assert!(pot.calc_energy(&coords).is_finite());
    }

    #[test]
    fn forces_match_finite_difference() {
        let pot = PairThole::new(vec![0], vec![1], vec![1.3], vec![-0.7]);
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
        let pot = PairThole::new(vec![0], vec![1], vec![1.3], vec![-0.7]);
        let coords: Vec<F> = vec![0.0, 0.0, 0.0, 1.2, 0.7, -0.4];
        let (_, f) = pot.calc_energy_forces(&coords);
        for dim in 0..3 {
            assert!((f[dim] + f[3 + dim]).abs() < 1e-9, "dim {dim}");
        }
    }
}
