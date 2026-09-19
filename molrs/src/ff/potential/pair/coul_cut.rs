//! Buffered Coulomb pair potential with a hard distance cutoff — `pair/coul/cut`:
//!
//! ```text
//! E(r) = k · qᵢqⱼ / (D · (r + δ))      r < r_cut,  else 0
//! ```
//!
//! Every constant in that formula is **force-field data, read from the style**:
//! the Coulomb constant `k`, the dielectric `D` and the buffering distance `δ`.
//! The kernel holds none of them.
//!
//! # One kernel, two force fields
//!
//! `δ = 0` degenerates the buffered form into the textbook Coulomb `k·qᵢqⱼ/r`,
//! which is what OPLS, LAMMPS and every other non-buffered force field mean. MMFF's
//! electrostatics (Halgren, *MMFF.I* eq. 5) is the *same kernel* at
//! `k = 332.0716`, `D = 1.0`, `δ = 0.05 Å` — a parameterization, not a kernel of
//! its own. The buffer is what keeps `E` finite at `r = 0`, where MMFF's charges
//! sit on the nuclei.
//!
//! # Why `k` is a style param and not a shared constant
//!
//! MMFF uses Halgren's **332.0716**; OPLS/LAMMPS use CODATA's
//! [`COULOMB_REAL`](molrs::units::constants::COULOMB_REAL) = **332.06371**. The 2.4e-5
//! relative difference is worth 0.0036 kcal/mol on caffeine's −150.48 kcal/mol
//! electrostatic term — **above** the 1e-3 RDKit parity tolerance. Both values are
//! correct; the force field decides. A kernel holding either one would be choosing a
//! force field for its caller, so there is no default: a style that does not say is
//! an [`Err`].
//!
//! The kernel is topology-blind: it consumes a pre-resolved pair list whose per-pair
//! charge products `qᵢqⱼ` already include any exclusion / 1-4 scaling.

use crate::ff::forcefield::Params;
use crate::ff::potential::Potential;
use crate::ff::potential::gather_copies;
use crate::ff::potential::geometry::validate_coords;
use crate::ff::potential::pair::energy_forces;
use molrs::math::Virial;
use molrs::spatial::neighbors::Neighbors;
use molrs::store::frame::Frame;
use molrs::types::F;

/// Below this squared separation a pair's force has no direction, so only the
/// (buffered) energy is accumulated. Unchanged from the unbuffered kernel.
const R_MIN2: F = 1e-24;

/// Buffered Coulomb with cutoff, over pre-resolved flat arrays.
///
/// `E = k·qᵢqⱼ / (D·(r + δ))`. All four scalars come from the force field; see the
/// module docs.
pub struct PairCoulCut {
    charges: Charges,
    /// Coulomb constant `k` in kcal·Å·mol⁻¹·e⁻² — the force field's, not the kernel's.
    coulomb: F,
    /// Dielectric constant `D` of the medium the force field was parameterised in.
    dielectric: F,
    /// Buffering distance δ in Å; `0.0` is the unbuffered, textbook Coulomb.
    delta: F,
    /// Distance cutoff in Å; `f64::INFINITY` disables it.
    cutoff: F,
}

/// Where a pair's charge product comes from.
enum Charges {
    /// Products resolved against one fixed pair list at construction, already
    /// carrying any exclusion or 1-4 scaling.
    Compiled {
        atom_i: Vec<usize>,
        atom_j: Vec<usize>,
        qiqj: Vec<F>,
    },
    /// Per-atom charges; the product is formed when a pair turns up.
    ///
    /// This is what a neighbour-driven evaluation needs — a neighbour table is
    /// a different list of pairs every rebuild, so a product resolved against
    /// an older one names the wrong atoms. Under a ghost régime the vector
    /// covers the copies too, each carrying its owner's charge.
    ///
    /// It carries **no** 1-4 scaling: there is nothing on a neighbour table to
    /// carry it. Scaling close neighbours is the caller's, through the
    /// special-bonds weights.
    PerAtom {
        q: Vec<F>,
        /// How many of the entries above are atoms; the rest are copies, and
        /// are rebuilt from their owners whenever the copy list is.
        n_owned: usize,
    },
}

impl PairCoulCut {
    pub fn new(
        atom_i: Vec<usize>,
        atom_j: Vec<usize>,
        qiqj: Vec<F>,
        coulomb: F,
        dielectric: F,
        delta: F,
        cutoff: F,
    ) -> Self {
        assert_eq!(atom_i.len(), atom_j.len());
        assert_eq!(atom_i.len(), qiqj.len());
        Self {
            charges: Charges::Compiled {
                atom_i,
                atom_j,
                qiqj,
            },
            coulomb,
            dielectric,
            delta,
            cutoff,
        }
    }

    /// A kernel that forms `qᵢqⱼ` from the atoms a pair names.
    ///
    /// The form a neighbour-driven evaluation needs.
    ///
    /// A neighbour table is a different list of pairs every rebuild, so a
    /// parameter resolved against an older one belongs to different atoms.
    /// Keyed on the atoms instead, it can be found for whatever pair turns up
    /// — including one that names a periodic copy.
    pub fn typed(q: Vec<F>, coulomb: F, dielectric: F, delta: F, cutoff: F) -> Self {
        let n_owned = q.len();
        Self {
            charges: Charges::PerAtom { q, n_owned },
            coulomb,
            dielectric,
            delta,
            cutoff,
        }
    }

    /// The pair term: buffered Coulomb for one already-reduced separation.
    ///
    /// `qq` is `k·qᵢqⱼ/D` — the style constants are hoisted by the caller, once
    /// per evaluation rather than once per pair. Returns the energy always and
    /// the force only when the separation has a direction.
    fn pair_kernel(&self, r2: F, disp: [F; 3], qq: F) -> Option<(F, [F; 3])> {
        if r2 >= self.cutoff * self.cutoff {
            return None;
        }
        let r = r2.sqrt();
        let r_buf = r + self.delta;
        // Unbuffered (δ = 0) coincident charges: `k·qᵢqⱼ/0` is undefined, so
        // skip the pair entirely. With a buffer this cannot trigger — keeping
        // the term finite at r = 0 is the buffer's whole purpose.
        if r_buf <= 0.0 {
            return None;
        }
        let energy = qq / r_buf;
        // The force is `−dE/dr · r̂` with `dE/dr = −k·qᵢqⱼ/(D·(r+δ)²)`, i.e. the
        // BUFFERED distance squared in the denominator — not `r²`. At r = 0 it
        // has no direction (the energy above is still real), so leave it zero.
        if r2 < R_MIN2 {
            return Some((energy, [0.0, 0.0, 0.0]));
        }
        let factor = qq / (r_buf * r_buf * r);
        Some((
            energy,
            [factor * disp[0], factor * disp[1], factor * disp[2]],
        ))
    }

    /// The accumulation, once. What differs between the two entry points is
    /// which pairs turn up and where `qᵢqⱼ` comes from — never the force.
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
}

impl Potential for PairCoulCut {
    fn calc_energy_forces(&self, coords: &[F]) -> (F, Vec<F>) {
        let n_atoms = validate_coords(coords);
        // `k/D` is a style-level constant: hoist it out of the pair loop. At
        // D = 1 this is exactly `k` (IEEE: `x / 1.0 == x`), so the δ = 0, D = 1
        // path stays bit-for-bit identical to the unbuffered kernel this
        // generalizes.
        let k_over_d = self.coulomb / self.dielectric;
        let Charges::Compiled {
            atom_i,
            atom_j,
            qiqj,
        } = &self.charges
        else {
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
            (i, j, k_over_d * qiqj[idx], d, r2)
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
        let k_over_d = self.coulomb / self.dielectric;
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
            // Parenthesised the way the compiled path forms it: `k/D · (qᵢqⱼ)`
            // and `(k/D · qᵢ) · qⱼ` differ in the last bit, and the two paths
            // are meant to be the same number.
            let qq = k_over_d * (q[i] * q[j]);
            (i, j, qq, [disp[[p, 0]], disp[[p, 1]], disp[[p, 2]]], d2[p])
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

/// Read a style param the **force field must supply**, or say which one it did not.
///
/// There is no `unwrap_or` here on purpose. A kernel that defaults `coulomb` or
/// `dielectric` is a kernel answering a question only the force field can answer: it
/// computes plausible numbers from constants nobody handed it, and every energy test
/// still passes. (`coulomb14scale` is projected out of the force field's
/// `SpecialBonds` by `Style::to_potential`, so through the documented route it is
/// always present — which is exactly why defaulting it here would be invisible.)
fn required(style_params: &Params, key: &str) -> Result<F, String> {
    style_params.get(key).map(|v| v as F).ok_or_else(|| {
        format!(
            "PairCoulCut: style params missing \"{key}\" — it is force-field data and this kernel \
             has no default for it. `pair/coul/cut` evaluates E = coulomb·qᵢqⱼ/(dielectric·(r+delta)); \
             the force field must declare `coulomb` (MMFF: 332.0716, OPLS/LAMMPS: 332.06371), \
             `dielectric`, and — via `special_bonds` — `coulomb14scale`."
        )
    })
}

/// Construct a [`PairCoulCut`] from **per-atom charges** + a neighbour list.
///
/// Reads per-atom `charge` from the atoms block (the molecule carries its own
/// charges — RESP/AM1-BCC for GAFF, MMFF's bond-increment charges, …), forms `qᵢqⱼ`
/// for each interacting pair, and scales 1-4-flagged pairs by the force field's 1-4
/// Coulomb weight.
///
/// # Style params
///
/// | param | meaning | missing |
/// |---|---|---|
/// | `coulomb` | Coulomb constant `k` | **`Err`** — the force field's to choose |
/// | `dielectric` | dielectric `D` | **`Err`** — a property of the medium, not of the kernel |
/// | `coulomb14scale` | 1-4 weight | **`Err`** — projected from `special_bonds` by `Style::to_potential` |
/// | `delta` | buffering distance δ (Å) | `0.0` — *semantic* default: no buffer, the textbook Coulomb |
/// | `cutoff` | cutoff (Å) | `∞` — *semantic* default: do not truncate |
///
/// The last two are real, meaningful choices a force field is entitled to leave
/// unsaid. The first three are not: a default there is the kernel pretending the
/// force field spoke.
///
/// The `pairs` block is the consumer-built neighbour list (`atomi`/`atomj`/`is_14`)
/// from `intramolecular_pairs`; 1-2/1-3 are already excluded. Charge-free pair types
/// are not consulted — this kernel is per-atom
/// ([`ParamSource::PerInstance`](crate::ff::potential::ParamSource::PerInstance)).
pub fn pair_coul_cut_ctor(
    style_params: &Params,
    _type_params: &[(&str, &Params)],
    frame: &Frame,
) -> Result<Box<dyn Potential>, String> {
    let coulomb = required(style_params, "coulomb")?;
    let dielectric = required(style_params, "dielectric")?;
    let scale_14 = required(style_params, "coulomb14scale")?;
    // The two genuine semantic defaults.
    let delta = style_params.get("delta").map(|d| d as F).unwrap_or(0.0);
    let cutoff = style_params
        .get("cutoff")
        .map(|c| c as F)
        .unwrap_or(F::INFINITY);

    let atoms = frame
        .get("atoms")
        .ok_or_else(|| "PairCoulCut: frame missing \"atoms\" block".to_string())?;
    let charges = atoms
        .get_float("charge")
        .ok_or_else(|| "PairCoulCut: atoms block missing \"charge\" column".to_string())?;
    let block = frame
        .get("pairs")
        .ok_or_else(|| "PairCoulCut: frame missing \"pairs\" block".to_string())?;
    let i_col = block
        .get_uint("atomi")
        .ok_or_else(|| "PairCoulCut: pairs block missing \"atomi\" column".to_string())?;
    let j_col = block
        .get_uint("atomj")
        .ok_or_else(|| "PairCoulCut: pairs block missing \"atomj\" column".to_string())?;
    let is_14 = block.get_bool("is_14");

    let n = i_col.len();
    let mut atom_i = Vec::with_capacity(n);
    let mut atom_j = Vec::with_capacity(n);
    let mut qiqj = Vec::with_capacity(n);

    for idx in 0..n {
        let i = i_col[idx] as usize;
        let j = j_col[idx] as usize;
        let mut qq = charges[i] as F * charges[j] as F;
        if is_14.is_some_and(|b| b[idx]) {
            qq *= scale_14;
        }
        atom_i.push(i);
        atom_j.push(j);
        qiqj.push(qq);
    }

    Ok(Box::new(PairCoulCut::new(
        atom_i, atom_j, qiqj, coulomb, dielectric, delta, cutoff,
    )))
}

/// Construct a neighbour-driven [`PairCoulCut`] from per-atom parameters.
///
/// The counterpart of [`pair_coul_cut_ctor`]: the same force field, keyed on the atoms
/// instead of on a pair list, so it can answer for whatever pairs a neighbour
/// search turns up. It reads no `pairs` block — there is none to read when the
/// list is rebuilt every few steps.
pub fn pair_coul_cut_typed_ctor(
    style_params: &Params,
    _type_params: &[(&str, &Params)],
    frame: &Frame,
) -> Result<Box<dyn Potential>, String> {
    let coulomb = required(style_params, "coulomb")?;
    let dielectric = required(style_params, "dielectric")?;
    let delta = style_params.get("delta").map(|d| d as F).unwrap_or(0.0);
    let cutoff = style_params
        .get("cutoff")
        .map(|c| c as F)
        .unwrap_or(F::INFINITY);
    // `coulomb14scale` is deliberately not read here. On this path the 1-4
    // weight is applied to the pair table, not baked into a charge product, so
    // requiring the kernel to know it would be asking it a question it no
    // longer answers.
    let atoms = frame
        .get("atoms")
        .ok_or_else(|| "PairCoulCut: frame missing \"atoms\" block".to_string())?;
    let charge = atoms
        .get_float("charge")
        .ok_or_else(|| "PairCoulCut: atoms block missing \"charge\" column".to_string())?;
    let q: Vec<F> = charge.iter().map(|&c| c as F).collect();
    Ok(Box::new(PairCoulCut::typed(
        q, coulomb, dielectric, delta, cutoff,
    )))
}

#[cfg(test)]
mod tests {
    use crate::ff::potential::pair::testing::{
        assert_same, assert_virial_matches_forces, table_over,
    };

    /// Forming `qᵢqⱼ` from the atoms is the same number as having formed it
    /// earlier against a fixed list — bit for bit, on the same pairs.
    ///
    /// The difference is only *when*, and that matters because a neighbour
    /// table is a different list of pairs every rebuild: a product resolved
    /// against an older one names the wrong atoms, silently.
    #[test]
    fn per_atom_charges_score_a_pair_exactly_as_compiled_products() {
        let q = vec![0.4_f64, -0.7, 0.3, -0.2];
        let coords: Vec<F> = vec![
            0.0, 0.0, 0.0, //
            2.1, 0.4, 0.2, //
            1.2, 1.9, 0.7, //
            3.0, 2.3, 1.1,
        ];
        let links = [(0_usize, 1_usize), (0, 2), (1, 3), (2, 3)];
        let (k, d, delta, cutoff) = (332.0716_f64, 1.0_f64, 0.05_f64, 10.0_f64);

        let (ai, aj): (Vec<usize>, Vec<usize>) = links.iter().copied().unzip();
        let qiqj: Vec<F> = links.iter().map(|&(i, j)| q[i] * q[j]).collect();
        let compiled = PairCoulCut::new(ai, aj, qiqj, k, d, delta, cutoff);
        let typed = PairCoulCut::typed(q, k, d, delta, cutoff);

        let table = table_over(&coords, &links);
        assert_same(
            "coul/cut",
            compiled.calc_energy_forces(&coords),
            typed.calc_energy_forces_with_pairs(&coords, &table),
        );
        assert_virial_matches_forces(
            "coul/cut",
            &coords,
            typed.calc_energy_forces_with_pairs_virial(&coords, &table),
        );
    }

    use super::*;
    use molrs::units::constants::COULOMB_REAL;

    /// MMFF's parameterization: Halgren's constant, unit dielectric, 0.05 Å buffer.
    const MMFF: (F, F, F) = (332.0716, 1.0, 0.05);

    fn unbuffered(qiqj: F, cutoff: F) -> PairCoulCut {
        PairCoulCut::new(vec![0], vec![1], vec![qiqj], COULOMB_REAL, 1.0, 0.0, cutoff)
    }

    #[test]
    fn energy_and_sign() {
        // Unit positive charges 2 Å apart: E = k_e / 2.
        let pot = unbuffered(1.0, F::INFINITY);
        let coords: Vec<F> = vec![0.0, 0.0, 0.0, 2.0, 0.0, 0.0];
        let (e, forces) = pot.calc_energy_forces(&coords);
        assert!((e - COULOMB_REAL / 2.0).abs() < 1e-9, "E got {e}");
        // Like charges repel: force on j (+x atom) points +x (away from i).
        assert!(
            forces[3] > 0.0,
            "like charges should repel, fxj={}",
            forces[3]
        );
        // Unlike charges attract.
        let pot2 = unbuffered(-1.0, F::INFINITY);
        let (_, f2) = pot2.calc_energy_forces(&coords);
        assert!(f2[3] < 0.0, "unlike charges should attract, fxj={}", f2[3]);
    }

    #[test]
    fn cutoff_and_zero_distance() {
        let pot = unbuffered(1.0, 1.5);
        // 2 Å apart, cutoff 1.5 -> excluded, E=0.
        let far: Vec<F> = vec![0.0, 0.0, 0.0, 2.0, 0.0, 0.0];
        assert_eq!(pot.calc_energy_forces(&far).0, 0.0);
        // Coincident atoms, UNBUFFERED -> skipped, finite.
        let zero: Vec<F> = vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let (e, f) = pot.calc_energy_forces(&zero);
        assert_eq!(e, 0.0);
        assert!(f.iter().all(|x| x.is_finite()));
    }

    /// The buffer's reason to exist: at r = 0 the energy is `k·qᵢqⱼ/(D·δ)`, finite.
    #[test]
    fn buffered_zero_distance_is_finite() {
        let (k, d, delta) = MMFF;
        let pot = PairCoulCut::new(vec![0], vec![1], vec![-0.25], k, d, delta, F::INFINITY);
        let zero: Vec<F> = vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let (e, f) = pot.calc_energy_forces(&zero);
        assert!((e - k * -0.25 / (d * delta)).abs() < 1e-9, "E got {e}");
        assert!(f.iter().all(|x| x.is_finite()), "forces {f:?}");
    }

    /// `δ = 0` reproduces the unbuffered kernel bit-for-bit (r = 2 is exact in f64).
    #[test]
    fn delta_zero_is_the_unbuffered_kernel() {
        let pot = unbuffered(-0.48, F::INFINITY);
        let coords: Vec<F> = vec![0.0, 0.0, 0.0, 2.0, 0.0, 0.0];
        let (e, _) = pot.calc_energy_forces(&coords);
        assert_eq!(e, COULOMB_REAL * -0.48 / 2.0);
    }

    /// The force must use `(r + δ)²`, not `r²` — buffering the energy and forgetting
    /// the gradient passes every energy assertion and fails here.
    #[test]
    fn numerical_gradient() {
        for (k, d, delta) in [(COULOMB_REAL, 1.0, 0.0), MMFF] {
            let pot = PairCoulCut::new(vec![0], vec![1], vec![0.8], k, d, delta, F::INFINITY);
            let coords: Vec<F> = vec![0.0, 0.0, 0.0, 1.7, 0.3, -0.4];
            let (_, forces) = pot.calc_energy_forces(&coords);
            let h = 1e-6;
            for dim in 0..coords.len() {
                let mut cp = coords.clone();
                let mut cm = coords.clone();
                cp[dim] += h;
                cm[dim] -= h;
                let fd =
                    -(pot.calc_energy_forces(&cp).0 - pot.calc_energy_forces(&cm).0) / (2.0 * h);
                assert!(
                    (forces[dim] - fd).abs() < 1e-5,
                    "delta {delta}, comp {dim}: analytic {} vs fd {fd}",
                    forces[dim]
                );
            }
            // Newton's third law.
            for dim in 0..3 {
                assert!((forces[dim] + forces[3 + dim]).abs() < 1e-9);
            }
        }
    }

    /// A style that does not carry `coulomb` / `dielectric` / `coulomb14scale` is an
    /// `Err`. The kernel must not supply the force field's own constants.
    #[test]
    fn missing_force_field_data_is_an_error() {
        let full = [
            ("coulomb", 332.0716),
            ("dielectric", 1.0),
            ("coulomb14scale", 0.75),
        ];
        for omitted in ["coulomb", "dielectric", "coulomb14scale"] {
            let pairs: Vec<(&str, f64)> = full
                .iter()
                .copied()
                .filter(|(k, _)| *k != omitted)
                .collect();
            let params = Params::from_pairs(&pairs);
            let err = required(&params, omitted).expect_err("must not default");
            assert!(err.contains(omitted), "error must name `{omitted}`: {err}");
        }
        // All three present -> read, not invented.
        let params = Params::from_pairs(&full);
        assert_eq!(required(&params, "coulomb").unwrap(), 332.0716);
    }
}
