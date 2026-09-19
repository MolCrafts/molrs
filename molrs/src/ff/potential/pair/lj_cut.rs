//! Unique LJ / Mie pair kernel (`lj/cut`).
//!
//! Pair source is fixed at construction:
//! - [`LJCut::new`] / [`LJCut::lj126`] — uniform ε/σ, loop-fed pairs (MD)
//! - [`LJCut::compiled`] — per-pair ε/σ from a ForceField `pairs` block
//!
//! Arithmetic uses `inv_r2 = 1/r2`. Degenerate pairs `r2 < 1e-24` are skipped.

use std::collections::HashMap;

use crate::ff::forcefield::Params;
use crate::ff::forcefield::mixing::Mixing;
use crate::ff::potential::gather_copies;
use crate::ff::potential::geometry::validate_coords;
use crate::ff::potential::pair::PairPotential;
use crate::ff::potential::pair::atom_type_index;
use crate::ff::potential::pair::fold_chunks;
use crate::ff::potential::{Member, PairDriven, Potential};
use molrs::math::Virial;
use molrs::spatial::neighbors::{Neighbors, VerletSkin};
use molrs::store::frame::Frame;
use molrs::types::F;
use ndarray::{Array2, ArrayView2};

const MIN_R2: F = 1e-24;

#[derive(Clone, Debug)]
enum PairSource {
    Loop,
    Compiled {
        atom_i: Vec<usize>,
        atom_j: Vec<usize>,
        epsilon: Vec<F>,
        sigma: Vec<F>,
    },
    /// Per-atom types plus a type-pair table.
    ///
    /// The difference from `Compiled` is *when* a parameter is chosen. A
    /// compiled kernel resolved its parameters against a pair list at
    /// construction, which is only meaningful while that exact list is the one
    /// being evaluated — and a neighbour table is rebuilt from scratch every
    /// few steps, with different rows in a different order. Keyed on the atoms
    /// instead, a parameter can be found for whatever pair turns up, including
    /// a pair that involves a periodic copy.
    Typed {
        /// Type index per atom. Under a ghost régime this covers the copies
        /// too, gathered from their owners.
        type_id: Vec<u32>,
        ntypes: usize,
        /// Flattened `ti * ntypes + tj`, each already mixed and each carrying
        /// the cutoff-dependent constants that go with it.
        sigma: Vec<F>,
        ceps: Vec<F>,
        e0: Vec<F>,
        f_rc: Vec<F>,
        /// How many of the entries above are atoms; the rest are copies, and
        /// are rebuilt from their owners whenever the copy list is.
        n_owned: usize,
    },
}

/// The cutoff-dependent constants for one mixed `(ε, σ)`.
///
/// Shared by the style-level constructor and the type-pair table so the two
/// cannot disagree about what shifting means.
fn shift_constants(
    epsilon: F,
    sigma: F,
    cutoff: F,
    n: i32,
    m: i32,
    shifted: bool,
    smeared: bool,
) -> (F, F, F) {
    let ceps = mie_c(n, m) * epsilon;
    let sr_c = sigma / cutoff;
    let sr_n_c = sr_c.powi(n);
    let sr_m_c = sr_c.powi(m);
    let u_c = ceps * (sr_n_c - sr_m_c);
    let fac_c = ceps * ((n as F) * sr_n_c - (m as F) * sr_m_c) / (cutoff * cutoff);
    let shift_energy = shifted || smeared;
    (
        ceps,
        if shift_energy { u_c } else { 0.0 },
        if smeared { fac_c * cutoff } else { 0.0 },
    )
}

/// LAMMPS `pair_style lj/cut`.
#[derive(Clone, Debug)]
pub struct LJCut {
    epsilon: F,
    sigma: F,
    cutoff: F,
    n: i32,
    m: i32,
    shifted: bool,
    smeared: bool,
    cutoff2: F,
    ceps: F,
    e0: F,
    f_rc: F,
    source: PairSource,
}

fn mie_c(n: i32, m: i32) -> F {
    let n = n as F;
    let m = m as F;
    (n / (n - m)) * (n / m).powf(m / (n - m))
}

impl LJCut {
    pub fn new(
        epsilon: F,
        sigma: F,
        cutoff: F,
        n: i32,
        m: i32,
        shifted: bool,
        smeared: bool,
    ) -> Result<Self, String> {
        if epsilon <= 0.0 || sigma <= 0.0 || cutoff <= 0.0 {
            return Err("LJCut requires epsilon, sigma, cutoff > 0".into());
        }
        if m <= 0 || n <= m {
            return Err(format!(
                "LJCut exponents must satisfy n > m > 0, got n={n}, m={m}"
            ));
        }
        let cutoff2 = cutoff * cutoff;
        let (ceps, e0, f_rc) = shift_constants(epsilon, sigma, cutoff, n, m, shifted, smeared);
        Ok(Self {
            epsilon,
            sigma,
            cutoff,
            n,
            m,
            shifted: shifted || smeared,
            smeared,
            cutoff2,
            ceps,
            e0,
            f_rc,
            source: PairSource::Loop,
        })
    }

    pub fn lj126(epsilon: F, sigma: F, cutoff: F) -> Result<Self, String> {
        Self::new(epsilon, sigma, cutoff, 12, 6, true, false)
    }

    pub fn compiled(
        atom_i: Vec<usize>,
        atom_j: Vec<usize>,
        epsilon: Vec<F>,
        sigma: Vec<F>,
    ) -> Self {
        assert_eq!(atom_i.len(), atom_j.len());
        assert_eq!(atom_i.len(), epsilon.len());
        assert_eq!(atom_i.len(), sigma.len());
        Self {
            epsilon: 1.0,
            sigma: 1.0,
            cutoff: F::INFINITY,
            n: 12,
            m: 6,
            shifted: false,
            smeared: false,
            cutoff2: F::INFINITY,
            ceps: 4.0,
            e0: 0.0,
            f_rc: 0.0,
            source: PairSource::Compiled {
                atom_i,
                atom_j,
                epsilon,
                sigma,
            },
        }
    }

    /// A kernel that finds its parameters from the atoms a pair names.
    ///
    /// `type_id` is one type index per atom — including, under a ghost régime,
    /// the copies, which carry their owners'. `per_type` is `(ε, σ)` per type,
    /// combined by `mixing` into the type-pair table.
    ///
    /// This is the form a neighbour-driven evaluation needs.
    /// [`compiled`](Self::compiled) resolved its parameters against one fixed
    /// pair list, so it can only answer for that list; a neighbour table is a
    /// different list every rebuild.
    #[allow(clippy::too_many_arguments)]
    pub fn typed(
        type_id: Vec<u32>,
        per_type: &[(F, F)],
        mixing: Mixing,
        cutoff: F,
        n: i32,
        m: i32,
        shifted: bool,
        smeared: bool,
    ) -> Result<Self, String> {
        if cutoff <= 0.0 {
            return Err("LJCut requires cutoff > 0".into());
        }
        if m <= 0 || n <= m {
            return Err(format!(
                "LJCut exponents must satisfy n > m > 0, got n={n}, m={m}"
            ));
        }
        let ntypes = per_type.len();
        if ntypes == 0 {
            return Err("LJCut::typed needs at least one type".into());
        }
        if let Some(&t) = type_id.iter().max()
            && t as usize >= ntypes
        {
            return Err(format!(
                "LJCut::typed: atom type {t} has no parameters (only {ntypes} types)"
            ));
        }
        let mut sigma = vec![0.0; ntypes * ntypes];
        let mut ceps = vec![0.0; ntypes * ntypes];
        let mut e0 = vec![0.0; ntypes * ntypes];
        let mut f_rc = vec![0.0; ntypes * ntypes];
        for ti in 0..ntypes {
            for tj in 0..ntypes {
                let (eps_ij, sig_ij) = mixing.combine(per_type[ti], per_type[tj]);
                let (c, e, fr) = shift_constants(eps_ij, sig_ij, cutoff, n, m, shifted, smeared);
                let t = ti * ntypes + tj;
                sigma[t] = sig_ij;
                ceps[t] = c;
                e0[t] = e;
                f_rc[t] = fr;
            }
        }
        let n_owned = type_id.len();
        Ok(Self {
            epsilon: 1.0,
            sigma: 1.0,
            cutoff,
            n,
            m,
            shifted: shifted || smeared,
            smeared,
            cutoff2: cutoff * cutoff,
            ceps: 0.0,
            e0: 0.0,
            f_rc: 0.0,
            source: PairSource::Typed {
                type_id,
                ntypes,
                sigma,
                ceps,
                e0,
                f_rc,
                n_owned,
            },
        })
    }

    pub fn epsilon(&self) -> F {
        self.epsilon
    }
    pub fn sigma(&self) -> F {
        self.sigma
    }
    pub fn cutoff(&self) -> F {
        self.cutoff
    }
    pub fn n(&self) -> i32 {
        self.n
    }
    pub fn m(&self) -> i32 {
        self.m
    }
    pub fn shifted(&self) -> bool {
        self.shifted
    }
    pub fn smeared(&self) -> bool {
        self.smeared
    }

    #[allow(clippy::too_many_arguments)]
    fn pair_kernel_params(
        &self,
        r2: F,
        disp: [F; 3],
        sigma: F,
        ceps: F,
        e0: F,
        f_rc: F,
        cutoff2: F,
        n: i32,
        m: i32,
    ) -> Option<(F, [F; 3])> {
        if r2 < MIN_R2 || r2 > cutoff2 {
            return None;
        }
        if n == 12 && m == 6 {
            let inv_r2 = 1.0 / r2;
            let sr2 = sigma * sigma * inv_r2;
            let sr6 = sr2 * sr2 * sr2;
            let sr12 = sr6 * sr6;
            let mut energy = ceps * (sr12 - sr6) - e0;
            let mut fac = ceps * (12.0 * sr12 - 6.0 * sr6) * inv_r2;
            if f_rc != 0.0 {
                let r = r2.sqrt();
                energy += (r - self.cutoff) * f_rc;
                fac -= f_rc / r;
            }
            return Some((energy, [fac * disp[0], fac * disp[1], fac * disp[2]]));
        }
        let sr = sigma / r2.sqrt();
        let sr_n = sr.powi(n);
        let sr_m = sr.powi(m);
        let mut energy = ceps * (sr_n - sr_m) - e0;
        let mut fac = ceps * ((n as F) * sr_n - (m as F) * sr_m) / r2;
        if f_rc != 0.0 {
            let r = r2.sqrt();
            energy += (r - self.cutoff) * f_rc;
            fac -= f_rc / r;
        }
        Some((energy, [fac * disp[0], fac * disp[1], fac * disp[2]]))
    }

    fn pair_kernel(&self, r2: F, disp: [F; 3]) -> Option<(F, [F; 3])> {
        self.pair_kernel_params(
            r2,
            disp,
            self.sigma,
            self.ceps,
            self.e0,
            self.f_rc,
            self.cutoff2,
            self.n,
            self.m,
        )
    }

    fn fold_compiled(&self, coords: &[F]) -> (F, Vec<F>) {
        let PairSource::Compiled {
            atom_i,
            atom_j,
            epsilon,
            sigma,
        } = &self.source
        else {
            return (0.0, vec![0.0; coords.len()]);
        };
        let n_atoms = validate_coords(coords);
        let mut energy = 0.0;
        let mut forces = vec![0.0; coords.len()];
        for idx in 0..atom_i.len() {
            let i = atom_i[idx];
            let j = atom_j[idx];
            debug_assert!(i < n_atoms && j < n_atoms);
            let dx = coords[j * 3] - coords[i * 3];
            let dy = coords[j * 3 + 1] - coords[i * 3 + 1];
            let dz = coords[j * 3 + 2] - coords[i * 3 + 2];
            let r2 = dx * dx + dy * dy + dz * dz;
            let ceps = 4.0 * epsilon[idx];
            let Some((e, f)) = self.pair_kernel_params(
                r2,
                [dx, dy, dz],
                sigma[idx],
                ceps,
                0.0,
                0.0,
                F::INFINITY,
                12,
                6,
            ) else {
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

    /// Fold a neighbour table with the parameters the atoms' types select.
    ///
    /// The only difference from [`fold_neighbors`](Self::fold_neighbors) is
    /// where `(σ, ε)` comes from: there, one style-level pair; here, the
    /// type-pair table. The geometry is already reduced either way — this
    /// kernel never learns whether a neighbour is an owned atom or a copy.
    fn fold_typed(&self, out: &mut [F], factor: &[F], pairs: &Neighbors) -> (F, Virial) {
        let n_pairs = pairs.query_point_indices().len();
        fold_chunks(out, n_pairs, |acc, rows| {
            self.fold_typed_rows(acc, factor, pairs, rows)
        })
    }

    fn fold_typed_rows(
        &self,
        out: &mut [F],
        factor: &[F],
        pairs: &Neighbors,
        rows: std::ops::Range<usize>,
    ) -> (F, Virial) {
        let mut virial = Virial::ZERO;
        let PairSource::Typed {
            type_id,
            ntypes,
            sigma,
            ceps,
            e0,
            f_rc,
            ..
        } = &self.source
        else {
            return (0.0, virial);
        };
        let Some(disp) = pairs.disp() else {
            return (0.0, virial);
        };
        let i = pairs.query_point_indices();
        let j = pairs.point_indices();
        let d2 = pairs.dist_sq();
        let mut energy = 0.0;
        for p in rows {
            let w = if factor.is_empty() { 1.0 } else { factor[p] };
            // Exactly zero skips: a bonded pair sits at bond length,
            // where this term is enormous.
            if w == 0.0 {
                continue;
            }
            let ia = i[p] as usize;
            let ja = j[p] as usize;
            let (Some(&ti), Some(&tj)) = (type_id.get(ia), type_id.get(ja)) else {
                // A pair naming an atom the type table does not cover cannot be
                // scored. Under a ghost régime that means the copies were not
                // gathered, which is a wiring fault and not a zero.
                debug_assert!(
                    false,
                    "pair ({ia}, {ja}) is outside the {} type ids",
                    type_id.len()
                );
                continue;
            };
            let t = ti as usize * ntypes + tj as usize;
            let d = [disp[[p, 0]], disp[[p, 1]], disp[[p, 2]]];
            let r2 = match d2 {
                Some(col) => col[p],
                None => d[0] * d[0] + d[1] * d[1] + d[2] * d[2],
            };
            let Some((e, f)) = self.pair_kernel_params(
                r2,
                d,
                sigma[t],
                ceps[t],
                e0[t],
                f_rc[t],
                self.cutoff2,
                self.n,
                self.m,
            ) else {
                continue;
            };
            let f = [w * f[0], w * f[1], w * f[2]];
            energy += w * e;
            virial.add_outer(f, d);
            let (bj, bi) = (3 * ja, 3 * ia);
            out[bj] += f[0];
            out[bj + 1] += f[1];
            out[bj + 2] += f[2];
            out[bi] -= f[0];
            out[bi + 1] -= f[1];
            out[bi + 2] -= f[2];
        }
        (energy, virial)
    }

    fn fold_neighbors(&self, out: &mut [F], factor: &[F], pairs: &Neighbors) -> (F, Virial) {
        let n_pairs = pairs.query_point_indices().len();
        fold_chunks(out, n_pairs, |acc, rows| {
            self.fold_neighbors_rows(acc, factor, pairs, rows)
        })
    }

    fn fold_neighbors_rows(
        &self,
        out: &mut [F],
        factor: &[F],
        pairs: &Neighbors,
        rows: std::ops::Range<usize>,
    ) -> (F, Virial) {
        let mut virial = Virial::ZERO;
        let Some(disp) = pairs.disp() else {
            return (0.0, virial);
        };
        let i = pairs.query_point_indices();
        let j = pairs.point_indices();
        let d2 = pairs.dist_sq();
        let mut energy = 0.0;
        for p in rows {
            let w = if factor.is_empty() { 1.0 } else { factor[p] };
            // Exactly zero skips: a bonded pair sits at bond length,
            // where this term is enormous.
            if w == 0.0 {
                continue;
            }
            let ia = i[p] as usize;
            let ja = j[p] as usize;
            let d = [disp[[p, 0]], disp[[p, 1]], disp[[p, 2]]];
            let r2 = match d2 {
                Some(col) => col[p],
                None => d[0] * d[0] + d[1] * d[1] + d[2] * d[2],
            };
            let Some((e, f)) = self.pair_kernel(r2, d) else {
                continue;
            };
            let f = [w * f[0], w * f[1], w * f[2]];
            energy += w * e;
            virial.add_outer(f, d);
            let bj = 3 * ja;
            let bi = 3 * ia;
            out[bj] += f[0];
            out[bj + 1] += f[1];
            out[bj + 2] += f[2];
            out[bi] -= f[0];
            out[bi + 1] -= f[1];
            out[bi + 2] -= f[2];
        }
        (energy, virial)
    }

    /// Compose `pairs_at` + table fold. Neighbour search stays the caller's.
    pub fn eval(
        &self,
        neighbors: &mut VerletSkin,
        pos: ArrayView2<'_, F>,
    ) -> Result<(F, Array2<F>), String> {
        let table = neighbors.pairs_at(pos).map_err(|e| e.to_string())?;
        let n = pos.nrows();
        let coords: Vec<F> = match pos.as_slice() {
            Some(s) => s.to_vec(),
            None => pos.iter().copied().collect(),
        };
        let (e, f) = self.calc_energy_forces_with_pairs(&coords, table);
        let forces = Array2::from_shape_vec((n, 3), f).map_err(|err| err.to_string())?;
        Ok((e, forces))
    }
}

impl PairPotential for LJCut {
    fn pair_energy(&self, r2: F, disp: [F; 3]) -> Option<F> {
        self.pair_kernel(r2, disp).map(|(e, _)| e)
    }
    fn pair_force(&self, r2: F, disp: [F; 3]) -> Option<[F; 3]> {
        self.pair_kernel(r2, disp).map(|(_, f)| f)
    }
    fn pair_eval(&self, r2: F, disp: [F; 3]) -> Option<(F, [F; 3])> {
        self.pair_kernel(r2, disp)
    }
}

impl Potential for LJCut {
    fn calc_energy_forces(&self, coords: &[F]) -> (F, Vec<F>) {
        match &self.source {
            PairSource::Compiled { .. } => self.fold_compiled(coords),
            // Both need a pair table nobody handed over.
            PairSource::Loop | PairSource::Typed { .. } => (0.0, vec![0.0; coords.len()]),
        }
    }

    fn calc_energy_forces_with_pairs(&self, coords: &[F], pairs: &Neighbors) -> (F, Vec<F>) {
        let (e, f, _) = self.calc_energy_forces_with_pairs_virial(coords, pairs);
        (e, f)
    }
}

impl PairDriven for LJCut {
    fn accumulate_pairs(
        &self,
        coords: &[F],
        pairs: &Neighbors,
        factor: &[F],
        out: &mut [F],
    ) -> (F, Option<Virial>) {
        match &self.source {
            // A compiled list is an intramolecular sum with no cutoff and no
            // periodicity; a virial from its raw differences would be a number
            // about nothing, and it cannot read a per-pair weight either.
            PairSource::Compiled { .. } => {
                debug_assert!(factor.is_empty());
                let (e, f) = self.fold_compiled(coords);
                for (acc, v) in out.iter_mut().zip(&f) {
                    *acc += v;
                }
                (e, None)
            }
            PairSource::Loop => {
                let (e, w) = self.fold_neighbors(out, factor, pairs);
                (e, Some(w))
            }
            PairSource::Typed { .. } => {
                let (e, w) = self.fold_typed(out, factor, pairs);
                (e, Some(w))
            }
        }
    }
    fn binds_a_fixed_pair_list(&self) -> bool {
        matches!(self.source, PairSource::Compiled { .. })
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
    fn gather_onto_copies(&mut self, owner: &[u32]) {
        let PairSource::Typed {
            type_id, n_owned, ..
        } = &mut self.source
        else {
            // Nothing per atom to extend.
            return;
        };
        gather_copies(type_id, *n_owned, owner);
    }
}

/// Construct a compiled [`LJCut`] from per-atom-type params + a neighbour list.
pub fn pair_lj_cut_ctor(
    style_params: &Params,
    type_params: &[(&str, &Params)],
    frame: &Frame,
) -> Result<Member, String> {
    let type_map: HashMap<&str, &Params> = type_params.iter().copied().collect();
    let scale_14 = style_params.get("lj14scale").unwrap_or(1.0) as F;
    // Absent `mixing` keeps Lorentz-Berthelot, the rule every reader that does
    // not declare one (AMBER prmtop, GAFF) actually means.
    let mixing = match style_params.get_str("mixing") {
        Some(name) => Mixing::parse(name).map_err(|e| format!("LJCut: {e}"))?,
        None => Mixing::Arithmetic,
    };

    let atoms = frame
        .get("atoms")
        .ok_or_else(|| "LJCut: frame missing \"atoms\" block".to_string())?;
    let atom_types = atoms
        .get_string("type")
        .ok_or_else(|| "LJCut: atoms block missing \"type\" column".to_string())?;
    let block = frame
        .get("pairs")
        .ok_or_else(|| "LJCut: frame missing \"pairs\" block".to_string())?;
    let i_col = block
        .get_uint("atomi")
        .ok_or_else(|| "LJCut: pairs block missing \"atomi\" column".to_string())?;
    let j_col = block
        .get_uint("atomj")
        .ok_or_else(|| "LJCut: pairs block missing \"atomj\" column".to_string())?;
    let is_14 = block.get_bool("is_14");

    let n = i_col.len();
    let mut atom_i = Vec::with_capacity(n);
    let mut atom_j = Vec::with_capacity(n);
    let mut eps_vec = Vec::with_capacity(n);
    let mut sig_vec = Vec::with_capacity(n);

    let per_atom = |t: &str| -> Result<(F, F), String> {
        let p = type_map
            .get(t)
            .ok_or_else(|| format!("LJCut: unknown atom type '{t}'"))?;
        let eps = p
            .get("epsilon")
            .ok_or_else(|| format!("LJCut type '{t}': missing 'epsilon'"))? as F;
        let sigma = p
            .get("sigma")
            .ok_or_else(|| format!("LJCut type '{t}': missing 'sigma'"))? as F;
        Ok((eps, sigma))
    };

    for idx in 0..n {
        let ij = per_atom(&atom_types[i_col[idx] as usize])?;
        let jj = per_atom(&atom_types[j_col[idx] as usize])?;
        let (mut eps, sigma) = mixing.combine(ij, jj);
        if is_14.is_some_and(|b| b[idx]) {
            eps *= scale_14;
        }
        atom_i.push(i_col[idx] as usize);
        atom_j.push(j_col[idx] as usize);
        eps_vec.push(eps);
        sig_vec.push(sigma);
    }

    Ok(Member::pair(LJCut::compiled(
        atom_i, atom_j, eps_vec, sig_vec,
    )))
}

/// Construct a neighbour-driven [`LJCut`] from per-atom parameters.
///
/// The counterpart of [`pair_lj_cut_ctor`]: the same force field, keyed on the atoms
/// instead of on a pair list, so it can answer for whatever pairs a neighbour
/// search turns up. It reads no `pairs` block — there is none to read when the
/// list is rebuilt every few steps.
pub fn pair_lj_cut_typed_ctor(
    style_params: &Params,
    type_params: &[(&str, &Params)],
    frame: &Frame,
) -> Result<Member, String> {
    let type_map: HashMap<&str, &Params> = type_params.iter().copied().collect();
    let mixing = match style_params.get_str("mixing") {
        Some(name) => Mixing::parse(name).map_err(|e| format!("LJCut: {e}"))?,
        None => Mixing::Arithmetic,
    };
    // Required, where the compiled form has no cutoff at all: an intramolecular
    // list is finite by construction, a periodic neighbour sum is not.
    let cutoff = style_params
        .get("cutoff")
        .ok_or_else(|| "LJCut: a neighbour-driven pair style must declare 'cutoff'".to_string())?
        as F;
    let n = style_params.get("n").unwrap_or(12.0).round() as i32;
    let m = style_params.get("m").unwrap_or(6.0).round() as i32;
    let shifted = style_params.get("shift").unwrap_or(0.0) != 0.0;

    let (type_id, labels) = atom_type_index(frame)?;
    let mut per_type = Vec::with_capacity(labels.len());
    for l in &labels {
        let p = type_map
            .get(l.as_str())
            .ok_or_else(|| format!("LJCut: unknown atom type '{l}'"))?;
        let eps = p
            .get("epsilon")
            .ok_or_else(|| format!("LJCut type '{l}': missing 'epsilon'"))? as F;
        let sigma = p
            .get("sigma")
            .ok_or_else(|| format!("LJCut type '{l}': missing 'sigma'"))? as F;
        per_type.push((eps, sigma));
    }
    Ok(Member::pair(LJCut::typed(
        type_id, &per_type, mixing, cutoff, n, m, shifted, false,
    )?))
}

#[cfg(test)]
mod tests {

    /// The fold gives the same number, bit for bit, however many threads ran.
    ///
    /// Above a pair-count threshold the fold is split across threads, and a
    /// sum of floats is not associative — so the split has to be by a fixed
    /// arithmetic boundary and the partials merged in chunk order. rayon's own
    /// adaptive split would reorder the additions and move the last bits of
    /// every energy, force and virial with them. This asserts the property
    /// rather than the intent: the table here is deliberately larger than the
    /// threshold, so the parallel path is the one being measured.
    #[cfg(feature = "rayon")]
    #[test]
    fn the_fold_does_not_depend_on_the_thread_count() {
        use molrs::spatial::neighbors::{NeighborList, NeighborPolicy, VerletSkin};
        use molrs::spatial::simbox::SimBox;

        // 10³ atoms at 3 Å with a 6 Å cutoff clears 8192 pairs comfortably.
        let side = 10_usize;
        let n = side * side * side;
        let l = side as F * 3.0;
        let bx = SimBox::cube(l, ndarray::array![0.0, 0.0, 0.0], [true; 3]).unwrap();
        let pos = ndarray::Array2::from_shape_fn((n, 3), |(a, k)| {
            let (i, j, m) = (a % side, (a / side) % side, a / (side * side));
            let base = [i, j, m][k] as F * 3.0;
            // A deterministic jitter, so no two separations coincide and a
            // reordering cannot be masked by equal terms.
            base + ((a * 7 + k * 13) as F * 0.618).fract() * 0.4
        });
        let mut skin = VerletSkin::new(
            NeighborList::new(6.0),
            6.0,
            NeighborPolicy {
                skin: 0.0,
                ..NeighborPolicy::default()
            },
            pos.view(),
            bx,
        )
        .unwrap();
        let table = skin.pairs_at(pos.view()).unwrap().clone();
        assert!(
            table.query_point_indices().len() > 8_192,
            "the fixture must cross the parallel threshold"
        );

        let kernel = LJCut::typed(
            (0..n).map(|i| (i % 2) as u32).collect(),
            &[(0.3, 3.4), (0.5, 3.0)],
            Mixing::Arithmetic,
            6.0,
            12,
            6,
            false,
            false,
        )
        .unwrap();
        let flat: Vec<F> = pos.iter().copied().collect();

        let run = |threads: usize| {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap();
            pool.install(|| {
                let mut out = vec![0.0; flat.len()];
                let (e, w) = kernel.accumulate_pairs(&flat, &table, &[], &mut out);
                (e, out, w.unwrap())
            })
        };

        let (e1, f1, w1) = run(1);
        assert!(e1.abs() > 1.0, "the fixture must interact");
        for threads in [2, 4, 8] {
            let (e, f, w) = run(threads);
            assert_eq!(e1.to_bits(), e.to_bits(), "{threads} threads: energy");
            for (k, (a, b)) in f1.iter().zip(&f).enumerate() {
                assert_eq!(a.to_bits(), b.to_bits(), "{threads} threads: force {k}");
            }
            for c in 0..6 {
                assert_eq!(
                    w1.components[c].to_bits(),
                    w.components[c].to_bits(),
                    "{threads} threads: virial {c}"
                );
            }
        }
    }

    use super::*;

    #[test]
    fn mie_c_is_four_for_12_6() {
        assert!((mie_c(12, 6) - 4.0).abs() < 1e-12);
    }

    #[test]
    fn loop_without_pairs_is_explicit_zero() {
        let lj = LJCut::lj126(1.0, 1.0, 2.5).unwrap();
        let coords: Vec<F> = vec![0.0, 0.0, 0.0, 1.1, 0.0, 0.0];
        let (e, f) = Potential::calc_energy_forces(&lj, &coords);
        assert_eq!(e, 0.0);
        assert!(f.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn compiled_newton_third_law() {
        let pot = LJCut::compiled(vec![0], vec![1], vec![0.5], vec![1.0]);
        let coords: Vec<F> = vec![0.0, 0.0, 0.0, 1.5, 0.3, 0.1];
        let (e, forces) = pot.calc_energy_forces(&coords);
        assert!(e.is_finite());
        for dim in 0..3 {
            assert!((forces[dim] + forces[3 + dim]).abs() < 1e-12);
        }
    }

    #[test]
    fn compiled_ignores_loop_pairs() {
        let pot = LJCut::compiled(vec![0], vec![1], vec![1.0], vec![1.0]);
        let coords: Vec<F> = vec![0.0, 0.0, 0.0, 2.0, 0.0, 0.0];
        let (e0, _) = pot.calc_energy_forces(&coords);
        let extra = Neighbors::from_pairs(
            [molrs::spatial::neighbors::NeighborPair {
                i: 0,
                j: 1,
                dist_sq: 1.0,
                disp: [1.0, 0.0, 0.0],
            }],
            molrs::spatial::neighbors::NeighborsStorage::FULL,
            molrs::spatial::neighbors::QueryMode::SelfQuery { num_points: 2 },
        );
        let (e1, _) = pot.calc_energy_forces_with_pairs(&coords, &extra);
        assert_eq!(e0, e1);
    }

    /// The typed table and the compiled list are two ways of finding the same
    /// number, and on the same pairs they must find it bit for bit.
    ///
    /// This is the whole claim of the typed form: nothing about the physics
    /// changed, only *when* a parameter is chosen. Anything else that moved
    /// would show up here, and it is checked on identical arithmetic — same
    /// mixing, same exponents, no cutoff, no shift — so bit equality is the
    /// right bar rather than a tolerance.
    ///
    /// Free boundary on purpose: this is about the lookup, not periodicity.
    #[test]
    fn a_typed_kernel_scores_a_pair_exactly_as_a_compiled_one() {
        use molrs::spatial::neighbors::{NeighborPair, NeighborsStorage, QueryMode};

        // Two types, deliberately unlike each other, so a table indexed the
        // wrong way round would give a different answer.
        let per_type = [(0.3_f64, 3.4_f64), (0.9, 2.6)];
        let type_id = vec![0_u32, 1, 0, 1];
        let coords: Vec<F> = vec![
            0.0, 0.0, 0.0, //
            3.1, 0.4, 0.2, //
            1.2, 2.9, 0.7, //
            4.0, 3.3, 1.1,
        ];
        let links = [(0_usize, 1_usize), (0, 2), (1, 3), (2, 3)];
        let mixing = Mixing::Arithmetic;

        let mut ai = Vec::new();
        let mut aj = Vec::new();
        let mut eps = Vec::new();
        let mut sig = Vec::new();
        let mut table = Vec::new();
        for &(i, j) in &links {
            let (e, sg) =
                mixing.combine(per_type[type_id[i] as usize], per_type[type_id[j] as usize]);
            ai.push(i);
            aj.push(j);
            eps.push(e);
            sig.push(sg);
            let d = [
                coords[j * 3] - coords[i * 3],
                coords[j * 3 + 1] - coords[i * 3 + 1],
                coords[j * 3 + 2] - coords[i * 3 + 2],
            ];
            table.push(NeighborPair {
                i: i as u32,
                j: j as u32,
                dist_sq: d[0] * d[0] + d[1] * d[1] + d[2] * d[2],
                disp: d,
            });
        }

        let compiled = LJCut::compiled(ai, aj, eps, sig);
        let typed =
            LJCut::typed(type_id, &per_type, mixing, F::INFINITY, 12, 6, false, false).unwrap();

        let neighbors = Neighbors::from_pairs(
            table,
            NeighborsStorage::FULL,
            QueryMode::SelfQuery { num_points: 4 },
        );

        let (e_c, f_c) = compiled.calc_energy_forces(&coords);
        let (e_t, f_t) = typed.calc_energy_forces_with_pairs(&coords, &neighbors);

        assert!(
            e_c.abs() > 1e-6,
            "the configuration must interact for this to assert anything; got {e_c}"
        );
        assert_eq!(e_c.to_bits(), e_t.to_bits(), "energy {e_c} vs {e_t}");
        assert_eq!(f_c.len(), f_t.len());
        for (c, (a, b)) in f_c.iter().zip(&f_t).enumerate() {
            assert_eq!(a.to_bits(), b.to_bits(), "force component {c}: {a} vs {b}");
        }

        crate::ff::potential::pair::testing::assert_virial_matches_forces(
            "lj/cut",
            &coords,
            typed.calc_energy_forces_with_pairs_virial(&coords, &neighbors),
        );
    }

    /// A typed kernel declares its cutoff, where a compiled one has none.
    ///
    /// `compiled` exists for an intramolecular list with no spatial cutoff at
    /// all — it passes `INFINITY` — and that is correct for what it is. A
    /// neighbour-driven kernel must not inherit it: every pair inside the
    /// cutoff interacts and nothing outside it does, which is what makes the
    /// sum finite in a periodic system.
    #[test]
    fn a_typed_kernel_stops_at_its_cutoff() {
        let typed = LJCut::typed(
            vec![0_u32, 0],
            &[(1.0, 1.0)],
            Mixing::Arithmetic,
            2.5,
            12,
            6,
            false,
            false,
        )
        .unwrap();
        assert!(typed.pair_eval(4.0, [2.0, 0.0, 0.0]).is_some());
        assert!(
            typed.pair_eval(9.0, [3.0, 0.0, 0.0]).is_none(),
            "3 Å is past the 2.5 Å cutoff"
        );
    }

    #[test]
    fn unshifted_energy_at_sigma_is_zero() {
        let lj = LJCut::new(1.5, 2.0, 5.0, 12, 6, false, false).unwrap();
        let (e, f) = lj.pair_eval(4.0, [2.0, 0.0, 0.0]).unwrap();
        assert!(e.abs() < 1e-12);
        assert!(f[0] > 0.0);
    }
}
