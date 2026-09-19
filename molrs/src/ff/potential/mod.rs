//! Potential energy evaluation traits and kernel registry.
//!
//! A [`Potential`] stores pre-resolved topology indices and parameters.
//! Callers pass only flat coordinates — no [`Frame`] in the hot loop.
//! Construction from a [`Frame`] happens once via [`ForceField::to_potentials`].

pub mod geometry;

pub mod angle;
pub mod bond;
pub mod dihedral;
pub mod improper;
pub mod kspace;
pub mod pair;
pub mod registry;
pub mod soft;

pub use registry::{
    KernelConstructor, KernelRegistry, ParamSource, lookup_kernel, lookup_param_source,
    register_kernel, register_kernel_with,
};

use std::borrow::Cow;
use std::collections::HashSet;

use ndarray::{Array1, Array2, ArrayView2};

use crate::ff::forcefield::{ForceField, Params, SpecialBonds};
use molrs::math::Virial;
use molrs::spatial::neighbors::Neighbors;
use molrs::store::block::Block;
use molrs::store::frame::Frame;
use molrs::system::bond_weights::BondDistanceWeights;
use molrs::types::{F, Idx};

/// Build the intramolecular non-bonded `pairs` block (`atomi`, `atomj`, `is_14`)
/// from a frame's bond/angle/dihedral topology: every `i < j` pair, excluding
/// 1-2 (bonded) and 1-3 (angle) pairs and flagging 1-4 (dihedral-end) pairs.
///
/// This is the single, force-field-agnostic neighbour list that
/// [`ForceField::to_potentials`] hands to every pair kernel — the same logic the
/// MMFF frame builder used to compute privately, lifted here so every force field
/// (GAFF/LAMMPS, OPLS, MMFF, …) shares one path. Per-pair scaling of the flagged
/// 1-4 pairs is applied by the pair kernels using the force field's special-bonds
/// weights, not baked into this list.
pub fn intramolecular_pairs(frame: &Frame) -> Block {
    let n_atoms = frame.get("atoms").and_then(|b| b.nrows()).unwrap_or(0);
    let excluded_12 = end_pairs(frame, "bonds", "atomi", "atomj");
    let excluded_13 = end_pairs(frame, "angles", "atomi", "atomk");
    let set_14 = end_pairs(frame, "dihedrals", "atomi", "atoml");

    let mut pi: Vec<Idx> = Vec::new();
    let mut pj: Vec<Idx> = Vec::new();
    let mut p14: Vec<bool> = Vec::new();
    for a in 0..n_atoms {
        for b in (a + 1)..n_atoms {
            let key = (a, b);
            if excluded_12.contains(&key) || excluded_13.contains(&key) {
                continue;
            }
            pi.push(a as Idx);
            pj.push(b as Idx);
            p14.push(set_14.contains(&key));
        }
    }

    let mut pairs = Block::new();
    if !pi.is_empty() {
        pairs
            .insert("atomi", Array1::from_vec(pi).into_dyn())
            .expect("fresh pairs block");
        pairs
            .insert("atomj", Array1::from_vec(pj).into_dyn())
            .expect("fresh pairs block");
        pairs
            .insert("is_14", Array1::from_vec(p14).into_dyn())
            .expect("fresh pairs block");
    }
    pairs
}

/// Sorted `(lo, hi)` end-atom pairs of a topology block (bond ends, angle i–k,
/// dihedral i–l). A missing block or column yields an empty set.
fn end_pairs(frame: &Frame, block: &str, col_a: &str, col_b: &str) -> HashSet<(usize, usize)> {
    let Some(b) = frame.get(block) else {
        return HashSet::new();
    };
    let (Some(a_col), Some(b_col)) = (b.get_uint(col_a), b.get_uint(col_b)) else {
        return HashSet::new();
    };
    a_col
        .iter()
        .zip(b_col.iter())
        .map(|(&i, &j)| {
            let (i, j) = (i as usize, j as usize);
            if i < j { (i, j) } else { (j, i) }
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Potential trait
// ---------------------------------------------------------------------------

/// Interface for computing potential energy and forces.
///
/// A `Potential` is **molecule-bound**: its per-element parameters are expanded
/// against the molecule's topology once at [`ForceField::to_potentials`] (string
/// type labels resolved to per-bond/angle/… arrays). Evaluation therefore takes
/// only coordinates — there is no per-call topology resolution.
///
/// Implementors provide [`calc_energy_forces`](Potential::calc_energy_forces)
/// (both in one pass, avoiding redundant geometry); [`calc_energy`] and
/// [`calc_forces`] default to it.
///
/// The geometry optimizer ([`crate::optimize::LBFGS`]) depends on this trait —
/// not the other way around.
///
/// [`calc_energy`]: Potential::calc_energy
/// [`calc_forces`]: Potential::calc_forces
pub trait Potential: Send + Sync {
    /// Compute energy and forces (= -gradient) in one pass.
    /// Returns `(energy, forces)` where forces has length `coords.len()`.
    fn calc_energy_forces(&self, coords: &[F]) -> (F, Vec<F>);

    /// Compute total potential energy (kcal/mol).
    fn calc_energy(&self, coords: &[F]) -> F {
        self.calc_energy_forces(coords).0
    }

    /// Compute forces (= -gradient), a length-3N vector.
    fn calc_forces(&self, coords: &[F]) -> Vec<F> {
        self.calc_energy_forces(coords).1
    }

    /// Evaluate with a per-step pair table the loop computed once and shares
    /// with every pair potential. Default ignores `pairs` and calls
    /// [`calc_energy_forces`](Potential::calc_energy_forces).
    fn calc_energy_forces_with_pairs(&self, coords: &[F], pairs: &Neighbors) -> (F, Vec<F>) {
        let _ = pairs;
        self.calc_energy_forces(coords)
    }

    /// The atom indices this kernel resolved at construction, `(n_terms,
    /// arity)` — arity 2 for a bond, 3 for an angle, 4 for a dihedral or an
    /// improper. `None` for a kernel whose state is not a fixed index list.
    ///
    /// A kernel that answers this is declaring that *which atoms* is separable
    /// from *what the parameters are*: row `r` here belongs with row `r` of
    /// its parameters, and a caller may hand back a different table naming
    /// different atoms for the same rows. That is what lets a periodic régime
    /// point a bond at a copy without the kernel learning that copies exist.
    fn terms(&self) -> Option<Array2<u32>> {
        None
    }

    /// Evaluate with `terms` in place of the indices resolved at construction.
    ///
    /// `terms` must have the shape [`terms`](Potential::terms) returned: the
    /// row *set* is fixed — it is the terms the force field declared — and
    /// only which atoms each row names may differ. Default ignores it and
    /// calls [`calc_energy_forces`](Potential::calc_energy_forces), which is
    /// right for a kernel that holds no indices.
    fn calc_energy_forces_with_terms(
        &self,
        coords: &[F],
        terms: ArrayView2<'_, u32>,
    ) -> (F, Vec<F>) {
        let _ = terms;
        self.calc_energy_forces(coords)
    }

    /// Energy, forces, and the virial `Σ f ⊗ r`, over a pair table.
    ///
    /// The virial cannot be recovered from the forces afterwards. Under
    /// periodic boundaries `Σ_a f_a ⊗ x_a` over the stored coordinates depends
    /// on where the cell's origin happens to be, and a sum over separations
    /// does not — so a kernel that can tally one tallies it *here*, in the same
    /// loop that made the forces, where both terms of each pair are still in
    /// hand.
    ///
    /// Default: evaluate normally and report none. `None` is not zero: a
    /// pressure computed from a fabricated zero is wrong and looks entirely
    /// plausible.
    fn calc_energy_forces_with_pairs_virial(
        &self,
        coords: &[F],
        pairs: &Neighbors,
    ) -> (F, Vec<F>, Option<Virial>) {
        let (e, f) = self.calc_energy_forces_with_pairs(coords, pairs);
        (e, f, None)
    }

    /// Add this kernel's contribution over a pair table into `out`, scaling
    /// each pair.
    ///
    /// Three things at once, and each of them is the reason the other two are
    /// here:
    ///
    /// * **`factor`** is one weight per pair, aligned with the table's rows, or
    ///   empty meaning all ones. A weight of exactly zero *skips* the pair
    ///   rather than multiplying it, because a bonded pair sits at bond length
    ///   where a repulsive term is enormous. Carrying the weights per pair is
    ///   what lets a caller stop rebuilding the pair table once per distinct
    ///   weight.
    /// * **`out`** is the caller's accumulator, `3 · n_atoms` long, added into
    ///   rather than returned. A provider summing several members then owns one
    ///   buffer for the whole step instead of one allocation per member per
    ///   step — and a thread can be handed a slice of something the caller owns,
    ///   where it cannot be handed a slice of something the callee allocates.
    /// * **the virial** comes back with the energy, from the same loop, for the
    ///   reason [`calc_energy_forces_with_pairs_virial`](Potential::calc_energy_forces_with_pairs_virial)
    ///   gives.
    ///
    /// Default: evaluate the ordinary way and add, **ignoring `factor`** —
    /// which is correct for a potential that does not sum over the pair table
    /// at all (an external field, a restraint, a constant force), and wrong for
    /// one that does.
    ///
    /// **A kernel that sums over the pair table must override this.** Nothing
    /// can check it: the default cannot tell a potential that has no pairs to
    /// weight from one that has and forgot to say so, and every in-tree pair
    /// kernel overrides. A third-party one that does not would silently lose
    /// its force field's exclusions.
    fn accumulate_pairs(
        &self,
        coords: &[F],
        pairs: &Neighbors,
        factor: &[F],
        out: &mut [F],
    ) -> (F, Option<Virial>) {
        let _ = factor;
        let (e, f, w) = self.calc_energy_forces_with_pairs_virial(coords, pairs);
        for (acc, v) in out.iter_mut().zip(&f) {
            *acc += v;
        }
        (e, w)
    }

    /// Whether this kernel's parameters are bound to one fixed pair list.
    ///
    /// Such a kernel answers for **that** list and no other. Handed a
    /// neighbour table it does not read it — it returns the frozen sum — and
    /// that is not a cheaper route to the same number, it is a different
    /// question silently unanswered. Worse, the caller goes on maintaining and
    /// reporting a live list that provably does not enter the answer.
    ///
    /// A periodic force path refuses such a member at construction. Default
    /// `false`: a kernel that reads geometry, or one keyed on the atoms, is
    /// bound to nothing.
    fn binds_a_fixed_pair_list(&self) -> bool {
        false
    }

    /// Extend per-atom state onto periodic copies.
    ///
    /// `owner[g]` is the atom copy `g` is a copy of, and the copies occupy
    /// rows `n_owned + g` of the coordinates a periodic régime evaluates over.
    /// A kernel that keeps anything per atom — a charge, a type index, a
    /// parameter record — has to cover those rows, because a pair table over
    /// copies will name them.
    ///
    /// Idempotent: the owned entries are the truth and the copies are derived
    /// from them, so calling this twice with the same map is the same as
    /// calling it once. Default does nothing, which is right for a kernel that
    /// keeps nothing per atom.
    fn gather_onto_copies(&mut self, owner: &[u32]) {
        let _ = owner;
    }
}

impl Potential for Box<dyn Potential> {
    fn calc_energy_forces(&self, coords: &[F]) -> (F, Vec<F>) {
        (**self).calc_energy_forces(coords)
    }

    fn calc_energy_forces_with_pairs(&self, coords: &[F], pairs: &Neighbors) -> (F, Vec<F>) {
        (**self).calc_energy_forces_with_pairs(coords, pairs)
    }

    fn terms(&self) -> Option<Array2<u32>> {
        (**self).terms()
    }

    fn calc_energy_forces_with_terms(
        &self,
        coords: &[F],
        terms: ArrayView2<'_, u32>,
    ) -> (F, Vec<F>) {
        (**self).calc_energy_forces_with_terms(coords, terms)
    }

    fn calc_energy_forces_with_pairs_virial(
        &self,
        coords: &[F],
        pairs: &Neighbors,
    ) -> (F, Vec<F>, Option<Virial>) {
        (**self).calc_energy_forces_with_pairs_virial(coords, pairs)
    }

    fn accumulate_pairs(
        &self,
        coords: &[F],
        pairs: &Neighbors,
        factor: &[F],
        out: &mut [F],
    ) -> (F, Option<Virial>) {
        (**self).accumulate_pairs(coords, pairs, factor, out)
    }

    fn binds_a_fixed_pair_list(&self) -> bool {
        (**self).binds_a_fixed_pair_list()
    }

    fn gather_onto_copies(&mut self, owner: &[u32]) {
        (**self).gather_onto_copies(owner)
    }
}

// ---------------------------------------------------------------------------
// Potentials collection
// ---------------------------------------------------------------------------

/// A kernel built for a neighbour-driven evaluation, and which of a force
/// field's special-bonds weight sets scales it. `None` for a bonded kernel:
/// it *is* the bonded interaction, not a scaled copy of one.
pub type TypedKernel = (Box<dyn Potential>, Option<registry::SpecialClass>);

/// One member of a neighbour-driven evaluation: the kernel, and the
/// bond-distance weights its non-bonded term takes.
///
/// The weights travel with the member because a force field may scale close
/// van-der-Waals and electrostatic neighbours differently, and in molrs those
/// are separate kernels.
pub type TypedMember = (Box<dyn Potential>, Option<BondDistanceWeights>);

/// Rebuild the copies' entries of a per-atom vector from their owners'.
///
/// `v` holds one entry per atom for the first `n_owned`, then one per copy.
/// The owned entries are the truth; the copies are derived, so this truncates
/// before extending and is therefore idempotent — which matters, because a
/// periodic régime calls it again every time the copy list is rebuilt and has
/// no way to know whether the last call took.
pub(crate) fn gather_copies<T: Clone>(v: &mut Vec<T>, n_owned: usize, owner: &[u32]) {
    debug_assert!(
        v.len() >= n_owned,
        "a per-atom vector shorter than the atoms it describes"
    );
    v.truncate(n_owned);
    v.reserve(owner.len());
    for &o in owner {
        debug_assert!(
            (o as usize) < n_owned,
            "a copy names an owner that is not an atom"
        );
        let x = v[o as usize].clone();
        v.push(x);
    }
}

/// Aggregates multiple potentials; energy/forces are summed.
///
/// Every member evaluates in one shared unit system. Unit conversion is the
/// caller's job (`UnitPreset`), never this type's.
pub struct Potentials {
    inner: Vec<Box<dyn Potential>>,
    /// Which members hold atom indices, recorded when each was pushed.
    ///
    /// `terms()` *allocates* the table it answers with, so asking it "is this a
    /// bonded term?" once per member per step is an allocation per bonded
    /// member per step. The question is settled when the member is. That it
    /// had to be asked at all is a symptom: a member's *role* is being
    /// recovered at runtime from a method whose job is to return data.
    holds_indices: Vec<bool>,
    /// Number of atoms the kernels were compiled against (`coords.len() / 3`).
    /// `0` when unknown (e.g. built incrementally via [`Potentials::push`]).
    n_atoms: usize,
}

impl std::fmt::Debug for Potentials {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Potentials")
            .field("len", &self.inner.len())
            .finish()
    }
}

impl Potentials {
    pub fn new() -> Self {
        Self {
            inner: Vec::new(),
            holds_indices: Vec::new(),
            n_atoms: 0,
        }
    }

    pub fn push(&mut self, pot: Box<dyn Potential>) {
        self.holds_indices.push(pot.terms().is_some());
        self.inner.push(pot);
    }

    pub fn len(&self) -> usize {
        self.inner.len()
    }

    /// Number of atoms the kernels were compiled against, i.e. the expected
    /// `coords.len() / 3`. Returns `0` if unknown (built via [`push`]).
    ///
    /// [`push`]: Potentials::push
    pub fn n_atoms(&self) -> usize {
        self.n_atoms
    }

    /// Record the compiled atom count (used by [`ForceField::to_potentials`]).
    pub fn set_n_atoms(&mut self, n_atoms: usize) {
        self.n_atoms = n_atoms;
    }

    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    /// The members, in the order the force field's styles produced them.
    ///
    /// A caller that needs to treat members differently — a periodic régime
    /// rebinding the bonded terms onto copies while the nonbonded one reads a
    /// neighbour table — needs them one at a time, because the index table a
    /// member wants is the member's own. Summing them all is what
    /// [`calc_energy_forces`](Self::calc_energy_forces) is for.
    pub fn members(&self) -> &[Box<dyn Potential>] {
        &self.inner
    }

    /// Give up the members, for a caller that wants to own them individually.
    pub fn into_members(self) -> Vec<Box<dyn Potential>> {
        self.inner
    }

    /// The members, mutably — for the per-atom gather a periodic régime runs.
    pub fn members_mut(&mut self) -> &mut [Box<dyn Potential>] {
        &mut self.inner
    }

    /// Same as [`calc_energy_forces`](Self::calc_energy_forces) but forwards
    /// one shared pair table to every member.
    pub fn calc_energy_forces_with_pairs(&self, coords: &[F], pairs: &Neighbors) -> (F, Vec<F>) {
        let n = coords.len();
        let mut total_e: F = 0.0;
        let mut total_f = vec![0.0; n];
        for p in &self.inner {
            let (e, f) = p.calc_energy_forces_with_pairs(coords, pairs);
            total_e += e;
            for (t, fi) in total_f.iter_mut().zip(f.iter()) {
                *t += fi;
            }
        }
        (total_e, total_f)
    }

    /// Compute total energy and forces in one pass over all potentials.
    pub fn calc_energy_forces(&self, coords: &[F]) -> (F, Vec<F>) {
        let n = coords.len();
        let mut total_e: F = 0.0;
        let mut total_f = vec![0.0; n];

        for p in &self.inner {
            let (e, f) = p.calc_energy_forces(coords);
            total_e += e;
            for (t, fi) in total_f.iter_mut().zip(f.iter()) {
                *t += fi;
            }
        }

        (total_e, total_f)
    }

    /// Total potential energy (kcal/mol).
    pub fn calc_energy(&self, coords: &[F]) -> F {
        self.calc_energy_forces(coords).0
    }

    /// Total forces (= -gradient), length 3N.
    pub fn calc_forces(&self, coords: &[F]) -> Vec<F> {
        self.calc_energy_forces(coords).1
    }
}

impl Default for Potentials {
    fn default() -> Self {
        Self::new()
    }
}

/// Make the aggregate usable wherever a single [`Potential`] is expected (the
/// geometry optimizer in [`crate::optimize`], an MD integrator, another
/// [`Potentials`]), forwarding to the summed evaluation over all kernels.
impl Potential for Potentials {
    fn calc_energy_forces(&self, coords: &[F]) -> (F, Vec<F>) {
        Potentials::calc_energy_forces(self, coords)
    }

    fn calc_energy_forces_with_pairs(&self, coords: &[F], pairs: &Neighbors) -> (F, Vec<F>) {
        Potentials::calc_energy_forces_with_pairs(self, coords, pairs)
    }

    /// The members' virials, summed — and `None` the moment one of them
    /// declines to report.
    ///
    /// Taking the default here would have been worse than wrong: an aggregate
    /// whose members all tally would report `None` and look like a kernel that
    /// simply cannot, so every pressure computed through a force field would
    /// be unavailable for no stated reason.
    fn calc_energy_forces_with_pairs_virial(
        &self,
        coords: &[F],
        pairs: &Neighbors,
    ) -> (F, Vec<F>, Option<Virial>) {
        let mut total_e: F = 0.0;
        let mut total_f = vec![0.0; coords.len()];
        let mut total_w = Some(Virial::ZERO);
        for p in &self.inner {
            let (e, f, w) = p.calc_energy_forces_with_pairs_virial(coords, pairs);
            total_e += e;
            for (t, fi) in total_f.iter_mut().zip(f.iter()) {
                *t += fi;
            }
            match (total_w.as_mut(), w) {
                (Some(acc), Some(part)) => {
                    for c in 0..6 {
                        acc.components[c] += part.components[c];
                    }
                }
                (_, None) => total_w = None,
                (None, _) => {}
            }
        }
        (total_e, total_f, total_w)
    }

    /// Every member gathers. An aggregate that swallowed this would leave a
    /// typed kernel's per-atom tables covering the owned atoms only, while the
    /// pair table it is handed names copies.
    fn gather_onto_copies(&mut self, owner: &[u32]) {
        for p in &mut self.inner {
            p.gather_onto_copies(owner);
        }
    }

    /// An aggregate holds no single index table — its members' are of
    /// different arities and different row sets — so it reports none, and the
    /// default [`calc_energy_forces_with_terms`](Potential::calc_energy_forces_with_terms)
    /// is then never asked for one. A caller that needs the bonded terms
    /// rebound takes the members individually, which is what
    /// [`into_members`](Potentials::into_members) is for.
    fn terms(&self) -> Option<Array2<u32>> {
        None
    }

    /// Every member accumulates into the same buffer, and one member that
    /// cannot report a virial makes the aggregate's `None`.
    fn accumulate_pairs(
        &self,
        coords: &[F],
        pairs: &Neighbors,
        factor: &[F],
        out: &mut [F],
    ) -> (F, Option<Virial>) {
        let mut total_e: F = 0.0;
        let mut total_w = Some(Virial::ZERO);
        for (k, p) in self.inner.iter().enumerate() {
            // A per-pair weight belongs to a member that reads the pair table.
            // A member holding atom indices is a bonded term: it ignores the
            // table, and its own interaction is the thing the weights exist to
            // avoid double-counting — scaling it would be scaling the wrong
            // side of that.
            let mine: &[F] = if self.holds_indices[k] { &[] } else { factor };
            let (e, w) = p.accumulate_pairs(coords, pairs, mine, out);
            total_e += e;
            match (total_w.as_mut(), w) {
                (Some(acc), Some(part)) => {
                    for c in 0..6 {
                        acc.components[c] += part.components[c];
                    }
                }
                (_, None) => total_w = None,
                (None, _) => {}
            }
        }
        (total_e, total_w)
    }

    /// True if *any* member is. One compiled kernel is enough to make the
    /// aggregate's answer independent of the table it is handed.
    fn binds_a_fixed_pair_list(&self) -> bool {
        self.inner.iter().any(|p| p.binds_a_fixed_pair_list())
    }
}

// ---------------------------------------------------------------------------
// Style -> Potential construction (OOP — replaces the kernel-registry free-fn map)
// ---------------------------------------------------------------------------

impl crate::ff::forcefield::Style {
    /// Build this style's kernel for a **neighbour-driven** evaluation, and
    /// say which special-bonds weights scale it.
    ///
    /// The counterpart of [`to_potential`](Self::to_potential). A bonded style
    /// is built identically — it reads indices, and a neighbour table does not
    /// concern it. A pair style is built in its typed form, which reads no
    /// `pairs` block: there is none to read when the list is rebuilt every few
    /// steps, and a kernel whose parameters were resolved against an older one
    /// would be naming different atoms.
    ///
    /// A pair style with no typed form is an [`Err`], not a fallback to the
    /// compiled one. Falling back would hand back a kernel whose parameters
    /// belong to a pair list nobody is evaluating, and it would answer.
    pub fn to_typed_potential(&self, frame: &Frame) -> Result<Option<TypedKernel>, String> {
        let category = self.category();
        if category == "atom" {
            return Ok(None);
        }
        if category != "pair" {
            // Bonded styles are unchanged, and take no special-bonds weight:
            // the term *is* the bonded interaction, not a scaled copy of it.
            // `special_bonds` is irrelevant to them, so a default is honest.
            return Ok(self
                .to_potential(frame, &SpecialBonds::default())?
                .map(|p| (p, None)));
        }
        // No `pairs` gate: a typed pair kernel is built from the atoms, and a
        // frame with atoms always has those.
        let type_params = self.defs.collect_type_params();
        let param_source =
            registry::lookup_param_source(category, &self.name).unwrap_or(ParamSource::TypeRows);
        if type_params.is_empty() && param_source == ParamSource::TypeRows {
            return Err(format!(
                "Style '{}' ({}) has no type definitions",
                self.name, category
            ));
        }
        let type_refs: Vec<(&str, &Params)> = type_params
            .iter()
            .map(|(name, params)| (name.as_str(), params))
            .collect();
        let (ctor, special) =
            registry::lookup_typed_kernel(category, &self.name).ok_or_else(|| {
                format!(
                    "pair style '{}' has no neighbour-driven form, so it cannot be \
                     evaluated over a neighbour list; its compiled form answers only \
                     for the pair list it was built from",
                    self.name
                )
            })?;
        let pot = ctor(&self.params, &type_refs, frame)?;
        Ok(Some((pot, Some(special))))
    }

    /// Build this style's molecule-bound [`Potential`] by **expanding** its type
    /// parameters against `frame`'s topology — each bond/angle/… row's string
    /// type label is resolved to its parameters and stored as per-element
    /// arrays, so the resulting potential evaluates from coordinates alone.
    ///
    /// Returns `Ok(None)` for a style that carries no pairwise kernel (an atom
    /// style — types/charges only), `Err` for an unknown `(category, name)`.
    ///
    /// The `(category, name)` → constructor mapping lives in the [`registry`]; a
    /// new potential is added by registering its kernel, not by editing this
    /// dispatch.
    pub fn to_potential(
        &self,
        frame: &Frame,
        special_bonds: &SpecialBonds,
    ) -> Result<Option<Box<dyn Potential>>, String> {
        let category = self.category();
        if category == "atom" {
            return Ok(None);
        }
        // A style contributes nothing when the molecule carries no topology of its
        // kind: a bonded style with no bonds/angles/dihedrals/impropers, or a pair
        // style when the neighbour list is empty (e.g. methane, whose every atom
        // pair is 1-2 or 1-3 excluded). Skip it rather than letting the kernel ctor
        // fault on the absent/empty block.
        let topo_block = match category {
            "bond" => Some("bonds"),
            "angle" => Some("angles"),
            "dihedral" => Some("dihedrals"),
            "improper" => Some("impropers"),
            "pair" => Some("pairs"),
            _ => None,
        };
        if let Some(block_name) = topo_block {
            let rows = frame.get(block_name).and_then(|b| b.nrows()).unwrap_or(0);
            if rows == 0 {
                return Ok(None);
            }
        }
        let type_params = self.defs.collect_type_params();
        // A style whose kernel resolves its parameters from type rows
        // (`ParamSource::TypeRows`) can resolve nothing without them — so no rows
        // is an error, not a silently-zero potential. A `PerInstance` style
        // (MMFF's bonded terms, `coul/cut`, `pme`) reads its numbers from Frame
        // columns the typifier baked and ignores `tp` entirely, so zero rows is
        // its *normal* state. Asking the registry which one this is replaces the
        // old blanket `category != "pair"` escape hatch — the hatch that let MMFF
        // register as table-driven and then be fed 4,065 rows of XML no code reads.
        //
        // The registry is the authority because it is where the kernel is declared;
        // an unregistered style falls through to `TypeRows` here and then fails on
        // the kernel lookup below with a more specific message.
        let param_source =
            registry::lookup_param_source(category, &self.name).unwrap_or(ParamSource::TypeRows);
        if type_params.is_empty() && param_source == ParamSource::TypeRows {
            return Err(format!(
                "Style '{}' ({}) has no type definitions",
                self.name, category
            ));
        }
        let type_refs: Vec<(&str, &Params)> = type_params
            .iter()
            .map(|(name, params)| (name.as_str(), params))
            .collect();
        let ctor = registry::lookup_kernel(category, &self.name).ok_or_else(|| {
            format!(
                "no kernel for style category '{}' name '{}'",
                category, self.name
            )
        })?;
        // Project the ForceField's `special_bonds` 1-4 weights into the params the
        // pair kernel reads (`lj14scale` / `coulomb14scale`), so the kernel scales
        // 1-4-flagged pairs without the registry signature carrying special_bonds.
        // Bonded kernels see their params unchanged.
        let params: Cow<Params> = if category == "pair" {
            let mut p = self.params.clone();
            p.set("lj14scale", special_bonds.lj_14());
            p.set("coulomb14scale", special_bonds.coul_14());
            Cow::Owned(p)
        } else {
            Cow::Borrowed(&self.params)
        };
        let pot = ctor(&params, &type_refs, frame)?;
        Ok(Some(pot))
    }
}

// ---------------------------------------------------------------------------
// Frame helpers
// ---------------------------------------------------------------------------

/// Extract flat coordinate vector from Frame's `"atoms"` block.
///
/// Reads `"x"`, `"y"`, `"z"` float columns.
/// Returns `[x0,y0,z0, x1,y1,z1, ...]` as `Vec<F>`.
pub fn extract_coords(frame: &Frame) -> Result<Vec<F>, String> {
    let atoms = frame
        .get("atoms")
        .ok_or_else(|| "Frame has no \"atoms\" block".to_string())?;

    let (x, y, z) = (
        atoms.get_float("x"),
        atoms.get_float("y"),
        atoms.get_float("z"),
    );
    let (Some(x), Some(y), Some(z)) = (x, y, z) else {
        return Err("atoms block missing x/y/z float columns".into());
    };

    let xs: Vec<F> = x.iter().copied().collect();
    let ys: Vec<F> = y.iter().copied().collect();
    let zs: Vec<F> = z.iter().copied().collect();

    let n = xs.len();
    if ys.len() != n || zs.len() != n {
        return Err("atoms x/y/z columns have mismatched lengths".into());
    }

    let mut coords = Vec::with_capacity(n * 3);
    for i in 0..n {
        coords.push(xs[i]);
        coords.push(ys[i]);
        coords.push(zs[i]);
    }
    Ok(coords)
}

/// Write a flat `[x0,y0,z0, …]` coordinate vector into the Frame's `"atoms"` block.
pub fn write_coords(frame: &mut Frame, coords: &[F]) -> Result<(), String> {
    let n = coords.len() / 3;
    if coords.len() != n * 3 {
        return Err(format!(
            "coords length {} is not a multiple of 3",
            coords.len()
        ));
    }
    let atoms = frame
        .get_mut("atoms")
        .ok_or_else(|| "Frame has no \"atoms\" block".to_string())?;
    let x = atoms
        .get_float_mut("x")
        .ok_or_else(|| "atoms block missing float column x".to_string())?;
    if x.len() != n {
        return Err(format!("coords atom count {n} != frame atoms {}", x.len()));
    }
    for i in 0..n {
        x[[i]] = coords[3 * i];
    }
    let y = atoms
        .get_float_mut("y")
        .ok_or_else(|| "atoms block missing float column y".to_string())?;
    for i in 0..n {
        y[[i]] = coords[3 * i + 1];
    }
    let z = atoms
        .get_float_mut("z")
        .ok_or_else(|| "atoms block missing float column z".to_string())?;
    for i in 0..n {
        z[[i]] = coords[3 * i + 2];
    }
    Ok(())
}

impl ForceField {
    /// Build the members of a **neighbour-driven** force evaluation, each with
    /// the bond-distance weights its non-bonded term takes.
    ///
    /// The counterpart of [`to_potentials`](Self::to_potentials), and what
    /// periodic MD needs. That one resolves every pair style against the
    /// frame's `pairs` block — a fixed list, finite by construction and with no
    /// spatial cutoff, which is right for a free-boundary molecule and wrong
    /// for a periodic system. This one resolves them against the **atoms**, so
    /// the kernels can answer for whatever pairs a neighbour search turns up,
    /// and reads no `pairs` block at all.
    ///
    /// The weights come back per member rather than once, because a force field
    /// may scale close van-der-Waals and electrostatic neighbours differently —
    /// Amber uses `1/2` and `1/1.2` — and in molrs those are separate kernels.
    /// A bonded member takes `None`: it *is* the bonded interaction, not a
    /// scaled copy of it.
    ///
    /// # Why the weights matter here and not there
    ///
    /// A compiled list carries the exclusions by leaving the excluded rows out
    /// and baking the 1-4 factor into the parameters. A neighbour table has no
    /// such memory — it finds every pair inside the cutoff, bonded or not — so
    /// without the weights a bonded pair is counted twice: once by the bond
    /// term and once at full non-bonded strength, at bond length.
    pub fn to_typed_potentials(&self, frame: &Frame) -> Result<Vec<TypedMember>, String> {
        let mut out = Vec::new();
        for style in self.styles() {
            // A bonded style contributes nothing when the molecule carries no
            // topology of its kind. A pair style is never skipped: which pairs
            // exist is the neighbour search's answer, not the frame's.
            let block = match style.category() {
                "bond" => Some("bonds"),
                "angle" => Some("angles"),
                "dihedral" => Some("dihedrals"),
                "improper" => Some("impropers"),
                _ => None,
            };
            if let Some(b) = block
                && frame.get(b).is_none()
            {
                continue;
            }
            if let Some((pot, special)) = style.to_typed_potential(frame)? {
                let weights = special.map(|c| match c {
                    registry::SpecialClass::Vdw => self.special_bonds().lj_weights(),
                    registry::SpecialClass::Coulomb => self.special_bonds().coul_weights(),
                });
                out.push((pot, weights));
            }
        }
        Ok(out)
    }

    /// Build evaluable [`Potentials`] by expanding every style against a
    /// typed [`Frame`].
    ///
    /// Each style's `to_potential` resolves its string type labels to per-element
    /// parameter arrays (see [`Style::to_potential`](crate::ff::forcefield::Style::to_potential)),
    /// so the resulting potentials are **molecule-bound**: they retain no Frame
    /// and evaluate from coordinates alone. Styles with no kernel (atom styles)
    /// are skipped. This is the molpy-style `ForceField → Potentials` conversion;
    /// there is no separate "compile" step.
    pub fn to_potentials(&self, frame: &Frame) -> Result<Potentials, String> {
        let mut pots = Potentials::new();
        for style in self.styles() {
            // A style whose topology block is entirely absent contributes nothing
            // (the molecule simply has no bonds/angles/… of that kind) — skip it,
            // rather than error. A *present* block with an unknown type label is a
            // real error and still propagates from the kernel constructor.
            let block = match style.category() {
                "bond" => Some("bonds"),
                "angle" => Some("angles"),
                "dihedral" => Some("dihedrals"),
                "improper" => Some("impropers"),
                "pair" => Some("pairs"),
                _ => None,
            };
            if let Some(b) = block
                && frame.get(b).is_none()
            {
                continue;
            }
            if let Some(pot) = style.to_potential(frame, self.special_bonds())? {
                pots.push(pot);
            }
        }
        // Record the atom count so callers (e.g. the geometry optimizer's batch
        // path) can validate coordinate shapes against this topology.
        pots.set_n_atoms(frame.get("atoms").and_then(|b| b.nrows()).unwrap_or(0));
        Ok(pots)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use molrs::store::block::Block;
    use molrs::types::Idx;
    use ndarray::Array1;

    struct DummyPotential {
        value: F,
    }

    impl Potential for DummyPotential {
        fn calc_energy_forces(&self, coords: &[F]) -> (F, Vec<F>) {
            (self.value, vec![self.value; coords.len()])
        }
    }

    fn make_atoms_only_frame() -> Frame {
        let mut frame = Frame::new();
        let mut atoms = Block::new();
        atoms
            .insert("x", Array1::from_vec(vec![0.0 as F, 2.0 as F]).into_dyn())
            .unwrap();
        atoms
            .insert("y", Array1::from_vec(vec![0.0 as F, 0.0 as F]).into_dyn())
            .unwrap();
        atoms
            .insert("z", Array1::from_vec(vec![0.0 as F, 0.0 as F]).into_dyn())
            .unwrap();
        frame.insert("atoms", atoms);
        frame
    }

    /// A four-atom frame carrying one term of every bonded arity, so a single
    /// force field produces a bond, an angle, a dihedral and an improper
    /// kernel and the rebinding contract can be checked on all four at once.
    /// Gathering twice is gathering once.
    ///
    /// A periodic régime calls this again every time the copy list is rebuilt
    /// and has no way to check whether the last call took — so appending
    /// rather than rebuilding would grow the vector without bound, and every
    /// index past the first rebuild would name the wrong atom.
    #[test]
    fn gathering_onto_copies_is_idempotent() {
        let owned = vec![10_u8, 20, 30];
        let owner = [2_u32, 0, 2];

        let mut v = owned.clone();
        gather_copies(&mut v, owned.len(), &owner);
        assert_eq!(v, vec![10, 20, 30, 30, 10, 30]);

        gather_copies(&mut v, owned.len(), &owner);
        assert_eq!(
            v,
            vec![10, 20, 30, 30, 10, 30],
            "a second gather changed it"
        );

        // A shorter copy list shrinks it back rather than leaving a tail.
        gather_copies(&mut v, owned.len(), &owner[..1]);
        assert_eq!(v, vec![10, 20, 30, 30]);
    }

    fn make_all_bonded_frame() -> Frame {
        let mut frame = Frame::new();
        let mut atoms = Block::new();
        // A non-planar, non-collinear arrangement: a dihedral and an improper
        // are both undefined on degenerate geometry, and a test that fed them
        // one would be asserting against NaN.
        atoms
            .insert(
                "x",
                Array1::from_vec(vec![0.0 as F, 1.5, 2.1, 3.4]).into_dyn(),
            )
            .unwrap();
        atoms
            .insert(
                "y",
                Array1::from_vec(vec![0.0 as F, 0.2, 1.4, 1.1]).into_dyn(),
            )
            .unwrap();
        atoms
            .insert(
                "z",
                Array1::from_vec(vec![0.0 as F, 0.9, 0.3, 1.8]).into_dyn(),
            )
            .unwrap();
        atoms
            .insert(
                "type",
                Array1::from_vec(vec!["A".to_string(); 4]).into_dyn(),
            )
            .unwrap();
        frame.insert("atoms", atoms);

        let idx = |v: Vec<Idx>| Array1::from_vec(v).into_dyn();
        let ty = |n: usize, name: &str| Array1::from_vec(vec![name.to_string(); n]).into_dyn();

        let mut bonds = Block::new();
        bonds.insert("atomi", idx(vec![0, 1, 2])).unwrap();
        bonds.insert("atomj", idx(vec![1, 2, 3])).unwrap();
        bonds.insert("type", ty(3, "A-A")).unwrap();
        frame.insert("bonds", bonds);

        let mut angles = Block::new();
        angles.insert("atomi", idx(vec![0, 1])).unwrap();
        angles.insert("atomj", idx(vec![1, 2])).unwrap();
        angles.insert("atomk", idx(vec![2, 3])).unwrap();
        angles.insert("type", ty(2, "A-A-A")).unwrap();
        frame.insert("angles", angles);

        let mut dihedrals = Block::new();
        dihedrals.insert("atomi", idx(vec![0])).unwrap();
        dihedrals.insert("atomj", idx(vec![1])).unwrap();
        dihedrals.insert("atomk", idx(vec![2])).unwrap();
        dihedrals.insert("atoml", idx(vec![3])).unwrap();
        dihedrals.insert("type", ty(1, "A-A-A-A")).unwrap();
        frame.insert("dihedrals", dihedrals);

        let mut impropers = Block::new();
        impropers.insert("atomi", idx(vec![1])).unwrap();
        impropers.insert("atomj", idx(vec![0])).unwrap();
        impropers.insert("atomk", idx(vec![2])).unwrap();
        impropers.insert("atoml", idx(vec![3])).unwrap();
        impropers.insert("type", ty(1, "A-A-A-A")).unwrap();
        frame.insert("impropers", impropers);

        frame
    }

    /// Handing a kernel back the very indices it resolved at construction must
    /// change nothing — bit for bit.
    ///
    /// This is the contract that makes a periodic régime possible: if the two
    /// entry points can disagree on identical input, they can disagree on
    /// remapped input too, and the difference would be indistinguishable from
    /// physics. Bit equality rather than a tolerance because it is meant to be
    /// the same arithmetic in the same order — a kernel that had grown a
    /// second copy of its force expression would show up here as a few ulp.
    #[test]
    fn rebinding_a_kernel_to_its_own_indices_changes_nothing() {
        let mut ff = ForceField::new("all-bonded");
        ff.def_bondstyle("harmonic")
            .def_type("A-A", &[("k", 300.0), ("r0", 1.5)]);
        ff.def_anglestyle("harmonic")
            .def_type("A-A-A", &[("k", 50.0), ("theta0", 1.911)]);
        ff.def_dihedralstyle("opls").def_type(
            "A-A-A-A",
            &[("k1", 1.3), ("k2", -0.05), ("k3", 0.24), ("k4", 0.0)],
        );
        ff.def_improperstyle("harmonic")
            .def_type("A-A-A-A", &[("k", 10.0), ("chi0", 0.0)]);

        let frame = make_all_bonded_frame();
        let pots = ff.to_potentials(&frame).unwrap();
        let coords = extract_coords(&frame).unwrap();

        let mut checked = 0;
        for (m, member) in pots.members().iter().enumerate() {
            let Some(terms) = member.terms() else {
                continue;
            };
            let (e0, f0) = member.calc_energy_forces(&coords);
            let (e1, f1) = member.calc_energy_forces_with_terms(&coords, terms.view());
            assert_eq!(
                e0.to_bits(),
                e1.to_bits(),
                "member {m}: energy {e0} vs {e1} on identical indices"
            );
            assert_eq!(f0.len(), f1.len(), "member {m}: force length");
            for (c, (a, b)) in f0.iter().zip(&f1).enumerate() {
                assert_eq!(
                    a.to_bits(),
                    b.to_bits(),
                    "member {m}: force component {c}: {a} vs {b}"
                );
            }
            assert!(
                e0.is_finite() && e0 != 0.0,
                "member {m} contributed nothing, so it asserted nothing"
            );
            checked += 1;
        }
        assert_eq!(
            checked, 4,
            "expected a bond, an angle, a dihedral and an improper to answer `terms()`"
        );
    }

    fn make_bond_frame() -> Frame {
        let mut frame = make_atoms_only_frame();
        let mut bonds = Block::new();
        bonds
            .insert("atomi", Array1::from_vec(vec![0 as Idx]).into_dyn())
            .unwrap();
        bonds
            .insert("atomj", Array1::from_vec(vec![1 as Idx]).into_dyn())
            .unwrap();
        bonds
            .insert("type", Array1::from_vec(vec!["A-A".to_string()]).into_dyn())
            .unwrap();
        frame.insert("bonds", bonds);
        frame
    }

    fn make_lj_frame() -> Frame {
        // Two atoms 2 Å apart, each of per-atom `type` "A" (the kernel reads
        // per-atom LJ params keyed by it and Lorentz-Berthelot-combines).
        let mut frame = make_atoms_only_frame();
        frame
            .get_mut("atoms")
            .unwrap()
            .insert(
                "type",
                Array1::from_vec(vec!["A".to_string(), "A".to_string()]).into_dyn(),
            )
            .unwrap();

        // Build the neighbour list the real way — `intramolecular_pairs` owns
        // the atomi/atomj/is_14 convention; with no bonds it yields the single
        // unexcluded (0,1) pair, is_14 = false.
        let pairs = intramolecular_pairs(&frame);
        frame.insert("pairs", pairs);

        frame
    }

    #[test]
    fn test_potentials_collection() {
        let mut pots = Potentials::new();
        pots.push(Box::new(DummyPotential { value: 1.0 }));
        pots.push(Box::new(DummyPotential { value: 2.0 }));

        assert_eq!(pots.len(), 2);

        let coords: Vec<F> = vec![0.0; 6];
        assert!((pots.calc_energy(&coords) - 3.0).abs() < 1e-5);

        let forces = pots.calc_forces(&coords);
        for f in &forces {
            assert!((*f - 3.0).abs() < 1e-5);
        }
    }

    struct PairCounting;

    impl Potential for PairCounting {
        fn calc_energy_forces(&self, coords: &[F]) -> (F, Vec<F>) {
            (0.0, vec![0.0; coords.len()])
        }

        fn calc_energy_forces_with_pairs(&self, coords: &[F], pairs: &Neighbors) -> (F, Vec<F>) {
            (
                pairs.query_point_indices().len() as F,
                vec![0.0; coords.len()],
            )
        }
    }

    #[test]
    fn one_pair_table_is_shared_with_every_member() {
        let pairs = Neighbors::from_pairs(
            [
                molrs::spatial::neighbors::NeighborPair {
                    i: 0,
                    j: 1,
                    dist_sq: 1.0,
                    disp: [1.0, 0.0, 0.0],
                },
                molrs::spatial::neighbors::NeighborPair {
                    i: 0,
                    j: 2,
                    dist_sq: 4.0,
                    disp: [2.0, 0.0, 0.0],
                },
            ],
            molrs::spatial::neighbors::NeighborsStorage::FULL,
            molrs::spatial::neighbors::QueryMode::SelfQuery { num_points: 3 },
        );
        let mut pots = Potentials::new();
        pots.push(Box::new(PairCounting));
        pots.push(Box::new(DummyPotential { value: 1.0 }));
        let coords: Vec<F> = vec![0.0; 9];
        let (e, _) = pots.calc_energy_forces_with_pairs(&coords, &pairs);
        assert!((e - 3.0).abs() < 1e-12);
    }

    #[test]
    fn unknown_style_kernel_is_error() {
        // A style whose (category, name) has no kernel -> Err from to_potential.
        let mut ff = ForceField::new("test");
        ff.def_bondstyle("nonexistent")
            .def_type("A-A", &[("k", 1.0)]);
        let frame = make_bond_frame();
        let err = ff.to_potentials(&frame).unwrap_err();
        assert!(err.contains("no kernel"), "{err}");
    }

    #[test]
    fn register_kernel_extends_dispatch() {
        // A custom (category, name) with no built-in kernel becomes usable by
        // registering its constructor — no edit to to_potential required.
        fn my_ctor(
            _sp: &Params,
            _tp: &[(&str, &Params)],
            _f: &Frame,
        ) -> Result<Box<dyn Potential>, String> {
            Ok(Box::new(DummyPotential { value: 42.0 }))
        }
        register_kernel("pair", "test/custom", my_ctor);

        let mut ff = ForceField::new("test");
        ff.def_pairstyle("test/custom", &[]).def_type("A", &[]);
        let frame = make_lj_frame();
        let pots = ff.to_potentials(&frame).unwrap();
        let coords = extract_coords(&frame).unwrap();
        // the custom kernel ran: DummyPotential yields its constant value.
        assert!((pots.calc_energy(&coords) - 42.0).abs() < 1e-9);
    }

    #[test]
    fn atom_style_is_skipped() {
        // Atom styles carry types/charges, not a pairwise kernel -> skipped.
        let ff = ForceField::new("test").with_atomstyle("full");
        let frame = make_atoms_only_frame();
        let pots = ff.to_potentials(&frame).unwrap();
        assert_eq!(pots.len(), 0);
    }

    #[test]
    fn test_compile_requires_types() {
        let mut ff = ForceField::new("test");
        ff.def_bondstyle("harmonic");
        let frame = make_bond_frame();
        let err = ff
            .to_potentials(&frame)
            .expect_err("expected compile to fail");
        assert!(err.contains("has no type definitions"));
    }

    #[test]
    fn test_compile_energy() {
        let mut ff = ForceField::new("test");
        ff.def_pairstyle("lj/cut", &[("cutoff", 10.0)])
            .def_type("A", &[("epsilon", 1.0), ("sigma", 1.0)]);

        let frame = make_lj_frame();
        let pots = ff.to_potentials(&frame).unwrap();
        let coords = extract_coords(&frame).unwrap();

        let (energy, _) = pots.calc_energy_forces(&coords);
        let expected: F = 4.0 * (1.0 / 4096.0 - 1.0 / 64.0);
        assert!((energy - expected).abs() < 1e-5);
    }

    #[test]
    fn test_compile_forces() {
        let mut ff = ForceField::new("test");
        ff.def_pairstyle("lj/cut", &[("cutoff", 10.0)])
            .def_type("A", &[("epsilon", 1.0), ("sigma", 1.0)]);

        let frame = make_lj_frame();
        let pots = ff.to_potentials(&frame).unwrap();
        let coords = extract_coords(&frame).unwrap();

        let (_, forces) = pots.calc_energy_forces(&coords);

        for dim in 0..3 {
            let sum = forces[dim] + forces[3 + dim];
            assert!(sum.abs() < 1e-5);
        }
    }

    #[test]
    fn lj_cut_combines_distinct_types_lorentz_berthelot() {
        // Two atoms 2.5 Å apart of distinct types A and B; the kernel must
        // Lorentz-Berthelot-combine their per-atom params.
        let mut frame = make_atoms_only_frame();
        frame
            .get_mut("atoms")
            .unwrap()
            .insert("x", Array1::from_vec(vec![0.0 as F, 2.5 as F]).into_dyn())
            .unwrap();
        frame
            .get_mut("atoms")
            .unwrap()
            .insert(
                "type",
                Array1::from_vec(vec!["A".to_string(), "B".to_string()]).into_dyn(),
            )
            .unwrap();
        frame.insert("pairs", intramolecular_pairs(&frame));

        let mut ff = ForceField::new("test");
        ff.def_pairstyle("lj/cut", &[])
            .def_type("A", &[("epsilon", 1.0), ("sigma", 1.0)])
            .def_type("B", &[("epsilon", 4.0), ("sigma", 3.0)]);

        let pots = ff.to_potentials(&frame).unwrap();
        let coords = extract_coords(&frame).unwrap();
        let (energy, _) = pots.calc_energy_forces(&coords);

        // ε = √(1·4) = 2, σ = (1+3)/2 = 2, at r = 2.5.
        let (eps, sigma, r) = (2.0_f64, 2.0_f64, 2.5_f64);
        let sr6 = (sigma / r).powi(6);
        let expected = 4.0 * eps * (sr6 * sr6 - sr6);
        assert!(
            (energy - expected).abs() < 1e-9,
            "energy={energy} expected={expected}"
        );
    }

    #[test]
    fn lj_cut_applies_special_bonds_14_scaling() {
        // A 4-atom chain 0-1-2-3 (bonds 0-1-2-3, angles, one dihedral) so the
        // neighbour list excludes 1-2/1-3 and flags only the (0,3) pair is_14.
        // With special_bonds lj 1-4 = 0.5, the (0,3) LJ energy is halved.
        let mut frame = Frame::new();
        let mut atoms = Block::new();
        atoms
            .insert(
                "x",
                Array1::from_vec(vec![0.0 as F, 1.0, 2.0, 3.0]).into_dyn(),
            )
            .unwrap();
        for col in ["y", "z"] {
            atoms
                .insert(col, Array1::from_vec(vec![0.0 as F; 4]).into_dyn())
                .unwrap();
        }
        atoms
            .insert(
                "type",
                Array1::from_vec(vec!["A".to_string(); 4]).into_dyn(),
            )
            .unwrap();
        frame.insert("atoms", atoms);

        let mut bonds = Block::new();
        bonds
            .insert("atomi", Array1::from_vec(vec![0 as Idx, 1, 2]).into_dyn())
            .unwrap();
        bonds
            .insert("atomj", Array1::from_vec(vec![1 as Idx, 2, 3]).into_dyn())
            .unwrap();
        frame.insert("bonds", bonds);

        let mut angles = Block::new();
        angles
            .insert("atomi", Array1::from_vec(vec![0 as Idx, 1]).into_dyn())
            .unwrap();
        angles
            .insert("atomk", Array1::from_vec(vec![2 as Idx, 3]).into_dyn())
            .unwrap();
        frame.insert("angles", angles);

        let mut dihedrals = Block::new();
        dihedrals
            .insert("atomi", Array1::from_vec(vec![0 as Idx]).into_dyn())
            .unwrap();
        dihedrals
            .insert("atoml", Array1::from_vec(vec![3 as Idx]).into_dyn())
            .unwrap();
        frame.insert("dihedrals", dihedrals);

        // The real neighbour list: only (0,3), flagged is_14.
        let pairs = intramolecular_pairs(&frame);
        assert_eq!(
            pairs.nrows(),
            Some(1),
            "expected exactly the (0,3) 1-4 pair"
        );
        frame.insert("pairs", pairs);

        let mut ff = ForceField::new("test");
        ff.def_pairstyle("lj/cut", &[])
            .def_type("A", &[("epsilon", 1.0), ("sigma", 1.0)]);
        let mut sb = *ff.special_bonds();
        sb.lj[2] = 0.5;
        ff.set_special_bonds(sb);

        let pots = ff.to_potentials(&frame).unwrap();
        let coords = extract_coords(&frame).unwrap();
        let (energy, _) = pots.calc_energy_forces(&coords);

        // (0,3) at r = 3, ε = σ = 1, scaled by the 0.5 1-4 weight.
        let sr6 = (1.0_f64 / 3.0).powi(6);
        let expected = 0.5 * 4.0 * (sr6 * sr6 - sr6);
        assert!(
            (energy - expected).abs() < 1e-12,
            "energy={energy} expected={expected}"
        );
    }

    #[test]
    fn test_compile_empty_ff() {
        let ff = ForceField::new("test");
        let frame = make_lj_frame();
        let pots = ff.to_potentials(&frame).unwrap();
        let coords = extract_coords(&frame).unwrap();

        let (energy, forces) = pots.calc_energy_forces(&coords);
        assert!(energy.abs() < 1e-5);
        assert_eq!(forces.len(), 6);
        assert!(forces.iter().all(|x| x.abs() < 1e-5));
    }

    #[test]
    fn test_compile_skips_absent_topology() {
        // A style whose topology block is absent from the frame contributes
        // nothing (no rows of that kind) — skipped, not an error.
        let mut ff = ForceField::new("test");
        ff.def_pairstyle("lj/cut", &[("cutoff", 10.0)])
            .def_type("A", &[("epsilon", 1.0), ("sigma", 1.0)]);

        let frame = make_atoms_only_frame();
        let pots = ff.to_potentials(&frame).unwrap();
        assert_eq!(pots.len(), 0);
        let coords = extract_coords(&frame).unwrap();
        assert!(pots.calc_energy(&coords).abs() < 1e-9);
    }
}
