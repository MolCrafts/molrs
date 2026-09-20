//! Potential energy evaluation traits and kernel registry.
//!
//! A [`Potential`] stores pre-resolved topology indices and parameters.
//! Callers pass only flat coordinates — no [`Frame`] in the hot loop.
//! Construction from a [`Frame`] happens once via [`ForceField::to_potentials`](crate::ff::forcefield::ForceField::to_potentials).

pub mod geometry;

pub mod angle;
pub mod bond;
pub mod compile;
pub mod dihedral;
pub mod improper;
pub mod kspace;
pub mod pair;
pub mod registry;
pub mod soft;

pub use registry::{
    KernelConstructor, KernelRegistry, ParamSource, RowSource, lookup_kernel, lookup_param_source,
    lookup_row_source, register_kernel, register_kernel_with,
};

use std::collections::HashSet;

use ndarray::{Array1, Array2, ArrayView2};

use crate::ff::forcefield::SpecialBonds;
use molrs::math::Virial;
use molrs::spatial::neighbors::Neighbors;
use molrs::store::block::Block;
use molrs::store::frame::Frame;
use molrs::system::bond_weights::BondDistanceWeights;
use molrs::types::{F, Idx};

/// Above this many atoms, [`intramolecular_pairs`] refuses rather than
/// enumerating.
///
/// The list is every pair in the molecule with no cutoff, so it is `N(N-1)/2`
/// rows — at 50 000 atoms that is 1.2 billion rows and about 11 GiB, which
/// arrives as an OOM kill rather than as an answer. 20 000 is where the block
/// is still under 2 GiB and the caller is still plausibly asking for what this
/// function is for: the intramolecular pairs of one molecule, in free space.
/// A periodic or larger system wants a neighbour list and
/// [`ForceField::to_typed_potentials`](crate::ff::forcefield::ForceField::to_typed_potentials).
///
/// molrs-wasm caps the same path at 2 000 for its own memory budget; this is
/// the native ceiling, not a duplicate of that policy.
pub const MAX_ATOMS_FOR_A_FULL_PAIR_LIST: usize = 20_000;

/// `atomi` + `atomj` + `is_14`, as they land in the block's columns.
const BYTES_PER_PAIR_ROW: usize = 4 + 4 + 1;

/// Build the intramolecular non-bonded `pairs` block (`atomi`, `atomj`, `is_14`)
/// from a frame's bond/angle/dihedral topology: every `i < j` pair, excluding
/// 1-2 (bonded) and 1-3 (angle) pairs and flagging 1-4 (dihedral-end) pairs.
///
/// This is the neighbour list that [`ForceField::to_potentials`](crate::ff::forcefield::ForceField::to_potentials) hands to every
/// pair kernel — the same logic the MMFF frame builder used to compute
/// privately, lifted here so every force field (GAFF/LAMMPS, OPLS, MMFF, …)
/// shares one path. Per-pair scaling of the flagged 1-4 pairs is applied by the
/// pair kernels using the force field's 1-4 weight, not baked into this list.
///
/// # Why this needs the force field's weights
///
/// Which 1-2 / 1-3 pairs belong in the list *is* a force-field decision, and
/// this function used to make it — always excluding both, whatever the force
/// field said. LAMMPS's `special_bonds fene` (`[0, 1, 1]`) keeps 1-3 pairs at
/// full strength, and a FENE chain without them has nothing holding it open.
/// `special` answers it instead, via
/// [`SpecialBonds::compiled_inclusion`](crate::ff::forcefield::SpecialBonds::compiled_inclusion),
/// which is also where weights this list cannot express become an [`Err`]
/// rather than a silently different force field.
///
/// [`SpecialBonds::default`](crate::ff::forcefield::SpecialBonds::default)
/// reproduces the historical behaviour exactly: both classes excluded.
pub fn intramolecular_pairs(frame: &Frame, special: &SpecialBonds) -> Result<Block, String> {
    let [keep_12, keep_13] = special.compiled_inclusion()?;
    let n_atoms = frame.get("atoms").and_then(|b| b.nrows()).unwrap_or(0);
    if n_atoms > MAX_ATOMS_FOR_A_FULL_PAIR_LIST {
        return Err(format!(
            "intramolecular_pairs: {n_atoms} atoms would enumerate {} pairs \
             (~{} GiB) — this list is every pair in the molecule, with no cutoff. \
             Above {MAX_ATOMS_FOR_A_FULL_PAIR_LIST} atoms build a neighbour list \
             instead and evaluate through ForceField::to_typed_potentials.",
            n_atoms * (n_atoms - 1) / 2,
            (n_atoms * (n_atoms - 1) / 2 * BYTES_PER_PAIR_ROW) >> 30,
        ));
    }
    let pairs_12 = end_pairs(frame, "bonds", "atomi", "atomj");
    let pairs_13 = end_pairs(frame, "angles", "atomi", "atomk");
    let set_14 = end_pairs(frame, "dihedrals", "atomi", "atoml");

    let mut pi: Vec<Idx> = Vec::new();
    let mut pj: Vec<Idx> = Vec::new();
    let mut p14: Vec<bool> = Vec::new();
    for a in 0..n_atoms {
        for b in (a + 1)..n_atoms {
            let key = (a, b);
            if (!keep_12 && pairs_12.contains(&key)) || (!keep_13 && pairs_13.contains(&key)) {
                continue;
            }
            pi.push(a as Idx);
            pj.push(b as Idx);
            // A pair that is *both* 1-3 and 1-4 (a four-ring closes one) takes
            // the closer class: it is the 1-3 weight LAMMPS applies there, and
            // this list already decided 1-3 by presence.
            p14.push(set_14.contains(&key) && !pairs_13.contains(&key) && !pairs_12.contains(&key));
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
    Ok(pairs)
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

/// Energy and forces from coordinates alone.
///
/// A `Potential` is **molecule-bound**: its per-element parameters are expanded
/// against the molecule's topology once at [`ForceField::to_potentials`](crate::ff::forcefield::ForceField::to_potentials)
/// (string type labels resolved to per-bond/angle/… arrays). Evaluation
/// therefore takes only coordinates — there is no per-call topology resolution.
///
/// This is the whole of what every potential can do. Two capabilities that only
/// some have are separate traits, so that a caller which needs one says so in a
/// type instead of asking at run time:
///
/// * [`IndexedTerms`] — the rows are named by an index table the caller may
///   replace. Every bonded kernel.
/// * [`PairDriven`] — the sum runs over whatever pairs a neighbour search turns
///   up. Every pair kernel.
///
/// [`Member`] is the three of them as one value, chosen when the kernel is
/// built. It exists because a `Box<dyn Potential>` cannot be asked which of the
/// two it also is — the question used to be put to `terms()`, whose job is to
/// return a table and which allocated one per member per step to answer it.
///
/// The geometry optimizer ([`crate::optimize::LBFGS`]) depends on this trait —
/// not the other way around.
pub trait Potential: Send + Sync {
    /// Compute energy and forces (= -gradient) in one pass.
    /// Returns `(energy, forces)` where forces has length `coords.len()`.
    fn calc_energy_forces(&self, coords: &[F]) -> (F, Vec<F>);

    /// Compute total potential energy (kcal/mol).
    fn calc_energy(&self, coords: &[F]) -> F {
        self.calc_energy_forces(coords).0
    }

    /// Evaluate with a per-step pair table the loop computed once and shares
    /// with every pair potential. Default ignores `pairs` and calls
    /// [`calc_energy_forces`](Potential::calc_energy_forces).
    fn calc_energy_forces_with_pairs(&self, coords: &[F], pairs: &Neighbors) -> (F, Vec<F>) {
        let _ = pairs;
        self.calc_energy_forces(coords)
    }
}

/// A potential whose rows are named by an index table the caller can replace.
///
/// Implementing this is a declaration that *which atoms* is separable from
/// *what the parameters are*: row `r` of [`terms`](IndexedTerms::terms) belongs
/// with row `r` of the parameters, and a caller may hand back a different table
/// naming different atoms for the same rows. That is what lets a periodic
/// régime point a bond at a periodic copy without the kernel ever learning that
/// copies exist.
///
/// Every bonded kernel implements it; no pair kernel does, because a pair
/// kernel's rows are not fixed — they are whatever the neighbour search found.
pub trait IndexedTerms: Potential {
    /// The atom indices this kernel resolved at construction, `(n_terms,
    /// arity)` — arity 2 for a bond, 3 for an angle, 4 for a dihedral or an
    /// improper.
    fn terms(&self) -> Array2<u32>;

    /// Evaluate with `terms` in place of the indices resolved at construction.
    ///
    /// `terms` must have the shape [`terms`](IndexedTerms::terms) returned: the
    /// row *set* is fixed — it is the terms the force field declared — and only
    /// which atoms each row names may differ.
    ///
    /// There is no default. A default that ignored `terms` would be a correct
    /// fallback for a kernel holding no indices and a silently wrong answer for
    /// one that does, and only the second kind is in this trait.
    fn calc_energy_forces_with_terms(
        &self,
        coords: &[F],
        terms: ArrayView2<'_, u32>,
    ) -> (F, Vec<F>);
}

/// A potential summed over whatever pairs a neighbour search turns up.
///
/// Every pair kernel implements it. Nothing else should: the two required
/// methods are the ones a caller maintaining a live neighbour table needs
/// answered honestly, and a potential that ignores the table cannot answer
/// them.
pub trait PairDriven: Potential {
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
    ///   reason [`calc_energy_forces_with_pairs_virial`](PairDriven::calc_energy_forces_with_pairs_virial)
    ///   gives.
    ///
    /// No default: this used to have one that ignored `factor`, which was right
    /// for a potential that does not sum over the pair table and silently lost
    /// a force field's exclusions for one that does. Only the second kind is in
    /// this trait, so the question is now asked of every implementor.
    fn accumulate_pairs(
        &self,
        coords: &[F],
        pairs: &Neighbors,
        factor: &[F],
        out: &mut [F],
    ) -> (F, Option<Virial>);

    /// Whether this kernel's parameters are bound to one fixed pair list.
    ///
    /// Such a kernel answers for **that** list and no other. Handed a
    /// neighbour table it does not read it — it returns the frozen sum — and
    /// that is not a cheaper route to the same number, it is a different
    /// question silently unanswered. Worse, the caller goes on maintaining and
    /// reporting a live list that provably does not enter the answer.
    ///
    /// A periodic force path refuses such a member at construction. No default:
    /// a kernel built from one pair list and a kernel keyed on the atoms are
    /// the same Rust type in molrs (the list is a private source variant), so
    /// the answer cannot be inferred and has to be given.
    fn binds_a_fixed_pair_list(&self) -> bool;

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

/// One member of a force evaluation, with its role fixed when it was built.
///
/// A force evaluation sums members that play different parts — a bonded term
/// reads an index table, a pair term reads a neighbour table, an external field
/// reads neither. Which part a member plays is settled by its constructor, and
/// this is where that answer is kept, so no loop has to re-derive it per step
/// and no method has to carry a default that is wrong for half its implementors.
pub enum Member {
    /// A bonded term: evaluated against an index table the caller supplies.
    Indexed(Box<dyn IndexedTerms>),
    /// A non-bonded term: evaluated against a neighbour table, with per-pair
    /// weights.
    Pair(Box<dyn PairDriven>),
    /// Everything else — an external field, a restraint, a constant force.
    /// Evaluated from coordinates alone.
    Plain(Box<dyn Potential>),
}

impl Member {
    /// A bonded term — one whose rows are named by an index table.
    pub fn indexed(p: impl IndexedTerms + 'static) -> Self {
        Member::Indexed(Box::new(p))
    }

    /// A non-bonded term — one summed over a neighbour table.
    pub fn pair(p: impl PairDriven + 'static) -> Self {
        Member::Pair(Box::new(p))
    }

    /// Anything else — evaluated from coordinates alone.
    pub fn plain(p: impl Potential + 'static) -> Self {
        Member::Plain(Box::new(p))
    }

    /// This member as a plain potential, whatever part it plays.
    pub fn as_potential(&self) -> &dyn Potential {
        match self {
            Member::Indexed(p) => &**p,
            Member::Pair(p) => &**p,
            Member::Plain(p) => &**p,
        }
    }

    /// The index table, for a bonded member.
    pub fn terms(&self) -> Option<Array2<u32>> {
        match self {
            Member::Indexed(p) => Some(p.terms()),
            _ => None,
        }
    }

    /// Whether this member is bound to one fixed pair list — see
    /// [`PairDriven::binds_a_fixed_pair_list`]. A member that reads no pair
    /// table is bound to none.
    pub fn binds_a_fixed_pair_list(&self) -> bool {
        match self {
            Member::Pair(p) => p.binds_a_fixed_pair_list(),
            _ => false,
        }
    }

    /// Extend per-atom state onto periodic copies. Only a pair member keeps
    /// any; see [`PairDriven::gather_onto_copies`].
    pub fn gather_onto_copies(&mut self, owner: &[u32]) {
        if let Member::Pair(p) = self {
            p.gather_onto_copies(owner);
        }
    }
}

impl std::fmt::Debug for Member {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Member::Indexed(_) => "Member::Indexed",
            Member::Pair(_) => "Member::Pair",
            Member::Plain(_) => "Member::Plain",
        })
    }
}

impl Potential for Member {
    fn calc_energy_forces(&self, coords: &[F]) -> (F, Vec<F>) {
        self.as_potential().calc_energy_forces(coords)
    }

    fn calc_energy_forces_with_pairs(&self, coords: &[F], pairs: &Neighbors) -> (F, Vec<F>) {
        self.as_potential()
            .calc_energy_forces_with_pairs(coords, pairs)
    }
}

impl Potential for Box<dyn Potential> {
    fn calc_energy_forces(&self, coords: &[F]) -> (F, Vec<F>) {
        (**self).calc_energy_forces(coords)
    }

    fn calc_energy_forces_with_pairs(&self, coords: &[F], pairs: &Neighbors) -> (F, Vec<F>) {
        (**self).calc_energy_forces_with_pairs(coords, pairs)
    }
}

// ---------------------------------------------------------------------------
// Potentials collection
// ---------------------------------------------------------------------------

/// A kernel built for a neighbour-driven evaluation, and which of a force
/// field's special-bonds weight sets scales it. `None` for a bonded kernel:
/// it *is* the bonded interaction, not a scaled copy of one.
pub type TypedKernel = (Member, Option<registry::SpecialClass>);

/// One member of a neighbour-driven evaluation: the kernel, and the
/// bond-distance weights its non-bonded term takes.
///
/// The weights travel with the member because a force field may scale close
/// van-der-Waals and electrostatic neighbours differently, and in molrs those
/// are separate kernels.
pub type TypedMember = (Member, Option<BondDistanceWeights>);

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
    /// Each member with the part it plays, settled when it was built.
    ///
    /// This used to be a `Vec<Box<dyn Potential>>` beside a `Vec<bool>` saying
    /// which of them held atom indices, because the role had to be recovered by
    /// calling `terms()` — a method whose job is to return a table, and which
    /// allocated one per bonded member per step to answer a question that was
    /// settled at construction. [`Member`] is that answer, kept.
    inner: Vec<Member>,
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
            n_atoms: 0,
        }
    }

    /// Add a member. Which part it plays is [`Member`]'s to say, and its
    /// constructor already said it.
    pub fn push(&mut self, member: Member) {
        self.inner.push(member);
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

    /// Record the compiled atom count (used by [`ForceField::to_potentials`](crate::ff::forcefield::ForceField::to_potentials)).
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
    pub fn members(&self) -> &[Member] {
        &self.inner
    }

    /// Give up the members, for a caller that wants to own them individually.
    pub fn into_members(self) -> Vec<Member> {
        self.inner
    }

    /// The members, mutably — for the per-atom gather a periodic régime runs.
    pub fn members_mut(&mut self) -> &mut [Member] {
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
}

/// An aggregate is pair-driven when it is asked to be: it forwards to the
/// members that read a pair table and evaluates the rest the ordinary way.
///
/// The split used to be a `Vec<bool>` filled by calling `terms()` on every
/// member; it is now the member's own [`Member`] variant, which its
/// constructor chose.
impl PairDriven for Potentials {
    /// Every member accumulates into the same buffer, and one member that
    /// cannot report a virial makes the aggregate's `None`.
    ///
    /// A per-pair weight belongs to a member that reads the pair table. A
    /// bonded member ignores the table, and its own interaction is the thing
    /// the weights exist to avoid double-counting — scaling it would be
    /// scaling the wrong side of that.
    fn accumulate_pairs(
        &self,
        coords: &[F],
        pairs: &Neighbors,
        factor: &[F],
        out: &mut [F],
    ) -> (F, Option<Virial>) {
        let mut total_e: F = 0.0;
        let mut total_w = Some(Virial::ZERO);
        for m in &self.inner {
            let (e, w) = match m {
                Member::Pair(p) => p.accumulate_pairs(coords, pairs, factor, out),
                other => {
                    let (e, f) = other.calc_energy_forces_with_pairs(coords, pairs);
                    for (acc, v) in out.iter_mut().zip(&f) {
                        *acc += v;
                    }
                    (e, None)
                }
            };
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
        self.inner.iter().any(Member::binds_a_fixed_pair_list)
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
        for m in &self.inner {
            let (e, f, w) = match m {
                Member::Pair(p) => p.calc_energy_forces_with_pairs_virial(coords, pairs),
                other => {
                    let (e, f) = other.calc_energy_forces_with_pairs(coords, pairs);
                    (e, f, None)
                }
            };
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
        for m in &mut self.inner {
            m.gather_onto_copies(owner);
        }
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

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ff::forcefield::{ForceField, Params};
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
            let Member::Indexed(pot) = member else {
                unreachable!("only an indexed member answers with a table")
            };
            let (e0, f0) = pot.calc_energy_forces(&coords);
            let (e1, f1) = pot.calc_energy_forces_with_terms(&coords, terms.view());
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
        let pairs = intramolecular_pairs(&frame, &SpecialBonds::default()).unwrap();
        frame.insert("pairs", pairs);

        frame
    }

    #[test]
    fn test_potentials_collection() {
        let mut pots = Potentials::new();
        pots.push(Member::plain(DummyPotential { value: 1.0 }));
        pots.push(Member::plain(DummyPotential { value: 2.0 }));

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
        pots.push(Member::plain(PairCounting));
        pots.push(Member::plain(DummyPotential { value: 1.0 }));
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
        fn my_ctor(_sp: &Params, _tp: &[(&str, &Params)], _f: &Frame) -> Result<Member, String> {
            Ok(Member::plain(DummyPotential { value: 42.0 }))
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
        frame.insert(
            "pairs",
            intramolecular_pairs(&frame, &SpecialBonds::default()).unwrap(),
        );

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
        let pairs = intramolecular_pairs(&frame, &SpecialBonds::default()).unwrap();
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

    /// A frame too large for a full pair list is refused, not enumerated.
    ///
    /// `N(N-1)/2` with no cutoff is the shape of this list, and the failure
    /// mode past a certain `N` is the OOM killer, which tells the caller
    /// nothing about what to do instead. The refusal names the alternative.
    #[test]
    fn a_frame_too_large_for_a_full_pair_list_is_refused() {
        use molrs::store::block::Block;
        use ndarray::Array1;

        let n = MAX_ATOMS_FOR_A_FULL_PAIR_LIST + 1;
        let mut frame = Frame::new();
        let mut atoms = Block::new();
        atoms
            .insert("type", Array1::from(vec!["a".to_string(); n]).into_dyn())
            .unwrap();
        frame.insert("atoms", atoms);

        let err = intramolecular_pairs(&frame, &SpecialBonds::default())
            .expect_err("a list this size is not an answer");
        assert!(err.contains("neighbour list"), "{err}");
        assert!(err.contains("to_typed_potentials"), "{err}");

        // And one atom under the ceiling still builds, so the bound is a
        // ceiling and not an off-by-one that refuses the supported case.
        let mut small = Frame::new();
        let mut atoms = Block::new();
        atoms
            .insert(
                "type",
                Array1::from(vec!["a".to_string(); MAX_ATOMS_FOR_A_FULL_PAIR_LIST]).into_dyn(),
            )
            .unwrap();
        small.insert("atoms", atoms);
        assert!(intramolecular_pairs(&small, &SpecialBonds::default()).is_ok());
    }
}

#[cfg(test)]
pub(crate) mod test_util {
    use super::Potential;
    use molrs::types::F;

    /// Central-difference check that every force component is `-dE/dx`.
    pub(crate) fn assert_forces_are_negative_gradient(pot: &dyn Potential, coords: &[F], tol: F) {
        let h: F = 1e-6;
        let (_, f) = pot.calc_energy_forces(coords);
        let mut worst: F = 0.0;
        for i in 0..coords.len() {
            let mut plus = coords.to_vec();
            let mut minus = coords.to_vec();
            plus[i] += h;
            minus[i] -= h;
            let numeric = -(pot.calc_energy(&plus) - pot.calc_energy(&minus)) / (2.0 * h);
            worst = worst.max((f[i] - numeric).abs());
        }
        assert!(worst < tol, "max |F + dE/dx| = {worst:.3e}");
    }
}
