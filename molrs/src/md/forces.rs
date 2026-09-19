//! The force-field seam: one trait between the integrator and everything that
//! makes a force.
//!
//! An integrator asks exactly one question — *what are the energy, the forces
//! and the virial at this configuration?* — and [`ForceProvider`] is that
//! question. Everything a force evaluation needs on the way to answering it
//! (the potential, the neighbour bookkeeping, the periodic régime, the
//! fold-back of copies onto owners) lives behind the trait, and the integrator
//! sees none of it. That is what makes the engine force-field agnostic: a new
//! way to make a force is a new implementor, not a new enum variant and a new
//! arm in a `match` that every caller must be recompiled against.
//!
//! # It is also the parallelisation boundary
//!
//! One `compute` call is one unit of work an implementation may split across
//! threads however it likes. This is not incidental — it is the reason the
//! trait has the shape it has.
//!
//! The per-pair fold that dominates a step happens *inside*
//! [`Potential::calc_energy_forces_with_pairs`], and `Potential` is a stable
//! interface with dozens of implementors that this module does not get to
//! change. So a parallel fold has to be owned by something above `Potential`
//! and below the integrator. Before this trait there was no such object.
//!
//! Three consequences are load-bearing:
//!
//! * **`&mut self`, and no interior mutability.** A provider is not stateless:
//!   a Verlet skin holds its reference coordinates and its age, a halo holds
//!   its copies and a generation counter, and all of it is bound to *one*
//!   trajectory. Exclusive access for the duration of a call is what lets an
//!   implementation own reusable scratch and mutate it from rayon with no
//!   locks, and split its own data with `par_chunks_mut` with no
//!   synchronisation at all. `&self` plus interior mutability would buy only
//!   concurrent `compute` calls on one shared provider — meaningless here,
//!   because two replicas at different positions cannot share a skin whose
//!   held coordinates name one of them — at the price of a lock every step,
//!   forever. One provider per replica is the right shape, and that needs
//!   `Send`, not `&self`.
//! * **`Send + Sync` on the trait.** A boxed provider must be movable to a
//!   worker thread; just as importantly, the bound *forbids* an implementor
//!   from hiding an `Rc` or a `RefCell` and foreclosing that later.
//! * **An owned [`ForceOutput`] out, borrowing nothing.** No lifetime crosses
//!   the boundary, so a provider may re-partition itself between calls —
//!   rebuild its halo, change its domain decomposition, migrate atoms — with
//!   no caller pinned to its internal layout.

use ndarray::{Array2, ArrayView2};

use molrs::ff::potential::Potential;
use molrs::math::Virial;
use molrs::spatial::neighbors::VerletSkin;
use molrs::types::{F, FNx3View};

use super::error::MdError;
use super::pairs::{BondedLists, Comm, SpecialWeights};
use super::types::ForceOutput;

/// What an integrator asks of a force field.
///
/// See the [module documentation](self) for why the signature is what it is.
pub trait ForceProvider: Send + Sync {
    /// Energy, forces on the **owned** atoms, and the virial, at `pos`.
    ///
    /// `forces` has exactly `pos.nrows()` rows. A provider that evaluates over
    /// a larger set — periodic copies, a padded domain — folds the result back
    /// before returning: that the extended set ever existed is not the
    /// caller's business, and an integrator that had to know would not be
    /// force-field agnostic.
    ///
    /// `wrap_shifts` is the lattice shift the caller's wrap just applied, one
    /// signed count per axis per atom, all zeros when nothing folded. It is a
    /// parameter rather than something the provider re-derives because a fold
    /// *relabels* an atom without moving it: no displacement test can see one,
    /// and a provider holding copies must reconcile it in the same breath.
    fn compute(
        &mut self,
        pos: FNx3View<'_>,
        wrap_shifts: ArrayView2<'_, i64>,
    ) -> Result<ForceOutput, MdError>;

    /// Neighbour-bookkeeping counters, for logs and tests — never for physics.
    fn neighbor_stats(&self) -> NeighborStats {
        NeighborStats::default()
    }
}

impl ForceProvider for Box<dyn ForceProvider> {
    fn compute(
        &mut self,
        pos: FNx3View<'_>,
        wrap_shifts: ArrayView2<'_, i64>,
    ) -> Result<ForceOutput, MdError> {
        (**self).compute(pos, wrap_shifts)
    }

    fn neighbor_stats(&self) -> NeighborStats {
        (**self).neighbor_stats()
    }
}

/// What a provider's neighbour bookkeeping has been doing.
///
/// Every field is optional because a provider that keeps no list has no
/// answer, and zero would be an answer — the same distinction
/// `Neighbors::disp` draws, for the same reason.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct NeighborStats {
    /// Pairs in the current table.
    pub edges: Option<usize>,
    /// Times the list has been rebuilt over the run.
    pub rebuilds: Option<usize>,
    /// Steps since the last rebuild.
    pub ago: Option<usize>,
}

// ---------------------------------------------------------------------------
// Direct
// ---------------------------------------------------------------------------

/// No neighbour list: the potential is handed coordinates and enumerates its
/// own pairs.
pub struct Direct {
    potential: Box<dyn Potential>,
}

impl Direct {
    /// Evaluate `potential` at the raw coordinates, every step.
    pub fn new(potential: impl Potential + 'static) -> Self {
        Self {
            potential: Box::new(potential),
        }
    }
}

impl ForceProvider for Direct {
    fn compute(
        &mut self,
        pos: FNx3View<'_>,
        _wrap_shifts: ArrayView2<'_, i64>,
    ) -> Result<ForceOutput, MdError> {
        let (energy, forces) = match pos.as_slice() {
            Some(flat) => self.potential.calc_energy_forces(flat),
            None => {
                let flat: Vec<F> = pos.iter().copied().collect();
                self.potential.calc_energy_forces(&flat)
            }
        };
        owned_output(energy, forces, pos.nrows())
    }
}

// ---------------------------------------------------------------------------
// MicPairs
// ---------------------------------------------------------------------------

/// Minimum-image pairs over the owned atoms.
///
/// The potential sees the owned atoms and a pair table whose displacements
/// have already been folded — it is never asked to know that a boundary
/// exists.
///
/// Members are evaluated one at a time, as under [`GhostPairs`], and for the
/// same reason: what each one needs differs. A member holding atom indices is
/// a bonded term and reads no pair table; the rest read the table, weighted by
/// the force field's exclusions.
///
/// # What this route cannot do
///
/// A bonded term here is measured on the **stored** coordinates, so a molecule
/// that straddles a face is measured across the cell. The minimum image fixes
/// up *pair* displacements and nothing else. A system whose molecules cross
/// boundaries wants [`GhostPairs`], which resolves the bonded indices onto
/// copies.
pub struct MicPairs {
    /// The force evaluation's members, in the order the caller gave them.
    members: Vec<Box<dyn Potential>>,
    /// Per member, the weights on its close non-bonded neighbours.
    special: Vec<SpecialWeights>,
    /// Which members hold atom indices, and so take no pair table.
    holds_indices: Vec<bool>,
    skin: VerletSkin,
}

impl MicPairs {
    /// Evaluate one potential over the pairs `skin` maintains, with no
    /// special-bonds weights.
    pub fn new(potential: impl Potential + 'static, skin: VerletSkin) -> Self {
        Self::from_members(
            vec![(
                Box::new(potential) as Box<dyn Potential>,
                SpecialWeights::default(),
            )],
            skin,
        )
    }

    /// Evaluate several members — what a force field compiles to — over the
    /// pairs `skin` maintains.
    ///
    /// Each member comes with the weights on its close non-bonded neighbours;
    /// a bonded member takes [`SpecialWeights::default`], which scales nothing.
    pub fn from_members(
        members: Vec<(Box<dyn Potential>, SpecialWeights)>,
        skin: VerletSkin,
    ) -> Self {
        let (members, special): (Vec<Box<dyn Potential>>, Vec<_>) = members.into_iter().unzip();
        let holds_indices = members.iter().map(|m| m.terms().is_some()).collect();
        Self {
            members,
            special,
            holds_indices,
            skin,
        }
    }
}

impl ForceProvider for MicPairs {
    fn compute(
        &mut self,
        pos: FNx3View<'_>,
        _wrap_shifts: ArrayView2<'_, i64>,
    ) -> Result<ForceOutput, MdError> {
        let n_atoms = pos.nrows();
        let pairs = self.skin.pairs_at(pos)?;
        let flat: Vec<F> = match pos.as_slice() {
            Some(s) => s.to_vec(),
            None => pos.iter().copied().collect(),
        };

        let mut energy = 0.0;
        let mut forces = vec![0.0; flat.len()];
        // `None` the moment any member declines to report one: a virial missing
        // a term is not a small error, it is a different quantity.
        let mut virial = Some(Virial::ZERO);
        let mut accumulate =
            |m: usize, (e, f, w_pair): (F, Vec<F>, Option<Virial>), w: F| -> Result<(), MdError> {
                if f.len() != forces.len() {
                    return Err(MdError::Invalid(format!(
                        "member {m} returned {} force components for {n_atoms} atoms",
                        f.len()
                    )));
                }
                energy += w * e;
                for (acc, v) in forces.iter_mut().zip(&f) {
                    *acc += w * v;
                }
                match (virial.as_mut(), w_pair) {
                    (Some(total), Some(part)) => {
                        for c in 0..6 {
                            total.components[c] += w * part.components[c];
                        }
                    }
                    // One member that cannot tally makes the whole sum
                    // unreportable; it does not make it smaller.
                    (_, None) => virial = None,
                    (None, _) => {}
                }
                Ok(())
            };
        for (m, member) in self.members.iter().enumerate() {
            if self.holds_indices[m] {
                // A bonded term reads its own indices, not the pair table — and
                // it needs no kernel-side tally. Its forces sum to zero term by
                // term and no periodic image entered the geometry, so
                // `Σ_a f_a ⊗ x_a` over the stored coordinates *is* its virial,
                // and is independent of where the cell's origin falls.
                let (e, f) = member.calc_energy_forces(&flat);
                let mut w_term = Virial::ZERO;
                for a in 0..n_atoms {
                    w_term.add_outer(
                        [f[a * 3], f[a * 3 + 1], f[a * 3 + 2]],
                        [flat[a * 3], flat[a * 3 + 1], flat[a * 3 + 2]],
                    );
                }
                accumulate(m, (e, f, Some(w_term)), 1.0)?;
            } else if self.special[m].is_empty() {
                accumulate(
                    m,
                    member.calc_energy_forces_with_pairs_virial(&flat, pairs),
                    1.0,
                )?;
            } else {
                // No copies here, so an index *is* its own owner.
                let (main, scaled) = self.special[m].split(pairs, n_atoms, &[]);
                accumulate(
                    m,
                    member.calc_energy_forces_with_pairs_virial(&flat, &main),
                    1.0,
                )?;
                for (w, table) in &scaled {
                    accumulate(
                        m,
                        member.calc_energy_forces_with_pairs_virial(&flat, table),
                        *w,
                    )?;
                }
            }
        }

        let forces = Array2::from_shape_vec((n_atoms, 3), forces).map_err(|_| {
            MdError::Invalid(format!("force components do not fit {n_atoms} atoms"))
        })?;
        Ok(ForceOutput {
            energy,
            forces,
            virial,
        })
    }

    fn neighbor_stats(&self) -> NeighborStats {
        NeighborStats {
            edges: Some(self.skin.num_edges()),
            rebuilds: Some(self.skin.rebuild_count()),
            ago: Some(self.skin.ago()),
        }
    }
}

// ---------------------------------------------------------------------------
// GhostPairs
// ---------------------------------------------------------------------------

/// Periodic copies: the potential sees an ordinary local cluster.
///
/// Owned atoms *and their copies* go in, with plain differences; the forces
/// come back over that extended set and are folded onto the owners before
/// anything physical is read off them.
///
/// Members are evaluated one at a time rather than through an aggregate,
/// because what each one needs differs: a member holding atom indices is
/// handed the indices resolved against the copies that exist now, and a member
/// reading geometry is handed the pair table. An aggregate would have to pick
/// one table for everybody, and the bonded ones are not interchangeable — a
/// bond list is not an angle list.
pub struct GhostPairs {
    comm: Comm,
    /// The force evaluation's members, in the order the caller gave them.
    members: Vec<Box<dyn Potential>>,
    /// Per member, the weights on its close non-bonded neighbours.
    ///
    /// Per member and not one shared table, because a force field may scale
    /// van der Waals and electrostatics differently — Amber uses `1/2` and
    /// `1/1.2` — and in molrs those are separate kernels. One table would have
    /// to be wrong for one of them.
    special: Vec<SpecialWeights>,
    /// Which members keep atom indices, and what those indices are now.
    lists: BondedLists,
    /// The copy-list generation the members' per-atom state was gathered for,
    /// or `None` before the first gather.
    gathered: Option<u64>,
}

impl GhostPairs {
    /// Evaluate one potential over the copies `comm` maintains, with no
    /// special-bonds weights — right for a system with no bonded topology.
    pub fn new(potential: impl Potential + 'static, comm: Comm) -> Self {
        Self::from_members(
            vec![(
                Box::new(potential) as Box<dyn Potential>,
                SpecialWeights::default(),
            )],
            comm,
        )
    }

    /// Evaluate several members — what a force field compiles to — over the
    /// copies `comm` maintains.
    ///
    /// The members are ordinary potentials. This takes no force field and no
    /// frame: whichever of them hold atom indices say so through
    /// [`Potential::terms`], and that is the whole of what MD needs to know
    /// about a force field.
    ///
    /// Each member comes with the weights on its close non-bonded neighbours.
    /// A neighbour table finds every pair inside the cutoff, bonded or not, so
    /// without them a bonded pair would be counted twice — once by the bonded
    /// term and once at full non-bonded strength. A bonded member takes
    /// [`SpecialWeights::default`], which scales nothing.
    pub fn from_members(members: Vec<(Box<dyn Potential>, SpecialWeights)>, comm: Comm) -> Self {
        let (members, special): (Vec<_>, Vec<_>) = members.into_iter().unzip();
        let lists = BondedLists::new(&members);
        Self {
            comm,
            members,
            lists,
            special,
            gathered: None,
        }
    }

    /// The halo, for tests that read its counters.
    pub fn comm(&self) -> &Comm {
        &self.comm
    }

    /// The bonded index lists as they stand.
    pub fn lists(&self) -> &BondedLists {
        &self.lists
    }
}

impl ForceProvider for GhostPairs {
    fn compute(
        &mut self,
        pos: FNx3View<'_>,
        wrap_shifts: ArrayView2<'_, i64>,
    ) -> Result<ForceOutput, MdError> {
        // Move the copies first, then re-resolve the indices against them, then
        // read the pairs. Each step depends on the one before it, and doing
        // them in one call would hide that.
        self.comm.advance(pos, wrap_shifts)?;
        // A member holding anything per atom — a charge, a type index — has to
        // cover the copies, because the pair table names them. That state is
        // *derived* from the owners', so a rebuild invalidates it and nothing
        // else does: a fold relabels an atom without changing what it is.
        // (An index resolution is invalidated by both, which is why
        // `BondedLists` below is asked every step and this is not.)
        let generation = self.comm.ghosts().generation();
        if self.gathered != Some(generation) {
            let owner = self.comm.ghosts().owner();
            for member in &mut self.members {
                member.gather_onto_copies(owner);
            }
            self.gathered = Some(generation);
        }
        self.lists.refresh(&self.comm, pos, wrap_shifts)?;
        let pairs = self.comm.pairs(pos)?;
        let all = self.comm.combined(pos)?;
        let n_all = all.nrows();
        let flat: Vec<F> = all.iter().copied().collect();

        let mut energy = 0.0;
        let mut forces = vec![0.0; flat.len()];
        let mut accumulate = |m: usize, (e, f): (F, Vec<F>), w: F| -> Result<(), MdError> {
            if f.len() != forces.len() {
                return Err(MdError::Invalid(format!(
                    "member {m} returned {} force components for {n_all} owned+ghost rows",
                    f.len()
                )));
            }
            energy += w * e;
            for (acc, v) in forces.iter_mut().zip(&f) {
                *acc += w * v;
            }
            Ok(())
        };
        for (m, member) in self.members.iter().enumerate() {
            match self.lists.current(m) {
                // A member holding atom indices is a bonded term: it reads no
                // pair table, so no weight applies to it.
                Some(terms) => {
                    accumulate(m, member.calc_energy_forces_with_terms(&flat, terms), 1.0)?
                }
                None if self.special[m].is_empty() => {
                    accumulate(m, member.calc_energy_forces_with_pairs(&flat, &pairs), 1.0)?
                }
                None => {
                    let set = self.comm.ghosts();
                    let (main, scaled) = self.special[m].split(&pairs, set.n_owned(), set.owner());
                    accumulate(m, member.calc_energy_forces_with_pairs(&flat, &main), 1.0)?;
                    for (w, table) in &scaled {
                        accumulate(m, member.calc_energy_forces_with_pairs(&flat, table), *w)?;
                    }
                }
            }
        }

        let mut f = Array2::from_shape_vec((n_all, 3), forces).map_err(|_| {
            MdError::Invalid(format!(
                "force components do not fit {n_all} owned+ghost rows"
            ))
        })?;
        // Reverse accumulation: a copy's force belongs to the atom it copies.
        // The virial is tallied from the copy forces *before* they are reduced
        // — afterwards the information it needs is gone.
        let virial = self.comm.reverse_comm_with_virial(&mut f, all.view())?;
        f.slice_collapse(ndarray::s![..pos.nrows(), ..]);
        Ok(ForceOutput {
            energy,
            forces: f.to_owned(),
            virial: Some(virial),
        })
    }

    fn neighbor_stats(&self) -> NeighborStats {
        NeighborStats {
            edges: None,
            rebuilds: Some(self.comm.rebuilds()),
            ago: None,
        }
    }
}

/// Shape a flat force vector into the owned `(N, 3)` block, or say why it does
/// not fit.
fn owned_output(energy: F, forces: Vec<F>, n_atoms: usize) -> Result<ForceOutput, MdError> {
    let n_components = forces.len();
    let forces = Array2::from_shape_vec((n_atoms, 3), forces).map_err(|_| {
        MdError::Invalid(format!(
            "potential returned {n_components} force components for {n_atoms} atoms"
        ))
    })?;
    Ok(ForceOutput {
        energy,
        forces,
        virial: None,
    })
}

#[cfg(test)]
mod tests {
    use ndarray::array;

    use molrs::ff::potential::Potentials;
    use molrs::ff::potential::pair::LJCut;
    use molrs::spatial::neighbors::{NeighborList, NeighborPolicy};
    use molrs::spatial::simbox::SimBox;

    use super::super::pairs::SpecialWeights;
    use super::*;

    fn cell() -> SimBox {
        SimBox::cube(12.0, array![0.0_f64, 0.0, 0.0], [true; 3]).unwrap()
    }

    fn four_atoms() -> ndarray::Array2<F> {
        array![
            [11.0_f64, 11.0, 11.0],
            [3.0, 11.0, 11.0],
            [11.0, 3.0, 11.0],
            [3.0, 3.0, 11.0],
        ]
    }

    fn lj() -> LJCut {
        LJCut::new(0.3, 3.4, 5.0, 12, 6, false, false).unwrap()
    }

    fn skin(pos: ndarray::ArrayView2<'_, F>) -> molrs::spatial::neighbors::VerletSkin {
        molrs::spatial::neighbors::VerletSkin::new(
            NeighborList::new(5.0),
            5.0,
            NeighborPolicy {
                skin: 0.0,
                ..NeighborPolicy::default()
            },
            pos,
            cell(),
        )
        .unwrap()
    }

    /// Every provider answers over the owned atoms, whatever it evaluated over.
    ///
    /// The ghost route is the one this can catch out: it evaluates over
    /// `owned + copies` and has to fold that back. An integrator handed the
    /// extended block would write forces onto rows that are not atoms, and the
    /// shapes would go on agreeing with each other for a while.
    #[test]
    fn a_provider_answers_with_exactly_the_owned_rows() {
        let pos = four_atoms();
        let n = pos.nrows();
        let no_fold = Array2::zeros((n, 3));

        let mut direct = Direct::new(Potentials::new());
        assert_eq!(
            direct
                .compute(pos.view(), no_fold.view())
                .unwrap()
                .forces
                .nrows(),
            n
        );

        let mut mic = MicPairs::new(lj(), skin(pos.view()));
        assert_eq!(
            mic.compute(pos.view(), no_fold.view())
                .unwrap()
                .forces
                .nrows(),
            n
        );

        let comm = Comm::new(cell(), pos.view(), 5.0, 0.0).unwrap();
        let mut ghosts = GhostPairs::new(lj(), comm);
        let out = ghosts.compute(pos.view(), no_fold.view()).unwrap();
        assert_eq!(out.forces.nrows(), n);
        assert!(
            !ghosts.comm().ghosts().is_empty(),
            "this cell must actually produce copies, or the fold-back is untested"
        );
    }

    /// A molecule that straddles a face is scored exactly as the same molecule
    /// sitting in the middle of the cell.
    ///
    /// This is the whole point of rebinding the indices, and it is the
    /// assertion that the kernels stayed ignorant of periodicity while it
    /// happened. A rigid translation of a periodic system changes nothing
    /// physical, so the bonded energy and every force component must come back
    /// the same — while the bookkeeping underneath is completely different:
    /// crossing, the bond and the angle are measured through copies; centred,
    /// they are measured between owned atoms.
    ///
    /// Without the rebinding the crossing case reads a bond stretched by the
    /// width of the box, which is not a small error — it is the failure the
    /// ghost régime exists to remove.
    #[test]
    fn a_molecule_across_a_face_scores_as_one_that_is_not() {
        use molrs::ff::forcefield::ForceField;
        use molrs::store::block::Block;
        use molrs::store::frame::Frame;
        use molrs::types::Idx;
        use ndarray::Array1;

        let l = 20.0_f64;
        let bx = SimBox::cube(l, array![0.0_f64, 0.0, 0.0], [true; 3]).unwrap();

        // A bent three-atom chain: two bonds and one angle, so both arities are
        // exercised and the angle's anchoring matters.
        let shape = array![[-1.0_f64, 0.0, 0.0], [0.0, 0.0, 0.0], [0.4, 0.9, 0.0]];

        let frame = |_: ()| {
            let mut f = Frame::new();
            let mut atoms = Block::new();
            atoms
                .insert("type", Array1::from(vec!["a".to_string(); 3]).into_dyn())
                .unwrap();
            f.insert("atoms", atoms);
            let mut bonds = Block::new();
            bonds
                .insert("atomi", Array1::from(vec![0 as Idx, 1]).into_dyn())
                .unwrap();
            bonds
                .insert("atomj", Array1::from(vec![1 as Idx, 2]).into_dyn())
                .unwrap();
            bonds
                .insert("type", Array1::from(vec!["a-a".to_string(); 2]).into_dyn())
                .unwrap();
            f.insert("bonds", bonds);
            let mut angles = Block::new();
            angles
                .insert("atomi", Array1::from(vec![0 as Idx]).into_dyn())
                .unwrap();
            angles
                .insert("atomj", Array1::from(vec![1 as Idx]).into_dyn())
                .unwrap();
            angles
                .insert("atomk", Array1::from(vec![2 as Idx]).into_dyn())
                .unwrap();
            angles
                .insert("type", Array1::from(vec!["a-a-a".to_string()]).into_dyn())
                .unwrap();
            f.insert("angles", angles);
            f
        };

        // Rest lengths deliberately off the actual geometry, so both terms
        // carry real energy and real force; a molecule at rest would satisfy
        // this test by contributing nothing.
        let mut field = ForceField::new("probe");
        field
            .def_bondstyle("harmonic")
            .def_bondtype("a", "a", &[("k", 100.0), ("r0", 1.2)]);
        field.def_anglestyle("harmonic").def_angletype(
            "a",
            "a",
            "a",
            &[("k", 40.0), ("theta0", 2.0)],
        );

        let run = |origin: [F; 3]| {
            let mut pts = shape.clone();
            for i in 0..pts.nrows() {
                for k in 0..3 {
                    pts[[i, k]] += origin[k];
                }
            }
            let (wrapped, _) = bx.wrap_shifts(pts.view());
            let comm = Comm::new(bx.clone(), wrapped.view(), 4.0, 0.0).unwrap();
            let members = field.to_potentials(&frame(())).unwrap().into_members();
            let members = members
                .into_iter()
                .map(|p| (p, SpecialWeights::default()))
                .collect();
            let mut provider = GhostPairs::from_members(members, comm);
            // The halo was built from these coordinates, so from its point of
            // view nothing has folded. A fold is reported exactly once, to the
            // halo that existed before it.
            let no_fold = Array2::<i64>::zeros((wrapped.nrows(), 3));
            let out = provider.compute(wrapped.view(), no_fold.view()).unwrap();
            (out.energy, out.forces)
        };

        // Straddling the low x face, and sitting in the middle.
        let (e_cross, f_cross) = run([0.2, 10.0, 10.0]);
        let (e_mid, f_mid) = run([10.0, 10.0, 10.0]);

        assert!(
            e_mid.abs() > 1.0,
            "the molecule must be strained for this to assert anything; got {e_mid}"
        );
        assert!(
            (e_cross - e_mid).abs() / e_mid.abs() < 1e-12,
            "energy {e_cross} across the face vs {e_mid} in the middle"
        );
        for i in 0..3 {
            for k in 0..3 {
                assert!(
                    (f_cross[[i, k]] - f_mid[[i, k]]).abs() < 1e-9,
                    "force on atom {i} component {k}: {} across the face vs {} in the middle",
                    f_cross[[i, k]],
                    f_mid[[i, k]]
                );
            }
        }
    }

    /// A kernel that reads per-atom types gives the same answer through copies
    /// as it does through the minimum image.
    ///
    /// The two régimes index differently: the minimum-image table names owned
    /// atoms, the ghost table names `[owned | copies]`. A typed kernel looks
    /// its parameters up *by index*, so the ghost route is only right if the
    /// copies carry their owners' types — and if they did not, the failure
    /// would be a wrong parameter rather than a crash: a silently different
    /// well depth for every pair that reaches through a face.
    ///
    /// Two unlike types, alternating, so picking up the wrong one is visible.
    #[test]
    fn a_typed_kernel_reads_the_same_types_through_copies_as_through_the_image() {
        use molrs::ff::potential::pair::lj_cut::Mixing;

        let l = 12.0_f64;
        let cutoff = 5.0;
        let bx = SimBox::cube(l, array![0.0_f64, 0.0, 0.0], [true; 3]).unwrap();
        // Eight atoms on a 4 Å cube whose every edge runs *through* a face, so
        // every interaction is periodic and every one of them needs a copy.
        let pos = array![
            [11.0_f64, 11.0, 11.0],
            [3.0, 11.0, 11.0],
            [11.0, 3.0, 11.0],
            [11.0, 11.0, 3.0],
            [3.0, 3.0, 11.0],
            [3.0, 11.0, 3.0],
            [11.0, 3.0, 3.0],
            [3.0, 3.0, 3.0],
        ];
        let n = pos.nrows();
        let per_type = [(0.3_f64, 3.4_f64), (0.9, 2.6)];
        let type_id: Vec<u32> = (0..n).map(|i| (i % 2) as u32).collect();
        let lj = || {
            LJCut::typed(
                type_id.clone(),
                &per_type,
                Mixing::Arithmetic,
                cutoff,
                12,
                6,
                false,
                false,
            )
            .unwrap()
        };

        let no_fold = Array2::<i64>::zeros((n, 3));

        // Zero skin on both sides: a skin is a caching policy, not a periodic
        // one, and a stale edge would be a difference this test is not about.
        let skin = molrs::spatial::neighbors::VerletSkin::new(
            NeighborList::new(cutoff),
            cutoff,
            NeighborPolicy {
                skin: 0.0,
                ..NeighborPolicy::default()
            },
            pos.view(),
            bx.clone(),
        )
        .unwrap();
        let mut mic = MicPairs::new(lj(), skin);
        let mic_out = mic.compute(pos.view(), no_fold.view()).unwrap();

        let comm = Comm::new(bx, pos.view(), cutoff, 0.0).unwrap();
        let mut ghosts = GhostPairs::new(lj(), comm);
        let ghost_out = ghosts.compute(pos.view(), no_fold.view()).unwrap();

        assert!(
            ghosts.comm().ghosts().len() > n,
            "this cell must materialise copies, or the gather is untested"
        );
        assert!(
            mic_out.energy.abs() > 1.0,
            "the system must interact for this to assert anything"
        );
        assert!(
            (ghost_out.energy - mic_out.energy).abs() / mic_out.energy.abs() < 1e-12,
            "energy {} through copies vs {} through the image",
            ghost_out.energy,
            mic_out.energy
        );
        for i in 0..n {
            for k in 0..3 {
                assert!(
                    (ghost_out.forces[[i, k]] - mic_out.forces[[i, k]]).abs() < 1e-10,
                    "force on atom {i} component {k}: {} vs {}",
                    ghost_out.forces[[i, k]],
                    mic_out.forces[[i, k]]
                );
            }
        }
    }

    /// An excluded pair contributes nothing — not a small number, nothing.
    ///
    /// A neighbour table finds every pair inside the cutoff, so a bonded pair
    /// turns up in it like any other. Left alone it would be counted twice:
    /// once by the bond term and once at full Lennard-Jones strength, at bond
    /// length, where that term is enormous. Here every pair of the chain is
    /// 1-2 or 1-3, so with an exclusion depth of 3 the non-bonded sum must be
    /// exactly zero.
    ///
    /// Exactly, because an exclusion drops the pair rather than scaling it.
    #[test]
    fn a_fully_excluded_molecule_has_no_non_bonded_energy() {
        use molrs::Topology;
        use molrs::ff::potential::pair::lj_cut::Mixing;
        use molrs::system::bond_weights::BondDistanceWeights;

        let bx = SimBox::cube(20.0, array![0.0_f64, 0.0, 0.0], [true; 3]).unwrap();
        // A bent chain, every atom well inside the 6 Å cutoff of the others.
        let pos = array![
            [9.0_f64, 10.0, 10.0],
            [10.5, 10.0, 10.0],
            [11.3, 11.2, 10.0]
        ];
        let n = pos.nrows();

        let topo = Topology::from_edges(n, &[[0, 1], [1, 2]]);
        let weights = BondDistanceWeights::from_exclusion_depth(3);
        let special = SpecialWeights::new(&topo.special_weights(&weights));

        let lj = LJCut::typed(
            vec![0_u32; n],
            &[(0.3_f64, 3.4_f64)],
            Mixing::Arithmetic,
            6.0,
            12,
            6,
            false,
            false,
        )
        .unwrap();

        let comm = Comm::new(bx.clone(), pos.view(), 6.0, 0.0).unwrap();
        let no_fold = Array2::<i64>::zeros((n, 3));
        let mut with =
            GhostPairs::from_members(vec![(Box::new(lj) as Box<dyn Potential>, special)], comm);
        let out = with.compute(pos.view(), no_fold.view()).unwrap();

        assert_eq!(
            out.energy, 0.0,
            "every pair of this chain is 1-2 or 1-3, so nothing may be left"
        );
        assert!(
            out.forces.iter().all(|&f| f == 0.0),
            "an excluded pair exerts no force either"
        );

        // Without the weights the same configuration is enormous — which is
        // what the exclusion is preventing, and what a silent failure would
        // have contributed instead.
        let lj = LJCut::typed(
            vec![0_u32; n],
            &[(0.3_f64, 3.4_f64)],
            Mixing::Arithmetic,
            6.0,
            12,
            6,
            false,
            false,
        )
        .unwrap();
        let comm = Comm::new(bx, pos.view(), 6.0, 0.0).unwrap();
        let mut without = GhostPairs::new(lj, comm);
        let bare = without.compute(pos.view(), no_fold.view()).unwrap();
        assert!(
            bare.energy > 100.0,
            "the unexcluded sum should be large; got {}",
            bare.energy
        );
    }

    /// The exclusions follow a molecule through a face.
    ///
    /// A pair naming a copy has to be weighted as the atom the copy is of — a
    /// bond graph knows owners, and a copy is the same atom seen through a
    /// face. If the lookup used the copy's index instead, the weight would come
    /// back as 1.0 and a bonded pair would be scored at full strength, but only
    /// for molecules near a boundary: a bug that hides everywhere except where
    /// it matters.
    #[test]
    fn exclusions_follow_a_molecule_through_a_face() {
        use molrs::Topology;
        use molrs::ff::potential::pair::lj_cut::Mixing;
        use molrs::system::bond_weights::BondDistanceWeights;

        let l = 20.0_f64;
        let cutoff = 6.0;
        let bx = SimBox::cube(l, array![0.0_f64, 0.0, 0.0], [true; 3]).unwrap();

        // Two bent chains: one straddles the low x face, the other does not.
        let shape = array![
            [-1.5_f64, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.8, 1.2, 0.0],
            [4.6, 0.3, 0.4],
            [6.1, 0.3, 0.4],
            [6.9, 1.5, 0.4],
        ];
        let n = shape.nrows();
        let topo = Topology::from_edges(n, &[[0, 1], [1, 2], [3, 4], [4, 5]]);
        let weights = BondDistanceWeights::from_exclusion_depth(3);
        let special = SpecialWeights::new(&topo.special_weights(&weights));

        let run = |origin: F| {
            let mut pts = shape.clone();
            for i in 0..pts.nrows() {
                pts[[i, 0]] += origin;
                pts[[i, 1]] += 10.0;
                pts[[i, 2]] += 10.0;
            }
            let (wrapped, _) = bx.wrap_shifts(pts.view());
            let lj = LJCut::typed(
                vec![0_u32; n],
                &[(0.3_f64, 3.4_f64)],
                Mixing::Arithmetic,
                cutoff,
                12,
                6,
                false,
                false,
            )
            .unwrap();
            let comm = Comm::new(bx.clone(), wrapped.view(), cutoff, 0.0).unwrap();
            let mut provider = GhostPairs::from_members(
                vec![(Box::new(lj) as Box<dyn Potential>, special.clone())],
                comm,
            );
            let no_fold = Array2::<i64>::zeros((n, 3));
            let out = provider.compute(wrapped.view(), no_fold.view()).unwrap();
            (out.energy, out.forces)
        };

        // Straddling (atom 0 lands at −0.5 and wraps), and shifted a third of
        // a cell so nothing crosses.
        let (e_cross, f_cross) = run(1.0);
        let (e_mid, f_mid) = run(1.0 + l / 3.0);

        assert!(
            e_mid.abs() > 1e-3,
            "the two chains must see each other for this to assert anything; got {e_mid}"
        );
        assert!(
            (e_cross - e_mid).abs() / e_mid.abs() < 1e-12,
            "energy {e_cross} across the face vs {e_mid} away from it"
        );
        for i in 0..n {
            for k in 0..3 {
                assert!(
                    (f_cross[[i, k]] - f_mid[[i, k]]).abs() < 1e-9,
                    "force on atom {i} component {k}: {} vs {}",
                    f_cross[[i, k]],
                    f_mid[[i, k]]
                );
            }
        }
    }

    /// The whole path, from a force field to a periodic force: bonded terms
    /// resolved onto copies, non-bonded ones found by a neighbour search and
    /// weighted by the force field's own exclusions.
    ///
    /// A rigid translation of a periodic system is a symmetry, so every number
    /// must come back the same — while underneath, one run measures two of its
    /// three bonds through copies and finds half its pairs through a face, and
    /// the other measures nothing through anything.
    ///
    /// This is the test that says the front door exists: nothing here builds a
    /// kernel by hand or knows which member is which.
    #[test]
    fn a_force_field_drives_a_periodic_evaluation_end_to_end() {
        use molrs::Topology;
        use molrs::ff::forcefield::ForceField;
        use molrs::store::block::Block;
        use molrs::store::frame::Frame;
        use molrs::types::Idx;
        use ndarray::Array1;

        let l = 20.0_f64;
        let cutoff = 6.0;
        let bx = SimBox::cube(l, array![0.0_f64, 0.0, 0.0], [true; 3]).unwrap();

        // Two bent chains close enough to see each other.
        let shape = array![
            [-1.5_f64, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.8, 1.2, 0.0],
            [4.6, 0.3, 0.4],
            [6.1, 0.3, 0.4],
            [6.9, 1.5, 0.4],
        ];
        let n = shape.nrows();
        let bonds = [[0usize, 1], [1, 2], [3, 4], [4, 5]];
        let topo = Topology::from_edges(n, &bonds);

        let mut frame = Frame::new();
        let mut atoms = Block::new();
        atoms
            .insert("type", Array1::from(vec!["a".to_string(); n]).into_dyn())
            .unwrap();
        frame.insert("atoms", atoms);
        let mut blk = Block::new();
        blk.insert(
            "atomi",
            Array1::from(bonds.iter().map(|b| b[0] as Idx).collect::<Vec<_>>()).into_dyn(),
        )
        .unwrap();
        blk.insert(
            "atomj",
            Array1::from(bonds.iter().map(|b| b[1] as Idx).collect::<Vec<_>>()).into_dyn(),
        )
        .unwrap();
        blk.insert(
            "type",
            Array1::from(vec!["a-a".to_string(); bonds.len()]).into_dyn(),
        )
        .unwrap();
        frame.insert("bonds", blk);

        // Rest length off the actual bond length, so the bonded term carries
        // real energy and a mis-resolved bond would be loud.
        let mut field = ForceField::new("probe");
        field
            .def_bondstyle("harmonic")
            .def_bondtype("a", "a", &[("k", 100.0), ("r0", 1.2)]);
        field
            .def_pairstyle("lj/cut", &[("cutoff", cutoff)])
            .def_type("a", &[("epsilon", 0.3), ("sigma", 3.4)]);

        let run = |origin: F| {
            let mut pts = shape.clone();
            for i in 0..pts.nrows() {
                pts[[i, 0]] += origin;
                pts[[i, 1]] += 10.0;
                pts[[i, 2]] += 10.0;
            }
            let (wrapped, _) = bx.wrap_shifts(pts.view());

            // The front door: no kernel built by hand, no member identified.
            let members: Vec<(Box<dyn Potential>, SpecialWeights)> = field
                .to_typed_potentials(&frame)
                .unwrap()
                .into_iter()
                .map(|(pot, weights)| {
                    let special = weights
                        .map(|w| SpecialWeights::new(&topo.special_weights(&w)))
                        .unwrap_or_default();
                    (pot, special)
                })
                .collect();
            assert_eq!(members.len(), 2, "a bond style and a pair style");

            let comm = Comm::new(bx.clone(), wrapped.view(), cutoff, 0.0).unwrap();
            let mut provider = GhostPairs::from_members(members, comm);
            let no_fold = Array2::<i64>::zeros((n, 3));
            let out = provider.compute(wrapped.view(), no_fold.view()).unwrap();
            (out.energy, out.forces)
        };

        let (e_cross, f_cross) = run(1.0);
        let (e_mid, f_mid) = run(1.0 + l / 3.0);

        assert!(
            e_mid.abs() > 1.0,
            "the system must be strained and interacting; got {e_mid}"
        );
        assert!(
            (e_cross - e_mid).abs() / e_mid.abs() < 1e-12,
            "energy {e_cross} across the face vs {e_mid} away from it"
        );
        for i in 0..n {
            for k in 0..3 {
                assert!(
                    (f_cross[[i, k]] - f_mid[[i, k]]).abs() < 1e-9,
                    "force on atom {i} component {k}: {} vs {}",
                    f_cross[[i, k]],
                    f_mid[[i, k]]
                );
            }
        }
    }

    /// The exclusions work the same without copies to map through.
    ///
    /// Under the minimum image an index is already its own owner, so the split
    /// runs with an empty owner map. That identity case is easy to get wrong in
    /// a way that only shows up here: a mapping that assumed copies exist would
    /// index past the end of an empty slice, or quietly weight the wrong pair.
    #[test]
    fn the_minimum_image_route_excludes_the_same_pairs() {
        use molrs::Topology;
        use molrs::ff::potential::pair::lj_cut::Mixing;
        use molrs::system::bond_weights::BondDistanceWeights;

        let bx = SimBox::cube(20.0, array![0.0_f64, 0.0, 0.0], [true; 3]).unwrap();
        let pos = array![
            [9.0_f64, 10.0, 10.0],
            [10.5, 10.0, 10.0],
            [11.3, 11.2, 10.0]
        ];
        let n = pos.nrows();
        let topo = Topology::from_edges(n, &[[0, 1], [1, 2]]);
        let weights = BondDistanceWeights::from_exclusion_depth(3);
        let special = SpecialWeights::new(&topo.special_weights(&weights));

        let lj = LJCut::typed(
            vec![0_u32; n],
            &[(0.3_f64, 3.4_f64)],
            Mixing::Arithmetic,
            6.0,
            12,
            6,
            false,
            false,
        )
        .unwrap();
        let skin = molrs::spatial::neighbors::VerletSkin::new(
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
        let mut mic =
            MicPairs::from_members(vec![(Box::new(lj) as Box<dyn Potential>, special)], skin);
        let out = mic
            .compute(pos.view(), Array2::<i64>::zeros((n, 3)).view())
            .unwrap();

        assert_eq!(
            out.energy, 0.0,
            "every pair of this chain is 1-2 or 1-3, so nothing may be left"
        );
        assert!(out.forces.iter().all(|&f| f == 0.0));
        assert_eq!(
            out.virial.expect("a typed kernel tallies one").components,
            [0.0; 6],
            "no pairs, no virial"
        );
    }

    /// A provider that keeps no list reports no counters — not zeroes.
    ///
    /// Zero rebuilds is a fact about a list that exists; a provider without one
    /// has no such fact, and saying `0` would let a caller conclude the list is
    /// fresh.
    #[test]
    fn counters_are_absent_rather_than_zero_when_there_is_no_list() {
        let pos = four_atoms();

        assert_eq!(
            Direct::new(Potentials::new()).neighbor_stats(),
            NeighborStats::default()
        );

        let mic = MicPairs::new(lj(), skin(pos.view()));
        let stats = mic.neighbor_stats();
        assert!(stats.edges.is_some());
        assert!(stats.rebuilds.is_some());
        assert!(stats.ago.is_some());

        let comm = Comm::new(cell(), pos.view(), 5.0, 0.0).unwrap();
        let stats = GhostPairs::new(lj(), comm).neighbor_stats();
        assert!(stats.rebuilds.is_some(), "a halo counts its rebuilds");
        assert!(
            stats.edges.is_none(),
            "the halo keeps no persistent edge list, so it has no edge count to report"
        );
    }
}
