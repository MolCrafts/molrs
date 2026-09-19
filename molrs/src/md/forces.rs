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
use molrs::spatial::neighbors::VerletSkin;
use molrs::types::{F, FNx3View};

use super::error::MdError;
use super::pairs::{BondedLists, Comm};
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
pub struct MicPairs {
    potential: Box<dyn Potential>,
    skin: VerletSkin,
}

impl MicPairs {
    /// Evaluate `potential` over the pairs `skin` maintains.
    pub fn new(potential: impl Potential + 'static, skin: VerletSkin) -> Self {
        Self {
            potential: Box::new(potential),
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
        let pairs = self.skin.pairs_at(pos)?;
        let (energy, forces) = match pos.as_slice() {
            Some(flat) => self.potential.calc_energy_forces_with_pairs(flat, pairs),
            None => {
                let flat: Vec<F> = pos.iter().copied().collect();
                self.potential.calc_energy_forces_with_pairs(&flat, pairs)
            }
        };
        owned_output(energy, forces, pos.nrows())
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
    /// Which members keep atom indices, and what those indices are now.
    lists: BondedLists,
}

impl GhostPairs {
    /// Evaluate one potential over the copies `comm` maintains.
    pub fn new(potential: impl Potential + 'static, comm: Comm) -> Self {
        Self::from_members(vec![Box::new(potential)], comm)
    }

    /// Evaluate several members — what a force field compiles to — over the
    /// copies `comm` maintains.
    ///
    /// The members are ordinary potentials. This takes no force field and no
    /// frame: whichever of them hold atom indices say so through
    /// [`Potential::terms`], and that is the whole of what MD needs to know
    /// about a force field.
    pub fn from_members(members: Vec<Box<dyn Potential>>, comm: Comm) -> Self {
        let lists = BondedLists::new(&members);
        Self {
            comm,
            members,
            lists,
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
        self.lists.refresh(&self.comm, pos, wrap_shifts)?;
        let pairs = self.comm.pairs(pos)?;
        let all = self.comm.combined(pos)?;
        let n_all = all.nrows();
        let flat: Vec<F> = all.iter().copied().collect();

        let mut energy = 0.0;
        let mut forces = vec![0.0; flat.len()];
        for (m, member) in self.members.iter().enumerate() {
            let (e, f) = match self.lists.current(m) {
                Some(terms) => member.calc_energy_forces_with_terms(&flat, terms),
                None => member.calc_energy_forces_with_pairs(&flat, &pairs),
            };
            if f.len() != forces.len() {
                return Err(MdError::Invalid(format!(
                    "member {m} returned {} force components for {n_all} owned+ghost rows",
                    f.len()
                )));
            }
            energy += e;
            for (acc, v) in forces.iter_mut().zip(&f) {
                *acc += v;
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
