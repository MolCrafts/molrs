//! Where an integrator gets its non-bonded pairs.
//!
//! Two periodic régimes reach the same physics by different routes, and an MD
//! run picks one:
//!
//! * [`PairSource::Mic`] keeps `N` atoms and fixes up every displacement with
//!   the minimum-image convention. Cheapest, and correct for any potential that
//!   consumes edge vectors.
//! * [`PairSource::Ghosts`] materialises the periodic copies and hands the
//!   potential an ordinary, non-periodic cluster. It costs the copies, and it
//!   is the only route that is correct for a potential reading *positions* —
//!   a many-body or machine-learned model cannot be told about a minimum-image
//!   fix-up it does not know to apply.
//!
//! They are variants of one enum rather than two optional fields because an
//! integrator has exactly one source: a pair of `Option`s would make "both" and
//! "neither, but configured" writable states that mean nothing.

use ndarray::ArrayView2;

use molrs::ff::forcefield::ForceField;
use molrs::ff::potential::Potentials;
use molrs::spatial::neighbors::{Neighbors, VerletSkin};
use molrs::spatial::periodic::{GhostError, GhostSet};
use molrs::spatial::simbox::SimBox;
use molrs::store::frame::Frame;
use molrs::types::{F, FNx3, FNx3View, Idx};

use super::error::MdError;

/// Owns the ghost atoms of an MD run and keeps them current.
///
/// [`GhostSet`] is a **snapshot** — which copies exist and where they are right
/// now. This keeps one current across a trajectory: it decides when the
/// snapshot has gone stale, rebuilds it, moves it every step, and hands out the
/// pair table and the re-resolved topology that follow. It is the ghost
/// régime's counterpart to
/// [`VerletSkin`] and answers the same
/// question: has anything moved far enough that what was frozen at the last
/// rebuild is no longer complete?
///
/// # The name, and the three operations
///
/// `Comm` is LAMMPS's name for this object, and the operations carry LAMMPS's
/// names too, because the point is that a future domain decomposition changes
/// none of them:
///
/// | LAMMPS | here | what it does |
/// |---|---|---|
/// | `comm->borders()` | [`GhostSet::borders`] | decide *which* copies exist — a search, at rebuild |
/// | `comm->forward_comm()` | [`GhostSet::forward_comm`] | move them to follow their owners — every step |
/// | `comm->reverse_comm()` | [`GhostSet::reverse_comm`] | sum the forces on copies back onto owners |
///
/// In a single process a forward communication is a translation and a reverse
/// one is a scatter-add. Under MPI they become messages, an owner may live on
/// another rank, and nothing above this layer notices — which is the whole
/// reason to spell them this way now rather than invent a local vocabulary and
/// translate later.
#[derive(Debug)]
pub struct Comm {
    set: GhostSet,
    bx: SimBox,
    cutoff: F,
    skin: F,
    /// Owned positions at the last halo build, for the displacement test.
    x_hold: FNx3,
    rebuilds: usize,
}

impl Comm {
    /// Build the halo for `owned` at `cutoff`, with `skin` of margin.
    ///
    /// The halo reaches `cutoff + skin`, so it stays complete while no atom has
    /// moved more than `skin/2` since the build — the same bookkeeping a Verlet
    /// skin does, applied to copies instead of to a pair list.
    pub fn new(bx: SimBox, owned: FNx3View<'_>, cutoff: F, skin: F) -> Result<Self, MdError> {
        if cutoff <= 0.0 {
            return Err(MdError::Invalid(format!(
                "cutoff must be > 0 Å, got {cutoff}"
            )));
        }
        if skin < 0.0 {
            return Err(MdError::Invalid(format!("skin must be >= 0 Å, got {skin}")));
        }
        let set = GhostSet::borders(&bx, owned, cutoff + skin).map_err(ghost_err)?;
        Ok(Self {
            set,
            bx,
            cutoff,
            skin,
            x_hold: owned.to_owned(),
            rebuilds: 0,
        })
    }

    /// The cell the copies are generated from.
    pub fn simbox(&self) -> &SimBox {
        &self.bx
    }

    /// Move the copies to follow their owners, and rebuild the set if it has
    /// gone stale.
    ///
    /// `wrap_shifts` is the shift the caller's wrap just applied. It is
    /// required, not optional: a copy replaced relative to an owner that has
    /// folded moves a whole cell while the pair list still points at it.
    ///
    /// The set is rebuilt when an atom has drifted more than half the skin
    /// since the last one, which is the same completeness argument a Verlet
    /// skin makes. Displacements are minimum-image, so a fold reads as the step
    /// the atom took rather than as a jump of one cell.
    pub fn advance(
        &mut self,
        owned: FNx3View<'_>,
        wrap_shifts: ArrayView2<'_, i64>,
    ) -> Result<(), MdError> {
        self.set
            .forward_comm(&self.bx, owned, wrap_shifts)
            .map_err(ghost_err)?;

        let mic = self.bx.mic();
        let half_skin_sq = (0.5 * self.skin) * (0.5 * self.skin);
        let mut max_d2 = 0.0;
        for i in 0..owned.nrows() {
            let d = mic.apply([
                owned[[i, 0]] - self.x_hold[[i, 0]],
                owned[[i, 1]] - self.x_hold[[i, 1]],
                owned[[i, 2]] - self.x_hold[[i, 2]],
            ]);
            let d2 = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
            if d2 > max_d2 {
                max_d2 = d2;
            }
        }
        if max_d2 > half_skin_sq {
            self.set =
                GhostSet::borders(&self.bx, owned, self.cutoff + self.skin).map_err(ghost_err)?;
            self.x_hold = owned.to_owned();
            self.rebuilds += 1;
        }
        Ok(())
    }

    /// The half-shell pair table over `[owned | ghost]`, centred on owned atoms.
    pub fn pairs(&self, owned: FNx3View<'_>) -> Result<Neighbors, MdError> {
        self.set.pairs(owned, self.cutoff).map_err(ghost_err)
    }

    /// The `[owned | ghost]` coordinates the pair table indexes.
    pub fn combined(&self, owned: FNx3View<'_>) -> Result<FNx3, MdError> {
        self.set.combined(owned).map_err(ghost_err)
    }

    /// How many times the halo has been rebuilt.
    pub fn rebuilds(&self) -> usize {
        self.rebuilds
    }

    /// The copies as they stand.
    pub fn ghosts(&self) -> &GhostSet {
        &self.set
    }
}

// ---------------------------------------------------------------------------
// Bonded topology through the ghost halo
// ---------------------------------------------------------------------------

/// Rewrite a frame's bonded topology so every term is resolved against the
/// nearest periodic copy of its partners.
///
/// A bonded kernel holds indices, not geometry: it is told *these two atoms are
/// bonded* and takes the plain difference of their coordinates. For a bond that
/// straddles a face those coordinates are a cell apart, so the kernel measures
/// a bond stretched by the width of the box and answers with a force to match.
/// Nothing raises — the number is simply wrong, and it is wrong by orders of
/// magnitude.
///
/// Remapping each partner to its closest copy makes the plain difference the
/// right one, and leaves all twenty-one bonded kernels exactly as they are.
/// They never learn that periodic boundaries exist, which is the whole point of
/// the ghost régime.
///
/// # Anchoring
///
/// Each term is resolved against **one** atom, not edge by edge:
///
/// * a bond `(i, j)` against `i`;
/// * an angle `(i, j, k)` against the vertex `j`, so both arms are placed
///   relative to the same point;
/// * a dihedral or improper `(i, j, k, l)` against `j`, the atom the others are
///   at most two bonds from.
///
/// Folding each edge against the previous atom instead can pick images that are
/// individually closest and jointly inconsistent — the three atoms of an angle
/// ending in three different cells — and the angle is then measured on a shape
/// that does not exist. An anchor cannot do that: every atom is placed relative
/// to a single point.
///
/// # When it refuses
///
/// A copy can only be chosen from the copies that exist. If the halo does not
/// reach as far as a bonded term does — a short pair cutoff with a long
/// molecule — the nearest available copy is still a cell away, and the term
/// stays stretched. That is reported as [`MdError::Invalid`] rather than
/// returned: a silently mismeasured bond is precisely the failure this function
/// exists to remove, and swapping it for a differently-silent one would be no
/// improvement.
fn ghost_err(e: GhostError) -> MdError {
    MdError::Invalid(e.to_string())
}

/// The pair source an integrator evaluates forces through.
///
/// The variants differ in size by a wide margin — a halo carries its copies, an
/// empty source carries nothing — so each is boxed and the enum stays one
/// pointer wide. An integrator holds one for the length of a run, so the
/// indirection is paid once and never in the step loop.
pub enum PairSource {
    /// No neighbour list — the potential enumerates its own pairs.
    None,
    /// Minimum-image pairs over the owned atoms.
    Mic(Box<VerletSkin>),
    /// Ghost atoms: the potential sees an ordinary local cluster.
    ///
    /// `bonded` is the force field resolved against the copies. When it is
    /// present it **is** the potential for this run: the integrator evaluates
    /// it and ignores the one it was constructed with, which is what keeps the
    /// two from being summed twice. Without it the integrator's own potential
    /// is used, which is right for a system with no bonded terms.
    Ghosts {
        /// The copies, and their lifecycle.
        comm: Box<Comm>,
        /// The force field, resolved against those copies.
        bonded: Option<Box<BondedTopology>>,
    },
}

impl PairSource {
    /// Minimum-image pairs from a Verlet-skinned neighbour list.
    pub fn mic(skin: VerletSkin) -> Self {
        Self::Mic(Box::new(skin))
    }

    /// Ghost-atom pairs, with the potential left to the integrator.
    pub fn ghosts(comm: Comm) -> Self {
        Self::Ghosts {
            comm: Box::new(comm),
            bonded: None,
        }
    }

    /// Ghost-atom pairs, with the force field resolved against the copies.
    pub fn bonded_ghosts(comm: Comm, bonded: BondedTopology) -> Self {
        Self::Ghosts {
            comm: Box::new(comm),
            bonded: Some(Box::new(bonded)),
        }
    }
}

/// The force field's view of the system, kept resolved against the copies.
///
/// A bonded term is a pair of *indices*, and which copy an index should name is
/// re-decided whenever the copy list is. Left to the caller that is a cadence
/// to get right and silent to get wrong: the kernels go on reading whatever
/// indices they were built with, measuring bonds against copies that have moved
/// on. This owns the cadence.
///
/// It is separate from [`Comm`] on purpose. `Comm` moves copies around and
/// knows nothing about force fields; this knows about force fields and owns no
/// copies. LAMMPS draws the same line — `Comm` communicates, and the `NTopo`
/// classes rebuild the bonded lists against `Domain::closest_image` — and a
/// type that did both would be answering to two masters.
pub struct BondedTopology {
    /// The frame as the caller gave it: owned indices, the source of truth.
    source: Frame,
    field: ForceField,
    /// Built from the resolved frame against `generation`.
    current: Potentials,
    /// The copy-list generation `current`'s indices were resolved against.
    generation: u64,
}

impl std::fmt::Debug for BondedTopology {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BondedTopology")
            .field("generation", &self.generation)
            .finish_non_exhaustive()
    }
}

impl BondedTopology {
    /// Resolve `source` against the copies `comm` currently holds.
    pub fn new(
        source: Frame,
        field: ForceField,
        comm: &Comm,
        owned: FNx3View<'_>,
    ) -> Result<Self, MdError> {
        let resolved = Self::resolve(&source, owned, comm)?;
        let current = field.to_potentials(&resolved).map_err(MdError::Invalid)?;
        Ok(Self {
            source,
            field,
            current,
            generation: comm.ghosts().generation(),
        })
    }

    /// Re-resolve if the copies have changed under it.
    ///
    /// Two things invalidate a resolved index, and only one is obvious.
    ///
    /// A **rebuild** replaces the copy list, and an index names a position in
    /// that list. This is checked rather than remembered: the resolution
    /// records the generation it was made against.
    ///
    /// A **fold** does not move an atom — it relabels it — so it costs no
    /// displacement and trips no rebuild. But "closest copy" is answered
    /// against the *stored* coordinate, and that has just jumped a lattice
    /// vector, so a bond resolved before the fold now reaches across the cell
    /// instead of through the face. Nothing is out of place; the index is
    /// answering an older question.
    pub fn refresh(
        &mut self,
        comm: &Comm,
        owned: FNx3View<'_>,
        wrap_shifts: ArrayView2<'_, i64>,
    ) -> Result<(), MdError> {
        let folded = wrap_shifts.iter().any(|&m| m != 0);
        let stale = self.generation != comm.ghosts().generation();
        if stale || folded {
            let resolved = Self::resolve(&self.source, owned, comm)?;
            self.current = self
                .field
                .to_potentials(&resolved)
                .map_err(MdError::Invalid)?;
            self.generation = comm.ghosts().generation();
        }
        debug_assert_eq!(
            self.generation,
            comm.ghosts().generation(),
            "bonded topology is resolved against a copy list that no longer exists"
        );
        Ok(())
    }

    /// The potentials, resolved against the current copies.
    pub fn potentials(&self) -> &Potentials {
        &self.current
    }

    /// The copy-list generation the indices name.
    pub fn generation(&self) -> u64 {
        self.generation
    }

    /// [`resolve`](Self::resolve), reachable from tests that check the
    /// rewriting itself rather than the potentials built from it.
    #[cfg(test)]
    pub fn resolve_for_test(
        frame: &Frame,
        owned: FNx3View<'_>,
        comm: &Comm,
    ) -> Result<Frame, MdError> {
        Self::resolve(frame, owned, comm)
    }

    #[allow(clippy::needless_range_loop)]
    fn resolve(frame: &Frame, owned: FNx3View<'_>, comm: &Comm) -> Result<Frame, MdError> {
        use molrs::store::keys;
        let set = comm.ghosts();

        // Beyond half the smallest plane spacing the minimum image is ambiguous, so
        // a term still that long after remapping has not been resolved — it has
        // been guessed.
        let min_width = comm
            .simbox()
            .nearest_plane_distance()
            .iter()
            .copied()
            .fold(F::INFINITY, F::min);
        let limit = (0.5 * min_width).min(set.reach());

        let mut out = frame.clone();
        for (block_name, cols, anchor_col) in [
            ("bonds", &[keys::ATOMI, keys::ATOMJ][..], 0usize),
            ("angles", &[keys::ATOMI, keys::ATOMJ, keys::ATOMK][..], 1),
            (
                "dihedrals",
                &[keys::ATOMI, keys::ATOMJ, keys::ATOMK, keys::ATOML][..],
                1,
            ),
            (
                "impropers",
                &[keys::ATOMI, keys::ATOMJ, keys::ATOMK, keys::ATOML][..],
                1,
            ),
        ] {
            let Some(block) = frame.get(block_name) else {
                continue;
            };
            let mut columns: Vec<Vec<Idx>> = Vec::with_capacity(cols.len());
            for c in cols {
                let Some(col) = block.get_uint(c) else {
                    columns.clear();
                    break;
                };
                columns.push(col.iter().copied().collect());
            }
            if columns.len() != cols.len() {
                continue; // an incomplete block is not this function's to repair
            }

            let n_terms = columns[0].len();
            for t in 0..n_terms {
                let anchor = columns[anchor_col][t] as usize;
                for (c, column) in columns.iter_mut().enumerate() {
                    if c == anchor_col {
                        continue;
                    }
                    let partner = column[t] as usize;
                    let mapped = set
                        .closest_image(owned, anchor, partner)
                        .map_err(|e| MdError::Invalid(e.to_string()))?;
                    column[t] = mapped as Idx;
                }
            }

            // Every term must now be compact. Checking it here, once per rebuild,
            // is cheaper than any kernel could and catches the case no kernel can
            // see: a halo that never reached far enough.
            let all = set
                .combined(owned)
                .map_err(|e| MdError::Invalid(e.to_string()))?;
            for t in 0..n_terms {
                let a = columns[anchor_col][t] as usize;
                for (c, column) in columns.iter().enumerate() {
                    if c == anchor_col {
                        continue;
                    }
                    let p = column[t] as usize;
                    let d2: F = (0..3)
                        .map(|k| {
                            let d = all[[p, k]] - all[[a, k]];
                            d * d
                        })
                        .sum();
                    if d2.sqrt() > limit {
                        return Err(MdError::Invalid(format!(
                            "{block_name} term {t} still spans {:.3} Å after remapping, \
                             past the {limit:.3} Å at which the periodic image stops being \
                             decidable. The halo does not reach as far as this molecule: \
                             build it with a reach that covers the bonded extent, not only \
                             the pair cutoff",
                            d2.sqrt()
                        )));
                    }
                }
            }

            let mut nb = block.clone();
            for (c, column) in cols.iter().zip(columns) {
                nb.insert(*c, ndarray::Array1::from(column).into_dyn())
                    .map_err(|e| MdError::Invalid(format!("{block_name}.{c}: {e}")))?;
            }
            out.insert(block_name, nb);
        }
        Ok(out)
    }
}

#[cfg(test)]
mod remap_tests {
    use super::*;
    use molrs::ff::potential::Potential;
    use molrs::ff::potential::bond::harmonic::BondHarmonic;
    use molrs::spatial::simbox::SimBox;
    use molrs::store::block::Block;
    use ndarray::{Array1, array};

    fn bonds_block(i: &[Idx], j: &[Idx]) -> Block {
        let mut b = Block::new();
        b.insert("atomi", Array1::from(i.to_vec()).into_dyn())
            .unwrap();
        b.insert("atomj", Array1::from(j.to_vec()).into_dyn())
            .unwrap();
        b
    }

    fn angles_block(i: &[Idx], j: &[Idx], k: &[Idx]) -> Block {
        let mut b = Block::new();
        b.insert("atomi", Array1::from(i.to_vec()).into_dyn())
            .unwrap();
        b.insert("atomj", Array1::from(j.to_vec()).into_dyn())
            .unwrap();
        b.insert("atomk", Array1::from(k.to_vec()).into_dyn())
            .unwrap();
        b
    }

    /// A bond whose two atoms sit on opposite sides of a face is rewritten to
    /// point at a copy, and the kernel that reads it — unchanged, and still
    /// ignorant of periodicity — then measures the bond that is actually there.
    #[test]
    fn a_crossing_bond_is_rewritten_to_its_closest_copy() {
        let bx = SimBox::cube(10.0, array![0.0_f64, 0.0, 0.0], [true; 3]).unwrap();
        let owned = array![[9.5_f64, 5.0, 5.0], [0.5, 5.0, 5.0]];
        let mut frame = Frame::new();
        frame.insert("bonds", bonds_block(&[0], &[1]));

        let comm = Comm::new(bx.clone(), owned.view(), 2.0, 0.0).unwrap();
        let remapped = BondedTopology::resolve_for_test(&frame, owned.view(), &comm).unwrap();

        let j = remapped.get("bonds").unwrap().get_uint("atomj").unwrap()[0] as usize;
        assert!(j >= 2, "atomj must now name a copy, not atom 1; got {j}");

        // Straight through a real kernel, on the combined coordinates.
        let all = comm.ghosts().combined(owned.view()).unwrap();
        let flat: Vec<F> = all.iter().copied().collect();
        let pot = BondHarmonic::new(vec![0], vec![j], vec![100.0], vec![1.0]);
        let (e, _) = pot.calc_energy_forces(&flat);
        assert!(e.abs() < 1e-9, "the bond is at rest length; got E = {e}");

        // The unrewritten frame is what the bug looks like: same kernel, same
        // coordinates, a bond read as eight times its rest length.
        let naive = BondHarmonic::new(vec![0], vec![1], vec![100.0], vec![1.0]);
        assert!((naive.calc_energy_forces(&flat).0 - 3200.0).abs() < 1e-9);
    }

    /// Both arms of an angle are resolved against the vertex, so the three
    /// atoms end up in one cell rather than in three.
    #[test]
    fn an_angle_is_anchored_on_its_vertex() {
        let bx = SimBox::cube(10.0, array![0.0_f64, 0.0, 0.0], [true; 3]).unwrap();
        // Vertex just inside the low face; the arms straddle it.
        let owned = array![[9.3_f64, 5.0, 5.0], [0.2, 5.0, 5.0], [1.2, 5.0, 5.0]];
        let mut frame = Frame::new();
        frame.insert("angles", angles_block(&[0], &[1], &[2]));

        let comm = Comm::new(bx.clone(), owned.view(), 3.0, 0.0).unwrap();
        let remapped = BondedTopology::resolve_for_test(&frame, owned.view(), &comm).unwrap();
        let blk = remapped.get("angles").unwrap();
        let (i, j, k) = (
            blk.get_uint("atomi").unwrap()[0] as usize,
            blk.get_uint("atomj").unwrap()[0] as usize,
            blk.get_uint("atomk").unwrap()[0] as usize,
        );
        assert_eq!(j, 1, "the vertex is the anchor and never moves");
        assert!(i >= 3, "the arm across the face must be a copy; got {i}");
        assert_eq!(k, 2, "the arm already beside the vertex is left alone");

        let all = comm.ghosts().combined(owned.view()).unwrap();
        let d = |a: usize, b: usize| {
            ((0..3)
                .map(|c| (all[[a, c]] - all[[b, c]]).powi(2))
                .sum::<F>())
            .sqrt()
        };
        assert!((d(i, j) - 0.9).abs() < 1e-9, "arm i-j is {}", d(i, j));
        assert!((d(k, j) - 1.0).abs() < 1e-9, "arm k-j is {}", d(k, j));
    }

    /// A molecule longer than the halo cannot be resolved, and saying so is the
    /// whole improvement: the alternative is a bond measured against whichever
    /// copy happened to exist.
    #[test]
    fn a_term_the_halo_cannot_reach_is_refused() {
        let bx = SimBox::cube(30.0, array![0.0_f64, 0.0, 0.0], [true; 3]).unwrap();
        // Bonded, but 14 Å apart — beyond a halo built for a 2 Å pair cutoff.
        let owned = array![[1.0_f64, 5.0, 5.0], [15.0, 5.0, 5.0]];
        let mut frame = Frame::new();
        frame.insert("bonds", bonds_block(&[0], &[1]));

        let comm = Comm::new(bx, owned.view(), 2.0, 0.0).unwrap();
        let err = BondedTopology::resolve_for_test(&frame, owned.view(), &comm).unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("still spans"), "{msg}");
        assert!(msg.contains("does not reach"), "{msg}");
    }

    /// A frame with no bonded blocks passes through untouched — the ghost path
    /// must not require a topology it was not given.
    #[test]
    fn a_frame_without_topology_is_unchanged() {
        let bx = SimBox::cube(10.0, array![0.0_f64, 0.0, 0.0], [true; 3]).unwrap();
        let owned = array![[1.0_f64, 1.0, 1.0], [2.0, 2.0, 2.0]];
        let comm = Comm::new(bx, owned.view(), 2.0, 0.0).unwrap();
        let out = BondedTopology::resolve_for_test(&Frame::new(), owned.view(), &comm).unwrap();
        assert!(out.get("bonds").is_none());
    }
}

#[cfg(test)]
mod owned_potential_tests {
    use super::*;
    use molrs::ff::forcefield::ForceField;
    use molrs::spatial::simbox::SimBox;
    use molrs::store::block::Block;
    use ndarray::{Array1, array};

    /// Two atoms bonded across a face, drifting until the halo rebuilds.
    ///
    /// The bond is a pair of *indices*, and which copy those indices should
    /// name is re-decided at every rebuild. Left to the caller that is a
    /// cadence to get right and silent to get wrong: the kernels keep reading
    /// whatever indices they were built with. So the halo re-resolves them in
    /// the same call that rebuilt the copies, and this checks that it does —
    /// the bonded energy must stay smooth across a rebuild, not step.
    #[test]
    fn the_halo_re_resolves_its_topology_when_it_rebuilds() {
        let l = 20.0_f64;
        let bx = SimBox::cube(l, array![0.0_f64, 0.0, 0.0], [true; 3]).unwrap();
        // Bonded, 1 Å apart the short way round the +x face — and parked in a
        // corner, with two spectator atoms beside them. Drifting on all three
        // axes churns which copies survive the cull, so the ghost list is
        // reordered from one rebuild to the next and an index kept across one
        // names a different copy than it did. A pair alone in the middle of a
        // face would not show that: its copy list is stable, and a stale index
        // would keep working by luck.
        let mut owned = array![
            [19.5_f64, 19.4, 19.6],
            [0.5, 19.4, 19.6],
            [19.3, 0.4, 19.2],
            [0.7, 19.8, 0.3],
        ];

        let mut bonds = Block::new();
        bonds
            .insert("atomi", Array1::from(vec![0 as Idx]).into_dyn())
            .unwrap();
        bonds
            .insert("atomj", Array1::from(vec![1 as Idx]).into_dyn())
            .unwrap();
        bonds
            .insert("type", Array1::from(vec!["a-a".to_string()]).into_dyn())
            .unwrap();
        let mut frame = Frame::new();
        frame.insert("bonds", bonds);
        let mut atoms = Block::new();
        atoms
            .insert("type", Array1::from(vec!["a".to_string(); 4]).into_dyn())
            .unwrap();
        frame.insert("atoms", atoms);

        // A real bond style, with a rest length equal to the bond's actual
        // length, so a correctly-resolved bond has zero energy and a
        // mis-resolved one does not.
        let mut field = ForceField::new("probe");
        let bs = field.def_bondstyle("harmonic");
        bs.def_bondtype("a", "a", &[("k", 100.0), ("r0", 1.0)]);

        // A small skin, so drifting forces several rebuilds over the run.
        let mut comm = Comm::new(bx.clone(), owned.view(), 3.0, 0.2).unwrap();
        let mut bonded = BondedTopology::new(frame, field, &comm, owned.view()).unwrap();

        let mut seen_rebuild = false;
        for step in 0..60 {
            // Both atoms drift together: the bond length never changes, so a
            // bonded energy that moves at all is the bookkeeping's doing.
            for i in 0..owned.nrows() {
                owned[[i, 0]] += 0.02;
                owned[[i, 1]] += 0.013;
                owned[[i, 2]] += 0.017;
            }
            let (wrapped, m) = bx.wrap_shifts(owned.view());
            owned = wrapped;

            let before = comm.rebuilds();
            comm.advance(owned.view(), m.view()).unwrap();
            bonded.refresh(&comm, owned.view(), m.view()).unwrap();
            let all = comm.combined(owned.view()).unwrap();
            let flat: Vec<F> = all.iter().copied().collect();
            let (e, _f) = bonded.potentials().calc_energy_forces(&flat);
            let n_terms = bonded.potentials().len();
            let all_rows = all.nrows();

            // Without this the assertion below is satisfied by an empty
            // potential, which would prove nothing at all.
            assert!(
                n_terms > 0,
                "step {step}: the force field produced no terms"
            );

            assert!(
                e.abs() < 1e-9,
                "step {step}: bonded energy {e}; the bond never changed length"
            );
            assert!(all_rows >= owned.nrows());
            if comm.rebuilds() > before {
                seen_rebuild = true;
            }
        }
        assert!(
            seen_rebuild,
            "the run must cross a rebuild, or it proves nothing"
        );
    }

    /// The invariant behind the re-resolution, asserted directly: a bonded
    /// topology is always resolved against the copy list that exists now.
    ///
    /// The re-resolution used to be conditioned on a boolean the caller set
    /// when it rebuilt. That is a thing to remember, and a test can only catch
    /// forgetting it if the fixture happens to reorder the list — which a small
    /// one does not. Keyed on the generation instead, the condition *is* the
    /// property, and this checks it every step rather than hoping a wrong
    /// answer shows up in an energy.
    #[test]
    fn the_topology_is_never_resolved_against_a_copy_list_that_is_gone() {
        let l = 20.0_f64;
        let bx = SimBox::cube(l, array![0.0_f64, 0.0, 0.0], [true; 3]).unwrap();
        let mut owned = array![
            [19.5_f64, 19.4, 19.6],
            [0.5, 19.4, 19.6],
            [19.3, 0.4, 19.2],
            [0.7, 19.8, 0.3],
        ];

        let mut bonds = Block::new();
        bonds
            .insert("atomi", Array1::from(vec![0 as Idx]).into_dyn())
            .unwrap();
        bonds
            .insert("atomj", Array1::from(vec![1 as Idx]).into_dyn())
            .unwrap();
        bonds
            .insert("type", Array1::from(vec!["a-a".to_string()]).into_dyn())
            .unwrap();
        let mut frame = Frame::new();
        frame.insert("bonds", bonds);
        let mut atoms = Block::new();
        atoms
            .insert("type", Array1::from(vec!["a".to_string(); 4]).into_dyn())
            .unwrap();
        frame.insert("atoms", atoms);

        let mut field = ForceField::new("probe");
        let bs = field.def_bondstyle("harmonic");
        bs.def_bondtype("a", "a", &[("k", 100.0), ("r0", 1.0)]);

        let mut comm = Comm::new(bx.clone(), owned.view(), 3.0, 0.2).unwrap();
        let mut bonded = BondedTopology::new(frame, field, &comm, owned.view()).unwrap();

        let mut generations: Vec<u64> = vec![comm.ghosts().generation()];
        for _ in 0..60 {
            for i in 0..owned.nrows() {
                owned[[i, 0]] += 0.02;
                owned[[i, 1]] += 0.013;
                owned[[i, 2]] += 0.017;
            }
            let (wrapped, m) = bx.wrap_shifts(owned.view());
            owned = wrapped;
            comm.advance(owned.view(), m.view()).unwrap();
            bonded.refresh(&comm, owned.view(), m.view()).unwrap();
            assert_eq!(
                bonded.generation(),
                comm.ghosts().generation(),
                "the topology must name copies that exist"
            );
            let g = comm.ghosts().generation();
            if *generations.last().unwrap() != g {
                generations.push(g);
            }
        }
        assert!(
            generations.len() > 2,
            "the copy list must have been rebuilt more than once; saw {} generations",
            generations.len()
        );
    }
}
