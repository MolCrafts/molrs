//! Ring detection for molecular graphs — the **handle-keyed chemistry
//! decoration** over the core graph primitive.
//!
//! There is exactly **one** SSSR implementation in the tree and it lives in
//! [`crate::system::topology`]: [`Topology::find_rings`] computes the
//! **Smallest Set of Smallest Rings** (equivalently the minimum cycle basis)
//! over contiguous `usize` vertex/edge indices. This module owns no ring
//! algorithm of its own; it only:
//!
//! 1. projects an [`Atomistic`] onto that index space — atom index `i` is the
//!    `i`-th atom of [`Atomistic::atoms`], edge index `i` the `i`-th bond of
//!    [`Atomistic::bonds`],
//! 2. calls [`Topology::find_rings`],
//! 3. lifts the resulting `usize` indices back onto the [`AtomId`] / [`BondId`]
//!    handles chemistry code (aromaticity, SMARTS, MMFF, AM1-BCC, the conformer
//!    pipeline) actually holds, as a [`RingInfo`].
//!
//! Ring selection, ordering and complexity are therefore whatever the core
//! primitive says they are — see [`Topology::find_rings`].

use std::collections::{HashMap, HashSet, VecDeque};

use crate::system::atomistic::{AtomId, Atomistic, BondId};
use crate::system::topology::Topology;

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// All ring information for an [`Atomistic`], produced by [`find_rings`].
///
/// The handle-keyed counterpart of
/// [`crate::system::topology::TopologyRingInfo`]: the same rings, addressed by
/// [`AtomId`] / [`BondId`] instead of by graph index.
#[derive(Debug, Clone)]
pub struct RingInfo {
    /// Each ring is an ordered list of `AtomId`s forming a closed path.
    rings: Vec<Vec<AtomId>>,
    /// atom → indices of rings that contain it.
    atom_rings: HashMap<AtomId, Vec<usize>>,
    /// bond → indices of rings that contain it.
    bond_rings: HashMap<BondId, Vec<usize>>,
}

impl RingInfo {
    /// Whether the atom belongs to any ring.
    pub fn is_atom_in_ring(&self, id: AtomId) -> bool {
        self.atom_rings.get(&id).is_some_and(|v| !v.is_empty())
    }

    /// Whether the bond belongs to any ring.
    pub fn is_bond_in_ring(&self, id: BondId) -> bool {
        self.bond_rings.get(&id).is_some_and(|v| !v.is_empty())
    }

    /// Number of rings containing this atom.
    pub fn num_atom_rings(&self, id: AtomId) -> usize {
        self.atom_rings.get(&id).map_or(0, Vec::len)
    }

    /// Number of rings containing this bond.
    pub fn num_bond_rings(&self, id: BondId) -> usize {
        self.bond_rings.get(&id).map_or(0, Vec::len)
    }

    /// Size (atom count) of every ring, in ascending order.
    pub fn ring_sizes(&self) -> Vec<usize> {
        self.rings.iter().map(Vec::len).collect()
    }

    /// All rings of exactly `n` atoms.
    pub fn rings_of_size(&self, n: usize) -> Vec<&Vec<AtomId>> {
        self.rings.iter().filter(|r| r.len() == n).collect()
    }

    /// Size of the smallest ring containing `id`, if any.
    pub fn smallest_ring_containing_atom(&self, id: AtomId) -> Option<usize> {
        self.atom_rings
            .get(&id)?
            .iter()
            .map(|&ri| self.rings[ri].len())
            .min()
    }

    /// Total number of rings detected.
    pub fn num_rings(&self) -> usize {
        self.rings.len()
    }

    /// All rings as slices of `AtomId`.
    pub fn rings(&self) -> &[Vec<AtomId>] {
        &self.rings
    }

    /// The **ring systems**: rings that share at least one atom, unioned into
    /// one atom list each. Systems are disjoint and cover every ring atom.
    ///
    /// Benzene → one system of 6; naphthalene → one of 10 (not two of 6);
    /// biphenyl → two of 6; a spiro pair → one system (they share their spiro
    /// atom). Acyclic molecules → empty.
    ///
    /// This is RDKit's `GetRingSystems` recipe (RDKit Cookbook, "Count Ring
    /// Systems"): walk the rings, merging any that intersect one already
    /// collected. The recipe's `includeSpiro` switch decides whether a
    /// single shared atom is enough to merge (`nInCommon > 1` when off); this
    /// method always merges on one, i.e. `includeSpiro=True`. The switch only
    /// changes how systems are *counted* — for any caller that asks "is this
    /// system wholly present?", a spiro atom belongs to both rings, so both
    /// are answered together either way.
    ///
    /// Which ring set feeds this does not matter: the true SSSR and RDKit's
    /// symmetrized `GetSymmSSSR` differ in *which* small rings they list, not
    /// in which atoms are cyclic, and the union over shared atoms collapses
    /// that difference.
    pub fn ring_systems(&self) -> Vec<Vec<AtomId>> {
        let n = self.rings.len();
        if n == 0 {
            return Vec::new();
        }
        // Union-find over rings: share an atom ⇒ same component.
        let mut parent: Vec<usize> = (0..n).collect();
        fn find(parent: &mut [usize], mut i: usize) -> usize {
            while parent[i] != i {
                parent[i] = parent[parent[i]];
                i = parent[i];
            }
            i
        }
        fn unite(parent: &mut [usize], a: usize, b: usize) {
            let ra = find(parent, a);
            let rb = find(parent, b);
            if ra != rb {
                parent[rb] = ra;
            }
        }
        // atom → first ring index seen
        let mut atom_owner: HashMap<AtomId, usize> = HashMap::new();
        for (ri, ring) in self.rings.iter().enumerate() {
            for &atom in ring {
                if let Some(&prev) = atom_owner.get(&atom) {
                    unite(&mut parent, prev, ri);
                } else {
                    atom_owner.insert(atom, ri);
                }
            }
        }
        // Component → atoms, in first-seen order so the result is deterministic
        // without leaking a handle ordering.
        let mut order: Vec<usize> = Vec::new();
        let mut systems: HashMap<usize, Vec<AtomId>> = HashMap::new();
        let mut seen: HashSet<AtomId> = HashSet::new();
        for (ri, ring) in self.rings.iter().enumerate() {
            let root = find(&mut parent, ri);
            let atoms = systems.entry(root).or_insert_with(|| {
                order.push(root);
                Vec::new()
            });
            for &atom in ring {
                if seen.insert(atom) {
                    atoms.push(atom);
                }
            }
        }
        order
            .into_iter()
            .map(|root| systems.remove(&root).expect("root inserted above"))
            .collect()
    }

    /// Size (atom count) of the largest [`ring system`](Self::ring_systems).
    ///
    /// Benzene → 6; naphthalene → 10 (not 6). Acyclic molecules → 0.
    pub fn max_ring_system_size(&self) -> usize {
        self.ring_systems().iter().map(Vec::len).max().unwrap_or(0)
    }
}

// ---------------------------------------------------------------------------
// Public entry point
// ---------------------------------------------------------------------------

/// Size of the largest fused/bridged ring system in `mol` (see
/// [`RingInfo::max_ring_system_size`]).
pub fn max_ring_system_size(mol: &Atomistic) -> usize {
    find_rings(mol).max_ring_system_size()
}

/// The ring systems a `radius`-ball around `centers` must not cut, counting
/// only rings of at most `max_ring_size` atoms.
///
/// Feed the result to
/// [`Atomistic::extract_subgraph`](crate::system::atomistic::Atomistic::extract_subgraph)
/// as its `whole_groups`.
///
/// # Why a size bound, and why local
///
/// "A cut ring is a different molecule" is a claim about **small** rings:
/// aromaticity, ring strain, SMARTS `r3`–`r8`, and MMFF's 3-/4-/6-membered ring
/// parameters all read a local cycle. It is false for a large one. A 5000-atom
/// macrocycle is locally indistinguishable from a chain — no typifier in this
/// tree can tell them apart — yet closing on it drags all 5000 atoms into a
/// ball that wanted 9, which is exactly what happens the moment a crosslink
/// joins two chains into a loop. Unbounded closure does not merely cost time:
/// it defeats region typing, whose whole premise is bounded work per edit.
///
/// With the bound, the question stops needing a global ring decomposition.
/// A bond lies on a ring of at most `max_ring_size` atoms **iff** its endpoints
/// stay connected within `max_ring_size - 1` hops after it is removed — a
/// bounded local search. Closure is then the flood from the ball along exactly
/// those bonds, so the cost follows the ball and its chemical neighbourhood,
/// not the size of the parent. Fused and spiro systems still arrive whole: the
/// flood crosses every small-ring bond it meets, which is what fuses them.
///
/// `max_ring_size < 3` disables closure (no ring has fewer than three atoms).
/// Groups come back one per ring system touched; atoms on no small ring are
/// absent, and an acyclic molecule yields nothing at all.
pub fn small_ring_closure(
    mol: &Atomistic,
    centers: &[AtomId],
    radius: i64,
    max_ring_size: usize,
) -> Vec<Vec<AtomId>> {
    if max_ring_size < 3 || centers.is_empty() {
        return Vec::new();
    }

    // Seed atoms: the plain radius ball. `extract_ball` re-derives the same set;
    // recomputing it here keeps this function self-contained.
    //
    // Deliberately not `Atomistic::topo_distances`: that keys its BFS by a
    // slot-indexed `SecondaryMap`, so it allocates and drains one slot per atom
    // of the *parent* even when `max_hops` stops the search after nine. A
    // hash-keyed BFS costs the ball, which is what this function promises.
    let mut ball: Vec<AtomId> = Vec::new();
    let mut hops: HashMap<AtomId, i64> = HashMap::new();
    let mut queue: VecDeque<AtomId> = VecDeque::new();
    for &center in centers {
        if hops.insert(center, 0).is_none() {
            ball.push(center);
            queue.push_back(center);
        }
    }
    while let Some(atom) = queue.pop_front() {
        let hop = hops[&atom];
        if hop >= radius {
            continue;
        }
        let neighbors: Vec<AtomId> = mol.neighbor_bonds(atom).map(|(other, _)| other).collect();
        for other in neighbors {
            if let std::collections::hash_map::Entry::Vacant(slot) = hops.entry(other) {
                slot.insert(hop + 1);
                ball.push(other);
                queue.push_back(other);
            }
        }
    }

    // One flood per ring system, so an atom is emitted in exactly one group.
    let mut assigned: HashSet<AtomId> = HashSet::new();
    let mut on_small_ring: HashMap<(AtomId, AtomId), bool> = HashMap::new();
    let mut groups: Vec<Vec<AtomId>> = Vec::new();

    for &start in &ball {
        if !assigned.insert(start) {
            continue;
        }
        let mut system = vec![start];
        let mut frontier = vec![start];
        let mut in_system: HashSet<AtomId> = HashSet::from([start]);
        while let Some(atom) = frontier.pop() {
            let neighbors: Vec<AtomId> = mol.neighbor_bonds(atom).map(|(other, _)| other).collect();
            for other in neighbors {
                if in_system.contains(&other) {
                    continue;
                }
                let key = if atom < other {
                    (atom, other)
                } else {
                    (other, atom)
                };
                let cyclic = match on_small_ring.get(&key) {
                    Some(&known) => known,
                    None => {
                        let known = bond_on_small_ring(mol, atom, other, max_ring_size);
                        on_small_ring.insert(key, known);
                        known
                    }
                };
                if cyclic {
                    in_system.insert(other);
                    assigned.insert(other);
                    system.push(other);
                    frontier.push(other);
                }
            }
        }
        // A lone atom is not a ring system — it is an atom on no small ring.
        if system.len() > 1 {
            groups.push(system);
        }
    }
    groups
}

/// Whether the bond between `a` and `b` lies on a ring of at most
/// `max_ring_size` atoms.
///
/// Bounded BFS for an alternative `a`→`b` route: a route of `L` bonds closes a
/// ring of `L + 1` atoms, so the search stops at `max_ring_size - 1` hops. Cost
/// is the neighbourhood within that many bonds, never the whole molecule.
fn bond_on_small_ring(mol: &Atomistic, a: AtomId, b: AtomId, max_ring_size: usize) -> bool {
    let max_hops = max_ring_size - 1;
    let mut visited: HashSet<AtomId> = HashSet::from([a]);
    let mut queue: VecDeque<(AtomId, usize)> = VecDeque::from([(a, 0usize)]);
    while let Some((atom, hops)) = queue.pop_front() {
        if hops >= max_hops {
            continue;
        }
        for (other, _) in mol.neighbor_bonds(atom) {
            // The bond under test is not a route around itself.
            if atom == a && other == b {
                continue;
            }
            if other == b {
                return true;
            }
            if visited.insert(other) {
                queue.push_back((other, hops + 1));
            }
        }
    }
    false
}

/// Compute the ring information (SSSR / minimum cycle basis) for `mol`.
///
/// Delegates the actual cycle search to the core graph primitive
/// [`Topology::find_rings`] and lifts its `usize` indices back onto
/// [`AtomId`] / [`BondId`] handles. Rings come back smallest-first, exactly as
/// the primitive orders them.
pub fn find_rings(mol: &Atomistic) -> RingInfo {
    // ---- 1. Project onto the core index space ------------------------------
    // Atom index `i` == the `i`-th atom of `mol.atoms()`; edge index `i` == the
    // `i`-th bond of `mol.bonds()` (`Topology::from_edges` preserves edge
    // insertion order), so the mapping back below is unambiguous.
    let atom_vec: Vec<AtomId> = mol.atoms().map(|(id, _)| id).collect();
    let atom_to_idx: HashMap<AtomId, usize> = atom_vec
        .iter()
        .enumerate()
        .map(|(i, &id)| (id, i))
        .collect();

    let bond_vec: Vec<BondId> = mol.bonds().map(|(id, _)| id).collect();

    let mut edges: Vec<[usize; 2]> = Vec::with_capacity(bond_vec.len());
    // Fast (AtomId, AtomId) → BondId lookup, used to lift ring edges back.
    let mut bond_map: HashMap<(AtomId, AtomId), BondId> = HashMap::new();
    for &bid in &bond_vec {
        let (n0, n1) = mol.bond_endpoints(bid).expect("bond must exist");
        edges.push([atom_to_idx[&n0], atom_to_idx[&n1]]);
        bond_map.insert((n0, n1), bid);
        bond_map.insert((n1, n0), bid);
    }

    // ---- 2. The one and only SSSR in the tree -------------------------------
    let topo = Topology::from_edges(atom_vec.len(), &edges);
    let info = topo.find_rings();

    // ---- 3. Lift node-index cycles → AtomId rings ---------------------------
    let rings: Vec<Vec<AtomId>> = info
        .rings()
        .iter()
        .map(|cycle| cycle.iter().map(|&ni| atom_vec[ni]).collect())
        .collect();

    // ---- 4. Build the handle-keyed reverse-lookup maps ----------------------
    // Each ring is a closed path, so its cyclically consecutive atom pairs are
    // exactly its bonds.
    let mut atom_rings: HashMap<AtomId, Vec<usize>> = HashMap::new();
    let mut bond_rings: HashMap<BondId, Vec<usize>> = HashMap::new();

    for (ri, ring) in rings.iter().enumerate() {
        let n = ring.len();
        for i in 0..n {
            let a = ring[i];
            let b = ring[(i + 1) % n];
            atom_rings.entry(a).or_default().push(ri);
            if let Some(&bid) = bond_map.get(&(a, b)) {
                bond_rings.entry(bid).or_default().push(ri);
            }
        }
    }

    RingInfo {
        rings,
        atom_rings,
        bond_rings,
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::system::molgraph::Atom;

    fn cycle(n: usize) -> Atomistic {
        let mut g = Atomistic::new();
        let ids: Vec<AtomId> = (0..n).map(|_| g.add_atom(Atom::new())).collect();
        for i in 0..n {
            g.add_bond(ids[i], ids[(i + 1) % n])
                .expect("add cycle bond");
        }
        g
    }

    #[test]
    fn test_single_6ring() {
        let g = cycle(6);
        let ri = find_rings(&g);
        assert_eq!(ri.num_rings(), 1);
        assert_eq!(ri.ring_sizes(), vec![6]);
    }

    #[test]
    fn test_linear_no_rings() {
        let mut g = Atomistic::new();
        let ids: Vec<AtomId> = (0..6).map(|_| g.add_atom(Atom::new())).collect();
        for i in 0..5 {
            g.add_bond(ids[i], ids[i + 1]).expect("add chain bond");
        }
        assert_eq!(find_rings(&g).num_rings(), 0);
    }

    #[test]
    fn test_all_atoms_in_6ring() {
        let g = cycle(6);
        let ri = find_rings(&g);
        for (id, _) in g.atoms() {
            assert!(ri.is_atom_in_ring(id));
        }
    }

    #[test]
    fn test_all_bonds_in_6ring() {
        let g = cycle(6);
        let ri = find_rings(&g);
        for (bid, _) in g.bonds() {
            assert!(ri.is_bond_in_ring(bid));
        }
    }

    #[test]
    fn test_empty_mol() {
        assert_eq!(find_rings(&Atomistic::new()).num_rings(), 0);
    }

    #[test]
    fn test_naphthalene() {
        let mut g = Atomistic::new();
        let ids: Vec<AtomId> = (0..10).map(|_| g.add_atom(Atom::new())).collect();
        // Ring A: 0-1-2-3-4-5-0
        for i in 0..5 {
            g.add_bond(ids[i], ids[i + 1]).expect("bond");
        }
        g.add_bond(ids[5], ids[0]).expect("bond");
        // Ring B: 2-3-6-7-8-9-2
        g.add_bond(ids[3], ids[6]).expect("bond");
        g.add_bond(ids[6], ids[7]).expect("bond");
        g.add_bond(ids[7], ids[8]).expect("bond");
        g.add_bond(ids[8], ids[9]).expect("bond");
        g.add_bond(ids[9], ids[2]).expect("bond");

        let ri = find_rings(&g);
        assert_eq!(ri.num_rings(), 2);
        let mut sizes = ri.ring_sizes();
        sizes.sort_unstable();
        assert_eq!(sizes, vec![6, 6]);
        // Fused system: 10 carbons, not a single 6-ring.
        assert_eq!(ri.max_ring_system_size(), 10);
        assert_eq!(max_ring_system_size(&g), 10);
    }

    #[test]
    fn test_max_ring_system_size_benzene_and_acyclic() {
        assert_eq!(find_rings(&cycle(6)).max_ring_system_size(), 6);
        let mut chain = Atomistic::new();
        let ids: Vec<AtomId> = (0..3).map(|_| chain.add_atom(Atom::new())).collect();
        chain.add_bond(ids[0], ids[1]).unwrap();
        chain.add_bond(ids[1], ids[2]).unwrap();
        assert_eq!(find_rings(&chain).max_ring_system_size(), 0);
    }

    // -- ring systems -------------------------------------------------------

    /// Two `n`-cycles joined at `shared` consecutive atoms: `shared = 2` is
    /// ortho-fused (naphthalene-like), `shared = 1` is spiro.
    fn joined_cycles(n: usize, shared: usize) -> (Atomistic, Vec<AtomId>) {
        let mut g = Atomistic::new();
        let total = 2 * n - shared;
        let ids: Vec<AtomId> = (0..total).map(|_| g.add_atom(Atom::new())).collect();
        for i in 0..n {
            g.add_bond(ids[i], ids[(i + 1) % n]).expect("ring A bond");
        }
        // Ring B reuses ids[0..shared] and continues through the fresh atoms.
        let mut ring_b: Vec<AtomId> = (0..shared).map(|i| ids[i]).collect();
        ring_b.extend(ids[n..].iter().copied());
        for i in 0..ring_b.len() {
            g.add_bond(ring_b[i], ring_b[(i + 1) % ring_b.len()])
                .expect("ring B bond");
        }
        (g, ids)
    }

    #[test]
    fn test_ring_systems_benzene_is_one_system_of_six() {
        let systems = find_rings(&cycle(6)).ring_systems();
        assert_eq!(systems.len(), 1);
        assert_eq!(systems[0].len(), 6);
    }

    #[test]
    fn test_ring_systems_fused_pair_is_one_system() {
        let (g, _) = joined_cycles(6, 2);
        let systems = find_rings(&g).ring_systems();
        assert_eq!(systems.len(), 1, "naphthalene is one system, not two");
        assert_eq!(systems[0].len(), 10);
    }

    #[test]
    fn test_ring_systems_spiro_pair_is_one_system() {
        // RDKit's GetRingSystems would count these as two with the default
        // includeSpiro=False; for closure they are one, because the shared atom
        // sits in both rings and cannot be present without completing both.
        let (g, _) = joined_cycles(6, 1);
        let systems = find_rings(&g).ring_systems();
        assert_eq!(systems.len(), 1);
        assert_eq!(systems[0].len(), 11);
    }

    #[test]
    fn test_ring_systems_disconnected_rings_stay_separate() {
        let mut g = cycle(6);
        let ids: Vec<AtomId> = (0..6).map(|_| g.add_atom(Atom::new())).collect();
        for i in 0..6 {
            g.add_bond(ids[i], ids[(i + 1) % 6]).expect("second ring");
        }
        let systems = find_rings(&g).ring_systems();
        assert_eq!(systems.len(), 2);
        assert!(systems.iter().all(|s| s.len() == 6));
    }

    #[test]
    fn test_ring_systems_acyclic_is_empty() {
        let mut chain = Atomistic::new();
        let ids: Vec<AtomId> = (0..3).map(|_| chain.add_atom(Atom::new())).collect();
        chain.add_bond(ids[0], ids[1]).unwrap();
        chain.add_bond(ids[1], ids[2]).unwrap();
        assert!(find_rings(&chain).ring_systems().is_empty());
    }

    #[test]
    fn test_ring_systems_partition_every_ring_atom_exactly_once() {
        let (g, _) = joined_cycles(6, 2);
        let ri = find_rings(&g);
        let mut seen: HashSet<AtomId> = HashSet::new();
        for system in ri.ring_systems() {
            for atom in system {
                assert!(seen.insert(atom), "atom in two systems");
            }
        }
        for (id, _) in g.atoms() {
            assert_eq!(ri.is_atom_in_ring(id), seen.contains(&id));
        }
    }

    // -- the reason ring systems exist: extraction must not cut one ---------

    #[test]
    fn test_extraction_closed_on_ring_systems_never_cuts_a_ring() {
        // A methyl hanging off a 6-ring. A radius-2 ball from the methyl carbon
        // reaches three ring atoms — half a ring, which perceives as a different
        // molecule than the one it was cut from.
        // Extraction re-wraps as an Atomistic, so these atoms need an element.
        let mut g = Atomistic::new();
        let ring_ids: Vec<AtomId> = (0..6).map(|_| g.add_atom_bare("C")).collect();
        for i in 0..6 {
            g.add_bond(ring_ids[i], ring_ids[(i + 1) % 6])
                .expect("ring bond");
        }
        let methyl = g.add_atom_bare("C");
        g.add_bond(ring_ids[0], methyl).expect("methyl bond");

        let ri = find_rings(&g);
        let plain = g.extract_subgraph(&[methyl], 2, false, &[]).unwrap();
        assert_eq!(plain.graph.n_atoms(), 4, "half the ring, plus the methyl");

        let closed = g
            .extract_subgraph(&[methyl], 2, false, &ri.ring_systems())
            .unwrap();
        assert_eq!(closed.graph.n_atoms(), 7, "whole ring, plus the methyl");
        let selected: HashSet<AtomId> = closed.node_map.keys().copied().collect();
        for system in ri.ring_systems() {
            let inside = system.iter().filter(|a| selected.contains(a)).count();
            assert!(
                inside == 0 || inside == system.len(),
                "ring system split: {inside} of {} atoms",
                system.len()
            );
        }
        // The atoms the radius alone would have reached keep their true distance.
        assert_eq!(closed.hops[&methyl], 0);
        assert_eq!(closed.hops[&ring_ids[0]], 1);
        assert_eq!(closed.hops[&ring_ids[1]], 2);
        // The far side of the ring came in through the closure, beyond the radius.
        assert_eq!(closed.hops[&ring_ids[3]], 4);
    }

    // -----------------------------------------------------------------------
    // small_ring_closure — bounded, local
    // -----------------------------------------------------------------------

    /// A ring of `n` carbons with a methyl on atom 0. Returns `(graph, methyl)`.
    fn methyl_on_ring(n: usize) -> (Atomistic, Vec<AtomId>, AtomId) {
        let mut g = Atomistic::new();
        let ring: Vec<AtomId> = (0..n).map(|_| g.add_atom_bare("C")).collect();
        for i in 0..n {
            g.add_bond(ring[i], ring[(i + 1) % n]).expect("ring bond");
        }
        let methyl = g.add_atom_bare("C");
        g.add_bond(ring[0], methyl).expect("methyl bond");
        (g, ring, methyl)
    }

    #[test]
    fn test_small_ring_closure_returns_the_whole_benzene_from_its_methyl() {
        let (g, ring, methyl) = methyl_on_ring(6);
        let groups = small_ring_closure(&g, &[methyl], 1, 8);
        assert_eq!(groups.len(), 1, "one ring system");
        let closed: HashSet<AtomId> = groups[0].iter().copied().collect();
        assert_eq!(closed, ring.iter().copied().collect::<HashSet<_>>());
        // The exocyclic bond is on no ring, so the methyl is not part of it.
        assert!(!closed.contains(&methyl));
    }

    #[test]
    fn test_small_ring_closure_fuses_naphthalene_into_one_group() {
        let (g, ids) = joined_cycles(6, 2);
        let groups = small_ring_closure(&g, &[ids[0]], 0, 8);
        assert_eq!(groups.len(), 1, "fused rings are one system");
        assert_eq!(groups[0].len(), 10);
    }

    #[test]
    fn test_small_ring_closure_fuses_a_spiro_pair_into_one_group() {
        let (g, ids) = joined_cycles(6, 1);
        let groups = small_ring_closure(&g, &[ids[0]], 0, 8);
        assert_eq!(groups.len(), 1, "a shared atom fuses the two rings");
        assert_eq!(groups[0].len(), 11);
    }

    #[test]
    fn test_small_ring_closure_ignores_a_macrocycle() {
        // The whole point of the bound. A 50-membered ring is topologically a
        // ring and locally a chain; before the bound existed this pulled all 50
        // atoms in to type the handful the ball actually wanted.
        let (g, _ring, methyl) = methyl_on_ring(50);
        assert!(
            small_ring_closure(&g, &[methyl], 2, 8).is_empty(),
            "a 50-ring is not a small ring"
        );
        // Raise the bound past the ring and it comes back — the bound is the
        // only thing deciding this, not some other property of the graph.
        let groups = small_ring_closure(&g, &[methyl], 2, 51);
        assert_eq!(groups.len(), 1);
        assert_eq!(groups[0].len(), 50);
    }

    #[test]
    fn test_small_ring_closure_stops_at_the_bond_between_two_rings() {
        // Biphenyl: the inter-ring bond lies on no small ring, so a ball on one
        // ring must not drag in the other.
        let mut g = Atomistic::new();
        let ring_of = |g: &mut Atomistic| -> Vec<AtomId> {
            let r: Vec<AtomId> = (0..6).map(|_| g.add_atom_bare("C")).collect();
            for i in 0..6 {
                g.add_bond(r[i], r[(i + 1) % 6]).expect("ring bond");
            }
            r
        };
        let left = ring_of(&mut g);
        let right = ring_of(&mut g);
        g.add_bond(left[0], right[0]).expect("biaryl bond");

        let groups = small_ring_closure(&g, &[left[3]], 1, 8);
        assert_eq!(groups.len(), 1, "only the ring the ball is on");
        assert_eq!(groups[0].len(), 6);
    }

    #[test]
    fn test_small_ring_closure_is_empty_for_an_acyclic_molecule() {
        let mut g = Atomistic::new();
        let ids: Vec<AtomId> = (0..8).map(|_| g.add_atom_bare("C")).collect();
        for i in 0..7 {
            g.add_bond(ids[i], ids[i + 1]).expect("chain bond");
        }
        assert!(small_ring_closure(&g, &[ids[3]], 3, 8).is_empty());
    }

    #[test]
    fn test_small_ring_closure_is_disabled_below_three_atoms() {
        let (g, _ring, methyl) = methyl_on_ring(6);
        assert!(small_ring_closure(&g, &[methyl], 2, 2).is_empty());
    }

    #[test]
    fn test_small_ring_closure_agrees_with_sssr_ring_systems_on_small_rings() {
        // Where both are defined — every ring within the bound — the local
        // answer must be the global one. Otherwise the bound is not the only
        // difference between them.
        for (name, g, seed) in [
            ("benzene + methyl", methyl_on_ring(6).0, 0usize),
            ("naphthalene", joined_cycles(6, 2).0, 0),
            ("spiro pair", joined_cycles(6, 1).0, 0),
            ("cyclopropane", cycle(3), 0),
        ] {
            let atoms: Vec<AtomId> = g.atoms().map(|(id, _)| id).collect();
            let local = small_ring_closure(&g, &[atoms[seed]], 0, 8);
            let global: Vec<Vec<AtomId>> = find_rings(&g)
                .ring_systems()
                .into_iter()
                .filter(|system| system.contains(&atoms[seed]))
                .collect();
            let as_set = |groups: &[Vec<AtomId>]| -> HashSet<AtomId> {
                groups.iter().flatten().copied().collect()
            };
            assert_eq!(
                as_set(&local),
                as_set(&global),
                "{name}: local closure disagrees with the SSSR ring system"
            );
        }
    }

    #[test]
    fn test_extraction_with_small_ring_closure_keeps_the_ball_bounded() {
        // The end-to-end claim: on a small ring the ball still arrives whole,
        // and on a macrocycle it stays the size the radius asked for.
        let (benzene, _, methyl) = methyl_on_ring(6);
        let groups = small_ring_closure(&benzene, &[methyl], 2, 8);
        let closed = benzene
            .extract_subgraph(&[methyl], 2, false, &groups)
            .unwrap();
        assert_eq!(closed.graph.n_atoms(), 7, "whole ring, plus the methyl");

        let (macro_ring, _, methyl) = methyl_on_ring(500);
        let groups = small_ring_closure(&macro_ring, &[methyl], 2, 8);
        let closed = macro_ring
            .extract_subgraph(&[methyl], 2, false, &groups)
            .unwrap();
        assert_eq!(
            closed.graph.n_atoms(),
            4,
            "a macrocycle must not pull its whole loop into the ball"
        );
    }
}
