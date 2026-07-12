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

use std::collections::HashMap;

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
}

// ---------------------------------------------------------------------------
// Public entry point
// ---------------------------------------------------------------------------

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
    }
}
