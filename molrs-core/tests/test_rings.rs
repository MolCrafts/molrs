//! Integration tests for ring detection (src/core/rings.rs).

use molrs::{Atom, AtomId, MolGraph, find_rings};

fn carbon() -> Atom {
    let mut a = Atom::new();
    a.set("symbol", "C");
    a
}

fn bond(g: &mut MolGraph, a: AtomId, b: AtomId) {
    g.add_bond(a, b).expect("add bond");
}

/// Build a simple cycle graph (n atoms in a ring).
fn cycle(n: usize) -> MolGraph {
    let mut g = MolGraph::new();
    let ids: Vec<AtomId> = (0..n).map(|_| g.add_atom(carbon())).collect();
    for i in 0..n {
        bond(&mut g, ids[i], ids[(i + 1) % n]);
    }
    g
}

// ── Benzene (6-membered ring) ────────────────────────────────────────────────

#[test]
fn test_benzene_one_ring() {
    let g = cycle(6);
    let ri = find_rings(&g);
    assert_eq!(ri.num_rings(), 1, "benzene should have exactly 1 SSSR ring");
    assert_eq!(ri.ring_sizes(), vec![6]);
}

#[test]
fn test_benzene_all_atoms_in_ring() {
    let g = cycle(6);
    let ri = find_rings(&g);
    for (id, _) in g.atoms() {
        assert!(ri.is_atom_in_ring(id), "atom {:?} should be in a ring", id);
        assert_eq!(ri.num_atom_rings(id), 1);
    }
}

#[test]
fn test_benzene_all_bonds_in_ring() {
    let g = cycle(6);
    let ri = find_rings(&g);
    for (bid, _) in g.bonds() {
        assert!(
            ri.is_bond_in_ring(bid),
            "bond {:?} should be in a ring",
            bid
        );
        assert_eq!(ri.num_bond_rings(bid), 1);
    }
}

#[test]
fn test_benzene_smallest_ring_size() {
    let g = cycle(6);
    let ri = find_rings(&g);
    for (id, _) in g.atoms() {
        assert_eq!(ri.smallest_ring_containing_atom(id), Some(6));
    }
}

// ── n-Hexane (no rings) ──────────────────────────────────────────────────────

#[test]
fn test_hexane_no_rings() {
    let mut g = MolGraph::new();
    let ids: Vec<AtomId> = (0..6).map(|_| g.add_atom(carbon())).collect();
    for i in 0..5 {
        bond(&mut g, ids[i], ids[i + 1]);
    }
    let ri = find_rings(&g);
    assert_eq!(ri.num_rings(), 0);
    for (id, _) in g.atoms() {
        assert!(!ri.is_atom_in_ring(id));
        assert_eq!(ri.smallest_ring_containing_atom(id), None);
    }
}

// ── Naphthalene (two fused 6-rings sharing one bond) ────────────────────────
//
//  0-1-2-3-4-5-0   (ring A: atoms 0..5)
//  2-3-6-7-8-9-2   (ring B: atoms 2,3,6..9 — shares bond 2-3)

#[test]
fn test_naphthalene_two_rings() {
    let mut g = MolGraph::new();
    // 10 atoms (0-9)
    let ids: Vec<AtomId> = (0..10).map(|_| g.add_atom(carbon())).collect();

    // Ring A: 0-1-2-3-4-5-0
    for i in 0..5 {
        bond(&mut g, ids[i], ids[i + 1]);
    }
    bond(&mut g, ids[5], ids[0]);

    // Ring B: 2-3-6-7-8-9-2 (shared bond 2-3 already added)
    bond(&mut g, ids[3], ids[6]);
    bond(&mut g, ids[6], ids[7]);
    bond(&mut g, ids[7], ids[8]);
    bond(&mut g, ids[8], ids[9]);
    bond(&mut g, ids[9], ids[2]);

    let ri = find_rings(&g);
    assert_eq!(ri.num_rings(), 2, "naphthalene SSSR = 2 rings");
    let mut sizes = ri.ring_sizes();
    sizes.sort_unstable();
    assert_eq!(sizes, vec![6, 6]);
}

#[test]
fn test_naphthalene_shared_bond_in_two_rings() {
    let mut g = MolGraph::new();
    let ids: Vec<AtomId> = (0..10).map(|_| g.add_atom(carbon())).collect();
    for i in 0..5 {
        bond(&mut g, ids[i], ids[i + 1]);
    }
    bond(&mut g, ids[5], ids[0]);
    bond(&mut g, ids[3], ids[6]);
    bond(&mut g, ids[6], ids[7]);
    bond(&mut g, ids[7], ids[8]);
    bond(&mut g, ids[8], ids[9]);
    bond(&mut g, ids[9], ids[2]);

    let ri = find_rings(&g);

    // The shared bond (2-3) should be in 2 rings.
    let shared_bond = g
        .bonds()
        .find(|(_, b)| {
            (b.atoms[0] == ids[2] && b.atoms[1] == ids[3])
                || (b.atoms[0] == ids[3] && b.atoms[1] == ids[2])
        })
        .map(|(bid, _)| bid);

    assert!(shared_bond.is_some(), "bond 2-3 should exist");
    assert_eq!(ri.num_bond_rings(shared_bond.unwrap()), 2);
}

// ── Spiro[4.4]nonane (two 5-rings sharing one atom) ─────────────────────────
//
//  Ring A: atoms 0-1-2-3-4-0
//  Ring B: atoms 4-5-6-7-8-4   (atom 4 is the spiro centre)

#[test]
fn test_spiro_two_rings_one_shared_atom() {
    let mut g = MolGraph::new();
    let ids: Vec<AtomId> = (0..9).map(|_| g.add_atom(carbon())).collect();

    // Ring A
    for i in 0..4 {
        bond(&mut g, ids[i], ids[i + 1]);
    }
    bond(&mut g, ids[4], ids[0]);

    // Ring B
    for i in 4..8 {
        bond(&mut g, ids[i], ids[i + 1]);
    }
    bond(&mut g, ids[8], ids[4]);

    let ri = find_rings(&g);
    assert_eq!(ri.num_rings(), 2, "spiro compound has 2 SSSR rings");
    let mut sizes = ri.ring_sizes();
    sizes.sort_unstable();
    assert_eq!(sizes, vec![5, 5]);

    // Spiro centre is in both rings.
    assert_eq!(ri.num_atom_rings(ids[4]), 2);
}

// ── Large ring (15-membered) ─────────────────────────────────────────────────

#[test]
fn test_large_ring_15() {
    let g = cycle(15);
    let ri = find_rings(&g);
    assert_eq!(ri.num_rings(), 1);
    assert_eq!(ri.ring_sizes(), vec![15]);
    assert_eq!(ri.rings_of_size(15).len(), 1);
}

// ── rings_of_size helper ─────────────────────────────────────────────────────

#[test]
fn test_rings_of_size_helper() {
    let g = cycle(5);
    let ri = find_rings(&g);
    assert_eq!(ri.rings_of_size(5).len(), 1);
    assert_eq!(ri.rings_of_size(6).len(), 0);
}
