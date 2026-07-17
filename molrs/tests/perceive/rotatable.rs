//! `Perceive::find_rotatable` — graph-in / graph-out rotatable-bond detection.
//!
//! Category: basics + graph-out. A bond is rotatable when it is a single bond,
//! acyclic, and both endpoints are non-terminal — the criteria the wrapped
//! `detect_rotatable_bonds` already implements. Butane is used as its **heavy
//! skeleton** (C-C-C-C): the terminal carbons are degree-1, so only the central
//! bond qualifies.

use molrs::perceive::Perceive;

use crate::fixtures::{benzene, bond_between, bond_int, butane_skeleton};

// ── Butane skeleton ─────────────────────────────────────────────────────────

#[test]
fn test_butane_central_bond_is_rotatable() {
    let (mol, c) = butane_skeleton();
    let g = Perceive::new().find_rotatable(&mol);

    let central = bond_between(&g, c[1], c[2]);
    assert_eq!(
        bond_int(&g, central, "is_rotatable"),
        Some(1),
        "the central C-C bond of the butane skeleton is rotatable"
    );
}

#[test]
fn test_butane_terminal_bonds_are_not_rotatable() {
    let (mol, c) = butane_skeleton();
    let g = Perceive::new().find_rotatable(&mol);

    for (i, j) in [(0, 1), (2, 3)] {
        let bid = bond_between(&g, c[i], c[j]);
        assert_eq!(
            bond_int(&g, bid, "is_rotatable"),
            Some(0),
            "terminal C-C bond {i}-{j} has a degree-1 endpoint: not rotatable"
        );
    }
}

// ── Benzene: ring bonds never rotate ────────────────────────────────────────

#[test]
fn test_benzene_ring_bonds_are_not_rotatable() {
    let (mol, carbons) = benzene();
    let g = Perceive::new().find_rotatable(&mol);

    for i in 0..6 {
        let bid = bond_between(&g, carbons[i], carbons[(i + 1) % 6]);
        assert_eq!(
            bond_int(&g, bid, "is_rotatable"),
            Some(0),
            "benzene ring bond {i} is cyclic: not rotatable"
        );
    }
}

#[test]
fn test_benzene_has_no_rotatable_bond_at_all() {
    let (mol, _) = benzene();
    let g = Perceive::new().find_rotatable(&mol);

    for (bid, _) in g.bonds() {
        assert_eq!(
            bond_int(&g, bid, "is_rotatable"),
            Some(0),
            "benzene has no rotatable bond (ring bonds cyclic, C-H terminal)"
        );
    }
}

// ── graph-out shape ─────────────────────────────────────────────────────────

#[test]
fn test_find_rotatable_returns_graph_of_same_size() {
    let (mol, _) = butane_skeleton();
    let g = Perceive::new().find_rotatable(&mol);

    assert_eq!(g.n_atoms(), mol.n_atoms(), "find_rotatable adds no atoms");
    assert_eq!(g.n_bonds(), mol.n_bonds(), "find_rotatable adds no bonds");
}
