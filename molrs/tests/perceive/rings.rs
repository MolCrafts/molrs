//! `Perceive::find_rings` — graph-in / graph-out ring perception.
//!
//! Category: basics + graph-out. Expectations are the ones the existing
//! `molrs::find_rings` side-table already produces (see `tests/core/rings.rs`);
//! the builder only re-projects them onto the returned graph as props.

use molrs::perceive::Perceive;

use crate::fixtures::{atom_int, benzene, bond_between, bond_int, methane};

// ── Benzene: every ring atom / ring bond is flagged ─────────────────────────

#[test]
fn test_benzene_ring_atoms_flagged_in_one_ring() {
    let (mol, carbons) = benzene();
    let g = Perceive::new().find_rings(&mol);

    for &c in &carbons {
        assert_eq!(
            atom_int(&g, c, "is_in_ring"),
            Some(1),
            "benzene carbon must be flagged in a ring"
        );
    }
}

#[test]
fn test_benzene_ring_atoms_have_one_ring() {
    let (mol, carbons) = benzene();
    let g = Perceive::new().find_rings(&mol);

    for &c in &carbons {
        assert_eq!(
            atom_int(&g, c, "n_rings"),
            Some(1),
            "benzene carbon belongs to exactly 1 SSSR ring"
        );
    }
}

#[test]
fn test_benzene_ring_bonds_flagged_in_one_ring() {
    let (mol, carbons) = benzene();
    let g = Perceive::new().find_rings(&mol);

    for i in 0..6 {
        let bid = bond_between(&g, carbons[i], carbons[(i + 1) % 6]);
        assert_eq!(
            bond_int(&g, bid, "is_in_ring"),
            Some(1),
            "benzene C-C bond {i} must be flagged in a ring"
        );
        assert_eq!(bond_int(&g, bid, "n_rings"), Some(1));
    }
}

// ── Methane: nothing is in a ring ───────────────────────────────────────────

#[test]
fn test_methane_atoms_not_in_ring() {
    let (mol, _c) = methane();
    let g = Perceive::new().find_rings(&mol);

    for (id, _) in g.atoms() {
        assert_eq!(
            atom_int(&g, id, "is_in_ring"),
            Some(0),
            "acyclic atom must carry is_in_ring = 0"
        );
        assert_eq!(atom_int(&g, id, "n_rings"), Some(0));
    }
}

#[test]
fn test_methane_bonds_not_in_ring() {
    let (mol, _c) = methane();
    let g = Perceive::new().find_rings(&mol);

    for (bid, _) in g.bonds() {
        assert_eq!(
            bond_int(&g, bid, "is_in_ring"),
            Some(0),
            "acyclic bond must carry is_in_ring = 0"
        );
        assert_eq!(bond_int(&g, bid, "n_rings"), Some(0));
    }
}

// ── graph-out shape ─────────────────────────────────────────────────────────

#[test]
fn test_find_rings_returns_graph_of_same_size() {
    let (mol, _) = benzene();
    let g = Perceive::new().find_rings(&mol);

    assert_eq!(g.n_atoms(), mol.n_atoms(), "find_rings adds no atoms");
    assert_eq!(g.n_bonds(), mol.n_bonds(), "find_rings adds no bonds");
}
