//! `Perceive::find_aromaticity` — graph-in / graph-out aromaticity perception.
//!
//! Category: basics + graph-out. The wrapped `perceive_aromaticity` is
//! `&mut`-shaped, so the builder must clone first; that clone is asserted in
//! `perceive/immutability.rs`. Expectations mirror the RDKit-validated
//! `tests/core/aromaticity.rs` (benzene aromatic, cyclohexane not).

use molrs::perceive::Perceive;

use crate::fixtures::{atom_int, benzene, bond_between, bond_int, cyclohexane};

// ── Benzene is aromatic ─────────────────────────────────────────────────────

#[test]
fn test_benzene_carbons_are_aromatic() {
    let (mol, carbons) = benzene();
    let g = Perceive::new().find_aromaticity(&mol);

    for &c in &carbons {
        assert_eq!(
            atom_int(&g, c, "is_aromatic"),
            Some(1),
            "all six benzene carbons are aromatic"
        );
    }
}

#[test]
fn test_benzene_ring_bonds_are_aromatic() {
    let (mol, carbons) = benzene();
    let g = Perceive::new().find_aromaticity(&mol);

    for i in 0..6 {
        let bid = bond_between(&g, carbons[i], carbons[(i + 1) % 6]);
        assert_eq!(
            bond_int(&g, bid, "is_aromatic"),
            Some(1),
            "benzene C-C bond {i} is aromatic"
        );
    }
}

// ── Cyclohexane is not ──────────────────────────────────────────────────────

#[test]
fn test_cyclohexane_carbons_are_not_aromatic() {
    let (mol, carbons) = cyclohexane();
    let g = Perceive::new().find_aromaticity(&mol);

    for &c in &carbons {
        assert_eq!(
            atom_int(&g, c, "is_aromatic").unwrap_or(0),
            0,
            "cyclohexane carbon must not be aromatic"
        );
    }
}

#[test]
fn test_cyclohexane_ring_bonds_are_not_aromatic() {
    let (mol, carbons) = cyclohexane();
    let g = Perceive::new().find_aromaticity(&mol);

    for i in 0..6 {
        let bid = bond_between(&g, carbons[i], carbons[(i + 1) % 6]);
        assert_eq!(
            bond_int(&g, bid, "is_aromatic").unwrap_or(0),
            0,
            "cyclohexane C-C bond {i} must not be aromatic"
        );
    }
}

// ── graph-out shape ─────────────────────────────────────────────────────────

#[test]
fn test_find_aromaticity_returns_graph_of_same_size() {
    let (mol, _) = benzene();
    let g = Perceive::new().find_aromaticity(&mol);

    assert_eq!(g.n_atoms(), mol.n_atoms(), "find_aromaticity adds no atoms");
    assert_eq!(g.n_bonds(), mol.n_bonds(), "find_aromaticity adds no bonds");
}
