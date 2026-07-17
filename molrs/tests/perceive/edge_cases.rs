//! Edge cases — empty graph, single atom, disconnected fragments.
//!
//! Category: edge cases. Perception must be total: no panic on degenerate
//! input, and fragments are perceived independently.

use molrs::Atomistic;
use molrs::perceive::Perceive;

use crate::fixtures::{atom_int, push_benzene, push_methane};

// ── Empty molecule ──────────────────────────────────────────────────────────

#[test]
fn test_empty_molecule_survives_every_finder() {
    let mol = Atomistic::new();
    let p = Perceive::new();

    for g in [
        p.find_rings(&mol),
        p.find_aromaticity(&mol),
        p.find_hydrogens(&mol),
        p.find_stereo(&mol),
        p.find_rotatable(&mol),
    ] {
        assert_eq!(g.n_atoms(), 0, "empty in, empty out");
        assert_eq!(g.n_bonds(), 0);
    }
}

// ── Single atom ─────────────────────────────────────────────────────────────

#[test]
fn test_single_atom_is_not_in_a_ring() {
    let mut mol = Atomistic::new();
    let c = mol.add_atom_xyz("C", 0.0, 0.0, 0.0);

    let g = Perceive::new().find_rings(&mol);
    assert_eq!(atom_int(&g, c, "is_in_ring"), Some(0));
    assert_eq!(atom_int(&g, c, "n_rings"), Some(0));
}

#[test]
fn test_single_atom_is_not_aromatic() {
    let mut mol = Atomistic::new();
    let c = mol.add_atom_xyz("C", 0.0, 0.0, 0.0);

    let g = Perceive::new().find_aromaticity(&mol);
    assert_eq!(atom_int(&g, c, "is_aromatic").unwrap_or(0), 0);
}

#[test]
fn test_single_atom_has_no_rotatable_bond() {
    let mut mol = Atomistic::new();
    mol.add_atom_xyz("C", 0.0, 0.0, 0.0);

    let g = Perceive::new().find_rotatable(&mol);
    assert_eq!(g.n_bonds(), 0, "a lone atom has no bonds to flag");
    assert_eq!(g.n_atoms(), 1);
}

// ── Two disconnected fragments (benzene + methane) ──────────────────────────

#[test]
fn test_fragments_are_ring_perceived_independently() {
    let mut mol = Atomistic::new();
    let carbons = push_benzene(&mut mol, 0.0);
    let methyl = push_methane(&mut mol, 10.0);

    let g = Perceive::new().find_rings(&mol);

    for &c in &carbons {
        assert_eq!(
            atom_int(&g, c, "is_in_ring"),
            Some(1),
            "benzene fragment carbon is in a ring"
        );
    }
    assert_eq!(
        atom_int(&g, methyl, "is_in_ring"),
        Some(0),
        "methane fragment carbon is not in a ring"
    );
}

#[test]
fn test_fragments_are_aromaticity_perceived_independently() {
    let mut mol = Atomistic::new();
    let carbons = push_benzene(&mut mol, 0.0);
    let methyl = push_methane(&mut mol, 10.0);

    let g = Perceive::new().find_aromaticity(&mol);

    for &c in &carbons {
        assert_eq!(
            atom_int(&g, c, "is_aromatic"),
            Some(1),
            "benzene fragment carbon is aromatic"
        );
    }
    assert_eq!(
        atom_int(&g, methyl, "is_aromatic").unwrap_or(0),
        0,
        "methane fragment carbon is not aromatic"
    );
}
