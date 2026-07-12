//! Immutability — the sharpest contract of the builder.
//!
//! Category: immutability. Every `Perceive::find_*` is graph-in / graph-out:
//! it clones, writes onto the clone, and returns it. The input `mol` must come
//! back **byte-for-byte as it went in**. This matters most for
//! `find_aromaticity`, because the free function it wraps
//! (`perceive_aromaticity`) takes `&mut Atomistic` and overwrites bond `order`
//! with `1.5` — if the builder forgets to clone, the caller's Kekulé structure
//! is silently destroyed.
//!
//! Every assertion below is on the **original**, never on the result.

use molrs::perceive::Perceive;

use crate::fixtures::{atom_prop, benzene, bond_between, bond_f64, bond_prop, butane_skeleton};

// ── find_aromaticity ────────────────────────────────────────────────────────

#[test]
fn test_find_aromaticity_leaves_original_atoms_unflagged() {
    let (mol, _) = benzene();
    let _g = Perceive::new().find_aromaticity(&mol);

    for (id, _) in mol.atoms() {
        assert!(
            atom_prop(&mol, id, "is_aromatic").is_none(),
            "original atom {id:?} must NOT gain is_aromatic — the builder clones"
        );
    }
}

#[test]
fn test_find_aromaticity_leaves_original_bonds_unflagged() {
    let (mol, _) = benzene();
    let _g = Perceive::new().find_aromaticity(&mol);

    for (bid, _) in mol.bonds() {
        assert!(
            bond_prop(&mol, bid, "is_aromatic").is_none(),
            "original bond {bid:?} must NOT gain is_aromatic"
        );
        assert!(
            bond_prop(&mol, bid, "kekule_order").is_none(),
            "original bond {bid:?} must NOT gain kekule_order"
        );
    }
}

#[test]
fn test_find_aromaticity_leaves_original_kekule_orders_intact() {
    let (mol, carbons) = benzene();
    let _g = Perceive::new().find_aromaticity(&mol);

    // Benzene was built with alternating Kekulé orders 2, 1, 2, 1, 2, 1.
    // `perceive_aromaticity` rewrites ring-bond order to 1.5 in place — the
    // original must still show its integer orders.
    for i in 0..6 {
        let bid = bond_between(&mol, carbons[i], carbons[(i + 1) % 6]);
        let want = if i % 2 == 0 { 2.0 } else { 1.0 };
        assert_eq!(
            bond_f64(&mol, bid, "order"),
            Some(want),
            "original C-C bond {i} order must stay Kekulé, not become 1.5"
        );
    }
}

// ── find_rings ──────────────────────────────────────────────────────────────

#[test]
fn test_find_rings_leaves_original_atoms_unflagged() {
    let (mol, _) = benzene();
    let _g = Perceive::new().find_rings(&mol);

    for (id, _) in mol.atoms() {
        assert!(
            atom_prop(&mol, id, "is_in_ring").is_none(),
            "original atom {id:?} must NOT gain is_in_ring"
        );
        assert!(
            atom_prop(&mol, id, "n_rings").is_none(),
            "original atom {id:?} must NOT gain n_rings"
        );
    }
}

#[test]
fn test_find_rings_leaves_original_bonds_unflagged() {
    let (mol, _) = benzene();
    let _g = Perceive::new().find_rings(&mol);

    for (bid, _) in mol.bonds() {
        assert!(
            bond_prop(&mol, bid, "is_in_ring").is_none(),
            "original bond {bid:?} must NOT gain is_in_ring"
        );
        assert!(
            bond_prop(&mol, bid, "n_rings").is_none(),
            "original bond {bid:?} must NOT gain n_rings"
        );
    }
}

// ── find_rotatable ──────────────────────────────────────────────────────────

#[test]
fn test_find_rotatable_leaves_original_bonds_unflagged() {
    let (mol, _) = butane_skeleton();
    let _g = Perceive::new().find_rotatable(&mol);

    for (bid, _) in mol.bonds() {
        assert!(
            bond_prop(&mol, bid, "is_rotatable").is_none(),
            "original bond {bid:?} must NOT gain is_rotatable"
        );
    }
}

// ── find_hydrogens ──────────────────────────────────────────────────────────

#[test]
fn test_find_hydrogens_leaves_original_size_unchanged() {
    let (mol, _) = butane_skeleton();
    let n_atoms = mol.n_atoms();
    let n_bonds = mol.n_bonds();

    let g = Perceive::new().find_hydrogens(&mol);
    assert!(
        g.n_atoms() > n_atoms,
        "sanity: the clone did gain hydrogens"
    );

    assert_eq!(mol.n_atoms(), n_atoms, "original must not gain atoms");
    assert_eq!(mol.n_bonds(), n_bonds, "original must not gain bonds");
}
