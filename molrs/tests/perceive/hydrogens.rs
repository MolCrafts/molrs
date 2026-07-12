//! `Perceive::find_hydrogens` — graph-in / graph-out hydrogen addition.
//!
//! Category: basics + graph-out. Wraps the already non-mutating
//! `molrs::add_hydrogens`. We assert only what the brief pins down — the
//! skeleton *gains* hydrogens and keeps its heavy atoms — never a specific
//! filled valence count.

use molrs::PropValue;
use molrs::perceive::Perceive;

use crate::fixtures::{atom_prop, butane_skeleton};

/// Element symbol of an atom, as written by `add_atom_xyz`.
fn element(g: &molrs::Atomistic, id: molrs::AtomId) -> String {
    match atom_prop(g, id, "element") {
        Some(PropValue::Str(s)) => s,
        _ => String::new(),
    }
}

#[test]
fn test_heavy_skeleton_gains_hydrogens() {
    let (mol, _) = butane_skeleton();
    let g = Perceive::new().find_hydrogens(&mol);

    assert!(
        g.n_atoms() > mol.n_atoms(),
        "butane skeleton ({} atoms) must gain hydrogens, got {} atoms",
        mol.n_atoms(),
        g.n_atoms()
    );
}

#[test]
fn test_added_atoms_are_all_hydrogen() {
    let (mol, _) = butane_skeleton();
    let g = Perceive::new().find_hydrogens(&mol);

    let n_h = g.atoms().filter(|(id, _)| element(&g, *id) == "H").count();
    assert_eq!(
        n_h,
        g.n_atoms() - mol.n_atoms(),
        "every atom added by find_hydrogens is a hydrogen"
    );
}

#[test]
fn test_heavy_atoms_are_preserved() {
    let (mol, carbons) = butane_skeleton();
    let g = Perceive::new().find_hydrogens(&mol);

    let n_c = g.atoms().filter(|(id, _)| element(&g, *id) == "C").count();
    assert_eq!(
        n_c,
        carbons.len(),
        "find_hydrogens preserves the heavy-atom skeleton"
    );
}

#[test]
fn test_new_hydrogens_are_bonded() {
    let (mol, _) = butane_skeleton();
    let g = Perceive::new().find_hydrogens(&mol);

    assert!(
        g.n_bonds() > mol.n_bonds(),
        "each added hydrogen brings a bond"
    );
}
