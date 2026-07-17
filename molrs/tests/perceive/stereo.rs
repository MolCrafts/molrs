//! `Perceive::find_stereo` — graph-in / graph-out stereo assignment.
//!
//! Category: basics + graph-out, deliberately minimal.
//!
//! The wrapped functions return **maps**, and `assign_bond_stereo_from_3d` in
//! particular returns a *total* one: it inserts `BondStereo::None` for every
//! bond that is not a double bond (`src/perceive/stereo.rs`, the
//! `(order - 2.0).abs() > 0.5` arm). `None` there is a **sentinel meaning "not a
//! stereo bond"**, not a perceived fact — so the builder must skip it rather
//! than materialise it as a `stereo` prop. Normalising that map shape is
//! precisely what `find_stereo` exists to do: a molecule with no stereocentre
//! and no double bond must come back carrying no `stereo` prop at all.
//!
//! We do not hand-build a chiral centre here: that would mean inventing 3D
//! coordinates whose R/S label is not established by any existing fixture.

use molrs::perceive::Perceive;

use crate::fixtures::{atom_prop, bond_prop, butane_skeleton};

#[test]
fn test_no_stereocentre_yields_no_atom_stereo_prop() {
    let (mol, _) = butane_skeleton();
    let g = Perceive::new().find_stereo(&mol);

    for (id, _) in g.atoms() {
        assert!(
            atom_prop(&g, id, "stereo").is_none(),
            "atom {id:?} has no stereocentre, so no stereo prop must be written"
        );
    }
}

#[test]
fn test_no_stereo_bond_yields_no_bond_stereo_prop() {
    let (mol, _) = butane_skeleton();
    let g = Perceive::new().find_stereo(&mol);

    for (bid, _) in g.bonds() {
        assert!(
            bond_prop(&g, bid, "stereo").is_none(),
            "bond {bid:?} is a single bond, so no stereo prop must be written"
        );
    }
}

#[test]
fn test_find_stereo_returns_graph_of_same_size() {
    let (mol, _) = butane_skeleton();
    let g = Perceive::new().find_stereo(&mol);

    assert_eq!(g.n_atoms(), mol.n_atoms(), "find_stereo adds no atoms");
    assert_eq!(g.n_bonds(), mol.n_bonds(), "find_stereo adds no bonds");
}
