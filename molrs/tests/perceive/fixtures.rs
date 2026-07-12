//! Shared molecule builders and property readers for the `perceive` test target.
//!
//! Perception in this target is topology-driven — rings, aromaticity and
//! rotatable-bond detection read connectivity and bond orders only — so the
//! coordinates below are idealised planar / zig-zag layouts and **no assertion
//! depends on them**. `find_stereo` is exercised only on a skeleton that has no
//! stereocentre, so it is coordinate-independent too.
//!
//! Aromaticity perception assumes **all hydrogens are explicit** (an atom's
//! total degree is its neighbour count), so every fixture that is fed to
//! `find_aromaticity` carries its hydrogens.

use std::f64::consts::PI;

use molrs::store::keys;
use molrs::{AtomId, Atomistic, BondId, PropValue};

// ── Molecule builders ───────────────────────────────────────────────────────

/// Push a Kekulé benzene (6 C, alternating bond orders 2/1, 6 explicit H)
/// centred at `(x0, 0, 0)`. Returns the six carbon ids in ring order.
pub fn push_benzene(g: &mut Atomistic, x0: f64) -> Vec<AtomId> {
    const R_CC: f64 = 1.39;
    const R_CH: f64 = 2.48;

    let carbons: Vec<AtomId> = (0..6)
        .map(|i| {
            let th = PI / 3.0 * i as f64;
            g.add_atom_xyz("C", x0 + R_CC * th.cos(), R_CC * th.sin(), 0.0)
        })
        .collect();

    let hydrogens: Vec<AtomId> = (0..6)
        .map(|i| {
            let th = PI / 3.0 * i as f64;
            g.add_atom_xyz("H", x0 + R_CH * th.cos(), R_CH * th.sin(), 0.0)
        })
        .collect();

    // Ring bonds: Kekulé alternation (2, 1, 2, 1, 2, 1).
    for i in 0..6 {
        let order = if i % 2 == 0 { 2.0 } else { 1.0 };
        let bid = g
            .add_bond(carbons[i], carbons[(i + 1) % 6])
            .expect("benzene C-C bond");
        g.set_bond_prop(bid, keys::ORDER, order)
            .expect("benzene C-C order");
    }
    for i in 0..6 {
        let bid = g
            .add_bond(carbons[i], hydrogens[i])
            .expect("benzene C-H bond");
        g.set_bond_prop(bid, keys::ORDER, 1.0)
            .expect("benzene C-H order");
    }

    carbons
}

/// Push a methane (C + 4 explicit H) centred at `(x0, 0, 0)`. Returns the carbon.
pub fn push_methane(g: &mut Atomistic, x0: f64) -> AtomId {
    const D: f64 = 0.63; // C-H 1.09 Å along the tetrahedral diagonals

    let c = g.add_atom_xyz("C", x0, 0.0, 0.0);
    for (sx, sy, sz) in [
        (1.0, 1.0, 1.0),
        (-1.0, -1.0, 1.0),
        (-1.0, 1.0, -1.0),
        (1.0, -1.0, -1.0),
    ] {
        let h = g.add_atom_xyz("H", x0 + sx * D, sy * D, sz * D);
        let bid = g.add_bond(c, h).expect("methane C-H bond");
        g.set_bond_prop(bid, keys::ORDER, 1.0)
            .expect("methane C-H order");
    }
    c
}

/// Benzene on its own. Returns the graph and its six carbons in ring order.
pub fn benzene() -> (Atomistic, Vec<AtomId>) {
    let mut g = Atomistic::new();
    let carbons = push_benzene(&mut g, 0.0);
    (g, carbons)
}

/// Methane on its own. Returns the graph and its carbon.
pub fn methane() -> (Atomistic, AtomId) {
    let mut g = Atomistic::new();
    let c = push_methane(&mut g, 0.0);
    (g, c)
}

/// Cyclohexane: 6 C ring, all single bonds, 2 explicit H per carbon.
/// Returns the graph and its six carbons in ring order.
pub fn cyclohexane() -> (Atomistic, Vec<AtomId>) {
    const R_CC: f64 = 1.53;
    const R_CH: f64 = 2.60;

    let mut g = Atomistic::new();
    let carbons: Vec<AtomId> = (0..6)
        .map(|i| {
            let th = PI / 3.0 * i as f64;
            g.add_atom_xyz("C", R_CC * th.cos(), R_CC * th.sin(), 0.0)
        })
        .collect();

    for i in 0..6 {
        let bid = g
            .add_bond(carbons[i], carbons[(i + 1) % 6])
            .expect("cyclohexane C-C bond");
        g.set_bond_prop(bid, keys::ORDER, 1.0)
            .expect("cyclohexane C-C order");
    }
    for (i, &c) in carbons.iter().enumerate() {
        let th = PI / 3.0 * i as f64;
        for z in [0.9, -0.9] {
            let h = g.add_atom_xyz("H", R_CH * th.cos(), R_CH * th.sin(), z);
            let bid = g.add_bond(c, h).expect("cyclohexane C-H bond");
            g.set_bond_prop(bid, keys::ORDER, 1.0)
                .expect("cyclohexane C-H order");
        }
    }
    (g, carbons)
}

/// n-Butane **heavy-atom skeleton** (C-C-C-C, no hydrogens): the terminal
/// carbons are degree-1, so only the central bond is rotatable. Returns the
/// graph and the four carbons in chain order.
pub fn butane_skeleton() -> (Atomistic, Vec<AtomId>) {
    let mut g = Atomistic::new();
    let carbons: Vec<AtomId> = [(0.00, 0.00), (1.26, 0.87), (2.52, 0.00), (3.78, 0.87)]
        .iter()
        .map(|&(x, y)| g.add_atom_xyz("C", x, y, 0.0))
        .collect();

    for i in 0..3 {
        let bid = g
            .add_bond(carbons[i], carbons[i + 1])
            .expect("butane C-C bond");
        g.set_bond_prop(bid, keys::ORDER, 1.0)
            .expect("butane C-C order");
    }
    (g, carbons)
}

// ── Lookup / property readers ───────────────────────────────────────────────

/// The bond joining `a` and `b` (panics if absent).
pub fn bond_between(g: &Atomistic, a: AtomId, b: AtomId) -> BondId {
    g.bonds()
        .find(|(_, bond)| {
            (bond.nodes[0] == a && bond.nodes[1] == b) || (bond.nodes[0] == b && bond.nodes[1] == a)
        })
        .map(|(bid, _)| bid)
        .expect("bond between the two atoms")
}

/// An atom property, cloned out of the graph (`None` when unset).
pub fn atom_prop(g: &Atomistic, id: AtomId, key: &str) -> Option<PropValue> {
    g.get_atom(id).expect("atom exists").get(key).cloned()
}

/// A bond property, cloned out of the graph (`None` when unset).
pub fn bond_prop(g: &Atomistic, bid: BondId, key: &str) -> Option<PropValue> {
    g.get_bond(bid)
        .expect("bond exists")
        .props
        .get(key)
        .cloned()
}

/// Integer-valued atom property (accepts `Int` or `F64` storage, as the
/// existing perception readers do).
pub fn atom_int(g: &Atomistic, id: AtomId, key: &str) -> Option<i64> {
    as_int(atom_prop(g, id, key))
}

/// Integer-valued bond property (accepts `Int` or `F64` storage).
pub fn bond_int(g: &Atomistic, bid: BondId, key: &str) -> Option<i64> {
    as_int(bond_prop(g, bid, key))
}

/// Float-valued bond property (e.g. `order`).
pub fn bond_f64(g: &Atomistic, bid: BondId, key: &str) -> Option<f64> {
    bond_prop(g, bid, key).and_then(|v| v.as_f64())
}

fn as_int(v: Option<PropValue>) -> Option<i64> {
    match v {
        Some(PropValue::Int(i)) => Some(i64::from(i)),
        Some(PropValue::F64(f)) => Some(f as i64),
        _ => None,
    }
}
