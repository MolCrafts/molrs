//! Integration tests for stereochemistry (src/core/stereo.rs).

use molrs::{
    Atom, AtomId, BondStereo, MolGraph, PropValue, TetrahedralStereo, assign_bond_stereo_from_3d,
    assign_stereo_from_3d, chiral_volume, find_chiral_centers,
};

fn atom_at(sym: &str, x: f64, y: f64, z: f64) -> Atom {
    Atom::xyz(sym, x, y, z)
}

fn add_double_bond(g: &mut MolGraph, a: AtomId, b: AtomId) {
    if let Ok(bid) = g.add_bond(a, b)
        && let Ok(bnd) = g.get_bond_mut(bid)
    {
        bnd.props.insert("order".to_string(), PropValue::F64(2.0));
    }
}

// ── Chiral volume ────────────────────────────────────────────────────────────

#[test]
fn test_chiral_volume_right_handed_positive() {
    // Right-handed tetrahedron: v1=(1,0,0), v2=(0,1,0), v3=(0,0,1)
    // Triple product = 1 > 0
    let mut g = MolGraph::new();
    let c = g.add_atom(atom_at("C", 0.0, 0.0, 0.0));
    let n1 = g.add_atom(atom_at("H", 1.0, 0.0, 0.0));
    let n2 = g.add_atom(atom_at("H", 0.0, 1.0, 0.0));
    let n3 = g.add_atom(atom_at("H", 0.0, 0.0, 1.0));
    let n4 = g.add_atom(atom_at("H", -1.0, -1.0, -1.0));
    for &n in &[n1, n2, n3, n4] {
        g.add_bond(c, n).expect("add bond");
    }
    let vol = chiral_volume(&g, c, &[n1, n2, n3, n4]);
    assert!(vol > 0.0, "expected positive volume, got {vol}");
}

#[test]
fn test_chiral_volume_swap_flips_sign() {
    let mut g = MolGraph::new();
    let c = g.add_atom(atom_at("C", 0.0, 0.0, 0.0));
    let n1 = g.add_atom(atom_at("H", 1.0, 0.0, 0.0));
    let n2 = g.add_atom(atom_at("H", 0.0, 1.0, 0.0));
    let n3 = g.add_atom(atom_at("H", 0.0, 0.0, 1.0));
    let n4 = g.add_atom(atom_at("H", -1.0, -1.0, -1.0));
    for &n in &[n1, n2, n3, n4] {
        g.add_bond(c, n).expect("add bond");
    }
    let v_orig = chiral_volume(&g, c, &[n1, n2, n3, n4]);
    let v_swap = chiral_volume(&g, c, &[n2, n1, n3, n4]); // swap first two
    assert!(
        v_orig * v_swap < 0.0,
        "swapping two neighbours should flip sign: {v_orig} vs {v_swap}"
    );
}

#[test]
fn test_chiral_volume_missing_coords_returns_zero() {
    let mut g = MolGraph::new();
    // atoms without coordinates
    let c = g.add_atom(Atom::new());
    let n1 = g.add_atom(Atom::new());
    let n2 = g.add_atom(Atom::new());
    let n3 = g.add_atom(Atom::new());
    let n4 = g.add_atom(Atom::new());
    for &n in &[n1, n2, n3, n4] {
        g.add_bond(c, n).expect("add bond");
    }
    let vol = chiral_volume(&g, c, &[n1, n2, n3, n4]);
    assert_eq!(vol, 0.0);
}

// ── find_chiral_centers ──────────────────────────────────────────────────────

#[test]
fn test_no_centers_in_ethane() {
    let mut g = MolGraph::new();
    let c1 = g.add_atom(atom_at("C", 0.0, 0.0, 0.0));
    let c2 = g.add_atom(atom_at("C", 1.5, 0.0, 0.0));
    g.add_bond(c1, c2).expect("add bond");
    assert!(find_chiral_centers(&g).is_empty());
}

#[test]
fn test_no_centers_for_3_neighbors() {
    // Carbon with only 3 neighbours (e.g., sp2)
    let mut g = MolGraph::new();
    let c = g.add_atom(atom_at("C", 0.0, 0.0, 0.0));
    for i in 0..3 {
        let h = g.add_atom(atom_at("H", i as f64, 0.0, 0.0));
        g.add_bond(c, h).expect("add bond");
    }
    assert!(find_chiral_centers(&g).is_empty());
}

#[test]
fn test_4_neighbor_atom_is_center() {
    let mut g = MolGraph::new();
    let c = g.add_atom(atom_at("C", 0.0, 0.0, 0.0));
    for i in 0..4_usize {
        // Give each neighbour a unique position so they are distinct IDs
        let h = g.add_atom(atom_at("H", i as f64, 0.0, 0.0));
        g.add_bond(c, h).expect("add bond");
    }
    let centers = find_chiral_centers(&g);
    assert_eq!(centers.len(), 1);
    assert_eq!(centers[0], c);
}

// ── assign_stereo_from_3d ────────────────────────────────────────────────────

#[test]
fn test_stereo_assigned_for_tetrahedral_center() {
    let mut g = MolGraph::new();
    let c = g.add_atom(atom_at("C", 0.0, 0.0, 0.0));
    let n1 = g.add_atom(atom_at("F", 1.0, 0.0, 0.0));
    let n2 = g.add_atom(atom_at("Cl", 0.0, 1.0, 0.0));
    let n3 = g.add_atom(atom_at("Br", 0.0, 0.0, 1.0));
    let n4 = g.add_atom(atom_at("H", -1.0, -1.0, -1.0));
    for &n in &[n1, n2, n3, n4] {
        g.add_bond(c, n).expect("add bond");
    }
    let stereo = assign_stereo_from_3d(&g);
    assert!(stereo.contains_key(&c), "center should have a stereo entry");
    assert_ne!(
        stereo[&c],
        TetrahedralStereo::Unspecified,
        "non-planar center should not be Unspecified"
    );
}

#[test]
fn test_achiral_molecule_no_centers() {
    let mut g = MolGraph::new();
    let c1 = g.add_atom(atom_at("C", 0.0, 0.0, 0.0));
    let c2 = g.add_atom(atom_at("C", 1.5, 0.0, 0.0));
    g.add_bond(c1, c2).expect("add bond");
    let stereo = assign_stereo_from_3d(&g);
    assert!(stereo.is_empty(), "no stereocentres expected");
}

#[test]
fn test_stereo_cw_vs_ccw_differ() {
    // Two mirror-image tetrahedra should give opposite chirality labels.
    let make = |sign: f64| {
        let mut g = MolGraph::new();
        let c = g.add_atom(atom_at("C", 0.0, 0.0, 0.0));
        let n1 = g.add_atom(atom_at("F", 1.0 * sign, 0.0, 0.0));
        let n2 = g.add_atom(atom_at("Cl", 0.0, 1.0, 0.0));
        let n3 = g.add_atom(atom_at("Br", 0.0, 0.0, 1.0));
        let n4 = g.add_atom(atom_at("H", -1.0, -1.0, -1.0));
        for &n in &[n1, n2, n3, n4] {
            g.add_bond(c, n).expect("add bond");
        }
        (g, c)
    };

    let (g1, c1) = make(1.0);
    let (g2, c2) = make(-1.0);

    let s1 = assign_stereo_from_3d(&g1);
    let s2 = assign_stereo_from_3d(&g2);

    assert_ne!(
        s1[&c1], s2[&c2],
        "mirror images should have opposite chirality"
    );
}

// ── assign_bond_stereo_from_3d ───────────────────────────────────────────────

#[test]
fn test_cis_2_butene_z() {
    // cis-2-butene: both methyl groups on the same side (+y).
    let mut g = MolGraph::new();
    let c1 = g.add_atom(atom_at("C", 0.0, 0.0, 0.0));
    let c2 = g.add_atom(atom_at("C", 1.34, 0.0, 0.0));
    let s1 = g.add_atom(atom_at("C", -0.5, 1.0, 0.0)); // +y
    let s2 = g.add_atom(atom_at("C", 1.84, 1.0, 0.0)); // +y
    add_double_bond(&mut g, c1, c2);
    g.add_bond(c1, s1).expect("add bond");
    g.add_bond(c2, s2).expect("add bond");

    let stereo = assign_bond_stereo_from_3d(&g);
    let double_bid = stereo
        .iter()
        .find(|&(_, v)| *v == BondStereo::Z || *v == BondStereo::E)
        .map(|(&k, _)| k)
        .expect("should find an E/Z bond");
    assert_eq!(stereo[&double_bid], BondStereo::Z, "cis should be Z");
}

#[test]
fn test_trans_2_butene_e() {
    // trans-2-butene: methyl groups on opposite sides.
    let mut g = MolGraph::new();
    let c1 = g.add_atom(atom_at("C", 0.0, 0.0, 0.0));
    let c2 = g.add_atom(atom_at("C", 1.34, 0.0, 0.0));
    let s1 = g.add_atom(atom_at("C", -0.5, 1.0, 0.0)); // +y
    let s2 = g.add_atom(atom_at("C", 1.84, -1.0, 0.0)); // -y
    add_double_bond(&mut g, c1, c2);
    g.add_bond(c1, s1).expect("add bond");
    g.add_bond(c2, s2).expect("add bond");

    let stereo = assign_bond_stereo_from_3d(&g);
    let double_bid = stereo
        .iter()
        .find(|&(_, v)| *v == BondStereo::E || *v == BondStereo::Z)
        .map(|(&k, _)| k)
        .expect("should find an E/Z bond");
    assert_eq!(stereo[&double_bid], BondStereo::E, "trans should be E");
}

#[test]
fn test_single_bond_is_none() {
    let mut g = MolGraph::new();
    let a = g.add_atom(atom_at("C", 0.0, 0.0, 0.0));
    let b = g.add_atom(atom_at("C", 1.5, 0.0, 0.0));
    g.add_bond(a, b).expect("add bond");
    let stereo = assign_bond_stereo_from_3d(&g);
    let bid = g.bonds().next().unwrap().0;
    assert_eq!(stereo[&bid], BondStereo::None);
}

#[test]
fn test_double_bond_no_substituents_is_none() {
    // C=C with no other substituents → BondStereo::None
    let mut g = MolGraph::new();
    let c1 = g.add_atom(atom_at("C", 0.0, 0.0, 0.0));
    let c2 = g.add_atom(atom_at("C", 1.34, 0.0, 0.0));
    add_double_bond(&mut g, c1, c2);
    let stereo = assign_bond_stereo_from_3d(&g);
    let bid = g.bonds().next().unwrap().0;
    assert_eq!(
        stereo[&bid],
        BondStereo::None,
        "C=C with no subs has no stereo"
    );
}
