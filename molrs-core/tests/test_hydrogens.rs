//! Integration tests for hydrogen addition (src/core/hydrogens.rs).

use molrs::{Atom, AtomId, MolGraph, PropValue, add_hydrogens, implicit_h_count, remove_hydrogens};

fn atom(sym: &str) -> Atom {
    let mut a = Atom::new();
    a.set("symbol", sym);
    a
}

fn bond_order(g: &mut MolGraph, a: AtomId, b: AtomId, order: f64) {
    if let Ok(bid) = g.add_bond(a, b)
        && let Ok(bnd) = g.get_bond_mut(bid)
    {
        bnd.props.insert("order".to_string(), PropValue::F64(order));
    }
}

fn count_symbol(g: &MolGraph, sym: &str) -> usize {
    g.atoms()
        .filter(|(_, a)| {
            a.get_str("symbol")
                .is_some_and(|s| s.eq_ignore_ascii_case(sym))
        })
        .count()
}

// ── Methane skeleton ────────────────────────────────────────────────────────

#[test]
fn test_isolated_carbon_gets_4h() {
    let mut g = MolGraph::new();
    g.add_atom(atom("C"));
    let result = add_hydrogens(&g);
    assert_eq!(result.n_atoms(), 5, "C + 4H");
    assert_eq!(count_symbol(&result, "H"), 4);
    // Orignal is unchanged
    assert_eq!(g.n_atoms(), 1);
}

// ── Ethane (C-C) ─────────────────────────────────────────────────────────────

#[test]
fn test_ethane_c_c_6_hydrogens() {
    let mut g = MolGraph::new();
    let c1 = g.add_atom(atom("C"));
    let c2 = g.add_atom(atom("C"));
    bond_order(&mut g, c1, c2, 1.0);
    let result = add_hydrogens(&g);
    assert_eq!(result.n_atoms(), 8, "2C + 6H");
    assert_eq!(count_symbol(&result, "H"), 6);
}

// ── Ethylene (C=C) ───────────────────────────────────────────────────────────

#[test]
fn test_ethylene_double_bond_4_hydrogens() {
    let mut g = MolGraph::new();
    let c1 = g.add_atom(atom("C"));
    let c2 = g.add_atom(atom("C"));
    bond_order(&mut g, c1, c2, 2.0);
    let result = add_hydrogens(&g);
    assert_eq!(result.n_atoms(), 6, "2C + 4H");
    assert_eq!(count_symbol(&result, "H"), 4);
}

// ── Acetylene (C≡C) ──────────────────────────────────────────────────────────

#[test]
fn test_acetylene_triple_bond_2_hydrogens() {
    let mut g = MolGraph::new();
    let c1 = g.add_atom(atom("C"));
    let c2 = g.add_atom(atom("C"));
    bond_order(&mut g, c1, c2, 3.0);
    let result = add_hydrogens(&g);
    assert_eq!(result.n_atoms(), 4, "2C + 2H");
    assert_eq!(count_symbol(&result, "H"), 2);
}

// ── Benzene (aromatic, bond order 1.5) ───────────────────────────────────────

#[test]
fn test_benzene_aromatic_6_hydrogens() {
    let mut g = MolGraph::new();
    let ids: Vec<AtomId> = (0..6).map(|_| g.add_atom(atom("C"))).collect();
    for i in 0..6 {
        bond_order(&mut g, ids[i], ids[(i + 1) % 6], 1.5);
    }
    let result = add_hydrogens(&g);
    assert_eq!(result.n_atoms(), 12, "6C + 6H");
    assert_eq!(count_symbol(&result, "H"), 6);
}

// ── Water (O) ────────────────────────────────────────────────────────────────

#[test]
fn test_isolated_oxygen_gets_2h() {
    let mut g = MolGraph::new();
    g.add_atom(atom("O"));
    let result = add_hydrogens(&g);
    assert_eq!(result.n_atoms(), 3, "O + 2H");
    assert_eq!(count_symbol(&result, "H"), 2);
}

// ── Ammonia-like (N with 1 C bond) ──────────────────────────────────────────

#[test]
fn test_nitrogen_1_bond_gets_2h() {
    let mut g = MolGraph::new();
    let n = g.add_atom(atom("N"));
    let c = g.add_atom(atom("C"));
    bond_order(&mut g, n, c, 1.0);
    let result = add_hydrogens(&g);
    // N gets 2H, C gets 3H → total 2 + 5 = 7 atoms
    assert_eq!(result.n_atoms(), 7);
    let _ = (n, c);
}

// ── NH4+ formal charge ────────────────────────────────────────────────────────

#[test]
fn test_nh4_plus_implicit_count_is_4() {
    let mut g = MolGraph::new();
    let mut n_atom = Atom::new();
    n_atom.set("symbol", "N");
    n_atom.set("formal_charge", 1.0_f64);
    let n = g.add_atom(n_atom);
    let h_count = implicit_h_count(&g, n).unwrap();
    assert_eq!(h_count, 4, "NH4+ should need 4 hydrogens");
}

#[test]
fn test_nh4_plus_add_hydrogens() {
    let mut g = MolGraph::new();
    let mut n_atom = Atom::new();
    n_atom.set("symbol", "N");
    n_atom.set("formal_charge", 1.0_f64);
    g.add_atom(n_atom);
    let result = add_hydrogens(&g);
    assert_eq!(result.n_atoms(), 5, "N + 4H");
    assert_eq!(count_symbol(&result, "H"), 4);
}

// ── No extra H on existing hydrogens ─────────────────────────────────────────

#[test]
fn test_existing_hydrogens_not_modified() {
    let mut g = MolGraph::new();
    let c = g.add_atom(atom("C"));
    let h = g.add_atom(atom("H"));
    bond_order(&mut g, c, h, 1.0);
    let result = add_hydrogens(&g);
    // C (1 existing bond) → 3 more H; existing H → no change
    assert_eq!(count_symbol(&result, "H"), 4); // 1 original + 3 new
    assert_eq!(result.n_atoms(), 5); // 1C + 1original_H + 3new_H
    let _ = (c, h);
}

// ── Noble gas — no valence model ─────────────────────────────────────────────

#[test]
fn test_noble_gas_no_hydrogens() {
    let mut g = MolGraph::new();
    g.add_atom(atom("Ne"));
    let result = add_hydrogens(&g);
    assert_eq!(result.n_atoms(), 1, "Ne should receive no H");
}

// ── Sulfur (valence 2 / 4 / 6) ───────────────────────────────────────────────

#[test]
fn test_isolated_sulfur_gets_2h() {
    let mut g = MolGraph::new();
    g.add_atom(atom("S"));
    let result = add_hydrogens(&g);
    assert_eq!(count_symbol(&result, "H"), 2, "S (no bonds) → 2H");
}

// ── Bond order absent defaults to 1.0 ────────────────────────────────────────

#[test]
fn test_missing_bond_order_defaults_to_single() {
    let mut g = MolGraph::new();
    let c1 = g.add_atom(atom("C"));
    let c2 = g.add_atom(atom("C"));
    g.add_bond(c1, c2).expect("add bond"); // no order property
    let result = add_hydrogens(&g);
    // Each C treated as having 1 single bond → 3H each
    assert_eq!(count_symbol(&result, "H"), 6);
}

// ── Immutability: original mol unchanged ─────────────────────────────────────

#[test]
fn test_add_hydrogens_immutable() {
    let mut g = MolGraph::new();
    let c = g.add_atom(atom("C"));
    let before_atoms = g.n_atoms();
    let before_bonds = g.n_bonds();
    let _result = add_hydrogens(&g);
    assert_eq!(g.n_atoms(), before_atoms);
    assert_eq!(g.n_bonds(), before_bonds);
    let _ = c;
}

// ═════════════════════════════════════════════════════════════════════════════
// remove_hydrogens tests
// ═════════════════════════════════════════════════════════════════════════════

// ── Roundtrip: add then remove on methane skeleton ───────────────────────

#[test]
fn test_remove_hydrogens_roundtrip_methane() {
    let mut g = MolGraph::new();
    g.add_atom(atom("C"));
    let with_h = add_hydrogens(&g);
    assert_eq!(with_h.n_atoms(), 5, "C + 4H");
    let stripped = remove_hydrogens(&with_h);
    assert_eq!(stripped.n_atoms(), 1, "back to C only");
    assert_eq!(stripped.n_bonds(), 0);
}

// ── Roundtrip: benzene ──────────────────────────────────────────────────

#[test]
fn test_remove_hydrogens_roundtrip_benzene() {
    let mut g = MolGraph::new();
    let ids: Vec<AtomId> = (0..6).map(|_| g.add_atom(atom("C"))).collect();
    for i in 0..6 {
        bond_order(&mut g, ids[i], ids[(i + 1) % 6], 1.5);
    }
    let with_h = add_hydrogens(&g);
    assert_eq!(with_h.n_atoms(), 12, "6C + 6H");
    let stripped = remove_hydrogens(&with_h);
    assert_eq!(stripped.n_atoms(), 6, "back to 6C");
    assert_eq!(stripped.n_bonds(), 6, "ring bonds preserved");
}

// ── Partial explicit H: only terminal H removed ─────────────────────────

#[test]
fn test_remove_hydrogens_partial_explicit() {
    // C with 1 explicit H + add_hydrogens → 5 atoms; remove → back to 1C
    let mut g = MolGraph::new();
    let c = g.add_atom(atom("C"));
    let h = g.add_atom(atom("H"));
    bond_order(&mut g, c, h, 1.0);
    let with_h = add_hydrogens(&g);
    assert_eq!(count_symbol(&with_h, "H"), 4);
    let stripped = remove_hydrogens(&with_h);
    assert_eq!(stripped.n_atoms(), 1, "all H removed");
    assert_eq!(count_symbol(&stripped, "C"), 1);
}

// ── Immutability: remove_hydrogens does not mutate input ────────────────

#[test]
fn test_remove_hydrogens_immutable() {
    let mut g = MolGraph::new();
    g.add_atom(atom("C"));
    let with_h = add_hydrogens(&g);
    let before = with_h.n_atoms();
    let _stripped = remove_hydrogens(&with_h);
    assert_eq!(with_h.n_atoms(), before, "input unchanged");
}

// ── Water roundtrip ─────────────────────────────────────────────────────

#[test]
fn test_remove_hydrogens_water_roundtrip() {
    let mut g = MolGraph::new();
    g.add_atom(atom("O"));
    let with_h = add_hydrogens(&g);
    assert_eq!(with_h.n_atoms(), 3, "O + 2H");
    let stripped = remove_hydrogens(&with_h);
    assert_eq!(stripped.n_atoms(), 1, "back to O only");
}

// ── Noble gas: no H to remove ───────────────────────────────────────────

#[test]
fn test_remove_hydrogens_noble_gas_unchanged() {
    let mut g = MolGraph::new();
    g.add_atom(atom("Ne"));
    let stripped = remove_hydrogens(&g);
    assert_eq!(stripped.n_atoms(), 1, "Ne unchanged");
}
