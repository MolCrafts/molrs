//! # Building Molecules with MolGraph
//!
//! Demonstrates how to construct molecules from scratch using MolGraph.
//! Covers atom creation, property system, bonds/angles/dihedrals,
//! neighbor queries, cascading deletion, and coarse-grained beads.
//!
//! Run with: `cargo run -p molrs-core --example molgraph_building`

use molrs::{Atom, Bead, MolGraph, PropValue, add_hydrogens, remove_hydrogens};

fn main() {
    water();
    ethane();
    coarse_grained();
    hydrogen_roundtrip();
}

// ─── Water (H2O) ────────────────────────────────────────────────────────────

fn water() {
    println!("=== Water (H2O) ===\n");

    // --- 1. Create atoms with Atom::xyz(symbol, x, y, z) ---
    let mut mol = MolGraph::new();

    let o = mol.add_atom(Atom::xyz("O", 0.0, 0.0, 0.0));
    let h1 = mol.add_atom(Atom::xyz("H", 0.9572, 0.0, 0.0));
    let h2 = mol.add_atom(Atom::xyz("H", -0.2400, 0.9266, 0.0));

    println!("After adding 3 atoms:");
    println!("  n_atoms  = {}", mol.n_atoms());

    // --- 2. Atom property system ---
    // set() / get_f64() / get_str()
    mol.get_atom_mut(o).expect("O atom").set("charge", -0.8476);
    mol.get_atom_mut(h1).expect("H1 atom").set("charge", 0.4238);
    mol.get_atom_mut(h2).expect("H2 atom").set("charge", 0.4238);

    let o_atom = mol.get_atom(o).expect("O atom");
    println!(
        "\n  O atom: symbol={:?}, x={:.4}, charge={:.4}",
        o_atom.get_str("symbol").unwrap(),
        o_atom.get_f64("x").unwrap(),
        o_atom.get_f64("charge").unwrap(),
    );

    // Index trait for quick access (panics if key missing)
    assert_eq!(mol.get_atom(o).expect("O atom")["x"], PropValue::F64(0.0));

    // keys() and contains_key()
    let keys: Vec<&str> = mol.get_atom(o).expect("O atom").keys().collect();
    println!("  O atom keys: {:?}", keys);
    println!(
        "  contains 'charge'? {}",
        mol.get_atom(o).expect("O atom").contains_key("charge")
    );

    // remove() a property
    mol.get_atom_mut(o).expect("O atom").remove("charge");
    println!(
        "  After removing 'charge': contains_key = {}",
        mol.get_atom(o).expect("O atom").contains_key("charge")
    );

    // --- 3. Bonds and topology ---
    let b1 = mol.add_bond(o, h1).expect("add O-H1 bond");
    let b2 = mol.add_bond(o, h2).expect("add O-H2 bond");
    mol.add_angle(h1, o, h2).expect("add H-O-H angle");

    println!("\nTopology:");
    println!("  n_bonds  = {}", mol.n_bonds());
    println!("  n_angles = {}", mol.n_angles());

    // Set bond order
    mol.get_bond_mut(b1)
        .expect("bond b1")
        .props
        .insert("order".into(), PropValue::F64(1.0));
    mol.get_bond_mut(b2)
        .expect("bond b2")
        .props
        .insert("order".into(), PropValue::F64(1.0));

    // --- 4. Iteration ---
    println!("\nAll atoms:");
    for (_id, atom) in mol.atoms() {
        println!(
            "  {} at ({:.4}, {:.4}, {:.4})",
            atom.get_str("symbol").unwrap(),
            atom.get_f64("x").unwrap(),
            atom.get_f64("y").unwrap(),
            atom.get_f64("z").unwrap(),
        );
    }

    println!("\nAll bonds:");
    for (_bid, bond) in mol.bonds() {
        let order = bond
            .props
            .get("order")
            .and_then(|v| {
                if let PropValue::F64(f) = v {
                    Some(*f)
                } else {
                    None
                }
            })
            .unwrap_or(1.0);
        println!("  bond order = {:.1}", order);
    }

    // --- 5. Neighbor queries ---
    println!("\nNeighbors of O:");
    for nid in mol.neighbors(o) {
        let sym = mol
            .get_atom(nid)
            .expect("neighbor atom")
            .get_str("symbol")
            .unwrap();
        println!("  -> {}", sym);
    }

    println!("\nNeighbor bonds of O (with bond order):");
    for (nid, order) in mol.neighbor_bonds(o) {
        let sym = mol
            .get_atom(nid)
            .expect("neighbor atom")
            .get_str("symbol")
            .unwrap();
        println!("  -> {} (order = {:.1})", sym, order);
    }

    println!();
}

// ─── Ethane (C2H6) ──────────────────────────────────────────────────────────

fn ethane() {
    println!("=== Ethane (C2H6) ===\n");

    let mut mol = MolGraph::new();

    // Two carbons along x-axis (C-C bond ~1.54 A)
    let c1 = mol.add_atom(Atom::xyz("C", 0.0, 0.0, 0.0));
    let c2 = mol.add_atom(Atom::xyz("C", 1.54, 0.0, 0.0));

    // Three hydrogens on each carbon (tetrahedral geometry)
    let h1 = mol.add_atom(Atom::xyz("H", -0.39, 0.93, 0.29));
    let h2 = mol.add_atom(Atom::xyz("H", -0.39, -0.29, -0.93));
    let h3 = mol.add_atom(Atom::xyz("H", -0.39, -0.64, 0.64));
    let h4 = mol.add_atom(Atom::xyz("H", 1.93, -0.93, -0.29));
    let h5 = mol.add_atom(Atom::xyz("H", 1.93, 0.29, 0.93));
    let h6 = mol.add_atom(Atom::xyz("H", 1.93, 0.64, -0.64));

    // Bonds
    let cc_bond = mol.add_bond(c1, c2).expect("add C-C bond");
    mol.add_bond(c1, h1).expect("add C1-H1 bond");
    mol.add_bond(c1, h2).expect("add C1-H2 bond");
    mol.add_bond(c1, h3).expect("add C1-H3 bond");
    mol.add_bond(c2, h4).expect("add C2-H4 bond");
    mol.add_bond(c2, h5).expect("add C2-H5 bond");
    mol.add_bond(c2, h6).expect("add C2-H6 bond");

    // Angles (H-C-C and H-C-H)
    mol.add_angle(h1, c1, c2).expect("add H-C-C angle");
    mol.add_angle(h2, c1, c2).expect("add H-C-C angle");
    mol.add_angle(h3, c1, c2).expect("add H-C-C angle");
    mol.add_angle(h1, c1, h2).expect("add H-C-H angle");

    // Dihedrals (H-C-C-H)
    mol.add_dihedral(h1, c1, c2, h4)
        .expect("add H-C-C-H dihedral");
    mol.add_dihedral(h2, c1, c2, h5)
        .expect("add H-C-C-H dihedral");

    println!(
        "Built ethane: {} atoms, {} bonds, {} angles, {} dihedrals",
        mol.n_atoms(),
        mol.n_bonds(),
        mol.n_angles(),
        mol.n_dihedrals(),
    );

    // Check connectivity
    println!(
        "C1 has {} neighbors, C2 has {} neighbors",
        mol.neighbors(c1).count(),
        mol.neighbors(c2).count(),
    );

    // --- 6. Cascading deletion ---
    println!("\nDemonstrating cascading deletion:");
    println!(
        "  Before removing C1: {} atoms, {} bonds, {} angles, {} dihedrals",
        mol.n_atoms(),
        mol.n_bonds(),
        mol.n_angles(),
        mol.n_dihedrals(),
    );

    mol.remove_atom(c1).expect("remove C1");

    println!(
        "  After removing C1:  {} atoms, {} bonds, {} angles, {} dihedrals",
        mol.n_atoms(),
        mol.n_bonds(),
        mol.n_angles(),
        mol.n_dihedrals(),
    );
    println!("  (All bonds, angles, and dihedrals referencing C1 were removed)");

    // Verify C-C bond was removed
    assert!(mol.get_bond(cc_bond).is_err());

    println!();
}

// ─── Coarse-grained beads ───────────────────────────────────────────────────

fn coarse_grained() {
    println!("=== Coarse-grained Beads ===\n");

    let mut mol = MolGraph::new();

    // Bead is a type alias for Atom — same API, different convention
    let mut w1 = Bead::new();
    w1.set("name", "W");
    w1.set("x", 0.0);
    w1.set("y", 0.0);
    w1.set("z", 0.0);
    w1.set("mass", 72.0);

    let mut w2 = Bead::new();
    w2.set("name", "W");
    w2.set("x", 4.7);
    w2.set("y", 0.0);
    w2.set("z", 0.0);
    w2.set("mass", 72.0);

    let id1 = mol.add_atom(w1);
    let id2 = mol.add_atom(w2);
    mol.add_bond(id1, id2).expect("add CG bond");

    println!("CG model: {} beads, {} bonds", mol.n_atoms(), mol.n_bonds());

    let bead = mol.get_atom(id1).expect("bead 1");
    println!(
        "Bead 1: name={:?}, mass={}, x={}",
        bead.get_str("name").unwrap(),
        bead.get_f64("mass").unwrap(),
        bead.get_f64("x").unwrap(),
    );
    println!("(Bead is just a type alias for Atom — same API, different property conventions)");
}

// ─── Hydrogen add / remove roundtrip ───────────────────────────────────────

fn hydrogen_roundtrip() {
    println!("=== Hydrogen Add / Remove Roundtrip ===\n");

    // Build ethanol skeleton: C-C-O
    let mut mol = MolGraph::new();
    let c1 = mol.add_atom(Atom::xyz("C", 0.0, 0.0, 0.0));
    let c2 = mol.add_atom(Atom::xyz("C", 1.54, 0.0, 0.0));
    let o = mol.add_atom(Atom::xyz("O", 2.40, 0.94, 0.0));
    mol.add_bond(c1, c2).expect("add C-C bond");
    mol.add_bond(c2, o).expect("add C-O bond");

    println!(
        "Skeleton:  {} atoms, {} bonds",
        mol.n_atoms(),
        mol.n_bonds(),
    );

    // Add hydrogens
    let with_h = add_hydrogens(&mol);
    println!(
        "After add_hydrogens:  {} atoms, {} bonds",
        with_h.n_atoms(),
        with_h.n_bonds(),
    );

    // Remove hydrogens
    let stripped = remove_hydrogens(&with_h);
    println!(
        "After remove_hydrogens: {} atoms, {} bonds",
        stripped.n_atoms(),
        stripped.n_bonds(),
    );

    // Verify immutability: original skeleton unchanged
    assert_eq!(mol.n_atoms(), 3, "original unchanged");
    assert!(with_h.n_atoms() > stripped.n_atoms());
    println!(
        "\nOriginal skeleton still has {} atoms (immutable).",
        mol.n_atoms()
    );
}
