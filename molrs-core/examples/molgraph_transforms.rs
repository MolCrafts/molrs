//! # Spatial Transforms & Composition with MolGraph
//!
//! Demonstrates translation, rotation, cloning, and merging of molecules.
//! Uses methane (CH4) as the example molecule.
//!
//! Run with: `cargo run -p molrs-core --example molgraph_transforms`

use molrs::{Atom, MolGraph};

fn main() {
    let mol = build_methane();
    translation(&mol);
    rotation(&mol);
    clone_independence(&mol);
    merge_molecules(&mol);
}

/// Build a methane molecule (CH4) with tetrahedral geometry.
fn build_methane() -> MolGraph {
    let mut mol = MolGraph::new();

    // Central carbon
    let c = mol.add_atom(Atom::xyz("C", 0.0, 0.0, 0.0));

    // Four hydrogens at tetrahedral positions (~1.09 A bond length)
    let h1 = mol.add_atom(Atom::xyz("H", 0.6293, 0.6293, 0.6293));
    let h2 = mol.add_atom(Atom::xyz("H", 0.6293, -0.6293, -0.6293));
    let h3 = mol.add_atom(Atom::xyz("H", -0.6293, 0.6293, -0.6293));
    let h4 = mol.add_atom(Atom::xyz("H", -0.6293, -0.6293, 0.6293));

    mol.add_bond(c, h1).expect("add C-H1 bond");
    mol.add_bond(c, h2).expect("add C-H2 bond");
    mol.add_bond(c, h3).expect("add C-H3 bond");
    mol.add_bond(c, h4).expect("add C-H4 bond");

    mol.add_angle(h1, c, h2).expect("add angle");
    mol.add_angle(h1, c, h3).expect("add angle");
    mol.add_angle(h1, c, h4).expect("add angle");
    mol.add_angle(h2, c, h3).expect("add angle");
    mol.add_angle(h2, c, h4).expect("add angle");
    mol.add_angle(h3, c, h4).expect("add angle");

    mol
}

/// Print all atom coordinates.
fn print_coords(mol: &MolGraph, label: &str) {
    println!("  {}:", label);
    for (_id, atom) in mol.atoms() {
        println!(
            "    {} ({:>8.4}, {:>8.4}, {:>8.4})",
            atom.get_str("symbol").unwrap(),
            atom.get_f64("x").unwrap(),
            atom.get_f64("y").unwrap(),
            atom.get_f64("z").unwrap(),
        );
    }
}

// ─── Translation ────────────────────────────────────────────────────────────

fn translation(original: &MolGraph) {
    println!("=== Translation ===\n");

    let mut mol = original.clone();
    print_coords(&mol, "Before translate");

    mol.translate([5.0, 0.0, 0.0]);
    print_coords(&mol, "After translate([5, 0, 0])");

    mol.translate([0.0, 3.0, -1.0]);
    print_coords(&mol, "After translate([0, 3, -1])");

    println!();
}

// ─── Rotation ───────────────────────────────────────────────────────────────

fn rotation(original: &MolGraph) {
    println!("=== Rotation ===\n");

    // --- Rotate 90 degrees around z-axis (about origin) ---
    let mut mol = original.clone();
    let half_pi = std::f64::consts::FRAC_PI_2;

    print_coords(&mol, "Before rotation");

    mol.rotate([0.0, 0.0, 1.0], half_pi, None);
    print_coords(&mol, "After 90deg rotation around z-axis (origin)");

    // --- Rotate around a custom center ---
    let mut mol2 = original.clone();
    let center = [5.0, 5.0, 0.0];

    print_coords(&mol2, "\n  Before rotation about center (5,5,0)");

    mol2.rotate([0.0, 0.0, 1.0], half_pi, Some(center));
    print_coords(&mol2, "After 90deg rotation around z-axis about (5,5,0)");

    // --- 180 degree rotation (flip) ---
    let mut mol3 = original.clone();
    let pi = std::f64::consts::PI;

    mol3.rotate([1.0, 0.0, 0.0], pi, None);
    print_coords(&mol3, "\n  After 180deg rotation around x-axis");

    println!();
}

// ─── Clone independence ────────────────────────────────────────────────────

fn clone_independence(original: &MolGraph) {
    println!("=== Clone Independence ===\n");

    let mut a = original.clone();
    let b = a.clone();

    // Mutate 'a'
    a.translate([100.0, 0.0, 0.0]);

    // 'b' should be unaffected
    let a_first_x = a.atoms().next().unwrap().1.get_f64("x").unwrap();
    let b_first_x = b.atoms().next().unwrap().1.get_f64("x").unwrap();

    println!("  After translating clone 'a' by [100, 0, 0]:");
    println!("    a first atom x = {:.4}", a_first_x);
    println!("    b first atom x = {:.4} (unchanged)", b_first_x);

    // Independent atom counts
    let (first_id, _) = a.atoms().next().unwrap();
    a.remove_atom(first_id).expect("remove first atom");
    println!(
        "\n  After removing an atom from 'a': a has {} atoms, b has {} atoms",
        a.n_atoms(),
        b.n_atoms()
    );

    println!();
}

// ─── Merge ──────────────────────────────────────────────────────────────────

fn merge_molecules(original: &MolGraph) {
    println!("=== Merge ===\n");

    // Create two copies, translate one
    let mut mol1 = original.clone();
    let mut mol2 = original.clone();
    mol2.translate([5.0, 0.0, 0.0]);

    println!("  mol1: {} atoms, {} bonds", mol1.n_atoms(), mol1.n_bonds());
    println!(
        "  mol2: {} atoms, {} bonds (translated by [5,0,0])",
        mol2.n_atoms(),
        mol2.n_bonds()
    );

    // Merge mol2 into mol1 (consumes mol2, IDs are remapped)
    mol1.merge(mol2);

    println!("\n  After merge:");
    println!(
        "    combined: {} atoms, {} bonds, {} angles",
        mol1.n_atoms(),
        mol1.n_bonds(),
        mol1.n_angles(),
    );

    // Verify both molecules are present by checking coordinates
    println!("\n  All atoms in merged molecule:");
    for (_id, atom) in mol1.atoms() {
        println!(
            "    {} ({:>8.4}, {:>8.4}, {:>8.4})",
            atom.get_str("symbol").unwrap(),
            atom.get_f64("x").unwrap(),
            atom.get_f64("y").unwrap(),
            atom.get_f64("z").unwrap(),
        );
    }
}
