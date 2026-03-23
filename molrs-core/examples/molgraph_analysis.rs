//! # Analysis & Conversion with MolGraph
//!
//! Demonstrates ring detection, Frame round-trip conversion, and
//! molecular summary statistics using a benzene (C6H6) molecule.
//!
//! Run with: `cargo run -p molrs-core --example molgraph_analysis`

use molrs::{Atom, MolGraph, PropValue, find_rings};

fn main() {
    let mol = build_benzene();
    ring_detection(&mol);
    frame_roundtrip(&mol);
    summary(&mol);
}

/// Build benzene (C6H6) with aromatic bonds.
fn build_benzene() -> MolGraph {
    let mut mol = MolGraph::new();

    // Six carbons in a regular hexagon (C-C ~1.40 A)
    let angles_rad: Vec<f64> = (0..6)
        .map(|i| i as f64 * std::f64::consts::TAU / 6.0)
        .collect();
    let r = 1.40;

    let carbons: Vec<_> = angles_rad
        .iter()
        .map(|&a| mol.add_atom(Atom::xyz("C", r * a.cos(), r * a.sin(), 0.0)))
        .collect();

    // C-C aromatic bonds (order 1.5)
    let mut bond_ids = Vec::new();
    for i in 0..6 {
        let bid = mol.add_bond(carbons[i], carbons[(i + 1) % 6]).unwrap();
        mol.get_bond_mut(bid)
            .expect("bond just added")
            .props
            .insert("order".into(), PropValue::F64(1.5));
        bond_ids.push(bid);
    }

    // Add angles (C-C-C)
    for i in 0..6 {
        mol.add_angle(carbons[(i + 5) % 6], carbons[i], carbons[(i + 1) % 6])
            .expect("add C-C-C angle");
    }

    // Six hydrogens pointing outward (~1.08 A from carbon)
    let r_h = r + 1.08;
    for (i, &a) in angles_rad.iter().enumerate() {
        let h = mol.add_atom(Atom::xyz("H", r_h * a.cos(), r_h * a.sin(), 0.0));
        let bid = mol.add_bond(carbons[i], h).expect("add C-H bond");
        mol.get_bond_mut(bid)
            .expect("bond just added")
            .props
            .insert("order".into(), PropValue::F64(1.0));
    }

    mol
}

// ─── Ring detection ─────────────────────────────────────────────────────────

fn ring_detection(mol: &MolGraph) {
    println!("=== Ring Detection (Benzene) ===\n");

    let rings = find_rings(mol);

    println!("  Total rings found: {}", rings.num_rings());
    println!("  Ring sizes: {:?}", rings.ring_sizes());

    // Check 6-membered rings
    let six_rings = rings.rings_of_size(6);
    println!("  6-membered rings: {}", six_rings.len());

    // Per-atom ring membership
    println!("\n  Per-atom ring info:");
    for (id, atom) in mol.atoms() {
        let sym = atom.get_str("symbol").unwrap();
        let in_ring = rings.is_atom_in_ring(id);
        let n_rings = rings.num_atom_rings(id);
        let smallest = rings.smallest_ring_containing_atom(id);
        println!(
            "    {} — in_ring={}, num_rings={}, smallest={:?}",
            sym, in_ring, n_rings, smallest,
        );
    }

    // Per-bond ring membership
    println!("\n  Per-bond ring info:");
    for (bid, bond) in mol.bonds() {
        let a_sym = mol
            .get_atom(bond.atoms[0])
            .expect("atom exists")
            .get_str("symbol")
            .unwrap();
        let b_sym = mol
            .get_atom(bond.atoms[1])
            .expect("atom exists")
            .get_str("symbol")
            .unwrap();
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
        println!(
            "    {}-{} (order={:.1}) — in_ring={}",
            a_sym,
            b_sym,
            order,
            rings.is_bond_in_ring(bid),
        );
    }

    println!();
}

// ─── Frame round-trip ───────────────────────────────────────────────────────

fn frame_roundtrip(mol: &MolGraph) {
    println!("=== Frame Round-trip ===\n");

    // --- Export to Frame ---
    let frame = mol.to_frame();

    println!("  Exported Frame blocks:");
    for (name, block) in frame.iter() {
        println!(
            "    {:>12}: nrows={:?}, ncols={}",
            name,
            block.nrows(),
            block.len(),
        );
    }

    // Inspect the atoms block
    if let Some(atoms_block) = frame.get("atoms") {
        let col_names: Vec<&str> = atoms_block.keys().collect();
        println!("\n  Atoms block columns: {:?}", col_names);

        if let Some(x_col) = atoms_block.get_float("x") {
            println!(
                "  First 3 x-coordinates: {:.4?}",
                &x_col.as_slice().unwrap()[..3]
            );
        }
    }

    // --- Import from Frame ---
    let mol2 = MolGraph::from_frame(&frame).expect("round-trip should succeed");

    println!("\n  Round-trip verification:");
    println!(
        "    original  — atoms={}, bonds={}, angles={}",
        mol.n_atoms(),
        mol.n_bonds(),
        mol.n_angles()
    );
    println!(
        "    from_frame — atoms={}, bonds={}, angles={}",
        mol2.n_atoms(),
        mol2.n_bonds(),
        mol2.n_angles()
    );

    assert_eq!(mol.n_atoms(), mol2.n_atoms());
    assert_eq!(mol.n_bonds(), mol2.n_bonds());
    assert_eq!(mol.n_angles(), mol2.n_angles());
    println!("    All counts match!");

    println!();
}

// ─── Summary statistics ─────────────────────────────────────────────────────

fn summary(mol: &MolGraph) {
    println!("=== Benzene Summary ===\n");

    println!("  Atoms:     {}", mol.n_atoms());
    println!("  Bonds:     {}", mol.n_bonds());
    println!("  Angles:    {}", mol.n_angles());
    println!("  Dihedrals: {}", mol.n_dihedrals());

    // Element breakdown
    let mut elements: std::collections::HashMap<&str, usize> = std::collections::HashMap::new();
    for (_id, atom) in mol.atoms() {
        if let Some(sym) = atom.get_str("symbol") {
            *elements.entry(sym).or_default() += 1;
        }
    }
    print!("  Formula:  ");
    let mut sorted: Vec<_> = elements.iter().collect();
    sorted.sort_by_key(|(sym, _)| *sym);
    for (sym, count) in &sorted {
        print!("{}{}", sym, count);
    }
    println!();

    // Ring info
    let rings = find_rings(mol);
    println!(
        "  Rings:     {} (sizes: {:?})",
        rings.num_rings(),
        rings.ring_sizes()
    );

    // Bond order distribution
    let mut single = 0;
    let mut aromatic = 0;
    let mut other = 0;
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
        if (order - 1.0).abs() < 0.01 {
            single += 1;
        } else if (order - 1.5).abs() < 0.01 {
            aromatic += 1;
        } else {
            other += 1;
        }
    }
    println!(
        "  Bond types: {} single, {} aromatic, {} other",
        single, aromatic, other,
    );
}
