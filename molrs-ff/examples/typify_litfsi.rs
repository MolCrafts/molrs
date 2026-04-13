//! LiTFSI (lithium bis(trifluoromethanesulfonyl)imide) MMFF94 typification.
//!
//! Demonstrates MMFF typing on an ionic molecule commonly used as a lithium
//! salt in battery electrolytes.
//!
//! Structure: Li⁺ [N(SO₂CF₃)₂]⁻
//!   - 16 atoms: Li, N, 2×S, 4×O, 2×C, 6×F
//!   - 15 bonds: Li-N, 2×N-S, 4×S=O, 2×S-C, 6×C-F
//!
//! Run:
//!   cargo run -p molcrafts-molrs-ff --example typify_litfsi

use molrs::atomistic::Atomistic;
use molrs::molgraph::{Atom, AtomId, MolGraph, PropValue};
use molrs_ff::typifier::Typifier;
use molrs_ff::typifier::mmff::MMFFTypifier;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== LiTFSI MMFF94 Typification ===\n");

    // =========================================================================
    // Step 1: Build molecular graph
    // =========================================================================

    let mut mol = MolGraph::new();

    let li = mol.add_atom(Atom::xyz("Li", -3.0, 0.0, 0.0));
    let n = mol.add_atom(Atom::xyz("N", -1.5, 0.0, 0.0));
    let s1 = mol.add_atom(Atom::xyz("S", 0.0, 1.0, 0.0));
    let s2 = mol.add_atom(Atom::xyz("S", 0.0, -1.0, 0.0));
    let o1 = mol.add_atom(Atom::xyz("O", 0.5, 1.8, 1.0));
    let o2 = mol.add_atom(Atom::xyz("O", 0.5, 1.8, -1.0));
    let o3 = mol.add_atom(Atom::xyz("O", 0.5, -1.8, 1.0));
    let o4 = mol.add_atom(Atom::xyz("O", 0.5, -1.8, -1.0));
    let c1 = mol.add_atom(Atom::xyz("C", 1.5, 0.5, 0.0));
    let c2 = mol.add_atom(Atom::xyz("C", 1.5, -0.5, 0.0));
    let f1 = mol.add_atom(Atom::xyz("F", 2.5, 0.5, 1.0));
    let f2 = mol.add_atom(Atom::xyz("F", 2.5, 0.5, -1.0));
    let f3 = mol.add_atom(Atom::xyz("F", 2.5, 1.5, 0.0));
    let f4 = mol.add_atom(Atom::xyz("F", 2.5, -0.5, 1.0));
    let f5 = mol.add_atom(Atom::xyz("F", 2.5, -0.5, -1.0));
    let f6 = mol.add_atom(Atom::xyz("F", 2.5, -1.5, 0.0));

    add_bond(&mut mol, li, n, 1.0);
    add_bond(&mut mol, n, s1, 1.0);
    add_bond(&mut mol, n, s2, 1.0);
    add_bond(&mut mol, s1, o1, 2.0);
    add_bond(&mut mol, s1, o2, 2.0);
    add_bond(&mut mol, s2, o3, 2.0);
    add_bond(&mut mol, s2, o4, 2.0);
    add_bond(&mut mol, s1, c1, 1.0);
    add_bond(&mut mol, s2, c2, 1.0);
    add_bond(&mut mol, c1, f1, 1.0);
    add_bond(&mut mol, c1, f2, 1.0);
    add_bond(&mut mol, c1, f3, 1.0);
    add_bond(&mut mol, c2, f4, 1.0);
    add_bond(&mut mol, c2, f5, 1.0);
    add_bond(&mut mol, c2, f6, 1.0);

    let (n_atoms, n_bonds) = (mol.atoms().count(), mol.bonds().count());
    println!("Built LiTFSI: {} atoms, {} bonds", n_atoms, n_bonds);

    // =========================================================================
    // Step 2: Create typifier and assign types
    // =========================================================================

    let typifier = MMFFTypifier::mmff94()?;

    let ring_info = molrs::find_rings(&mol);
    println!("Rings detected: {}", ring_info.num_rings());

    let atom_types = typifier.assign_atom_types(&mol, &ring_info);

    println!("\n--- MMFF94 Atom Type Assignments ---\n");
    println!("{:<6} {:<8} {:<6} Description", "Index", "Element", "Type");
    println!("{}", "-".repeat(50));

    let atoms_vec: Vec<_> = mol.atoms().collect();
    for (i, (aid, atom)) in atoms_vec.iter().enumerate() {
        let sym = atom.get_str("symbol").unwrap_or("?");
        let t = atom_types.get(aid).copied().unwrap_or(0);
        let desc = type_description(sym, t);
        println!("{:<6} {:<8} {:<6} {}", i, sym, t, desc);
    }

    // =========================================================================
    // Step 3: Inspect typed Frame
    // =========================================================================
    println!("\n--- Typed Frame ---\n");

    let frame = typifier.typify(&mol)?;

    let atoms_block = frame.get("atoms").expect("atoms block");
    let bonds_block = frame.get("bonds").expect("bonds block");
    let angles_block = frame.get("angles").expect("angles block");

    println!("Frame blocks:");
    println!("  atoms:     {} rows", atoms_block.nrows().unwrap_or(0));
    println!("  bonds:     {} rows", bonds_block.nrows().unwrap_or(0));
    println!("  angles:    {} rows", angles_block.nrows().unwrap_or(0));
    if let Some(dih) = frame.get("dihedrals") {
        println!("  dihedrals: {} rows", dih.nrows().unwrap_or(0));
    }
    if let Some(pairs) = frame.get("pairs") {
        println!("  pairs:     {} rows", pairs.nrows().unwrap_or(0));
    }

    let bond_types = bonds_block.get_string("type").expect("bond type");
    let bond_i = bonds_block.get_uint("i").expect("bond i");
    let bond_j = bonds_block.get_uint("j").expect("bond j");
    println!("\n--- Bond Type Labels ---\n");
    for idx in 0..bond_types.len() {
        println!(
            "  [{:>2}-{:>2}] type={}",
            bond_i[idx], bond_j[idx], bond_types[idx]
        );
    }

    // =========================================================================
    // Step 4: Build potentials and evaluate
    // =========================================================================
    println!("\n--- Build & Evaluate ---\n");

    let mol_aa = Atomistic::try_from_molgraph(mol.clone()).expect("mol is atomistic");
    match typifier.build(&mol_aa) {
        Ok(potentials) => {
            let coords = molrs_ff::potential::extract_coords(&frame)?;
            let (energy, _) = potentials.eval(&coords);
            println!(
                "Built {} kernel(s), energy = {:.4} kcal/mol",
                potentials.len(),
                energy
            );
        }
        Err(e) => {
            println!("Build skipped (incomplete parameter coverage): {}", e);
            println!("(Typification completed successfully)");
        }
    }

    Ok(())
}

fn add_bond(mol: &mut MolGraph, a: AtomId, b: AtomId, order: f64) {
    let bid = mol.add_bond(a, b).expect("add_bond");
    if (order - 1.0).abs() > 0.01 {
        mol.get_bond_mut(bid)
            .unwrap()
            .props
            .insert("order".to_string(), PropValue::F64(order));
    }
}

fn type_description(sym: &str, t: u32) -> &'static str {
    match (sym, t) {
        ("Li", 92) => "LI+ (monovalent lithium cation)",
        ("Li", _) => "Li (fallback — check parameters)",
        ("N", 43) => "NSO2 (N in sulfonamide)",
        ("N", 8) => "NR (generic sp3 nitrogen)",
        ("N", _) => "N (other nitrogen type)",
        ("S", 18) => "SO2 (sulfone sulfur)",
        ("S", 17) => "S=O (sulfinyl sulfur)",
        ("S", 15) => "S (generic sulfur)",
        ("S", _) => "S (other sulfur type)",
        ("O", 7) => "O=S / O=C (double-bonded oxygen)",
        ("O", 32) => "O2CM (carboxylate/sulfonate O)",
        ("O", 6) => "OR (sp3 oxygen)",
        ("O", _) => "O (other oxygen type)",
        ("C", 1) => "CR (sp3 carbon — tetrahedral CF₃)",
        ("C", _) => "C (other carbon type)",
        ("F", 11) => "F (fluorine)",
        ("F", _) => "F (other fluorine type)",
        _ => "(?)",
    }
}
