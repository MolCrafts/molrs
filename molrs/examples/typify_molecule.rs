//! Tutorial: How to Typify a Molecule
//!
//! This example demonstrates the complete MMFF94 typification pipeline in molrs:
//!
//!   MolGraph  →  typify()  →  Frame  →  to_potentials()  →  eval
//!
//! "Typification" assigns every atom a force-field integer type that encodes its
//! chemical environment (element, hybridization, ring membership, aromaticity, …).
//! These types are the lookup keys for force-field parameters (spring constants,
//! equilibrium bond lengths, van der Waals radii, partial charges, etc.).
//!
//! # MMFF94 Typing Algorithm
//!
//! Each atom is classified by an **8-tuple** of properties:
//!
//! | Prop  | Meaning                                 | Example           |
//! |-------|-----------------------------------------|-------------------|
//! | atno  | Atomic number                           | 6 (carbon)        |
//! | crd   | Coordination (# bonded neighbors)       | 3 (sp2)           |
//! | val   | Valence (Σ bond orders + implicit H)    | 4                 |
//! | pilp  | Pi lone pair available?                 | 1 for pyrrole-N   |
//! | mltb  | Multiple bond type                      | 2 = double/arom   |
//! | arom  | Aromatic?                               | 1 in benzene ring |
//! | linh  | Linear (sp)?                            | 1 for -C≡C-       |
//! | sbmb  | Single/multi bond mix?                  | 1 for C(=O)-O     |
//!
//! The 8-tuple is matched against the MMFF property table (~95 canonical types).
//! When multiple types share the same 8-tuple, ring-size disambiguation selects
//! the right one (e.g. type 22 for 3-ring carbon vs type 20 for 4-ring carbon).
//!
//! # The Pipeline
//!
//! ```text
//! MolGraph (atoms + bonds + coords)
//!     │
//!     ├─ typifier.typify(&mol)         → labeled Atomistic
//!     │     ├─ MmffMolProperties             (validated atom types + charges)
//!     │     └─ ff::mmff::params               per-instance kb/r0, ka/theta0,
//!     │         (the RDKit-faithful           v1/v2/v3, koop + the type codes
//!     │          resolver: aromaticity,       — one resolver for the numbers
//!     │          ring size, equivalence       AND the labels
//!     │          degradation, empirical
//!     │          fallbacks)
//!     │
//!     ├─ .to_frame()                   → typed Frame
//!     ├─ frame["pairs"] = intramolecular_pairs(&frame)   (the consumer's list)
//!     │
//!     └─ typifier.ff().to_potentials(&frame)  → Potentials (pre-resolved SoA)
//!           │
//!           potentials.calc_energy_forces(coords) → (energy, forces)
//! ```
//!
//! MMFF is **not** a special case: the last step is the same `to_potentials` call
//! every force field in molrs goes through. Its bonded kernels are `PerInstance`
//! styles — they read the parameter columns the typifier baked, not type rows.
//!
//! # Common MMFF Types
//!
//! | ID | Symbol | Description          |
//! |----|--------|----------------------|
//! |  1 | CR     | sp3 carbon           |
//! |  2 | C=C    | sp2 carbon (vinyl)   |
//! |  3 | C=O    | sp2 carbon (carbonyl)|
//! |  5 | HC     | hydrogen on carbon   |
//! |  6 | OR     | sp3 oxygen           |
//! |  7 | O=C    | carbonyl oxygen      |
//! |  8 | NR     | sp3 nitrogen         |
//! | 37 | CB     | aromatic carbon      |
//! | 44 | NB     | aromatic N (pyridine)|
//!
//! Run:
//!   cargo run -p molcrafts-molrs --example typify_molecule --features ff
//!
//! MMFF94 parameters are embedded in the binary — no external data files needed.

use molrs::Atomistic;
use molrs::ff::potential::{extract_coords, intramolecular_pairs};
use molrs::ff::typifier::mmff::MMFF94Typifier;
use molrs::perceive::rings::find_rings;
use molrs::system::molgraph::{Atom, PropValue};
use molrs::types::F;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // =========================================================================
    // Part 1: Build molecular graphs
    // =========================================================================

    println!("=== Part 1: Building Molecular Graphs ===\n");

    let ethane = build_ethane();
    let benzene = build_benzene();
    let acetic_acid = build_acetic_acid();

    let (na, nb) = count(&ethane);
    println!("Built ethane:      {} atoms, {} bonds", na, nb);
    let (na, nb) = count(&benzene);
    println!("Built benzene:     {} atoms, {} bonds", na, nb);
    let (na, nb) = count(&acetic_acid);
    println!("Built acetic acid: {} atoms, {} bonds", na, nb);

    // =========================================================================
    // Part 2: Ring perception
    // =========================================================================

    println!("\n=== Part 2: Ring Perception ===\n");

    let ethane_rings = find_rings(&ethane);
    let benzene_rings = find_rings(&benzene);

    println!("Ethane rings:  {}", ethane_rings.num_rings());
    println!("Benzene rings: {}", benzene_rings.num_rings());

    for (aid, atom) in benzene.atoms() {
        if atom.get_str("symbol") == Some("C") {
            let in_ring = benzene_rings.is_atom_in_ring(aid);
            let size = benzene_rings.smallest_ring_containing_atom(aid);
            println!(
                "  C atom {:?}: in_ring={}, smallest_ring={:?}",
                aid, in_ring, size
            );
            break;
        }
    }

    // =========================================================================
    // Part 3: Create MMFF94 typifier
    // =========================================================================
    // One call loads typing metadata + force-field parameters from embedded XML.

    println!("\n=== Part 3: Creating MMFF94 Typifier ===\n");

    let typifier = MMFF94Typifier::new();

    if let Some(prop) = typifier.params().get_prop(1) {
        println!(
            "Type 1 (CR): atno={}, crd={}, val={}, mltb={}, arom={}",
            prop.atno, prop.crd, prop.val, prop.mltb, prop.arom
        );
    }
    if let Some(prop) = typifier.params().get_prop(37) {
        println!(
            "Type 37 (CB): atno={}, crd={}, val={}, mltb={}, arom={}",
            prop.atno, prop.crd, prop.val, prop.mltb, prop.arom
        );
    }

    println!(
        "ForceField '{}': {} style(s)",
        typifier.ff().name,
        typifier.ff().styles().len()
    );

    // =========================================================================
    // Part 4: Assign atom types
    // =========================================================================

    println!("\n=== Part 4: Atom Type Assignment ===\n");

    print_atom_types("Ethane", &typifier, &ethane)?;
    print_atom_types("\nBenzene", &typifier, &benzene)?;
    print_atom_types("\nAcetic acid", &typifier, &acetic_acid)?;

    // =========================================================================
    // Part 5: Bond / Angle / Torsion typification
    //
    // These type codes are NOT a function of the atom types alone — which is why
    // there is no `typifier.typify_bond(t1, t2, order)` to call here. MMFF's bond
    // type needs perceived aromaticity (an aromatic ring bond is AROMATIC, never
    // SINGLE, so it is bond type 0 — not 1); its angle type needs ring membership
    // (a C-C-C angle in cyclopropane is type 3); its torsion type needs 4-/5-ring
    // detection. All three are resolved per interaction, against the topology, and
    // stamped on the typed Frame — so the way to see them is to read the Frame.
    // =========================================================================

    println!("\n=== Part 5: Interaction Typification ===\n");
    println!("Type codes are resolved against the topology (aromaticity, ring size),");
    println!("not from atom types alone — so they are read off the typed Frame below.");
    println!("Label grammar: `{{type_code}}_{{atom_types...}}`.\n");

    let benz_bonds = typifier.typify(&benzene)?.to_frame();
    let benz_bond_types = benz_bonds
        .get("bonds")
        .and_then(|b| b.get_string("type"))
        .expect("benzene bond type column");
    println!("Benzene bond labels (aromatic ring bonds are bond type 0):");
    for label in benz_bond_types.iter().take(4) {
        println!("  {label}");
    }

    // =========================================================================
    // Part 6: Inspect a typed Frame (optional — for debugging)
    // =========================================================================

    println!("\n=== Part 6: Inspecting Typed Frame ===\n");

    let mut frame = typifier.typify(&ethane)?.to_frame();
    // The neighbour list is the consumer's to build; add it now so the Frame below
    // is the complete input `to_potentials` (Part 7) is handed.
    frame.insert("pairs", intramolecular_pairs(&frame));

    let atoms = frame.get("atoms").expect("atoms block");
    let bonds = frame.get("bonds").expect("bonds block");
    let angles = frame.get("angles").expect("angles block");
    let dihedrals = frame.get("dihedrals").expect("dihedrals block");

    println!("Ethane Frame:");
    println!("  atoms:     {} rows", atoms.nrows().unwrap_or(0));
    println!("  bonds:     {} rows", bonds.nrows().unwrap_or(0));
    println!("  angles:    {} rows", angles.nrows().unwrap_or(0));
    println!("  dihedrals: {} rows", dihedrals.nrows().unwrap_or(0));

    let type_col = atoms.get_string("type").expect("type column");
    let charge_col = atoms.get_float("charge").expect("charge column");
    println!("\n  Atom details:");
    for i in 0..type_col.len() {
        println!(
            "    [{}] type={:>2}, charge={:+.4}",
            i, type_col[i], charge_col[i]
        );
    }

    let bond_type_col = bonds.get_string("type").expect("bond type column");
    let bond_i = bonds.get_uint("atomi").expect("bond atomi");
    let bond_j = bonds.get_uint("atomj").expect("bond atomj");
    println!("\n  Bond details:");
    for idx in 0..bond_type_col.len() {
        println!(
            "    [{}-{}] type={}",
            bond_i[idx], bond_j[idx], bond_type_col[idx]
        );
    }

    // =========================================================================
    // Part 7: Compile potentials and evaluate (the standard ForceField route)
    // =========================================================================

    println!("\n=== Part 7: Compile & Evaluate ===\n");

    match typifier.ff().to_potentials(&frame) {
        Ok(potentials) => {
            println!("Built {} potential kernel(s)", potentials.len());

            let coords = extract_coords(&frame)?;
            let (energy, forces) = potentials.calc_energy_forces(&coords);

            println!("Total energy: {:.4} kcal/mol", energy);

            let n_atoms_frame = coords.len() / 3;
            let mut fsum = [0.0 as F; 3];
            for i in 0..n_atoms_frame {
                fsum[0] += forces[3 * i];
                fsum[1] += forces[3 * i + 1];
                fsum[2] += forces[3 * i + 2];
            }
            println!(
                "Force sum: [{:+.2e}, {:+.2e}, {:+.2e}] (should be ~0)",
                fsum[0], fsum[1], fsum[2]
            );

            println!("\n  Per-atom forces:");
            for i in 0..n_atoms_frame {
                println!(
                    "    [{}] type={:>2}  f=[{:+8.3}, {:+8.3}, {:+8.3}]",
                    i,
                    type_col[i],
                    forces[3 * i],
                    forces[3 * i + 1],
                    forces[3 * i + 2],
                );
            }
        }
        Err(e) => {
            println!("Build skipped (incomplete parameter coverage): {}", e);
            println!("(Typification completed successfully)");
        }
    }

    // =========================================================================
    // Bonus: Benzene
    // =========================================================================
    println!("\n=== Bonus: Benzene ===\n");

    let benz_frame = typifier.typify(&benzene)?.to_frame();
    let benz_atoms = benz_frame.get("atoms").unwrap();
    let benz_types = benz_atoms.get_string("type").unwrap();
    println!("Benzene atom types: {:?}", benz_types.as_slice().unwrap());

    let benz_bonds = benz_frame.get("bonds").unwrap();
    let benz_bond_types = benz_bonds.get_string("type").unwrap();
    println!(
        "Benzene bond types: {:?}",
        benz_bond_types.as_slice().unwrap()
    );

    Ok(())
}

// =============================================================================
// Molecule builders
// =============================================================================

fn build_ethane() -> Atomistic {
    let mut mol = Atomistic::new();
    let c1 = mol.add_atom(Atom::xyz("C", 0.0, 0.0, 0.0));
    let c2 = mol.add_atom(Atom::xyz("C", 1.54, 0.0, 0.0));
    mol.add_bond(c1, c2).unwrap();

    let h_positions: [(_, [f64; 3]); 6] = [
        (c1, [0.0, 1.0, 0.0]),
        (c1, [0.0, -0.5, 0.87]),
        (c1, [0.0, -0.5, -0.87]),
        (c2, [1.54, 1.0, 0.0]),
        (c2, [1.54, -0.5, 0.87]),
        (c2, [1.54, -0.5, -0.87]),
    ];
    for (c, [x, y, z]) in h_positions {
        let h = mol.add_atom(Atom::xyz("H", x, y, z));
        mol.add_bond(c, h).unwrap();
    }
    mol
}

fn build_benzene() -> Atomistic {
    use std::f64::consts::PI;

    let mut mol = Atomistic::new();
    let r = 1.40;

    let mut carbons = Vec::new();
    for i in 0..6 {
        let angle = 2.0 * PI * (i as f64) / 6.0;
        let x = r * angle.cos();
        let y = r * angle.sin();
        carbons.push(mol.add_atom(Atom::xyz("C", x, y, 0.0)));
    }

    for i in 0..6 {
        let bid = mol.add_bond(carbons[i], carbons[(i + 1) % 6]).unwrap();
        mol.set_bond_prop(bid, "order", PropValue::F64(1.5))
            .unwrap();
    }

    let rh = 2.48;
    #[allow(clippy::needless_range_loop)]
    for i in 0..6 {
        let angle = 2.0 * PI * (i as f64) / 6.0;
        let hx = rh * angle.cos();
        let hy = rh * angle.sin();
        let h = mol.add_atom(Atom::xyz("H", hx, hy, 0.0));
        mol.add_bond(carbons[i], h).unwrap();
    }

    mol
}

fn build_acetic_acid() -> Atomistic {
    let mut mol = Atomistic::new();

    let c_me = mol.add_atom(Atom::xyz("C", 0.0, 0.0, 0.0));
    let h1 = mol.add_atom(Atom::xyz("H", -0.5, 0.9, 0.0));
    let h2 = mol.add_atom(Atom::xyz("H", -0.5, -0.45, 0.78));
    let h3 = mol.add_atom(Atom::xyz("H", -0.5, -0.45, -0.78));
    mol.add_bond(c_me, h1).unwrap();
    mol.add_bond(c_me, h2).unwrap();
    mol.add_bond(c_me, h3).unwrap();

    let c_co = mol.add_atom(Atom::xyz("C", 1.52, 0.0, 0.0));
    mol.add_bond(c_me, c_co).unwrap();

    let o_dbl = mol.add_atom(Atom::xyz("O", 2.1, 1.1, 0.0));
    let bid_co = mol.add_bond(c_co, o_dbl).unwrap();
    mol.set_bond_prop(bid_co, "order", PropValue::F64(2.0))
        .unwrap();

    let o_oh = mol.add_atom(Atom::xyz("O", 2.1, -1.1, 0.0));
    mol.add_bond(c_co, o_oh).unwrap();

    let h_oh = mol.add_atom(Atom::xyz("H", 3.0, -1.3, 0.0));
    mol.add_bond(o_oh, h_oh).unwrap();

    mol
}

// =============================================================================
// Utilities
// =============================================================================

fn count(mol: &Atomistic) -> (usize, usize) {
    (mol.atoms().count(), mol.bonds().count())
}

/// Typify `mol` and print each atom's MMFF numeric type. The validated types now
/// come from the typed frame's `atoms` `type` column (in graph atom order),
/// produced by reusing the RDKit-validated MMFF front-end.
fn print_atom_types(
    name: &str,
    typifier: &MMFF94Typifier,
    mol: &Atomistic,
) -> Result<(), Box<dyn std::error::Error>> {
    let frame = typifier.typify(mol)?.to_frame();
    let types = frame
        .get("atoms")
        .and_then(|b| b.get_string("type"))
        .ok_or("atoms type column")?;
    println!("{name} atom types:");
    for ((_, atom), ty) in mol.atoms().zip(types.iter()) {
        let sym = atom.get_str("symbol").unwrap_or("?");
        let t: u32 = ty.parse().unwrap_or(0);
        println!("  {} → type {} ({})", sym, t, type_label(t));
    }
    Ok(())
}

fn type_label(t: u32) -> &'static str {
    match t {
        1 => "CR (sp3 C)",
        2 => "C=C (vinyl)",
        3 => "C=O (carbonyl C)",
        4 => "CSP (sp C)",
        5 => "HC (H on C)",
        6 => "OR (sp3 O)",
        7 => "O=C (carbonyl O)",
        8 => "NR (sp3 N)",
        9 => "N=C",
        10 => "NC=O (amide N)",
        11 => "F",
        12 => "Cl",
        13 => "Br",
        14 => "I",
        15 => "S",
        16 => "S=O",
        20 => "CR4R (C in 4-ring)",
        21 => "HOR (H on O)",
        22 => "CR3R (C in 3-ring)",
        23 => "HNR (H on N)",
        24 => "HOCO (H on COOH)",
        37 => "CB (aromatic C)",
        44 => "NB (aromatic N, pyridine)",
        _ => "(?)",
    }
}
