//! ForceField API example: build, compile, and evaluate potentials.
//!
//! Demonstrates:
//!   1. Programmatic ForceField construction (harmonic bond + angle)
//!   2. compile(frame) → Potentials → eval(coords)
//!
//! Run: cargo run -p molrs-core --example forcefield

use molrs::block::Block;
use molrs_ff::forcefield::ForceField;
use molrs::frame::Frame;
use molrs_ff::potential::extract_coords;
use molrs::types::{F, U};
use ndarray::Array1;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // ---- 1. Build a simple water molecule ----
    let frame = build_water();

    // ---- 2. Define ForceField (programmatic) ----
    let mut ff = ForceField::new("TIP3P");

    ff.def_bondstyle("harmonic")
        .def_type("OW-HW", &[("k0", 450.0), ("r0", 0.9572)]);

    ff.def_anglestyle("harmonic")
        .def_type("HW-OW-HW", &[("k0", 55.0), ("theta0", 104.52)]);

    println!("ForceField: {}", ff.name);
    println!("  styles: {}", ff.styles().len());

    // ---- 3. Compile: ForceField + Frame → Potentials ----
    let potentials = ff.compile(&frame)?;
    println!("Compiled {} potential(s)", potentials.len());

    // ---- 4. Evaluate: coords → (energy, forces) ----
    let coords = extract_coords(&frame)?;
    let (energy, forces) = potentials.eval(&coords);

    println!("\nEnergy: {:.6} kcal/mol", energy);
    let labels = ["O", "H1", "H2"];
    for (i, label) in labels.iter().enumerate() {
        println!(
            "  {}: f = [{:+.4}, {:+.4}, {:+.4}]",
            label,
            forces[3 * i],
            forces[3 * i + 1],
            forces[3 * i + 2],
        );
    }

    // Verify forces sum to zero (Newton's third law)
    let fx_sum: F = (0..3).map(|i| forces[3 * i]).sum();
    let fy_sum: F = (0..3).map(|i| forces[3 * i + 1]).sum();
    let fz_sum: F = (0..3).map(|i| forces[3 * i + 2]).sum();
    println!(
        "\nForce sum: [{:+.2e}, {:+.2e}, {:+.2e}] (should be ~0)",
        fx_sum, fy_sum, fz_sum
    );

    Ok(())
}

/// Build a TIP3P water molecule: O at origin, two H atoms at bond length 0.9572
/// with H-O-H angle 104.52 degrees.
fn build_water() -> Frame {
    let half_angle = (104.52_f64 / 2.0).to_radians();
    let r = 0.9572_f64;
    let hx = r * half_angle.cos();
    let hy = r * half_angle.sin();

    let mut frame = Frame::new();

    // Atoms block: x, y, z, type columns
    let mut atoms = Block::new();
    atoms
        .insert(
            "x",
            Array1::from_vec(vec![0.0 as F, hx as F, hx as F]).into_dyn(),
        )
        .unwrap();
    atoms
        .insert(
            "y",
            Array1::from_vec(vec![0.0 as F, hy as F, -(hy as F)]).into_dyn(),
        )
        .unwrap();
    atoms
        .insert(
            "z",
            Array1::from_vec(vec![0.0 as F, 0.0 as F, 0.0 as F]).into_dyn(),
        )
        .unwrap();
    atoms
        .insert(
            "type",
            Array1::from_vec(vec!["OW".to_string(), "HW".to_string(), "HW".to_string()]).into_dyn(),
        )
        .unwrap();
    frame.insert("atoms", atoms);

    // Bonds block: i, j, type columns
    let mut bonds = Block::new();
    bonds
        .insert("i", Array1::from_vec(vec![0 as U, 0 as U]).into_dyn())
        .unwrap();
    bonds
        .insert("j", Array1::from_vec(vec![1 as U, 2 as U]).into_dyn())
        .unwrap();
    bonds
        .insert(
            "type",
            Array1::from_vec(vec!["OW-HW".to_string(), "OW-HW".to_string()]).into_dyn(),
        )
        .unwrap();
    frame.insert("bonds", bonds);

    // Angles block: i, j (central), k, type columns
    let mut angles = Block::new();
    angles
        .insert("i", Array1::from_vec(vec![1 as U]).into_dyn())
        .unwrap();
    angles
        .insert("j", Array1::from_vec(vec![0 as U]).into_dyn())
        .unwrap();
    angles
        .insert("k", Array1::from_vec(vec![2 as U]).into_dyn())
        .unwrap();
    angles
        .insert(
            "type",
            Array1::from_vec(vec!["HW-OW-HW".to_string()]).into_dyn(),
        )
        .unwrap();
    frame.insert("angles", angles);

    frame
}
