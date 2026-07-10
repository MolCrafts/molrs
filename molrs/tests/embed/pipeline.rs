//! End-to-end tests for the `generate_3d` embedding pipeline.
//!
//! Covers: seeded reproducibility and the empty-input error. The retired
//! FragmentRules / MMFF94->UFF-fallback / algorithm-selector cases were removed
//! (superseded by ETKDG); current coordinate-generation coverage lives in
//! `etkdg.rs`, `distgeom.rs`, and `torsions.rs`.

use molrs::conformer::{Conformer, ConformerOptions};
use molrs::{AtomId, Atomistic, PropValue};

fn bond(g: &mut Atomistic, a: AtomId, b: AtomId, order: f64) {
    let bid = g.add_bond(a, b).expect("add bond");
    g.set_bond_prop(bid, "order", PropValue::F64(order))
        .expect("set order");
}

fn coords_of(g: &Atomistic) -> Vec<[f64; 3]> {
    g.atoms()
        .map(|(_, atom)| {
            [
                atom.get_f64("x").unwrap_or(f64::NAN),
                atom.get_f64("y").unwrap_or(f64::NAN),
                atom.get_f64("z").unwrap_or(f64::NAN),
            ]
        })
        .collect()
}

#[test]
fn test_generate_3d_seed_reproducible() {
    // n-Butane skeleton: C-C-C-C
    let mut g = Atomistic::new();
    let a = g.add_atom_bare("C");
    let b = g.add_atom_bare("C");
    let c = g.add_atom_bare("C");
    let d = g.add_atom_bare("C");
    bond(&mut g, a, b, 1.0);
    bond(&mut g, b, c, 1.0);
    bond(&mut g, c, d, 1.0);

    let opts = ConformerOptions {
        add_hydrogens: false,
        rng_seed: Some(7),
        ..Default::default()
    };

    let (g1, _) = Conformer::new(opts.clone())
        .generate(&g)
        .expect("first embed");
    let (g2, _) = Conformer::new(opts.clone())
        .generate(&g)
        .expect("second embed");

    let c1 = coords_of(&g1);
    let c2 = coords_of(&g2);
    assert_eq!(c1.len(), c2.len());
    for i in 0..c1.len() {
        let dx = (c1[i][0] - c2[i][0]).abs();
        let dy = (c1[i][1] - c2[i][1]).abs();
        let dz = (c1[i][2] - c2[i][2]).abs();
        assert!(
            dx < 1e-12 && dy < 1e-12 && dz < 1e-12,
            "seeded runs should be deterministic"
        );
    }
}

#[test]
fn test_generate_3d_empty_molecule_returns_error() {
    let g = Atomistic::new();
    let err = Conformer::new(ConformerOptions::default().clone())
        .generate(&g)
        .expect_err("empty must error");
    assert!(
        err.to_string().contains("empty molecule"),
        "error should explain empty input"
    );
}
