//! End-to-end OPLS-style assembly: a hand-built typed Frame (harmonic
//! bond/angle + OPLS dihedral + lj/cut + coul/cut pairs) compiled through the
//! public `ForceField::compile` path, then evaluated, finite-difference
//! gradient-checked, and relaxed with the geometry optimizer.
//!
//! This mirrors what molpy's OPLS typifier would emit: the `pairs` block is the
//! pre-resolved, exclusion- and 1-4-scaled list, and the kernels are
//! topology-blind.

use crate::helpers::{flat_coords, numerical_forces, pairs_block, topo_block, typed_atoms_block};
use molrs::ff::ForceField;
use molrs::optimize::{LBFGS, LbfgsConfig};
use molrs::store::frame::Frame;
use molrs::types::F;

/// A 4-atom chain (butane-like backbone) with one of each OPLS term.
fn opls_chain_ff() -> ForceField {
    let mut ff = ForceField::new("opls-demo");
    ff.def_bondstyle("harmonic")
        .def_type("C-C", &[("k", 300.0), ("r0", 1.5)]);
    ff.def_anglestyle("harmonic")
        .def_type("C-C-C", &[("k", 60.0), ("theta0", 1.911)]);
    ff.def_dihedralstyle("opls").def_type(
        "C-C-C-C",
        &[("f1", 1.3), ("f2", -0.5), ("f3", 0.4), ("f4", 0.1)],
    );
    ff.def_pairstyle("lj/cut", &[("cutoff", 100.0)])
        .def_type("C", &[("epsilon", 0.05), ("sigma", 3.0)]);
    ff.def_pairstyle("coul/cut", &[("cutoff", 100.0)])
        .def_type("C", &[("qiqj", -0.05)]);
    ff
}

fn opls_chain_frame(coords: &[[F; 3]]) -> Frame {
    let mut frame = Frame::new();
    // Per-atom convention: `lj/cut` reads per-atom `type`, `coul/cut` reads
    // per-atom `charge` (the old per-pair `qiqj` is gone).
    frame.insert(
        "atoms",
        typed_atoms_block(coords, &["C", "C", "C", "C"], Some(&[0.1, -0.1, 0.1, -0.1])),
    );
    frame.insert(
        "bonds",
        topo_block(
            &[("atomi", &[0, 1, 2]), ("atomj", &[1, 2, 3])],
            &["C-C", "C-C", "C-C"],
        ),
    );
    frame.insert(
        "angles",
        topo_block(
            &[("atomi", &[0, 1]), ("atomj", &[1, 2]), ("atomk", &[2, 3])],
            &["C-C-C", "C-C-C"],
        ),
    );
    frame.insert(
        "dihedrals",
        topo_block(
            &[
                ("atomi", &[0]),
                ("atomj", &[1]),
                ("atomk", &[2]),
                ("atoml", &[3]),
            ],
            &["C-C-C-C"],
        ),
    );
    // Only non-excluded pair in a 4-atom chain: the 1-4 pair (0,3), flagged.
    frame.insert("pairs", pairs_block(&[0], &[3], &[true]));
    frame
}

/// A deliberately non-equilibrium starting geometry.
fn start_coords() -> [[F; 3]; 4] {
    [
        [0.1, 0.0, 0.0],
        [1.6, 0.1, 0.0],
        [1.5, 1.6, 0.2],
        [0.2, 1.5, 0.4],
    ]
}

/// ac-001 (ff-radians-convention): OPLS stores `theta0` in radians (GROMACS
/// native). With the kernel consuming radians directly — no double
/// `.to_radians()` in the ctor — the harmonic-angle energy at the reference
/// angle is ~0, not the ~100+ kcal/mol the old double-conversion produced
/// (1.911 rad mistakenly read as degrees → 0.0334 rad → dθ ≈ 1.878).
#[test]
fn opls_angle_energy_zero_at_reference() {
    let theta0 = 1.911_f64; // OPLS water/backbone reference, radians

    let mut ff = ForceField::new("opls-angle");
    ff.def_anglestyle("harmonic")
        .def_type("C-C-C", &[("k", 60.0), ("theta0", theta0)]);

    // Place three atoms so the i-j-k angle at vertex j is exactly theta0:
    // i and k each at distance r from j, splayed ±theta0/2 about the x axis.
    let r = 1.5_f64;
    let half = theta0 / 2.0;
    let xyz = [
        [r * half.cos(), r * half.sin(), 0.0],
        [0.0, 0.0, 0.0],
        [r * half.cos(), -r * half.sin(), 0.0],
    ];

    let mut frame = Frame::new();
    frame.insert(
        "angles",
        topo_block(
            &[("atomi", &[0]), ("atomj", &[1]), ("atomk", &[2])],
            &["C-C-C"],
        ),
    );

    let pots = ff
        .to_potentials(&frame)
        .expect("compile angle-only OPLS frame");
    let coords = flat_coords(&xyz);
    let (e, _) = pots.calc_energy_forces(&coords);
    assert!(
        e.abs() < 1e-6,
        "OPLS angle energy at the reference angle must be ~0, got {e} kcal/mol"
    );
}

#[test]
fn opls_assembly_compiles_and_force_matches_finite_difference() {
    let ff = opls_chain_ff();
    let xyz = start_coords();
    let frame = opls_chain_frame(&xyz);

    let pots = ff.to_potentials(&frame).expect("compile OPLS frame");
    // bond + angle + dihedral + lj/cut + coul/cut = 5 kernels.
    assert_eq!(pots.len(), 5);

    let coords = flat_coords(&xyz);
    let (energy, forces) = pots.calc_energy_forces(&coords);
    assert!(energy.is_finite());
    assert!(forces.iter().all(|f| f.is_finite()));

    // Whole-potential analytic force vs central finite difference.
    let fd = numerical_forces(|c| pots.calc_energy(c), &coords, 1e-6);
    for (d, (a, n)) in forces.iter().zip(&fd).enumerate() {
        assert!(
            (a - n).abs() < 1e-5,
            "component {d}: analytic {a} vs fd {n}"
        );
    }
}

#[test]
fn opls_minimize_single_and_batch() {
    let ff = opls_chain_ff();
    let xyz = start_coords();
    let frame = opls_chain_frame(&xyz);
    let pots = ff.to_potentials(&frame).expect("compile");

    let start = flat_coords(&xyz);
    let e_start = pots.calc_energy(&start);

    // Single relax.
    let mut single = start.clone();
    let report = LBFGS::new(&pots, LbfgsConfig::default())
        .run(&mut single)
        .expect("minimize");
    assert!(
        report.final_energy <= e_start + 1e-9,
        "energy must not increase: {e_start} -> {}",
        report.final_energy
    );
    assert!(report.converged, "should converge: {report:?}");

    // Homogeneous batch: block 0 identical to the single start.
    let n_atoms = 4;
    let b = 3;
    let mut batch: Vec<F> = Vec::new();
    for s in 0..b {
        for (i, &c) in start.iter().enumerate() {
            let pert = if s == 0 {
                0.0
            } else {
                0.01 * ((i % 3) as F - 1.0)
            };
            batch.push(c + pert);
        }
    }
    let reports = LBFGS::new(&pots, LbfgsConfig::default())
        .run_batch(&mut batch, n_atoms, b)
        .expect("batch");
    assert_eq!(reports.len(), b);
    assert!(
        (reports[0].final_energy - report.final_energy).abs() < 1e-9,
        "batch block 0 {} != single {}",
        reports[0].final_energy,
        report.final_energy
    );
}
