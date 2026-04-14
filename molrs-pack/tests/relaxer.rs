//! Tests for the per-target relaxer system and TorsionMcRelaxer.
//! The 13 unit tests inside relaxer.rs (mod tests) are preserved; these are
//! integration tests exercising the public API.

use molrs::molgraph::{Atom, MolGraph};
use molrs_pack::{
    F, Hook, InsideBoxRestraint, InsideSphereRestraint, Molpack, NullHandler, Target, TorsionMcHook,
};

// ── helpers ────────────────────────────────────────────────────────────────

/// Build a linear chain MolGraph + zigzag coords with tetrahedral angles.
fn chain(n: usize, bond_length: F) -> (MolGraph, Vec<[F; 3]>, Vec<F>) {
    let mut g = MolGraph::new();
    let mut ids = Vec::new();
    for _ in 0..n {
        ids.push(g.add_atom(Atom::new()));
    }
    for i in 0..n - 1 {
        let _ = g.add_bond(ids[i], ids[i + 1]);
    }

    let theta = 109.5 * std::f64::consts::PI as F / 180.0;
    let dx = bond_length * (-theta.cos());
    let dz = bond_length * theta.sin();

    let mut coords = Vec::with_capacity(n);
    coords.push([0.0, 0.0, 0.0]);
    for i in 1..n {
        let prev = coords[i - 1];
        let sign: F = if i % 2 == 0 { 1.0 } else { -1.0 };
        coords.push([prev[0] + dx, 0.0, prev[2] + sign * dz]);
    }

    let radii = vec![1.0; n];
    (g, coords, radii)
}

// ── bond detection ─────────────────────────────────────────────────────────

#[test]
fn detects_rotatable_bonds() {
    let (graph, _, _) = chain(10, 1.54);
    let hook = TorsionMcHook::new(&graph);
    assert_eq!(hook.bonds.len(), 7, "10-atom chain has 7 rotatable bonds");
}

#[test]
fn short_chain_fewer_bonds() {
    let (graph, _, _) = chain(3, 1.5);
    let hook = TorsionMcHook::new(&graph);
    // 3-atom chain: only terminal bonds → 0 rotatable
    assert_eq!(hook.bonds.len(), 0);
}

#[test]
fn four_atom_chain_has_one_bond() {
    let (graph, _, _) = chain(4, 1.5);
    let hook = TorsionMcHook::new(&graph);
    assert_eq!(hook.bonds.len(), 1);
}

// ── hook runner ────────────────────────────────────────────────────────────

#[test]
fn runner_modifies_coords() {
    let (graph, coords, _) = chain(5, 1.5);
    let hook = TorsionMcHook::new(&graph)
        .with_temperature(10.0)
        .with_steps(50);

    let mut runner = hook.build(&coords);
    let mut rng = rand::rng();
    let result = runner.on_iter(&coords, 1000.0, &mut |_| 999.0, &mut rng);

    assert!(result.is_some(), "runner should accept some moves");
    let new_coords = result.unwrap();
    assert_eq!(new_coords.len(), coords.len());

    let changed = new_coords.iter().zip(coords.iter()).any(|(a, b)| {
        let d = ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt();
        d > 1e-6
    });
    assert!(changed, "zigzag coords must change after torsion MC");
}

#[test]
fn runner_acceptance_rate_between_0_and_1() {
    let (graph, coords, _) = chain(8, 1.5);
    let hook = TorsionMcHook::new(&graph)
        .with_temperature(1.0)
        .with_steps(100);

    let mut runner = hook.build(&coords);
    let mut rng = rand::rng();
    let _ = runner.on_iter(&coords, 100.0, &mut |_| 50.0, &mut rng);

    let rate = runner.acceptance_rate();
    assert!(
        (0.0..=1.0).contains(&rate),
        "rate should be in [0,1]: {rate}"
    );
}

// ── target integration ─────────────────────────────────────────────────────

#[test]
fn target_with_hook_requires_count_1() {
    let (graph, coords, radii) = chain(5, 1.5);
    let hook = TorsionMcHook::new(&graph);
    let _target = Target::from_coords(&coords, &radii, 1).with_relaxer(hook);
}

#[test]
#[should_panic(expected = "relaxers require count == 1")]
fn target_with_hook_panics_for_count_gt_1() {
    let (graph, coords, radii) = chain(5, 1.5);
    let hook = TorsionMcHook::new(&graph);
    let _target = Target::from_coords(&coords, &radii, 2).with_relaxer(hook);
}

// ── packing with hooks ─────────────────────────────────────────────────────

#[test]
fn pack_with_torsion_hook_in_box() {
    let n = 10;
    let (graph, coords, radii) = chain(n, 1.5);

    let hook = TorsionMcHook::new(&graph)
        .with_temperature(1.0)
        .with_steps(10)
        .with_max_delta(std::f64::consts::PI as F / 4.0);

    let target = Target::from_coords(&coords, &radii, 1)
        .with_name("chain")
        .with_restraint(InsideBoxRestraint::new([0.0, 0.0, 0.0], [20.0, 20.0, 20.0]))
        .with_relaxer(hook);

    let result = Molpack::new()
        .add_handler(NullHandler)
        .pack(&[target], 10, Some(42))
        .expect("pack should not fail");

    assert_eq!(result.natoms(), n);
}

#[test]
fn pack_with_torsion_hook_in_sphere() {
    let n = 20;
    let (graph, coords, radii) = chain(n, 1.5);

    let hook = TorsionMcHook::new(&graph)
        .with_temperature(0.5)
        .with_steps(20);

    let target = Target::from_coords(&coords, &radii, 1)
        .with_name("polymer")
        .with_restraint(InsideSphereRestraint::new([0.0, 0.0, 0.0], 15.0))
        .with_relaxer(hook);

    let result = Molpack::new()
        .add_handler(NullHandler)
        .pack(&[target], 15, Some(123))
        .expect("pack should not fail");

    assert_eq!(result.natoms(), n);

    for pos in result.positions() {
        let r = (pos[0] * pos[0] + pos[1] * pos[1] + pos[2] * pos[2]).sqrt();
        assert!(r < 17.0, "atom at {pos:?} too far (r={r:.2})");
    }
}

#[test]
fn pack_hook_with_regular_target() {
    let n = 5;
    let (graph, chain_coords, chain_radii) = chain(n, 1.5);

    let restraint = InsideBoxRestraint::new([0.0, 0.0, 0.0], [30.0, 30.0, 30.0]);

    let chain_target = Target::from_coords(&chain_coords, &chain_radii, 1)
        .with_name("chain")
        .with_restraint(restraint)
        .with_relaxer(TorsionMcHook::new(&graph).with_steps(5));

    let point_target = Target::from_coords(&[[0.0, 0.0, 0.0]], &[1.0], 3)
        .with_name("points")
        .with_restraint(restraint);

    let result = Molpack::new()
        .add_handler(NullHandler)
        .pack(&[chain_target, point_target], 10, Some(99))
        .expect("pack should not fail");

    assert_eq!(result.natoms(), 8); // 5 chain + 3 point
}

// ── builder options ────────────────────────────────────────────────────────

#[test]
fn hook_builder_methods() {
    let (graph, _, _) = chain(10, 1.5);
    let hook = TorsionMcHook::new(&graph)
        .with_temperature(2.0)
        .with_steps(20)
        .with_max_delta(1.0)
        .with_self_avoidance(0.5);

    assert!((hook.temperature - 2.0).abs() < 1e-6);
    assert_eq!(hook.steps, 20);
    assert!((hook.max_delta - 1.0).abs() < 1e-6);
    assert!((hook.self_avoidance_radius - 0.5).abs() < 1e-6);
}

// ── StepInfo with hook acceptance ──────────────────────────────────────────

#[test]
fn step_info_hook_acceptance() {
    use molrs_pack::StepInfo;
    use molrs_pack::handler::PhaseInfo;

    let info = StepInfo {
        loop_idx: 0,
        max_loops: 10,
        phase: PhaseInfo {
            phase: 0,
            total_phases: 2,
            molecule_type: Some(0),
        },
        fdist: 1.0,
        frest: 0.5,
        improvement_pct: 5.0,
        radscale: 1.1,
        precision: 0.01,
        hook_acceptance: vec![(0, 0.35)],
    };

    assert_eq!(info.hook_acceptance.len(), 1);
    assert!((info.hook_acceptance[0].1 - 0.35).abs() < 1e-10);
}
