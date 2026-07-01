//! Integration tests for `compute::hbond` — detection, network, lifetimes.

use molrs::Frame;
use molrs::compute::hbond::{
    DistKind, HBondCriterion, HBonds, hbond_components, hbond_lifetimes, presence_from_hbonds,
};
use molrs::compute::traits::Compute;
use molrs::spatial::region::simbox::SimBox;
use molrs::store::block::Block;
use molrs::types::F;
use ndarray::{Array1, array};

/// Build a frame from `[x,y,z]` rows, optionally periodic with cube length `l`.
fn frame_of(coords: &[[F; 3]], cube: Option<F>) -> Frame {
    let n = coords.len();
    let mut block = Block::new();
    let xs = Array1::from_iter(coords.iter().map(|c| c[0]));
    let ys = Array1::from_iter(coords.iter().map(|c| c[1]));
    let zs = Array1::from_iter(coords.iter().map(|c| c[2]));
    block.insert("x", xs.into_dyn()).unwrap();
    block.insert("y", ys.into_dyn()).unwrap();
    block.insert("z", zs.into_dyn()).unwrap();
    let _ = n;
    let mut frame = Frame::new();
    frame.insert("atoms", block);
    if let Some(l) = cube {
        frame.simbox =
            Some(SimBox::cube(l, array![0.0 as F, 0.0, 0.0], [true, true, true]).unwrap());
    }
    frame
}

// ── ac-001: detection of a known dimer + cutoff boundaries ──────────────────

#[test]
fn water_dimer_detected_with_exact_geometry() {
    // Collinear D(O)–H···A(O): O at x=0, H at x=0.96, acceptor O at x=2.8.
    let coords = [[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [2.8, 0.0, 0.0]];
    let frame = frame_of(&coords, None);
    let hb = HBonds::new(vec![(0, 1)], vec![2], HBondCriterion::default());
    let res = hb.compute(&[&frame], ()).unwrap();
    assert_eq!(res.counts, vec![1]);
    let b = res.per_frame[0][0];
    assert_eq!((b.donor, b.hydrogen, b.acceptor), (0, 1, 2));
    assert!((b.distance - 2.8).abs() < 1e-9, "D···A = {}", b.distance);
    assert!((b.angle - 180.0).abs() < 1e-9, "angle = {}", b.angle);
}

#[test]
fn bond_dropped_when_angle_below_cutoff() {
    // Acceptor bent off-axis at H: angle at H ≈ 90° < 150° → no bond,
    // though D···A = sqrt(0.96² + 1.9²) ≈ 2.13 is inside the distance cutoff.
    let coords = [[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [0.96, 1.9, 0.0]];
    let frame = frame_of(&coords, None);
    let hb = HBonds::new(vec![(0, 1)], vec![2], HBondCriterion::default());
    let res = hb.compute(&[&frame], ()).unwrap();
    assert_eq!(res.counts, vec![0]);
}

#[test]
fn bond_dropped_when_distance_beyond_cutoff() {
    // Collinear but D···A = 4.0 > 3.5 → no candidate.
    let coords = [[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [4.0, 0.0, 0.0]];
    let frame = frame_of(&coords, None);
    let hb = HBonds::new(vec![(0, 1)], vec![2], HBondCriterion::default());
    let res = hb.compute(&[&frame], ()).unwrap();
    assert_eq!(res.counts, vec![0]);
}

// ── ac-002: minimum image under PBC ─────────────────────────────────────────

#[test]
fn detection_uses_minimum_image() {
    // Box L=10 on x. Donor O at x=9.5, H at x=0.46 (= 9.5 + 0.96 wrapped),
    // acceptor at x=1.9. Min-image D···A = 2.4; raw separation 7.6 > cutoff.
    let l = 10.0;
    let coords = [[9.5, 5.0, 5.0], [0.46, 5.0, 5.0], [1.9, 5.0, 5.0]];
    let frame = frame_of(&coords, Some(l));
    let hb = HBonds::new(vec![(0, 1)], vec![2], HBondCriterion::default());
    let res = hb.compute(&[&frame], ()).unwrap();
    assert_eq!(res.counts, vec![1], "min-image bond should be detected");
    let b = res.per_frame[0][0];
    assert!(
        (b.distance - 2.4).abs() < 1e-9,
        "min-image D···A = {}",
        b.distance
    );
    assert!((b.angle - 180.0).abs() < 1e-9, "angle = {}", b.angle);

    // Without PBC the same coordinates are too far apart → no bond.
    let free = frame_of(&coords, None);
    let res2 = hb.compute(&[&free], ()).unwrap();
    assert_eq!(res2.counts, vec![0]);
}

// ── ac-003: network components via core::Topology ───────────────────────────

#[test]
fn chain_forms_one_component_and_splits_when_broken() {
    // 4 molecules in a chain: 0-1-2-3.
    let edges = [(0, 1), (1, 2), (2, 3)];
    let net = hbond_components(4, &edges);
    assert_eq!(net.num_components, 1);
    assert_eq!(net.component_sizes, vec![4]);

    // Break the central bond (1,2) → two components of size 2.
    let broken = [(0, 1), (2, 3)];
    let net2 = hbond_components(4, &broken);
    assert_eq!(net2.num_components, 2);
    assert_eq!(net2.component_sizes, vec![2, 2]);
}

#[test]
fn isolated_nodes_count_as_singletons() {
    // 5 nodes, one bond 0-1; nodes 2,3,4 isolated.
    let net = hbond_components(5, &[(0, 1)]);
    assert_eq!(net.num_components, 4);
    assert_eq!(net.component_sizes, vec![2, 1, 1, 1]);
}

// ── ac-004: no petgraph in the compute hbond path ───────────────────────────

#[test]
fn hbond_source_has_no_petgraph() {
    let dir = concat!(env!("CARGO_MANIFEST_DIR"), "/src/compute/hbond");
    for file in [
        "mod.rs",
        "criterion.rs",
        "detect.rs",
        "network.rs",
        "lifetime.rs",
    ] {
        let path = format!("{dir}/{file}");
        let src = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {path}: {e}"));
        // Forbid real usage (imports / path refs); the word may appear in prose
        // documenting *why* petgraph is excluded.
        assert!(
            !src.contains("use petgraph") && !src.contains("petgraph::"),
            "{file} must not USE petgraph (it is smiles-only)"
        );
    }
}

// ── ac-005: continuous vs intermittent lifetimes ────────────────────────────

#[test]
fn always_on_bond_gives_unit_tcf() {
    let present = vec![vec![true; 6]];
    let r = hbond_lifetimes(&present, 1.0, 5).unwrap();
    for tau in 0..=5 {
        assert!((r.continuous[tau] - 1.0).abs() < 1e-12);
        assert!((r.intermittent[tau] - 1.0).abs() < 1e-12);
    }
    assert!((r.tau_continuous - r.tau_intermittent).abs() < 1e-12);
}

#[test]
fn intermittent_dominates_continuous_with_gaps() {
    // A bond that breaks and reforms: continuous must decay no slower than,
    // and the lifetime must order τ_intermittent ≥ τ_continuous.
    let present = vec![vec![true, true, false, true, true, false, true, true]];
    let r = hbond_lifetimes(&present, 0.5, 7).unwrap();
    for tau in 0..r.continuous.len() {
        assert!(
            r.intermittent[tau] >= r.continuous[tau] - 1e-12,
            "intermittent < continuous at tau={tau}"
        );
    }
    // Continuous is non-increasing.
    for tau in 1..r.continuous.len() {
        assert!(r.continuous[tau] <= r.continuous[tau - 1] + 1e-12);
    }
    assert!(r.tau_intermittent >= r.tau_continuous - 1e-12);
    assert!(
        r.tau_intermittent > r.tau_continuous,
        "gaps should separate the lifetimes"
    );
}

#[test]
fn lifetime_adapter_from_detection() {
    // Two frames where the same dimer bond is present, then absent.
    let on = [[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [2.8, 0.0, 0.0]];
    let off = [[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [2.8, 3.0, 0.0]]; // bent → no bond
    let f_on = frame_of(&on, None);
    let f_off = frame_of(&off, None);
    let hb = HBonds::new(vec![(0, 1)], vec![2], HBondCriterion::default());
    let res = hb.compute(&[&f_on, &f_off, &f_on, &f_on], ()).unwrap();
    let (keys, present) = presence_from_hbonds(&res);
    assert_eq!(keys, vec![(0, 2)]);
    assert_eq!(present, vec![vec![true, false, true, true]]);
    let life = hbond_lifetimes(&present, 1.0, 3).unwrap();
    assert!(life.tau_intermittent >= life.tau_continuous - 1e-12);
}

// ── ac-006: edge cases ──────────────────────────────────────────────────────

#[test]
fn empty_selections_yield_empty_result() {
    let frame = frame_of(&[[0.0, 0.0, 0.0]], None);
    let hb = HBonds::new(vec![], vec![], HBondCriterion::default());
    let res = hb.compute(&[&frame], ()).unwrap();
    assert_eq!(res.counts, vec![0]);
    assert!(res.per_frame[0].is_empty());
}

#[test]
fn self_bond_is_excluded() {
    // Acceptor list includes the donor heavy atom itself → must not self-bond.
    let coords = [[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [2.8, 0.0, 0.0]];
    let frame = frame_of(&coords, None);
    let hb = HBonds::new(vec![(0, 1)], vec![0, 2], HBondCriterion::default());
    let res = hb.compute(&[&frame], ()).unwrap();
    // Only the (0,1)->2 bond, never (0,1)->0.
    assert_eq!(res.counts, vec![1]);
    assert_eq!(res.per_frame[0][0].acceptor, 2);
}

#[test]
fn empty_frames_is_error() {
    let hb = HBonds::new(vec![(0, 1)], vec![2], HBondCriterion::default());
    let frames: Vec<&Frame> = Vec::new();
    assert!(hb.compute(&frames, ()).is_err());
}

#[test]
fn hydrogen_acceptor_distance_mode() {
    // Same collinear dimer, but gate on r(H···A) ≤ 2.0.
    let coords = [[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [2.8, 0.0, 0.0]];
    let frame = frame_of(&coords, None);
    let crit = HBondCriterion::new(2.0, DistKind::HydrogenAcceptor, 150.0);
    let hb = HBonds::new(vec![(0, 1)], vec![2], crit);
    let res = hb.compute(&[&frame], ()).unwrap();
    // r(H···A) = 2.8 - 0.96 = 1.84 ≤ 2.0, angle 180° → bond.
    assert_eq!(res.counts, vec![1]);
}
