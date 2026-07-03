//! End-to-end center-of-mass integration test.
//!
//! Exercises the public two-stage chain `Cluster -> CenterOfMass` against
//! known geometry: symmetric point clouds whose COM is their geometric centre.
//! This is the real-world usage pattern (cluster first, then reduce per
//! cluster) wired by hand against the public `Compute` API.

use molrs::Frame;
use molrs::spatial::neighbors::{LinkCell, NbListAlgo, NeighborList};
use molrs::spatial::region::simbox::SimBox;
use molrs::store::block::Block;
use molrs::types::F;
use ndarray::{Array1, Array2, array};

use molrs::compute::cluster::Cluster;
use molrs::compute::shape::center_of_mass::CenterOfMass;
use molrs::compute::traits::Compute;

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------

fn make_frame(positions: &[[F; 3]], box_len: F) -> Frame {
    let n = positions.len();
    let x = Array1::from_iter(positions.iter().map(|p| p[0]));
    let y = Array1::from_iter(positions.iter().map(|p| p[1]));
    let z = Array1::from_iter(positions.iter().map(|p| p[2]));
    let mut block = Block::new();
    block.insert("x", x.into_dyn()).unwrap();
    block.insert("y", y.into_dyn()).unwrap();
    block.insert("z", z.into_dyn()).unwrap();
    let mut frame = Frame::new();
    frame.insert("atoms", block);
    frame.simbox =
        Some(SimBox::cube(box_len, array![0.0 as F, 0.0, 0.0], [true, true, true]).unwrap());
    assert_eq!(n, frame.get("atoms").unwrap().nrows().unwrap());
    frame
}

fn build_nlist(frame: &Frame, cutoff: F) -> NeighborList {
    let atoms = frame.get("atoms").unwrap();
    let xs = atoms.get_float("x").unwrap();
    let ys = atoms.get_float("y").unwrap();
    let zs = atoms.get_float("z").unwrap();
    let n = xs.len();
    let mut pos = Array2::<F>::zeros((n, 3));
    for i in 0..n {
        pos[[i, 0]] = xs[[i]];
        pos[[i, 1]] = ys[[i]];
        pos[[i, 2]] = zs[[i]];
    }
    let simbox = frame.simbox.as_ref().unwrap();
    let mut lc = LinkCell::new().cutoff(cutoff);
    lc.build(pos.view(), simbox);
    lc.query().clone()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[test]
fn single_symmetric_cluster_com_is_geometric_center() {
    // Six points symmetric about (2, 2, 2) within cutoff 1.5 — one cluster.
    let c = [2.0, 2.0, 2.0];
    let pos = [
        [c[0] - 0.5, c[1], c[2]],
        [c[0] + 0.5, c[1], c[2]],
        [c[0], c[1] - 0.5, c[2]],
        [c[0], c[1] + 0.5, c[2]],
        [c[0], c[1], c[2] - 0.5],
        [c[0], c[1], c[2] + 0.5],
    ];
    let frame = make_frame(&pos, 20.0);
    let nlist = build_nlist(&frame, 1.5);

    let clusters = Cluster::new(1).compute(&[&frame], &vec![nlist]).unwrap();
    assert_eq!(clusters[0].num_clusters, 1, "all points should merge");

    let com = CenterOfMass::new().compute(&[&frame], &clusters).unwrap();
    assert_eq!(com.len(), 1);
    assert_eq!(com[0].centers_of_mass.len(), 1);

    let got = com[0].centers_of_mass[0];
    for axis in 0..3 {
        assert!(
            (got[axis] - c[axis]).abs() < 1e-9,
            "COM axis {axis} = {}, expected {}",
            got[axis],
            c[axis]
        );
    }
    // Uniform unit masses → total mass = particle count.
    assert!((com[0].cluster_masses[0] - 6.0).abs() < 1e-12);
}

#[test]
fn mass_weighting_shifts_com_toward_heavy_particle() {
    // Two particles 2 Å apart along x; with mass ratio 3:1 the COM sits at
    // x = (3·0 + 1·2)/4 = 0.5 from the heavy particle.
    let pos = [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]];
    let frame = make_frame(&pos, 20.0);
    let nlist = build_nlist(&frame, 2.5);

    let clusters = Cluster::new(1).compute(&[&frame], &vec![nlist]).unwrap();
    assert_eq!(clusters[0].num_clusters, 1);

    let com = CenterOfMass::new()
        .with_masses(&[3.0, 1.0])
        .compute(&[&frame], &clusters)
        .unwrap();

    let x = com[0].centers_of_mass[0][0];
    assert!(
        (x - 0.5).abs() < 1e-9,
        "mass-weighted COM x = {x}, expected 0.5"
    );
    assert!((com[0].cluster_masses[0] - 4.0).abs() < 1e-12);
}

#[test]
fn two_separated_clusters_resolve_independently() {
    // Two tight pairs far apart → two clusters, each COM at its pair midpoint.
    let pos = [
        [1.0, 1.0, 1.0],
        [1.0, 1.0, 1.4],
        [9.0, 9.0, 9.0],
        [9.0, 9.0, 9.4],
    ];
    let frame = make_frame(&pos, 20.0);
    let nlist = build_nlist(&frame, 1.0);

    let clusters = Cluster::new(1).compute(&[&frame], &vec![nlist]).unwrap();
    assert_eq!(clusters[0].num_clusters, 2);

    let com = CenterOfMass::new().compute(&[&frame], &clusters).unwrap();
    assert_eq!(com[0].centers_of_mass.len(), 2);

    // Each cluster's COM z is the midpoint of its pair (1.2 or 9.2).
    let mut zs: Vec<F> = com[0].centers_of_mass.iter().map(|c| c[2]).collect();
    zs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    assert!((zs[0] - 1.2).abs() < 1e-9, "cluster z = {}", zs[0]);
    assert!((zs[1] - 9.2).abs() < 1e-9, "cluster z = {}", zs[1]);
}
