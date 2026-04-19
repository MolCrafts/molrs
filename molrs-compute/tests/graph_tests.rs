//! Integration tests for the `Graph` DAG — diamond reuse, missing inputs,
//! node error propagation. These tests wire real Compute impls into a
//! `Graph` and assert semantic guarantees that can't be checked from any
//! single module's unit tests.

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use molrs::Frame;
use molrs::block::Block;
use molrs::frame_access::FrameAccess;
use molrs::neighbors::{LinkCell, NbListAlgo, NeighborList};
use molrs::region::simbox::SimBox;
use molrs::types::F;
use ndarray::{Array1, Array2, array};

use molrs_compute::center_of_mass::{COMResult, CenterOfMass};
use molrs_compute::cluster::{Cluster, ClusterResult};
use molrs_compute::cluster_centers::{ClusterCenters, ClusterCentersResult};
use molrs_compute::error::ComputeError;
use molrs_compute::graph::{Graph, Inputs, Store};
use molrs_compute::gyration_tensor::{GyrationTensor, GyrationTensorResult};
use molrs_compute::inertia_tensor::{InertiaTensor, InertiaTensorResult};
use molrs_compute::radius_of_gyration::{RadiusOfGyration, RgResult};
use molrs_compute::traits::Compute;

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
    frame.simbox = Some(
        SimBox::cube(
            box_len,
            array![0.0 as F, 0.0 as F, 0.0 as F],
            [true, true, true],
        )
        .unwrap(),
    );
    assert_eq!(n, frame.get("atoms").unwrap().nrows().unwrap());
    frame
}

fn build_nlist(frame: &Frame, cutoff: F) -> NeighborList {
    let xs = frame.get_float("atoms", "x").unwrap();
    let ys = frame.get_float("atoms", "y").unwrap();
    let zs = frame.get_float("atoms", "z").unwrap();
    let n = xs.len();
    let mut pos = Array2::<F>::zeros((n, 3));
    for i in 0..n {
        pos[[i, 0]] = xs[i];
        pos[[i, 1]] = ys[i];
        pos[[i, 2]] = zs[i];
    }
    let simbox = frame.simbox.as_ref().unwrap();
    let mut lc = LinkCell::new().cutoff(cutoff);
    lc.build(pos.view(), simbox);
    lc.query().clone()
}

// ---------------------------------------------------------------------------
// Counting wrapper — increments an atomic counter every time `compute` runs.
// ---------------------------------------------------------------------------

struct CountingCompute<C> {
    inner: C,
    counter: Arc<AtomicUsize>,
}

impl<C: Compute> Compute for CountingCompute<C> {
    type Args<'a> = C::Args<'a>;
    type Output = C::Output;

    fn compute<'a, FA: FrameAccess + Sync + 'a>(
        &self,
        frames: &[&'a FA],
        args: Self::Args<'a>,
    ) -> Result<Self::Output, ComputeError> {
        self.counter.fetch_add(1, Ordering::SeqCst);
        self.inner.compute(frames, args)
    }
}

// ---------------------------------------------------------------------------
// Diamond reuse — Rg + Inertia + Gyration share Cluster + COM + Centers
// ---------------------------------------------------------------------------

#[test]
fn diamond_reuse_runs_shared_nodes_exactly_once() {
    let pos = [
        [1.0, 1.0, 1.0],
        [1.5, 1.0, 1.0],
        [1.0, 1.5, 1.0],
        [5.0, 5.0, 5.0],
        [5.5, 5.0, 5.0],
        [5.0, 5.5, 5.0],
    ];
    let frame = make_frame(&pos, 10.0);
    let nlist = build_nlist(&frame, 1.0);

    let cluster_counter = Arc::new(AtomicUsize::new(0));
    let com_counter = Arc::new(AtomicUsize::new(0));
    let centers_counter = Arc::new(AtomicUsize::new(0));
    let rg_counter = Arc::new(AtomicUsize::new(0));
    let inertia_counter = Arc::new(AtomicUsize::new(0));
    let gyr_counter = Arc::new(AtomicUsize::new(0));

    let mut g = Graph::<Frame>::new();
    let nl_in = g.input::<Vec<NeighborList>>();

    let clusters = g.add(
        CountingCompute {
            inner: Cluster::new(1),
            counter: Arc::clone(&cluster_counter),
        },
        move |s: &Store| s.get(nl_in),
    );

    let com = g.add(
        CountingCompute {
            inner: CenterOfMass::new(),
            counter: Arc::clone(&com_counter),
        },
        move |s: &Store| s.get(clusters),
    );

    let centers = g.add(
        CountingCompute {
            inner: ClusterCenters::new(),
            counter: Arc::clone(&centers_counter),
        },
        move |s: &Store| s.get(clusters),
    );

    let rg_slot = g.add(
        CountingCompute {
            inner: RadiusOfGyration::new(),
            counter: Arc::clone(&rg_counter),
        },
        move |s: &Store| (s.get(clusters), s.get(com)),
    );

    let inertia_slot = g.add(
        CountingCompute {
            inner: InertiaTensor::new(),
            counter: Arc::clone(&inertia_counter),
        },
        move |s: &Store| (s.get(clusters), s.get(com)),
    );

    let gyr_slot = g.add(
        CountingCompute {
            inner: GyrationTensor::new(),
            counter: Arc::clone(&gyr_counter),
        },
        move |s: &Store| (s.get(clusters), s.get(centers)),
    );

    let store = g
        .run(&[&frame], Inputs::new().with(nl_in, vec![nlist]))
        .expect("graph run");

    // Shared intermediates: exactly once each, regardless of how many
    // downstream consumers capture them.
    assert_eq!(
        cluster_counter.load(Ordering::SeqCst),
        1,
        "Cluster should run once"
    );
    assert_eq!(com_counter.load(Ordering::SeqCst), 1, "COM should run once");
    assert_eq!(
        centers_counter.load(Ordering::SeqCst),
        1,
        "ClusterCenters should run once"
    );

    // Leaf nodes run once each too.
    assert_eq!(rg_counter.load(Ordering::SeqCst), 1);
    assert_eq!(inertia_counter.load(Ordering::SeqCst), 1);
    assert_eq!(gyr_counter.load(Ordering::SeqCst), 1);

    // Outputs are present and structurally non-empty.
    let rg: &Vec<RgResult> = store.get(rg_slot);
    let inertia: &Vec<InertiaTensorResult> = store.get(inertia_slot);
    let gyr: &Vec<GyrationTensorResult> = store.get(gyr_slot);
    assert_eq!(rg.len(), 1);
    assert_eq!(inertia.len(), 1);
    assert_eq!(gyr.len(), 1);
    assert_eq!(rg[0].0.len(), 2);
    assert_eq!(inertia[0].0.len(), 2);
    assert_eq!(gyr[0].0.len(), 2);
}

// ---------------------------------------------------------------------------
// Error paths
// ---------------------------------------------------------------------------

#[test]
fn missing_input_is_reported() {
    let pos = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]];
    let frame = make_frame(&pos, 5.0);

    let mut g = Graph::<Frame>::new();
    let _nl_in = g.input::<Vec<NeighborList>>();
    // Run without binding the input at all.
    let result = g.run(&[&frame], Inputs::new());
    match result {
        Err(ComputeError::MissingInput { .. }) => {}
        other => panic!("expected MissingInput, got {other:?}"),
    }
}

#[test]
fn node_error_is_wrapped_with_node_id() {
    // Feed Cluster a mismatched nlist count (0 nlists for 1 frame) — expects
    // DimensionMismatch inside Cluster::compute, which Graph should wrap
    // in ComputeError::Node { node_id, source: Box<DimensionMismatch> }.
    let pos = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]];
    let frame = make_frame(&pos, 5.0);

    let mut g = Graph::<Frame>::new();
    let nl_in = g.input::<Vec<NeighborList>>();
    let _clusters = g.add(Cluster::new(1), move |s: &Store| s.get(nl_in));

    let err = g
        .run(
            &[&frame],
            Inputs::new().with(nl_in, Vec::<NeighborList>::new()),
        )
        .unwrap_err();
    match err {
        ComputeError::Node { source, .. } => match *source {
            ComputeError::DimensionMismatch { .. } => {}
            other => panic!("expected wrapped DimensionMismatch, got {other:?}"),
        },
        other => panic!("expected Node wrapping, got {other:?}"),
    }
}

#[test]
fn single_node_graph_matches_direct_call() {
    // Graph::run with one Cluster node should produce an identical result to
    // calling Cluster::compute directly.
    let pos = [[0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [5.0, 5.0, 5.0]];
    let frame = make_frame(&pos, 10.0);
    let nlist = build_nlist(&frame, 1.0);

    let direct = Cluster::new(1)
        .compute(&[&frame], &vec![nlist.clone()])
        .unwrap();

    let mut g = Graph::<Frame>::new();
    let nl_in = g.input::<Vec<NeighborList>>();
    let c_slot = g.add(Cluster::new(1), move |s: &Store| s.get(nl_in));
    let store = g
        .run(&[&frame], Inputs::new().with(nl_in, vec![nlist]))
        .unwrap();

    let via_graph: &Vec<ClusterResult> = store.get(c_slot);
    assert_eq!(direct.len(), via_graph.len());
    assert_eq!(direct[0].num_clusters, via_graph[0].num_clusters);
    assert_eq!(direct[0].cluster_sizes, via_graph[0].cluster_sizes);
}

// Keep dead-code warning silent if future migrations drop these imports.
#[allow(dead_code)]
fn _assert_types_used(
    _rg: RgResult,
    _inertia: InertiaTensorResult,
    _gyr: GyrationTensorResult,
    _com: COMResult,
    _cc: ClusterCentersResult,
) {
}
