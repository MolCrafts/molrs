//! Graph benches: diamond reuse + single-node overhead.
//!
//! 1. **Diamond reuse**: Rg + Inertia + GyrationTensor share Cluster + COM +
//!    ClusterCenters. A `Graph` must beat the manual per-compute cascade by
//!    at least 30 % (gate ≤ 0.7×) because the shared intermediates run once
//!    instead of three times.
//!
//! 2. **Single-node overhead**: `Graph::run` with one RDF node must stay
//!    within 5 % of a direct `RDF::compute` call (gate ≤ 1.05×).
//!
//! Benches only measure the hot loop; setup (fixtures, nlist build) lives
//! outside the timed section.

use criterion::{Criterion, criterion_group};
use molrs::frame::Frame;
use molrs::neighbors::NeighborList;
use molrs_compute::center_of_mass::CenterOfMass;
use molrs_compute::cluster::Cluster;
use molrs_compute::cluster_centers::ClusterCenters;
use molrs_compute::graph::{Graph, Inputs, Store};
use molrs_compute::gyration_tensor::GyrationTensor;
use molrs_compute::inertia_tensor::InertiaTensor;
use molrs_compute::radius_of_gyration::RadiusOfGyration;
use molrs_compute::rdf::RDF;
use molrs_compute::traits::Compute;

use crate::helpers::{self, Fixture};

const N_ATOMS: usize = 2_000;
const N_BINS: usize = 100;

fn bench_diamond_reuse(c: &mut Criterion) {
    let mut group = c.benchmark_group("graph/diamond");
    let Fixture { frame, nlist, .. } = helpers::fixture(N_ATOMS, 42);
    let frames = [&frame];
    let nlists_owned = vec![nlist];

    // Baseline: manual cascade running Cluster / COM / ClusterCenters each
    // once, but re-running Cluster inside each of {Rg, Inertia, Gyr} makes
    // the comparison fair to the "diamond" scenario where users wire the
    // same dependency three times.
    group.bench_function("manual", |b| {
        b.iter(|| {
            let clusters_rg = Cluster::new(2).compute(&frames, &nlists_owned).unwrap();
            let com_rg = CenterOfMass::new().compute(&frames, &clusters_rg).unwrap();
            let rg = RadiusOfGyration::new()
                .compute(&frames, (&clusters_rg, &com_rg))
                .unwrap();

            let clusters_i = Cluster::new(2).compute(&frames, &nlists_owned).unwrap();
            let com_i = CenterOfMass::new().compute(&frames, &clusters_i).unwrap();
            let inertia = InertiaTensor::new()
                .compute(&frames, (&clusters_i, &com_i))
                .unwrap();

            let clusters_g = Cluster::new(2).compute(&frames, &nlists_owned).unwrap();
            let centers = ClusterCenters::new().compute(&frames, &clusters_g).unwrap();
            let gyr = GyrationTensor::new()
                .compute(&frames, (&clusters_g, &centers))
                .unwrap();

            std::hint::black_box((rg, inertia, gyr));
        });
    });

    // Graph path: Cluster / COM / ClusterCenters shared across all three.
    group.bench_function("graph", |b| {
        b.iter(|| {
            let mut g = Graph::<Frame>::new();
            let nl_in = g.input::<Vec<NeighborList>>();

            let clusters = g.add(Cluster::new(2), move |s: &Store| s.get(nl_in));
            let com = g.add(CenterOfMass::new(), move |s: &Store| s.get(clusters));
            let centers = g.add(ClusterCenters::new(), move |s: &Store| s.get(clusters));

            let rg = g.add(RadiusOfGyration::new(), move |s: &Store| {
                (s.get(clusters), s.get(com))
            });
            let inertia = g.add(InertiaTensor::new(), move |s: &Store| {
                (s.get(clusters), s.get(com))
            });
            let gyr = g.add(GyrationTensor::new(), move |s: &Store| {
                (s.get(clusters), s.get(centers))
            });

            let store = g
                .run(&frames, Inputs::new().with(nl_in, nlists_owned.clone()))
                .unwrap();
            std::hint::black_box((
                store.get(rg).clone(),
                store.get(inertia).clone(),
                store.get(gyr).clone(),
            ));
        });
    });

    group.finish();
}

fn bench_single_node_overhead(c: &mut Criterion) {
    let mut group = c.benchmark_group("graph/single_node_rdf");
    let Fixture { frame, nlist, .. } = helpers::fixture(N_ATOMS, 7);
    let frames = [&frame];
    let nlists_owned = vec![nlist];
    let rdf = RDF::new(N_BINS, helpers::CUTOFF, 0.0).unwrap();

    group.bench_function("direct", |b| {
        b.iter(|| {
            let r = rdf.compute(&frames, &nlists_owned).unwrap();
            std::hint::black_box(r);
        });
    });

    group.bench_function("graph", |b| {
        b.iter(|| {
            let mut g = Graph::<Frame>::new();
            let nl_in = g.input::<Vec<NeighborList>>();
            let rdf_slot = g.add(rdf.clone(), move |s: &Store| s.get(nl_in));
            let store = g
                .run(&frames, Inputs::new().with(nl_in, nlists_owned.clone()))
                .unwrap();
            std::hint::black_box(store.get(rdf_slot).clone());
        });
    });

    group.finish();
}

criterion_group!(benches, bench_diamond_reuse, bench_single_node_overhead);
