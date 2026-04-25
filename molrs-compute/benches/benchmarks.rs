mod center_of_mass;
mod cluster;
mod cluster_centers;
mod graph;
mod gyration_tensor;
mod helpers;
mod inertia_tensor;
mod msd;
mod radius_of_gyration;
mod rdf;

use criterion::criterion_main;

criterion_main!(
    rdf::benches,
    msd::benches,
    cluster::benches,
    cluster_centers::benches,
    center_of_mass::benches,
    gyration_tensor::benches,
    inertia_tensor::benches,
    radius_of_gyration::benches,
    graph::benches,
);
