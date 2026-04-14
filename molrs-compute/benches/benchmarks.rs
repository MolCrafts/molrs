mod cluster;
mod helpers;
mod msd;
mod rdf;
mod tensors;

use criterion::criterion_main;

criterion_main!(
    rdf::benches,
    msd::benches,
    cluster::benches,
    tensors::benches,
);
