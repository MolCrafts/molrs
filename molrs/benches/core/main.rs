mod core;
mod helpers;
mod ndarray_vs_vec;
mod neighbors;

use criterion::criterion_main;

criterion_main!(
    core::frame::benches,
    core::graph::benches,
    core::topology::benches,
    core::simbox::benches,
    neighbors::linkcell::benches,
    ndarray_vs_vec::primitives::benches,
);
