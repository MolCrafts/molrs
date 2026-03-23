mod core;
mod helpers;
mod ndarray_vs_vec;
mod neighbors;

use criterion::criterion_main;

criterion_main!(
    core::potential::benches,
    core::region::simbox::benches,
    neighbors::linkcell::benches,
    ndarray_vs_vec::primitives::benches,
);
