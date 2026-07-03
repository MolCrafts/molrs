//! `voronoi` category — native periodic radical-Voronoi tessellation (regression).
//!
//! The tessellation clips each cell against candidate neighbours in a growing
//! periodic shell (heavier per particle than the other kernels), so this uses a
//! single small configuration. Equal radii (= plain Voronoi) at liquid density.

use criterion::{Criterion, criterion_group};
use molrs::compute::voronoi::RadicalVoronoi;
use molrs::spatial::region::simbox::SimBox;
use molrs::types::F;
use ndarray::array;

use crate::helpers;

const VORO_N: usize = 500;

fn bench(c: &mut Criterion) {
    let mut group = c.benchmark_group("voronoi/build");
    helpers::configure(&mut group);
    let box_len = helpers::box_for_density(VORO_N);
    let simbox = SimBox::cube(box_len, array![0.0 as F, 0.0, 0.0], [true, true, true]).unwrap();
    let pts = helpers::random_positions(VORO_N, box_len, 7);
    let radii = vec![1.0 as F; VORO_N];

    group.bench_function("reg", |b| {
        b.iter(|| {
            std::hint::black_box(RadicalVoronoi.build(pts.view(), &radii, &simbox).unwrap());
        })
    });

    group.finish();
}

criterion_group!(benches, bench);
