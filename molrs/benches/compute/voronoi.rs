//! `RadicalVoronoi::build` — native periodic radical-Voronoi tessellation scaling.
//!
//! The tessellation clips each cell against candidate neighbours in a growing
//! periodic shell, so this size sweep is the headline check that the native
//! (WASM-clean, voro++-algorithm) path scales acceptably with particle count.
//! Equal radii (= plain Voronoi) at liquid-like density.

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group};
use molrs::compute::voronoi::RadicalVoronoi;
use molrs::spatial::region::simbox::SimBox;
use molrs::types::F;
use ndarray::array;

use crate::helpers;

/// Atom counts for the Voronoi sweep — smaller ceiling than RDF since the
/// tessellation is heavier per particle. Single configuration per point.
const VORO_SIZES: &[usize] = &[100, 500, 1_000, 2_000, 4_000];

fn size_sweep(c: &mut Criterion) {
    let mut group = c.benchmark_group("voronoi/size_sweep");
    helpers::configure(&mut group);

    for &n in VORO_SIZES {
        let box_len = helpers::box_for_density(n);
        let simbox = SimBox::cube(box_len, array![0.0 as F, 0.0, 0.0], [true, true, true]).unwrap();
        let pts = helpers::random_positions(n, box_len, 7);
        let radii = vec![1.0 as F; n];

        group.throughput(Throughput::Elements(n as u64));
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            b.iter(|| {
                std::hint::black_box(RadicalVoronoi.build(pts.view(), &radii, &simbox).unwrap());
            })
        });
    }

    group.finish();
}

criterion_group!(benches, size_sweep);
