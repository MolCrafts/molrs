//! `order` category — Steinhardt, hexatic, nematic, and solid-liquid order
//! parameters (regression). The neighbor-based kernels share the pooled
//! nlists; nematic consumes per-particle directors.

use criterion::{Criterion, criterion_group};
use molrs::compute::order::{Hexatic, Nematic, SolidLiquid, Steinhardt};
use molrs::compute::traits::Compute;
use molrs::types::F;

use crate::helpers;

fn bench_steinhardt(c: &mut Criterion) {
    let mut group = c.benchmark_group("order/steinhardt");
    helpers::configure(&mut group);
    let s = Steinhardt::new(&[4, 6]).unwrap();
    let (frames_owned, nlists) = helpers::reg_pool();
    let frames: Vec<&_> = frames_owned.iter().collect();

    group.bench_function("reg", |b| {
        b.iter(|| {
            std::hint::black_box(s.compute(&frames, &nlists).unwrap());
        })
    });
    group.finish();
}

fn bench_hexatic(c: &mut Criterion) {
    let mut group = c.benchmark_group("order/hexatic");
    helpers::configure(&mut group);
    let h = Hexatic::new(6).unwrap();
    let (frames_owned, nlists) = helpers::reg_pool();
    let frames: Vec<&_> = frames_owned.iter().collect();

    group.bench_function("reg", |b| {
        b.iter(|| {
            std::hint::black_box(h.compute(&frames, &nlists).unwrap());
        })
    });
    group.finish();
}

fn bench_nematic(c: &mut Criterion) {
    let mut group = c.benchmark_group("order/nematic");
    helpers::configure(&mut group);
    let nem = Nematic::new();
    let (frames_owned, _) = helpers::reg_pool();
    let frames: Vec<&_> = frames_owned.iter().collect();
    let dirs: Vec<[F; 3]> = vec![[0.0, 0.0, 1.0]; helpers::REG_N];

    group.bench_function("reg", |b| {
        b.iter(|| {
            std::hint::black_box(nem.compute(&frames, &dirs).unwrap());
        })
    });
    group.finish();
}

fn bench_solid_liquid(c: &mut Criterion) {
    let mut group = c.benchmark_group("order/solid_liquid");
    helpers::configure(&mut group);
    let sl = SolidLiquid::new(6);
    let (frames_owned, nlists) = helpers::reg_pool();
    let frames: Vec<&_> = frames_owned.iter().collect();

    group.bench_function("reg", |b| {
        b.iter(|| {
            std::hint::black_box(sl.compute(&frames, &nlists).unwrap());
        })
    });
    group.finish();
}

criterion_group!(
    benches,
    bench_steinhardt,
    bench_hexatic,
    bench_nematic,
    bench_solid_liquid,
);
