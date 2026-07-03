//! `environment` category — bond-order diagram + local (Steinhardt-style)
//! descriptors (regression). Both consume per-frame neighbor lists.

use criterion::{Criterion, criterion_group};
use molrs::compute::environment::{BondOrder, LocalDescriptors};
use molrs::compute::traits::Compute;

use crate::helpers;

fn bench_bond_order(c: &mut Criterion) {
    let mut group = c.benchmark_group("environment/bond_order");
    helpers::configure(&mut group);
    let bo = BondOrder::new(10, 10).unwrap();
    let (frames_owned, nlists) = helpers::reg_pool();
    let frames: Vec<&_> = frames_owned.iter().collect();

    group.bench_function("reg", |b| {
        b.iter(|| {
            std::hint::black_box(bo.compute(&frames, &nlists).unwrap());
        })
    });
    group.finish();
}

fn bench_local_descriptors(c: &mut Criterion) {
    let mut group = c.benchmark_group("environment/local_descriptors");
    helpers::configure(&mut group);
    let ld = LocalDescriptors::new(4);
    let (frames_owned, nlists) = helpers::reg_pool();
    let frames: Vec<&_> = frames_owned.iter().collect();

    group.bench_function("reg", |b| {
        b.iter(|| {
            std::hint::black_box(ld.compute(&frames, &nlists).unwrap());
        })
    });
    group.finish();
}

criterion_group!(benches, bench_bond_order, bench_local_descriptors);
