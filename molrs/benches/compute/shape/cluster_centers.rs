//! `ClusterCenters::compute` — MIC-aware geometric cluster centers (regression).

use criterion::{Criterion, criterion_group};
use molrs::compute::shape::cluster_centers::ClusterCenters;
use molrs::compute::traits::Compute;

use crate::helpers;

fn bench(c: &mut Criterion) {
    let mut group = c.benchmark_group("shape/cluster_centers");
    helpers::configure(&mut group);
    let centers = ClusterCenters::new();
    let (frames_owned, nlists) = helpers::reg_pool();
    let frames: Vec<&_> = frames_owned.iter().collect();
    let deps = helpers::build_deps(&frames, &nlists);

    group.bench_function("reg", |b| {
        b.iter(|| {
            std::hint::black_box(centers.compute(&frames, &deps.cluster).unwrap());
        })
    });

    group.finish();
}

criterion_group!(benches, bench);
