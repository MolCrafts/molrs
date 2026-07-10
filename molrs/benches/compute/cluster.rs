//! `cluster` category — distance-based connected-component clustering (regression).

use criterion::{Criterion, criterion_group};
use molrs::compute::cluster::Cluster;
use molrs::compute::traits::Compute;

use crate::helpers;

fn bench(c: &mut Criterion) {
    let mut group = c.benchmark_group("cluster/batch");
    helpers::configure(&mut group);
    let cluster = Cluster::new(2);
    let (frames_owned, nlists) = helpers::reg_pool();
    let frames: Vec<&_> = frames_owned.iter().collect();

    group.bench_function("reg", |b| {
        b.iter(|| {
            std::hint::black_box(cluster.compute(&frames, &nlists).unwrap());
        })
    });

    group.finish();
}

criterion_group!(benches, bench);
