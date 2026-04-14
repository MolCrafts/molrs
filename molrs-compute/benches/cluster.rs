//! Cluster::compute — distance-based cluster BFS on neighbor graph.

use criterion::{BenchmarkId, Criterion, criterion_group};
use molrs_compute::cluster::Cluster;
use molrs_compute::traits::Compute;

use crate::helpers::{self, Fixture};

const N_ATOMS: &[usize] = &[500, 2_000, 10_000];
const MIN_SIZE: usize = 2;

fn bench_compute(c: &mut Criterion) {
    let mut group = c.benchmark_group("cluster/compute");

    for &n in N_ATOMS {
        let Fixture { frame, nlist, .. } = helpers::fixture(n, 42);
        let cluster = Cluster::new(MIN_SIZE);
        let id = BenchmarkId::new(format!("n{n}"), n);
        group.bench_with_input(id, &(), |b, _| {
            b.iter(|| {
                let r = cluster.compute(&frame, &nlist).expect("cluster::compute");
                std::hint::black_box(r);
            });
        });
    }

    group.finish();
}

criterion_group!(benches, bench_compute);
