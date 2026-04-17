//! RDF::compute — the radial distribution function hot path.
//!
//! Sweeps atom count and bin count with a fixed PBC box + cutoff.

use criterion::{BenchmarkId, Criterion, criterion_group};
use molrs_compute::rdf::RDF;
use molrs_compute::traits::Compute;

use crate::helpers::{self, Fixture};

const N_ATOMS: &[usize] = &[500, 2_000, 10_000];
const N_BINS: &[usize] = &[50, 200];

fn bench_compute(c: &mut Criterion) {
    let mut group = c.benchmark_group("rdf/compute");

    for &n in N_ATOMS {
        let Fixture { frame, nlist, .. } = helpers::fixture(n, 42);
        for &bins in N_BINS {
            let rdf = RDF::new(bins, helpers::CUTOFF, 0.0).unwrap();
            let id = BenchmarkId::new(format!("n{n}_bins{bins}"), n);
            group.bench_with_input(id, &(), |b, _| {
                b.iter(|| {
                    let r = rdf.compute(&frame, &nlist).expect("rdf::compute");
                    std::hint::black_box(r);
                });
            });
        }
    }

    group.finish();
}

criterion_group!(benches, bench_compute);
