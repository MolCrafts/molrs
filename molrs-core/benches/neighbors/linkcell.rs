use criterion::{BenchmarkId, Criterion, criterion_group};
use molrs_core::neighbors::{BruteForce, LinkCell, NbList};
use molrs_core::types::F;

use crate::helpers;

const SIZES: &[usize] = &[500, 2000, 5000, 10000];
const BOX_SIZE: F = 30.0;
const CUTOFF: F = 4.0;

fn bench_build(c: &mut Criterion) {
    let mut group = c.benchmark_group("neighbors/build");

    for &n in SIZES {
        let pts = helpers::random_points(n, BOX_SIZE, 42);
        let bx = helpers::make_pbc_simbox(BOX_SIZE);

        group.bench_with_input(BenchmarkId::new("linkcell", n), &n, |b, _| {
            b.iter(|| {
                let mut nl = NbList(LinkCell::new().cutoff(CUTOFF));
                nl.build(pts.view(), &bx);
                std::hint::black_box(nl.query());
            });
        });

        group.bench_with_input(BenchmarkId::new("bruteforce", n), &n, |b, _| {
            b.iter(|| {
                let mut nl = NbList(BruteForce::new(CUTOFF));
                nl.build(pts.view(), &bx);
                std::hint::black_box(nl.query());
            });
        });
    }

    group.finish();
}

fn bench_update(c: &mut Criterion) {
    let mut group = c.benchmark_group("neighbors/update");

    for &n in SIZES {
        let pts = helpers::random_points(n, BOX_SIZE, 42);
        let pts2 = helpers::random_points(n, BOX_SIZE, 99);
        let bx = helpers::make_pbc_simbox(BOX_SIZE);

        {
            let mut nl = NbList(LinkCell::new().cutoff(CUTOFF));
            nl.build(pts.view(), &bx);

            group.bench_with_input(BenchmarkId::new("linkcell", n), &n, |b, _| {
                b.iter(|| {
                    nl.update(pts2.view(), &bx);
                    std::hint::black_box(nl.query());
                });
            });
        }

        {
            let mut nl = NbList(BruteForce::new(CUTOFF));
            nl.build(pts.view(), &bx);

            group.bench_with_input(BenchmarkId::new("bruteforce", n), &n, |b, _| {
                b.iter(|| {
                    nl.update(pts2.view(), &bx);
                    std::hint::black_box(nl.query());
                });
            });
        }
    }

    group.finish();
}

criterion_group!(benches, bench_build, bench_update);
