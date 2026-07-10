//! Neighbor-list regression benches: LinkCell / BruteForce build + update, plus
//! the SoA hot paths — `LinkCell::build_soa` (vs interleaved `build`) and the
//! high-level `NeighborQuery::from_columns` + `query_columns` (vs `new` +
//! `query`). Single small regression size.

use criterion::{BenchmarkId, Criterion, criterion_group};
use molrs::spatial::neighbors::{BruteForce, LinkCell, NbList, NbListAlgo, NeighborQuery};
use molrs::types::F;
use ndarray::Array2;

use crate::helpers;

const SIZES: &[usize] = &[helpers::REG_N];
const BOX_SIZE: F = 30.0;
const CUTOFF: F = 4.0;

/// Split an interleaved `[N, 3]` position array into SoA `x`/`y`/`z` columns.
fn columns(pts: &Array2<F>) -> (Vec<F>, Vec<F>, Vec<F>) {
    let n = pts.nrows();
    let (mut xs, mut ys, mut zs) = (
        Vec::with_capacity(n),
        Vec::with_capacity(n),
        Vec::with_capacity(n),
    );
    for i in 0..n {
        xs.push(pts[[i, 0]]);
        ys.push(pts[[i, 1]]);
        zs.push(pts[[i, 2]]);
    }
    (xs, ys, zs)
}

fn bench_build(c: &mut Criterion) {
    let mut group = c.benchmark_group("neighbors/build");
    helpers::configure(&mut group);

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
    helpers::configure(&mut group);

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

/// Interleaved `build` vs the SoA `build_soa` self-query hot path (the two are
/// byte-identical; this guards the SoA path against a regression).
fn bench_build_soa(c: &mut Criterion) {
    let mut group = c.benchmark_group("neighbors/build_soa");
    helpers::configure(&mut group);

    for &n in SIZES {
        let pts = helpers::random_points(n, BOX_SIZE, 42);
        let (xs, ys, zs) = columns(&pts);
        let bx = helpers::make_pbc_simbox(BOX_SIZE);

        group.bench_with_input(BenchmarkId::new("aos_build", n), &n, |b, _| {
            b.iter(|| {
                let mut nl = NbList(LinkCell::new().cutoff(CUTOFF));
                nl.build(pts.view(), &bx);
                std::hint::black_box(nl.query());
            });
        });

        group.bench_with_input(BenchmarkId::new("soa_build_soa", n), &n, |b, _| {
            b.iter(|| {
                let mut lc = LinkCell::new().cutoff(CUTOFF);
                lc.build_soa(&xs, &ys, &zs, &bx);
                std::hint::black_box(lc.query());
            });
        });
    }

    group.finish();
}

/// High-level `NeighborQuery` cross-query: interleaved `new` + `query` vs the
/// SoA `from_columns` + `query_columns` path.
fn bench_neighbor_query(c: &mut Criterion) {
    let mut group = c.benchmark_group("neighbors/query_columns");
    helpers::configure(&mut group);

    for &n in SIZES {
        let pts = helpers::random_points(n, BOX_SIZE, 42);
        let (xs, ys, zs) = columns(&pts);
        let bx = helpers::make_pbc_simbox(BOX_SIZE);

        group.bench_with_input(BenchmarkId::new("aos_new_query", n), &n, |b, _| {
            b.iter(|| {
                let nq = NeighborQuery::new(&bx, pts.view(), CUTOFF);
                std::hint::black_box(nq.query(pts.view()));
            });
        });

        group.bench_with_input(BenchmarkId::new("soa_from_columns", n), &n, |b, _| {
            b.iter(|| {
                let nq = NeighborQuery::from_columns(&bx, &xs, &ys, &zs, CUTOFF);
                std::hint::black_box(nq.query_columns(&xs, &ys, &zs));
            });
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_build,
    bench_update,
    bench_build_soa,
    bench_neighbor_query,
);
