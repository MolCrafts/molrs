//! Neighbor-list regression benches: LinkCell / BruteForce build + update, plus
//! the SoA hot paths — `LinkCell::build_soa` (vs interleaved `build`) and the
//! high-level `NeighborQuery::from_columns` + `query_columns` (vs `new` +
//! `query`). Single small regression size.

use criterion::{BenchmarkId, Criterion, criterion_group};
use molrs::spatial::neighbors::{
    BruteForce, CellGrid, LinkCell, NbList, NbListAlgo, NeighborQuery,
};
use molrs::spatial::region::simbox::SimBox;
use molrs::types::F;
use ndarray::{Array2, array};

use crate::helpers;

const SIZES: &[usize] = &[helpers::REG_N];
const BOX_SIZE: F = 30.0;
const CUTOFF: F = 4.0;

/// Hexagonal (γ = 120°) periodic box at the same edge length as the
/// orthorhombic fixture, so ortho and triclinic timings are comparable at equal
/// particle count.
fn make_triclinic_simbox(size: F) -> SimBox {
    let h = SimBox::matrix_from_lengths_angles([size, size, size], [90.0, 90.0, 120.0])
        .expect("hex matrix");
    SimBox::new(h, array![0.0 as F, 0.0 as F, 0.0 as F], [true, true, true]).expect("hex box")
}

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

/// Function-level microbenches for the `CellGrid` extraction.
///
/// Per `.claude/notes/performance.md`, an extraction is judged on per-function
/// evidence, not on an end-to-end bench whose ±3% noise floor is the same
/// magnitude as the gate. These are the permanent guards: `bench.yml` tracks
/// them over time, so a later change to the cell arithmetic shows up as a step
/// on the dashboard rather than being hidden inside a build timing.
fn bench_cellgrid(c: &mut Criterion) {
    let mut group = c.benchmark_group("neighbors/cellgrid");
    helpers::configure(&mut group);

    let pts = helpers::random_points_native(helpers::REG_N, BOX_SIZE, 42);
    let boxes = [
        ("ortho", helpers::make_pbc_simbox(BOX_SIZE)),
        ("triclinic", make_triclinic_simbox(BOX_SIZE)),
    ];

    for (label, bx) in &boxes {
        let g = CellGrid::for_cutoff(bx, CUTOFF);

        group.bench_function(BenchmarkId::new("cell_of/grid", label), |b| {
            b.iter(|| {
                let mut acc = 0usize;
                for &r in &pts {
                    acc ^= g.cell_of(bx, r);
                }
                std::hint::black_box(acc)
            });
        });

        let n_cells = g.n_cells();
        group.bench_function(BenchmarkId::new("stencil_fwd/grid", label), |b| {
            b.iter(|| {
                let mut buf = [0usize; 27];
                let mut acc = 0usize;
                for i in 0..n_cells {
                    acc += g.stencil_forward(i, &mut buf);
                }
                std::hint::black_box(acc)
            });
        });
    }

    group.finish();
}

/// Caller-level microbenches: the build path and the zero-allocation pair
/// traversal, each on an orthorhombic and a triclinic box.
///
/// A function microbench cannot see indirection, vtable or inlining-boundary
/// costs introduced at the call site, so the extraction needs both. The
/// triclinic entries also give the newly-reachable code path a permanent
/// baseline on the bench dashboard rather than a one-off measurement at merge.
fn bench_traversal(c: &mut Criterion) {
    let mut group = c.benchmark_group("neighbors/traversal");
    helpers::configure(&mut group);

    let boxes = [
        ("ortho", helpers::make_pbc_simbox(BOX_SIZE)),
        ("triclinic", make_triclinic_simbox(BOX_SIZE)),
    ];

    for &n in SIZES {
        let pts = helpers::random_points(n, BOX_SIZE, 42);

        for (label, bx) in &boxes {
            group.bench_function(BenchmarkId::new("build", label), |b| {
                b.iter(|| {
                    let mut nl = NbList(LinkCell::new().cutoff(CUTOFF));
                    nl.build(pts.view(), bx);
                    std::hint::black_box(nl.query());
                });
            });

            let mut nl = NbList(LinkCell::new().cutoff(CUTOFF));
            nl.build(pts.view(), bx);
            group.bench_function(BenchmarkId::new("visit_pairs", label), |b| {
                b.iter(|| {
                    let mut acc = 0.0 as F;
                    nl.visit_pairs(&mut |_i: u32, _j: u32, d2: F, _v: [F; 3]| acc += d2);
                    std::hint::black_box(acc)
                });
            });
        }
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_build,
    bench_update,
    bench_build_soa,
    bench_neighbor_query,
    bench_cellgrid,
    bench_traversal,
);
