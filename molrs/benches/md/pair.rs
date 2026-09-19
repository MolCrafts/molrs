//! The pair fold alone, at a fixed pair count — the arithmetic every other
//! cost in a step is measured against.

use std::time::Duration;

use criterion::{Criterion, criterion_group};
use molrs::ff::forcefield::mixing::Mixing;
use molrs::ff::potential::PairDriven;
use molrs::ff::potential::pair::LJCut;
use molrs::spatial::neighbors::{NeighborList, NeighborPolicy, VerletSkin};
use molrs::spatial::simbox::SimBox;
use molrs::types::{F, FNx3};
use ndarray::array;

const SIDE: usize = 8;
const SPACING: F = 3.0;
const CUTOFF: F = 6.0;

fn bench_fold(c: &mut Criterion) {
    let n = SIDE * SIDE * SIDE;
    let l = SIDE as F * SPACING;
    let bx = SimBox::cube(l, array![0.0, 0.0, 0.0], [true; 3]).unwrap();
    let mut pos = FNx3::zeros((n, 3));
    for a in 0..n {
        let (i, j, k) = (a % SIDE, (a / SIDE) % SIDE, a / (SIDE * SIDE));
        pos[[a, 0]] = i as F * SPACING;
        pos[[a, 1]] = j as F * SPACING;
        pos[[a, 2]] = k as F * SPACING;
    }
    let mut skin = VerletSkin::new(
        NeighborList::new(CUTOFF),
        CUTOFF,
        NeighborPolicy {
            skin: 0.0,
            ..NeighborPolicy::default()
        },
        pos.view(),
        bx,
    )
    .unwrap();
    let table = skin.pairs_at(pos.view()).unwrap().clone();
    let n_pairs = table.query_point_indices().len();

    let kernel = LJCut::typed(
        (0..n).map(|i| (i % 2) as u32).collect(),
        &[(0.3, 3.4), (0.5, 3.0)],
        Mixing::Arithmetic,
        CUTOFF,
        12,
        6,
        false,
        false,
    )
    .unwrap();
    let flat: Vec<F> = pos.iter().copied().collect();
    let half: Vec<F> = vec![0.5; n_pairs];
    let mut out = vec![0.0; flat.len()];

    let mut g = c.benchmark_group("md/pair");
    g.warm_up_time(Duration::from_millis(200));
    g.measurement_time(Duration::from_millis(600));
    g.sample_size(10);
    g.throughput(criterion::Throughput::Elements(n_pairs as u64));

    // What the fold is fed. If a step costs much more than these two together,
    // the difference is the provider's own bookkeeping and not the physics.
    g.bench_function("mic_table", |b| {
        b.iter(|| {
            skin.pairs_at(pos.view())
                .unwrap()
                .query_point_indices()
                .len()
        })
    });

    g.bench_function("lj_cut/fold", |b| {
        b.iter(|| {
            out.fill(0.0);
            kernel.accumulate_pairs(&flat, &table, &[], &mut out)
        })
    });
    // The weighted path costs one read per pair on top. If it ever costs more
    // than that again — a rebuilt table, say — this is where it shows.
    g.bench_function("lj_cut/fold_weighted", |b| {
        b.iter(|| {
            out.fill(0.0);
            kernel.accumulate_pairs(&flat, &table, &half, &mut out)
        })
    });
    g.finish();
}

criterion_group!(benches, bench_fold);
