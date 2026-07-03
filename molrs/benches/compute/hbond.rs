//! `hbond` category — hydrogen-bond detection + lifetime correlations
//! (regression).
//!
//! `detect` runs the geometric criterion over a pooled trajectory; the
//! lifetime kernel runs the continuous/intermittent TCF over a small synthetic
//! presence matrix (self-contained, independent of a specific detection run).

use criterion::{Criterion, criterion_group};
use molrs::compute::hbond::{HBondCriterion, HBonds, hbond_lifetimes};
use molrs::compute::traits::Compute;
use molrs::types::F;
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};

use crate::helpers;

fn bench_detect(c: &mut Criterion) {
    let mut group = c.benchmark_group("hbond/detect");
    helpers::configure(&mut group);
    let donors: Vec<(u32, u32)> = (0..200u32).map(|i| (2 * i, 2 * i + 1)).collect();
    let acceptors: Vec<u32> = (400..600u32).collect();
    let detector = HBonds::new(donors, acceptors, HBondCriterion::default());
    let (frames_owned, _) = helpers::reg_pool();
    let frames: Vec<&_> = frames_owned.iter().collect();

    group.bench_function("reg", |b| {
        b.iter(|| {
            std::hint::black_box(detector.compute(&frames, ()).unwrap());
        })
    });
    group.finish();
}

fn bench_lifetime(c: &mut Criterion) {
    let mut group = c.benchmark_group("hbond/lifetime");
    helpers::configure(&mut group);
    // Synthetic presence matrix: 64 time frames × 200 candidate bonds.
    let (n_time, n_bonds) = (64usize, 200usize);
    let mut rng = StdRng::seed_from_u64(42);
    let present: Vec<Vec<bool>> = (0..n_time)
        .map(|_| (0..n_bonds).map(|_| rng.random::<bool>()).collect())
        .collect();
    let dt: F = 1.0;
    let max_lag = 20;

    group.bench_function("reg", |b| {
        b.iter(|| {
            std::hint::black_box(hbond_lifetimes(&present, dt, max_lag).unwrap());
        })
    });
    group.finish();
}

criterion_group!(benches, bench_detect, bench_lifetime);
