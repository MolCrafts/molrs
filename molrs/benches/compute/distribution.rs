//! `distribution` category — distance/angle/dihedral distribution functions +
//! the combined (joint) distribution (regression).
//!
//! Distributions read explicit atom tuples ([`AtomGroups`]) — no neighbor list.
//! ONE small pooled frame set and a fixed tuple set per bench.

use criterion::{Criterion, criterion_group};
use molrs::compute::distribution::{
    AngleObservable, AtomGroups, AxisSpec, CombinedDistribution, DihedralObservable,
    DistanceObservable, DistributionFunction,
};
use molrs::compute::traits::Compute;

use crate::helpers;

const N_TUPLES: usize = 500;

fn pairs() -> AtomGroups {
    let t: Vec<(u64, u64)> = (0..N_TUPLES as u64).map(|i| (i, i + 1)).collect();
    AtomGroups::pairs(&t)
}

fn triples() -> AtomGroups {
    let t: Vec<(u64, u64, u64)> = (0..N_TUPLES as u64).map(|i| (i, i + 1, i + 2)).collect();
    AtomGroups::triples(&t)
}

fn quads() -> AtomGroups {
    let t: Vec<(u64, u64, u64, u64)> = (0..N_TUPLES as u64)
        .map(|i| (i, i + 1, i + 2, i + 3))
        .collect();
    AtomGroups::quads(&t)
}

fn bench_distance(c: &mut Criterion) {
    let mut group = c.benchmark_group("distribution/distance");
    helpers::configure(&mut group);
    let df = DistributionFunction::new(DistanceObservable, 50, 0.0, 5.0).unwrap();
    let groups = pairs();
    let (frames_owned, _) = helpers::reg_pool();
    let frames: Vec<&_> = frames_owned.iter().collect();

    group.bench_function("reg", |b| {
        b.iter(|| {
            std::hint::black_box(df.compute(&frames, &groups).unwrap());
        })
    });
    group.finish();
}

fn bench_angle(c: &mut Criterion) {
    let mut group = c.benchmark_group("distribution/angle");
    helpers::configure(&mut group);
    let df = DistributionFunction::over_natural_range(AngleObservable, 180).unwrap();
    let groups = triples();
    let (frames_owned, _) = helpers::reg_pool();
    let frames: Vec<&_> = frames_owned.iter().collect();

    group.bench_function("reg", |b| {
        b.iter(|| {
            std::hint::black_box(df.compute(&frames, &groups).unwrap());
        })
    });
    group.finish();
}

fn bench_dihedral(c: &mut Criterion) {
    let mut group = c.benchmark_group("distribution/dihedral");
    helpers::configure(&mut group);
    let df = DistributionFunction::over_natural_range(DihedralObservable, 180).unwrap();
    let groups = quads();
    let (frames_owned, _) = helpers::reg_pool();
    let frames: Vec<&_> = frames_owned.iter().collect();

    group.bench_function("reg", |b| {
        b.iter(|| {
            std::hint::black_box(df.compute(&frames, &groups).unwrap());
        })
    });
    group.finish();
}

fn bench_combined(c: &mut Criterion) {
    let mut group = c.benchmark_group("distribution/combined");
    helpers::configure(&mut group);
    let cdf = CombinedDistribution::new(
        vec![DistanceObservable.into(), AngleObservable.into()],
        vec![
            AxisSpec::new(10, 1.0, 3.0).unwrap(),
            AxisSpec::new(12, 0.0, std::f64::consts::PI)
                .unwrap()
                .with_sin_weight(true),
        ],
    )
    .unwrap();
    let groups = [pairs(), triples()];
    let (frames_owned, _) = helpers::reg_pool();
    let frames: Vec<&_> = frames_owned.iter().collect();

    group.bench_function("reg", |b| {
        b.iter(|| {
            std::hint::black_box(cdf.compute(&frames, &groups).unwrap());
        })
    });
    group.finish();
}

criterion_group!(
    benches,
    bench_distance,
    bench_angle,
    bench_dihedral,
    bench_combined,
);
