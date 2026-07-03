//! `density` category — correlation function, Gaussian/local density, and
//! spatial distribution (regression). ONE small pooled frame set per bench.

use criterion::{Criterion, criterion_group};
use molrs::compute::density::correlation_function::CorrelationArgs;
use molrs::compute::density::{
    CorrelationFunction, GaussianDensity, GridSpec, LocalDensity, SpatialDistribution,
};
use molrs::compute::traits::Compute;
use molrs::types::F;
use ndarray::arr2;

use crate::helpers;

fn bench_correlation_function(c: &mut Criterion) {
    let mut group = c.benchmark_group("density/correlation_function");
    helpers::configure(&mut group);
    let cf = CorrelationFunction::new(helpers::RDF_BINS, helpers::CUTOFF, 0.0).unwrap();
    let (frames_owned, nlists) = helpers::reg_pool();
    let frames: Vec<&_> = frames_owned.iter().collect();
    // Unit scalar field per atom, per frame.
    let vals: Vec<Vec<F>> = (0..frames.len())
        .map(|_| vec![1.0_f64; helpers::REG_N])
        .collect();

    group.bench_function("reg", |b| {
        b.iter(|| {
            std::hint::black_box(
                cf.compute(
                    &frames,
                    CorrelationArgs {
                        nlists: &nlists,
                        values_a: &vals,
                        values_b: &vals,
                    },
                )
                .unwrap(),
            );
        })
    });
    group.finish();
}

fn bench_gaussian_density(c: &mut Criterion) {
    let mut group = c.benchmark_group("density/gaussian_density");
    helpers::configure(&mut group);
    let gd = GaussianDensity::new(20, 20, 20, 0.5)
        .unwrap()
        .with_r_max(2.0);
    let (frames_owned, _) = helpers::reg_pool();
    let frames: Vec<&_> = frames_owned.iter().collect();

    group.bench_function("reg", |b| {
        b.iter(|| {
            std::hint::black_box(gd.compute(&frames, ()).unwrap());
        })
    });
    group.finish();
}

fn bench_local_density(c: &mut Criterion) {
    let mut group = c.benchmark_group("density/local_density");
    helpers::configure(&mut group);
    let ld = LocalDensity::new(helpers::CUTOFF).unwrap();
    let (frames_owned, nlists) = helpers::reg_pool();
    let frames: Vec<&_> = frames_owned.iter().collect();

    group.bench_function("reg", |b| {
        b.iter(|| {
            std::hint::black_box(ld.compute(&frames, &nlists).unwrap());
        })
    });
    group.finish();
}

fn bench_spatial_distribution(c: &mut Criterion) {
    let mut group = c.benchmark_group("density/spatial_distribution");
    helpers::configure(&mut group);
    // A 3-atom reference frame + a subset of target atoms binned in the local
    // coordinate system.
    let reference = vec![0usize, 1, 2];
    let template = arr2(&[[0.0_f64, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]);
    let target: Vec<usize> = (3..103).collect();
    let grid = GridSpec {
        n: [16, 16, 16],
        extent: [10.0, 10.0, 10.0],
    };
    let sd = SpatialDistribution::new(reference, template, target, grid).unwrap();
    let (frames_owned, _) = helpers::reg_pool();
    let frames: Vec<&_> = frames_owned.iter().collect();

    group.bench_function("reg", |b| {
        b.iter(|| {
            std::hint::black_box(sd.compute(&frames, ()).unwrap());
        })
    });
    group.finish();
}

criterion_group!(
    benches,
    bench_correlation_function,
    bench_gaussian_density,
    bench_local_density,
    bench_spatial_distribution,
);
