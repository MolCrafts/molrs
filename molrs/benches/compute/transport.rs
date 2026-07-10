//! `transport` category — diffusion, conductivity, Onsager, and VACF (regression).
//!
//! Covers the Einstein (MSD-slope) and Green–Kubo (ACF-integral) routes for
//! both self-diffusion and ionic conductivity, the Onsager cross-correlation,
//! the batch [`VACF`], and the streaming [`VACFAccumulator`] hot path. Every
//! series kernel takes a small synthetic `[n_samples, dof]` array and ignores
//! `frames`; the Einstein diffusion route consumes real frames (it delegates to
//! the windowed MSD).

use criterion::{Criterion, criterion_group};
use molrs::compute::traits::Compute;
use molrs::compute::transport::{
    EinsteinConductivity, EinsteinDiffusion, EinsteinDiffusionArgs, GreenKuboConductivity,
    GreenKuboDiffusion, OnsagerCorrelation, VACF, VACFAccumulator,
};
use molrs::store::frame::Frame as CoreFrame;
use ndarray::Array2;
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};

use crate::helpers;

/// Empty frame slice for the series-based transport kernels.
fn no_frames() -> Vec<&'static CoreFrame> {
    Vec::new()
}

// Single small regression series.
const N_SAMPLES: usize = 512;
const DOF: usize = 30; // 10 atoms × 3
const RES: usize = 100;
const DT: f64 = 1.0;

/// Synthetic `[n, cols]` fluctuating series in `[-1, 1)`.
fn rng_series(n: usize, cols: usize, seed: u64) -> Array2<f64> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut s = Array2::zeros((n, cols));
    for t in 0..n {
        for c in 0..cols {
            s[[t, c]] = rng.random::<f64>() * 2.0 - 1.0;
        }
    }
    s
}

fn bench_vacf_batch(c: &mut Criterion) {
    let mut group = c.benchmark_group("transport/vacf/batch");
    helpers::configure(&mut group);
    let v = rng_series(N_SAMPLES, DOF, 11);
    group.bench_function("reg", |b| {
        b.iter(|| {
            std::hint::black_box(VACF.compute(&no_frames(), (&v, DT, RES)).unwrap());
        })
    });
    group.finish();
}

fn bench_vacf_accumulator(c: &mut Criterion) {
    let mut group = c.benchmark_group("transport/vacf/accumulator");
    helpers::configure(&mut group);
    let v = rng_series(N_SAMPLES, DOF, 11);
    group.bench_function("reg", |b| {
        b.iter(|| {
            let mut acc = VACFAccumulator::new(RES).unwrap();
            for t in 0..N_SAMPLES {
                let frame: Vec<f64> = (0..DOF).map(|d| v[[t, d]]).collect();
                acc.accumulate(&frame).unwrap();
            }
            std::hint::black_box(acc.finalize().unwrap());
        })
    });
    group.finish();
}

fn bench_einstein_diffusion(c: &mut Criterion) {
    let mut group = c.benchmark_group("transport/einstein_diffusion");
    helpers::configure(&mut group);
    let (frames_owned, _) = helpers::reg_pool();
    let frames: Vec<&_> = frames_owned.iter().collect();
    group.bench_function("reg", |b| {
        b.iter(|| {
            std::hint::black_box(
                EinsteinDiffusion
                    .compute(&frames, EinsteinDiffusionArgs { dt: DT })
                    .unwrap(),
            );
        })
    });
    group.finish();
}

fn bench_green_kubo_diffusion(c: &mut Criterion) {
    let mut group = c.benchmark_group("transport/green_kubo_diffusion");
    helpers::configure(&mut group);
    let v = rng_series(N_SAMPLES, DOF, 11);
    group.bench_function("reg", |b| {
        b.iter(|| {
            std::hint::black_box(
                GreenKuboDiffusion
                    .compute(&no_frames(), (&v, DT, RES))
                    .unwrap(),
            );
        })
    });
    group.finish();
}

fn bench_einstein_conductivity(c: &mut Criterion) {
    let mut group = c.benchmark_group("transport/einstein_conductivity");
    helpers::configure(&mut group);
    let dipole = rng_series(N_SAMPLES, 3, 3);
    group.bench_function("reg", |b| {
        b.iter(|| {
            std::hint::black_box(
                EinsteinConductivity
                    .compute(&no_frames(), (&dipole, DT, RES))
                    .unwrap(),
            );
        })
    });
    group.finish();
}

fn bench_green_kubo_conductivity(c: &mut Criterion) {
    let mut group = c.benchmark_group("transport/green_kubo_conductivity");
    helpers::configure(&mut group);
    let current = rng_series(N_SAMPLES, 3, 5);
    group.bench_function("reg", |b| {
        b.iter(|| {
            std::hint::black_box(
                GreenKuboConductivity
                    .compute(&no_frames(), (&current, DT, RES))
                    .unwrap(),
            );
        })
    });
    group.finish();
}

fn bench_onsager(c: &mut Criterion) {
    let mut group = c.benchmark_group("transport/onsager");
    helpers::configure(&mut group);
    let p_i = rng_series(N_SAMPLES, 3, 7);
    let p_j = rng_series(N_SAMPLES, 3, 8);
    group.bench_function("reg", |b| {
        b.iter(|| {
            std::hint::black_box(
                OnsagerCorrelation
                    .compute(&no_frames(), (&p_i, &p_j, DT, RES))
                    .unwrap(),
            );
        })
    });
    group.finish();
}

criterion_group!(
    benches,
    bench_vacf_batch,
    bench_vacf_accumulator,
    bench_einstein_diffusion,
    bench_green_kubo_diffusion,
    bench_einstein_conductivity,
    bench_green_kubo_conductivity,
    bench_onsager,
);
