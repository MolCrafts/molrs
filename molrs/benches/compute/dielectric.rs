//! `dielectric` category — static dielectric constant + ε(ω) spectra (regression).
//!
//! Covers static dielectric constant (scalar + per-axis), the Einstein-Helfand
//! and Green-Kubo ε(ω) raw-compute + fit compositions, dipole moment, and
//! current density. ONE small synthetic dipole trajectory per bench.

use criterion::{Criterion, criterion_group};
use molrs::compute::dielectric::{
    compute_current_density, compute_dipole_moment, static_dielectric_constant,
    static_dielectric_constant_components,
};
use molrs::compute::spectroscopy::{EinsteinHelfandSpectrum, GreenKuboSpectrum};
use molrs::compute::traits::{Compute, Fit};
use molrs::compute::transport::{DebyeRelaxation, EwaldBoundary, GreenKuboConductivity};
use molrs::store::frame::Frame as CoreFrame;
use ndarray::{Array1, Array2};
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};

use crate::helpers;

/// Empty frame slice for the series-based raw computes (`frames` is unused).
fn no_frames() -> Vec<&'static CoreFrame> {
    Vec::new()
}

// Single small regression trajectory.
const N_FRAMES: usize = 512;
const N_ATOMS: usize = helpers::REG_N;
const VOLUME: f64 = 30_000.0; // Å³
const TEMPERATURE: f64 = 300.0; // K
const EPSILON_INF: f64 = 1.0;
const DT: f64 = 0.001; // ps
const MAX_CORRELATION: usize = 100;

/// Synthetic dipole trajectory `[n_frames, 3]` centred at 0.
fn synthetic_dipoles(n_frames: usize, seed: u64) -> Array2<f64> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut m = Array2::zeros((n_frames, 3));
    for t in 0..n_frames {
        for d in 0..3 {
            m[[t, d]] = rng.random::<f64>() - 0.5;
        }
    }
    m
}

/// Charges + a single-frame position set that produce a fluctuating dipole.
fn synthetic_charges_positions(n_atoms: usize, seed: u64) -> (Array1<f64>, Array2<f64>) {
    let mut rng = StdRng::seed_from_u64(seed);
    let charges: Array1<f64> = (0..n_atoms)
        .map(|_| if rng.random::<bool>() { 1.0 } else { -1.0 })
        .collect();
    let mut pos = Array2::zeros((n_atoms, 3));
    for i in 0..n_atoms {
        for d in 0..3 {
            pos[[i, d]] = rng.random::<f64>() * 30.0;
        }
    }
    (charges, pos)
}

fn bench_static_scalar(c: &mut Criterion) {
    let mut group = c.benchmark_group("dielectric/static_scalar");
    helpers::configure(&mut group);
    let dm = synthetic_dipoles(N_FRAMES, 42);
    group.bench_function("reg", |b| {
        b.iter(|| {
            std::hint::black_box(
                static_dielectric_constant(&dm, VOLUME, TEMPERATURE, EPSILON_INF).unwrap(),
            );
        })
    });
    group.finish();
}

fn bench_static_components(c: &mut Criterion) {
    let mut group = c.benchmark_group("dielectric/static_components");
    helpers::configure(&mut group);
    let dm = synthetic_dipoles(N_FRAMES, 42);
    group.bench_function("reg", |b| {
        b.iter(|| {
            std::hint::black_box(
                static_dielectric_constant_components(&dm, VOLUME, TEMPERATURE, EPSILON_INF)
                    .unwrap(),
            );
        })
    });
    group.finish();
}

fn bench_einstein_helfand(c: &mut Criterion) {
    let mut group = c.benchmark_group("dielectric/einstein_helfand");
    helpers::configure(&mut group);
    let dm = synthetic_dipoles(N_FRAMES, 42);
    group.bench_function("reg", |b| {
        b.iter(|| {
            let raw = DebyeRelaxation {
                volume: VOLUME,
                temperature: TEMPERATURE,
                boundary: EwaldBoundary::TinFoil,
            }
            .compute(&no_frames(), (&dm, DT, MAX_CORRELATION))
            .unwrap();
            std::hint::black_box(
                EinsteinHelfandSpectrum {
                    dt: DT,
                    volume: VOLUME,
                    temperature: TEMPERATURE,
                    epsilon_inf: EPSILON_INF,
                    zero_lag_variance: raw.zero_lag_variance,
                }
                .fit(&raw.acf)
                .unwrap(),
            );
        })
    });
    group.finish();
}

fn bench_green_kubo(c: &mut Criterion) {
    let mut group = c.benchmark_group("dielectric/green_kubo");
    helpers::configure(&mut group);
    let dm = synthetic_dipoles(N_FRAMES, 42);
    let j = compute_current_density(&dm, DT, VOLUME).unwrap();
    // Skip the NaN row 0 (no previous frame).
    let j_post: Array2<f64> = j.slice(ndarray::s![1.., ..]).to_owned();
    group.bench_function("reg", |b| {
        b.iter(|| {
            let raw = GreenKuboConductivity
                .compute(&no_frames(), (&j_post, DT, MAX_CORRELATION))
                .unwrap();
            std::hint::black_box(
                GreenKuboSpectrum {
                    dt: DT,
                    volume: VOLUME,
                    temperature: TEMPERATURE,
                    epsilon_inf: EPSILON_INF,
                    window_type: "hann".to_string(),
                }
                .fit(&raw.jacf)
                .unwrap(),
            );
        })
    });
    group.finish();
}

fn bench_dipole_moment(c: &mut Criterion) {
    let mut group = c.benchmark_group("dielectric/dipole_moment");
    helpers::configure(&mut group);
    let (charges, pos) = synthetic_charges_positions(N_ATOMS, 42);
    group.bench_function("reg", |b| {
        b.iter(|| {
            std::hint::black_box(compute_dipole_moment(&charges, &pos).unwrap());
        })
    });
    group.finish();
}

fn bench_current_density(c: &mut Criterion) {
    let mut group = c.benchmark_group("dielectric/current_density");
    helpers::configure(&mut group);
    let dm = synthetic_dipoles(N_FRAMES, 42);
    group.bench_function("reg", |b| {
        b.iter(|| {
            std::hint::black_box(compute_current_density(&dm, DT, VOLUME).unwrap());
        })
    });
    group.finish();
}

criterion_group!(
    benches,
    bench_static_scalar,
    bench_static_components,
    bench_einstein_helfand,
    bench_green_kubo,
    bench_dipole_moment,
    bench_current_density,
);
