//! `spectroscopy` category — vibrational + chiral spectra (regression).
//!
//! Each spectrum is the raw-correlator + spectral-fit composition. Covers
//! power spectrum (VDOS), IR, Raman, and the chiral kernels VCD, ROA, and
//! resonance Raman. ONE small synthetic time series per bench. (The ε(ω)
//! dielectric-spectrum fits live in the `dielectric` bench.)

use criterion::{Criterion, criterion_group};
use molrs::Frame;
use molrs::compute::traits::{Compute, Fit};
use molrs::compute::{
    IRFlux, IRSpectrum, PowerSpectrum, RamanSpectrum, RamanTensor, ResonanceRamanSpectrum,
    ResonanceRamanTensor, RoaCrossTensor, RoaSpectrum, VACF, VcdCrossFlux, VcdSpectrum,
};
use ndarray::Array2;
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};

use crate::helpers;

/// Empty frame slice for the series-based raw computes.
fn no_frames() -> Vec<&'static Frame> {
    Vec::new()
}

// Single small regression series.
const N_FRAMES: usize = 512;
const DT_FS: f64 = 0.5;
const RESOLUTION: usize = 100;
const POWER_DOF: usize = 30; // 10 atoms × 3

/// Oscillatory `[n_frames, n_cols]` series with per-column frequency + phase.
fn sine_series(n_frames: usize, n_cols: usize, seed: u64) -> Array2<f64> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut s = Array2::zeros((n_frames, n_cols));
    for c in 0..n_cols {
        let freq = 1.0 + rng.random::<f64>() * 30.0; // THz
        let phase = rng.random::<f64>() * 2.0 * std::f64::consts::PI;
        for t in 0..n_frames {
            let time = t as f64 * DT_FS;
            s[[t, c]] = (2.0 * std::f64::consts::PI * freq * 1e-3 * time + phase).sin();
        }
    }
    s
}

fn bench_power_spectrum(c: &mut Criterion) {
    let mut group = c.benchmark_group("spectroscopy/power_spectrum");
    helpers::configure(&mut group);
    let v = sine_series(N_FRAMES, POWER_DOF, 42);
    group.bench_function("reg", |b| {
        b.iter(|| {
            let raw = VACF.compute(&no_frames(), (&v, DT_FS, RESOLUTION)).unwrap();
            std::hint::black_box(PowerSpectrum.fit((&raw.acf, DT_FS)).unwrap());
        })
    });
    group.finish();
}

fn bench_ir_spectrum(c: &mut Criterion) {
    let mut group = c.benchmark_group("spectroscopy/ir_spectrum");
    helpers::configure(&mut group);
    let dm = sine_series(N_FRAMES, 3, 42);
    group.bench_function("reg", |b| {
        b.iter(|| {
            let raw = IRFlux
                .compute(&no_frames(), (&dm, DT_FS, RESOLUTION))
                .unwrap();
            std::hint::black_box(IRSpectrum.fit((&raw.acf, DT_FS)).unwrap());
        })
    });
    group.finish();
}

fn bench_raman_spectrum(c: &mut Criterion) {
    let mut group = c.benchmark_group("spectroscopy/raman_spectrum");
    helpers::configure(&mut group);
    let alpha = sine_series(N_FRAMES, 6, 42);
    group.bench_function("reg", |b| {
        b.iter(|| {
            let raw = RamanTensor
                .compute(&no_frames(), (&alpha, DT_FS, RESOLUTION))
                .unwrap();
            let fit = RamanSpectrum {
                incident_frequency_cm1: 10_000.0,
                temperature_k: 300.0,
                averaged: true,
            };
            std::hint::black_box(fit.fit((&raw.acf_iso, &raw.acf_aniso, DT_FS)).unwrap());
        })
    });
    group.finish();
}

fn bench_vcd_spectrum(c: &mut Criterion) {
    let mut group = c.benchmark_group("spectroscopy/vcd_spectrum");
    helpers::configure(&mut group);
    let mu = sine_series(N_FRAMES, 3, 42); // electric dipole
    let m = sine_series(N_FRAMES, 3, 43); // magnetic dipole
    group.bench_function("reg", |b| {
        b.iter(|| {
            let raw = VcdCrossFlux
                .compute(&no_frames(), (&mu, &m, DT_FS, RESOLUTION))
                .unwrap();
            std::hint::black_box(VcdSpectrum.fit((&raw.acf, DT_FS)).unwrap());
        })
    });
    group.finish();
}

fn bench_roa_spectrum(c: &mut Criterion) {
    let mut group = c.benchmark_group("spectroscopy/roa_spectrum");
    helpers::configure(&mut group);
    let alpha = sine_series(N_FRAMES, 6, 42); // electric polarizability (Voigt)
    let g = sine_series(N_FRAMES, 6, 43); // optical-activity G tensor (Voigt)
    group.bench_function("reg", |b| {
        b.iter(|| {
            let raw = RoaCrossTensor
                .compute(&no_frames(), (&alpha, &g, DT_FS, RESOLUTION))
                .unwrap();
            let fit = RoaSpectrum {
                incident_frequency_cm1: 0.0,
                temperature_k: 0.0,
                averaged: false,
            };
            std::hint::black_box(fit.fit((&raw.acf_iso, &raw.acf_aniso, DT_FS)).unwrap());
        })
    });
    group.finish();
}

fn bench_resonance_raman_spectrum(c: &mut Criterion) {
    let mut group = c.benchmark_group("spectroscopy/resonance_raman_spectrum");
    helpers::configure(&mut group);
    let alpha = sine_series(N_FRAMES, 6, 42); // resonant polarizabilities (Voigt)
    group.bench_function("reg", |b| {
        b.iter(|| {
            let raw = ResonanceRamanTensor
                .compute(&no_frames(), (&alpha, DT_FS, RESOLUTION))
                .unwrap();
            let fit = ResonanceRamanSpectrum {
                incident_frequency_cm1: 0.0,
                temperature_k: 0.0,
                averaged: false,
            };
            std::hint::black_box(fit.fit((&raw.acf_iso, &raw.acf_aniso, DT_FS)).unwrap());
        })
    });
    group.finish();
}

criterion_group!(
    benches,
    bench_power_spectrum,
    bench_ir_spectrum,
    bench_raman_spectrum,
    bench_vcd_spectrum,
    bench_roa_spectrum,
    bench_resonance_raman_spectrum,
);
